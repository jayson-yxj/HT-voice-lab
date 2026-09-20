"""Causal PCM replay with persistent NeMo speaker cache and bounded audio buffer."""
import math
import time

import numpy as np
import torch

from common import ROOT, SR, Session, replay, sync


def feature_window(model, audio, raw_start_frame, first_frame, last_frame, device):
    signal = torch.from_numpy(audio.copy()).unsqueeze(0).to(device)
    length = torch.tensor([len(audio)], dtype=torch.long, device=device)
    features, _ = model.process_signal(signal, length)
    trim = first_frame - raw_start_frame
    selected = features[:, :, trim:trim + last_frame - first_frame]
    if selected.shape[2] != last_frame - first_frame:
        raise RuntimeError("Insufficient causal mel features")
    return selected.transpose(1, 2), torch.tensor([selected.shape[2]], device=device, dtype=torch.long)


class SortformerStream:
    """One session's speaker cache, accepting only PCM that has arrived."""

    def __init__(self, device, threshold=0.5):
        from nemo.collections.asr.models import SortformerEncLabelModel
        self.device, self.threshold = device, threshold
        checkpoint = ROOT / "models/sortformer/diar_streaming_sortformer_4spk-v2.1.nemo"
        self.model = SortformerEncLabelModel.restore_from(str(checkpoint), map_location=torch.device(device))
        model = self.model
        model.eval().to(device)
        model.streaming_mode = True
        model.async_streaming = False
        module = model.sortformer_modules
        module.chunk_len, module.chunk_right_context = 6, 7
        module.fifo_len, module.spkcache_update_period, module.spkcache_len = 188, 144, 188
        module._check_streaming_parameters()
        if model.cfg.preprocessor.normalize not in (None, "NA"):
            raise RuntimeError("Unexpected normalization: would invalidate streaming feature equivalence")
        model.preprocessor.featurizer.dither = 0.0
        self.config = dict(chunk_len=6, right_context=7, fifo_len=188,
                           speaker_cache=188, update_period=144,
                           chunk_left_context=module.chunk_left_context,
                           threshold=threshold, frame_seconds=0.08,
                           input_buffer_seconds=1.04, stft_guard_seconds=0.02,
                           feature_normalization="NA", dither=0, speaker_limit=4)
        self.state = module.init_streaming_state(batch_size=1, async_streaming=False, device=model.device)
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = self.next_frame = 0
        self.closed = False

    def push(self, audio, eof=False):
        if self.closed:
            raise ValueError("Cannot append audio after EOF")
        self.closed = eof
        self.buffer = np.concatenate((self.buffer, audio))
        received = self.buffer_start + len(self.buffer)
        model = self.model
        module = model.sortformer_modules
        hop, step = 160, module.chunk_len * 8
        left_limit, right_limit = module.chunk_left_context * 8, module.chunk_right_context * 8
        final_frames = math.ceil(received / hop)
        updates = []
        while self.next_frame < final_frames:
            end_frame = self.next_frame + step
            required = (end_frame + right_limit) * hop + 320
            if not eof and received < required:
                break
            if eof:
                end_frame = min(end_frame, final_frames)
            left = min(left_limit, self.next_frame)
            right = min(right_limit, final_frames - end_frame) if eof else right_limit
            first, last = self.next_frame - left, end_frame + right
            raw_start_frame = max(0, first - 3)
            raw_start = raw_start_frame * hop
            raw_end = min(received, last * hop + 320)
            if raw_start < self.buffer_start:
                raise RuntimeError("Audio buffer discarded required context")
            wave = self.buffer[raw_start - self.buffer_start:raw_end - self.buffer_start]
            begin = time.monotonic()
            with torch.inference_mode():
                features, lengths = feature_window(model, wave, raw_start_frame, first, last, self.device)
                empty_preds = torch.zeros((1, 0, 4), device=model.device)
                self.state, probabilities = model.forward_streaming_step(
                    processed_signal=features, processed_signal_length=lengths,
                    streaming_state=self.state, total_preds=empty_preds,
                    left_offset=left, right_offset=right)
                probabilities = probabilities[0].cpu().numpy()
            sync(self.device)
            output, frames = [], []
            for index, probs in enumerate(probabilities):
                a = self.next_frame * hop + index * 1280
                b = min(a + 1280, received, end_frame * hop)
                if b <= a:
                    continue
                speakers = [int(s) for s in np.flatnonzero(probs >= self.threshold)]
                frames.append(dict(start_sample=a, end_sample=b, speakers=speakers))
                for speaker in speakers:
                    output.append(dict(start=a / SR, end=b / SR, speaker="speaker{}".format(speaker)))
            committed = min(received, end_frame * hop)
            if not frames or frames[-1]["end_sample"] != committed:
                raise RuntimeError("Sortformer frames do not cover the committed interval")
            updates.append(dict(committed_sample=committed, frames=frames, segments=output,
                                compute_s=time.monotonic() - begin, final_flush=eof))
            self.next_frame = end_frame
            keep_from = max(0, (self.next_frame - left_limit - 3) * hop)
            drop = min(len(self.buffer), max(0, keep_from - self.buffer_start))
            self.buffer = self.buffer[drop:]
            self.buffer_start += drop
        return updates


def run(args):
    start = time.monotonic()
    stream = SortformerStream(args.device, args.threshold)
    sync(args.device)
    session = Session(args, time.monotonic() - start, stream.config)
    segments = []
    for audio, available, eof in replay(args.audio, 320, session):
        for update in stream.push(audio, eof):
            segments.extend(update["segments"])
            session.emit(available, update["committed_sample"] / SR, update["compute_s"],
                         final_flush=eof, segments=update["segments"])
    session.finish(segments=segments)
