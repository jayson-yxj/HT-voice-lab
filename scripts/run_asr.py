import time

import numpy as np

from common import ROOT, SR, Session, replay, sync


def load_model(device):
    from funasr import AutoModel
    return AutoModel(model=str(ROOT / "models/paraformer"), device=device,
                     disable_update=True, disable_pbar=True)


def text_delta(previous, incoming):
    # FunASR strips the boundary spaces from each postprocessed chunk.
    if previous and incoming and all(c.isascii() and c.isalpha() for c in (previous[-1], incoming[0])):
        return " " + incoming
    return incoming


class SegmentASR:
    """Independent decoding cache for one contiguous attributed audio segment."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.cache = {}
        self.last_character = ""
        self.pending = np.empty(0, dtype=np.float32)
        self.closed = False

    def _decode(self, audio, final=False, padding=0):
        start = time.monotonic()
        result = self.model.generate(input=audio, cache=self.cache, is_final=final,
                                     chunk_size=[0, 10, 5], encoder_chunk_look_back=4,
                                     decoder_chunk_look_back=1, disable_pbar=True)
        sync(self.device)
        text = text_delta(self.last_character, "".join(x.get("text", "") for x in result))
        if text:
            self.last_character = text[-1]
        return dict(text=text,
                    compute_s=time.monotonic() - start, padding_samples=padding)

    def push(self, audio):
        if self.closed:
            raise ValueError("ASR segment already closed")
        self.pending = np.concatenate((self.pending, audio))
        updates = []
        while len(self.pending) >= 9600:
            updates.append(self._decode(self.pending[:9600]))
            self.pending = self.pending[9600:]
        return updates

    def finish(self):
        if self.closed:
            raise ValueError("ASR segment already closed")
        self.closed = True
        # FunASR 1.4.15 treats final input shorter than 960 samples as a cache-only
        # flush, dropping that input. Pad a nonempty short tail so it is decoded.
        padding = max(0, 960 - len(self.pending)) if len(self.pending) else 0
        audio = np.pad(self.pending, (0, padding))
        update = self._decode(audio, final=True, padding=padding)
        self.pending = np.empty(0, dtype=np.float32)
        self.cache = {}
        return update


def run(args):
    start = time.monotonic()
    model = load_model(args.device)
    sync(args.device)
    session = Session(args, time.monotonic() - start, dict(chunk_size=[0, 10, 5], sample_rate=SR))
    cache, texts, last_character = {}, [], ""
    for audio, end, last in replay(args.audio, 9600, session):
        start = time.monotonic()
        result = model.generate(input=audio, cache=cache, is_final=last,
                                chunk_size=[0, 10, 5], encoder_chunk_look_back=4,
                                decoder_chunk_look_back=1, disable_pbar=True)
        sync(args.device)
        text = text_delta(last_character, "".join(x.get("text", "") for x in result))
        if text:
            last_character = text[-1]
        texts.append(text)
        session.emit(end, None, time.monotonic() - start, text=text,
                     timestamp_scope="input progress only; not word timestamps", final=last)
        if text:
            print(text, flush=True)
    session.finish(text="".join(texts))
