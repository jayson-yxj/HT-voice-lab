"""Causal speaker-attributed streaming ASR; audio segment closure is not turn completion."""
from contextlib import ExitStack
import time

import numpy as np

from common import SR, Session, replay, sync, write_json
from run_asr import SegmentASR, load_model
from run_sortformer import SortformerStream


class AttributionRouter:
    """Route disjoint, committed PCM intervals to isolated ASR caches."""

    def __init__(self, make_asr, emit, silence_samples=12800,
                 window_pause_samples=None, window_limit_samples=None,
                 window_overlap_samples=0):
        self.make_asr, self.emit = make_asr, emit
        self.silence_samples = silence_samples
        self.window_pause_samples = window_pause_samples
        self.window_limit_samples = window_limit_samples
        self.window_overlap_samples = window_overlap_samples
        self.processed = 0
        self.current = self.asr = None
        self.gap = np.empty(0, dtype=np.float32)
        self.history = np.empty(0, dtype=np.float32)
        self.completed = []
        self.closed = False

    def _publish(self, update, final=False, reason=None):
        snapshot = update.get("text_mode") == "snapshot"
        if snapshot:
            self.current["text"] = update["text"]
        else:
            self.current["text"] += update["text"]
        if "language" in update:
            self.current["language"] = update["language"]
        self.current["revision"] += 1
        record = dict(self.current, status="final" if final else "partial",
                      start_s=self.current["start_sample"] / SR,
                      end_s=self.current["end_sample"] / SR,
                      close_reason=reason, text_delta=None if snapshot else update["text"],
                      text_update="replace" if snapshot else "append",
                      final_padding_samples=update.get("padding_samples", 0))
        self.emit(record, update["compute_s"])
        if final:
            self.completed.append(record)

    def _close(self, reason, keep_overlap=False):
        overlap = self.history.copy() if keep_overlap else np.empty(0, dtype=np.float32)
        previous_text = ""
        if self.current is not None:
            self._publish(self.asr.finish(), final=True, reason=reason)
            previous_text = self.current["text"]
        self.current = self.asr = None
        self.gap = np.empty(0, dtype=np.float32)
        self.history = np.empty(0, dtype=np.float32)
        return overlap, previous_text

    def _accept(self, audio, end):
        self.current["end_sample"] = end
        if self.window_overlap_samples:
            self.history = np.concatenate((self.history, audio))[-self.window_overlap_samples:]
        for update in self.asr.push(audio):
            self._publish(update)

    def push(self, frame, audio):
        start, end = frame["start_sample"], frame["end_sample"]
        slots = frame["speakers"]
        if self.closed or start != self.processed or end <= start or len(audio) != end - start:
            raise ValueError("Expected contiguous PCM frames before EOF")
        if slots != sorted(set(slots)) or any(s not in range(4) for s in slots):
            raise ValueError("Expected distinct Sortformer slots 0..3")
        self.processed = end
        if not slots:
            if self.current is not None:
                self.gap = np.concatenate((self.gap, audio))
                if len(self.gap) >= self.silence_samples:
                    self._close("silence_gap")
            return
        speakers = ["ABCD"[s] for s in slots]
        rollover = None
        if self.current is not None and self.current["speaker_candidates"] != speakers:
            reason = "speaker_change" if len(speakers) == len(self.current["speaker_candidates"]) == 1 else "overlap_boundary"
            self._close(reason)
        elif self.current is not None:
            duration = self.current["end_sample"] - self.current["start_sample"]
            if len(self.gap) and self.window_pause_samples is not None and duration >= self.window_pause_samples:
                rollover = self._close("asr_window_pause", keep_overlap=True)
            elif self.window_limit_samples is not None and end - self.current["start_sample"] > self.window_limit_samples:
                rollover = self._close("asr_window_limit", keep_overlap=True)
        if self.current is None:
            self.current = dict(id="segment-{:06d}".format(len(self.completed) + 1), revision=0,
                                speaker=speakers[0] if len(speakers) == 1 else None,
                                speaker_candidates=speakers,
                                attribution="single_speaker" if len(speakers) == 1 else "overlap_unresolved",
                                start_sample=start, end_sample=start, text="",
                                timestamp_scope="source_audio_segment_not_word_timestamps",
                                turn_complete=None)
            self.asr = self.make_asr()
            if rollover is not None and len(rollover[0]):
                self.asr.prime(*rollover)
        if len(self.gap):
            self._accept(self.gap, start)
            self.gap = np.empty(0, dtype=np.float32)
        self._accept(audio, end)

    def finish(self):
        if self.closed:
            raise ValueError("Conversation already closed")
        self.closed = True
        self._close("eof")


def format_segment(record):
    label = record["speaker"] or "未分离({})".format("+".join(record["speaker_candidates"]))
    return "[{:.2f}–{:.2f}] {}: {}".format(record["start_s"], record["end_s"], label, record["text"])


def join_window_text(previous, current):
    previous, current = previous.rstrip(), current.lstrip()
    if not previous or not current:
        return previous + current
    if current[0].isascii() and current[0].islower():
        return previous.rstrip(".!?") + " " + current
    if current[0].isascii() and current[0].isalpha() and (
            previous[-1] in ".!?" or previous[-1].isascii() and previous[-1].isalnum()):
        return previous + " " + current
    return previous + current


def merge_asr_windows(records):
    """Merge artificial ASR windows while keeping semantic completion unknown."""
    spans = []
    for record in records:
        if (spans and spans[-1]["close_reason"] in {"asr_window_pause", "asr_window_limit"}
                and spans[-1]["speaker_candidates"] == record["speaker_candidates"]):
            span = spans[-1]
            span["text"] = join_window_text(span["text"], record["text"])
            span["end_sample"] = record["end_sample"]
            span["end_s"] = record["end_s"]
            span["close_reason"] = record["close_reason"]
            span["source_segment_ids"].append(record["id"])
            span["acoustic_window_count"] += 1
            continue
        span = dict(record)
        span["id"] = "span-{:06d}".format(len(spans) + 1)
        span["source_segment_ids"] = [record["id"]]
        span["acoustic_window_count"] = 1
        spans.append(span)
    return spans


def run(args):
    with ExitStack() as resources:
        _run(args, resources)


def _run(args, resources):
    start = time.monotonic()
    diarizer = SortformerStream(args.device, args.threshold)
    asr_name = getattr(args, "asr_model", "paraformer")
    if asr_name == "qwen3":
        if args.device != "cuda":
            raise ValueError("This Qwen vLLM evaluation requires CUDA")
        from qwen_client import QwenWorker, QwenSegmentASR
        worker = QwenWorker(args.qwen_chunk_seconds)
        resources.callback(worker.close)
        make_asr = lambda: QwenSegmentASR(worker)
        asr_config = dict(name="qwen3-asr-0.6b", **worker.config,
                          bounded_window=dict(pause_seconds=args.qwen_window_pause,
                                              limit_seconds=args.qwen_window_limit,
                                              overlap_seconds=args.qwen_window_overlap),
                          worker_log=str(worker.log_path))
    else:
        model = load_model(args.device)
        make_asr = lambda: SegmentASR(model, args.device)
        asr_config = dict(name="paraformer", chunk_size=[0, 10, 5])
    sync(args.device)
    config = dict(diarizer=diarizer.config, asr=asr_config,
                  warmup_performed=bool(asr_config.get("warmup")),
                  routing="committed speaker frames -> original PCM -> isolated segment ASR cache",
                  silence_gap_seconds=args.silence_gap, speaker_labels={str(i): c for i, c in enumerate("ABCD")},
                  overlap="transcribe once, speaker=null, retain all candidate speakers",
                  timestamps="source audio segments; no word timestamps",
                  final_semantics="ASR segment closed; turn_complete remains null",
                  asr_short_tail_padding="Paraformer: up to 959 zeros at flush; Qwen: no padding; excluded from source timestamps",
                  gpu_metric_scope="parent process PyTorch only; excludes Qwen/vLLM worker")
    session = Session(args, time.monotonic() - start, config)
    available, eof, first_text = 0.0, False, None

    def emit(record, compute_seconds):
        nonlocal first_text
        if first_text is None and record["text"]:
            first_text = time.monotonic() - session.origin
        session.emit(available, None, compute_seconds, final_flush=eof,
                     event="transcript", text=record["text"], utterance=record)
        if record["status"] == "final":
            print(format_segment(record), flush=True)

    window_pause = args.qwen_window_pause if asr_name == "qwen3" else None
    window_limit = args.qwen_window_limit if asr_name == "qwen3" else None
    window_overlap = args.qwen_window_overlap if asr_name == "qwen3" else 0
    router = AttributionRouter(
        make_asr, emit, silence_samples=round(args.silence_gap * SR),
        window_pause_samples=round(window_pause * SR) if window_pause is not None else None,
        window_limit_samples=round(window_limit * SR) if window_limit is not None else None,
        window_overlap_samples=round(window_overlap * SR))
    pending = np.empty(0, dtype=np.float32)
    pending_start = max_pending = 0
    segments = []
    for audio, available, eof in replay(args.audio, 320, session):
        pending = np.concatenate((pending, audio))
        max_pending = max(max_pending, len(pending))
        for update in diarizer.push(audio, eof):
            segments.extend(update["segments"])
            session.emit(available, update["committed_sample"] / SR, update["compute_s"],
                         event="diarization", final_flush=eof, segments=update["segments"])
            for frame in update["frames"]:
                a, b = frame["start_sample"] - pending_start, frame["end_sample"] - pending_start
                if not 0 <= a < b <= len(pending):
                    raise RuntimeError("Diarization requested unavailable PCM")
                router.push(frame, pending[a:b])
            drop = update["committed_sample"] - pending_start
            pending = pending[drop:]
            pending_start = update["committed_sample"]
    if len(pending) or router.processed != round(session.duration * SR):
        raise RuntimeError("Unprocessed audio remains at EOF")
    router.finish()
    write_json(session.out / "utterances.json", router.completed)
    speech_spans = merge_asr_windows(router.completed)
    write_json(session.out / "speech_spans.json", speech_spans)
    session.finish(segments=segments, text="\n".join(format_segment(r) for r in speech_spans),
                   extra=dict(first_transcript_output_seconds=first_text,
                              utterance_count=len(router.completed),
                              speech_span_count=len(speech_spans),
                              unresolved_overlap_segments=sum(r["speaker"] is None for r in router.completed),
                              max_pending_audio_seconds=max_pending / SR,
                              output_delay_scope="diarization watermark only; transcript delay not word-aligned"))
