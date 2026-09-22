"""Causal speaker-attributed streaming ASR; audio segment closure is not turn completion."""
from contextlib import ExitStack
import time

import numpy as np

from common import ROOT, SR, Session, merge_segments, replay, sync, write_json
from run_asr import SegmentASR, load_model
from run_sortformer import SortformerStream


class PyannoteVoiceEncoder:
    """Small CPU voiceprint encoder used only to link provisional speaker slots."""

    def __init__(self):
        import torch
        from diart.models import EmbeddingModel
        checkpoint = ROOT / "models/embedding/pytorch_model.bin"
        if not checkpoint.is_file():
            raise RuntimeError("Missing pyannote embedding model; run './lab fetch --only embedding'")
        self.torch = torch
        self.model = EmbeddingModel.from_pyannote(
            str(checkpoint), use_hf_token=False).to(torch.device("cpu")).eval()

    def __call__(self, audio):
        tensor = self.torch.from_numpy(audio.copy()).view(1, 1, -1)
        with self.torch.inference_mode():
            embedding = self.model(tensor)[0].float().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0:
            raise RuntimeError("Voice encoder returned an invalid embedding")
        return embedding / norm


class SpeakerStabilizer:
    """Delay new Sortformer slots until they match a voiceprint or persist."""

    def __init__(self, encoder, confirm_samples=round(1.5 * SR),
                 match_threshold=0.30, match_margin=0.15,
                 persistent_match_threshold=0.45,
                 max_hold_samples=4 * SR, prototype_samples=8 * SR,
                 max_speakers=4, recheck_samples=round(0.5 * SR),
                 appearance_gap_samples=round(0.8 * SR)):
        self.encoder = encoder
        self.confirm_samples = confirm_samples
        self.match_threshold = match_threshold
        self.persistent_match_threshold = persistent_match_threshold
        self.match_margin = match_margin
        self.max_hold_samples = max_hold_samples
        self.prototype_samples = prototype_samples
        self.max_speakers = max_speakers
        self.recheck_samples = recheck_samples
        self.appearance_gap_samples = appearance_gap_samples
        self.pending = []
        self.active = {}
        self.trusted_slots = {}
        self.prototypes = {}
        self.audit = []
        self.processed = 0
        self.closed = False
        self.config = dict(
            enabled=True, encoder="pyannote/embedding-cpu",
            confirm_seconds=confirm_samples / SR,
            match_threshold=match_threshold, match_margin=match_margin,
            persistent_match_threshold=persistent_match_threshold,
            recheck_seconds=recheck_samples / SR,
            appearance_gap_seconds=appearance_gap_samples / SR,
            max_hold_seconds=max_hold_samples / SR,
            prototype_seconds=prototype_samples / SR,
            short_policy="unresolved_without_person_promotion")

    @property
    def buffered_samples(self):
        return sum(len(item["audio"]) for item in self.pending)

    def _audio(self, appearance):
        if not appearance["audio"]:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(appearance["audio"])[:self.prototype_samples]

    def _embed(self, appearance):
        audio = self._audio(appearance)
        if not len(audio):
            return None
        return self.encoder(audio)

    def _apply(self, appearance, canonical, reason, scores=None):
        appearance["resolved"] = True
        appearance["canonical"] = canonical
        appearance["reason"] = reason
        appearance["scores"] = scores or {}
        for item in appearance["items"]:
            item["decisions"][appearance["slot"]] = canonical
            item["reasons"][appearance["slot"]] = reason

    def _decide(self, appearance, allow_new, allow_borderline_match=False):
        embedding = self._embed(appearance)
        scores = {canonical: float(np.dot(embedding, prototype))
                  for canonical, prototype in self.prototypes.items()} if embedding is not None else {}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best = ranked[0] if ranked else None
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        threshold = (self.match_threshold if allow_borderline_match else
                     self.persistent_match_threshold)
        if (best and best[1] >= threshold
                and best[1] - second >= self.match_margin):
            self._apply(appearance, best[0], "voiceprint_match", scores)
            return True
        appearance["scores"] = scores
        if not allow_new:
            return False
        if embedding is not None and len(self.prototypes) < self.max_speakers:
            canonical = len(self.prototypes)
            self.prototypes[canonical] = embedding
            self.trusted_slots[appearance["slot"]] = canonical
            self._apply(appearance, canonical, "confirmed_new_speaker", scores)
            return True
        self._apply(appearance, None, "unresolved_speaker_limit", scores)
        return True

    def _close(self, slot):
        appearance = self.active.pop(slot)
        if not appearance["resolved"]:
            if appearance["single_samples"] >= self.confirm_samples:
                self._decide(appearance, allow_new=True, allow_borderline_match=True)
        if not appearance["resolved"]:
            self._apply(appearance, None, "short_unresolved")
        canonical = appearance["canonical"]
        if (canonical is not None and self.trusted_slots.get(slot) == canonical
                and appearance["single_samples"] >= self.confirm_samples):
            embedding = self._embed(appearance)
            if embedding is not None:
                previous = self.prototypes[canonical]
                combined = previous * 0.25 + embedding * 0.75
                self.prototypes[canonical] = combined / np.linalg.norm(combined)
        self.audit.append(dict(
            start_s=appearance["start_sample"] / SR,
            end_s=appearance["end_sample"] / SR,
            raw_speaker="speaker{}".format(slot),
            speaker=("ABCD"[canonical] if canonical is not None else None),
            decision=appearance["reason"],
            single_speaker_audio_s=appearance["single_samples"] / SR,
            similarities={"ABCD"[key]: round(value, 4)
                          for key, value in appearance["scores"].items()}))

    def _drain(self):
        output = []
        while self.pending:
            item = self.pending[0]
            if any(slot not in item["decisions"] for slot in item["raw_speakers"]):
                break
            self.pending.pop(0)
            canonical = sorted({value for value in item["decisions"].values()
                                if value is not None})
            unresolved = any(value is None for value in item["decisions"].values())
            labels = ["ABCD"[value] for value in canonical]
            if unresolved:
                candidates = labels + ["ABCD"[value] for value in sorted(self.prototypes)
                                       if "ABCD"[value] not in labels]
                candidates.append("未确认")
                attribution = "short_unresolved"
            else:
                candidates = labels
                attribution = ("single_speaker" if len(labels) == 1
                               else "overlap_unresolved" if len(labels) > 1 else "silence")
            frame = dict(
                start_sample=item["start_sample"], end_sample=item["end_sample"],
                speakers=canonical, speaker_labels=labels,
                speaker_candidates=candidates, attribution=attribution,
                raw_speakers=list(item["raw_speakers"]),
                stabilization_reasons=dict(item["reasons"]))
            output.append((frame, item["audio"]))
        return output

    def push(self, frame, audio):
        start, end = frame["start_sample"], frame["end_sample"]
        raw = frame["speakers"]
        if self.closed or start != self.processed or end <= start or len(audio) != end - start:
            raise ValueError("Expected contiguous unstabilized speaker frames")
        if raw != sorted(set(raw)) or any(slot not in range(4) for slot in raw):
            raise ValueError("Expected distinct raw Sortformer slots 0..3")
        self.processed = end
        for slot, appearance in list(self.active.items()):
            if slot not in raw:
                appearance["gap_samples"] += end - start
                if raw or appearance["gap_samples"] >= self.appearance_gap_samples:
                    self._close(slot)
        item = dict(start_sample=start, end_sample=end, audio=audio.copy(),
                    raw_speakers=list(raw), decisions={}, reasons={})
        self.pending.append(item)
        for slot in raw:
            appearance = self.active.get(slot)
            if appearance is None:
                appearance = dict(slot=slot, start_sample=start, end_sample=end,
                                  items=[], audio=[], single_samples=0,
                                  next_check_samples=self.confirm_samples,
                                  gap_samples=0,
                                  resolved=False, canonical=None, reason=None, scores={})
                self.active[slot] = appearance
                if slot in self.trusted_slots:
                    self._apply(appearance, self.trusted_slots[slot], "trusted_slot")
            appearance["end_sample"] = end
            appearance["gap_samples"] = 0
            appearance["items"].append(item)
            if appearance["resolved"]:
                item["decisions"][slot] = appearance["canonical"]
                item["reasons"][slot] = appearance["reason"]
            if len(raw) == 1 and appearance["single_samples"] < self.prototype_samples:
                remaining = self.prototype_samples - appearance["single_samples"]
                piece = audio[:remaining].copy()
                appearance["audio"].append(piece)
                appearance["single_samples"] += len(piece)
            if (not appearance["resolved"] and appearance["single_samples"] >=
                    appearance["next_check_samples"]):
                allow_new = (not self.prototypes or
                             appearance["single_samples"] >= self.max_hold_samples)
                if not self._decide(appearance, allow_new):
                    appearance["next_check_samples"] += self.recheck_samples
            if (not appearance["resolved"] and
                    end - appearance["start_sample"] >= self.max_hold_samples and
                    appearance["single_samples"] < self.confirm_samples):
                self._apply(appearance, None, "unresolved_hold_limit")
        return self._drain()

    def finish(self):
        if self.closed:
            raise ValueError("Speaker stabilizer already closed")
        self.closed = True
        for slot in list(self.active):
            self._close(slot)
        output = self._drain()
        if self.pending:
            raise RuntimeError("Speaker stabilizer retained unresolved frames at EOF")
        return output

    def stats(self):
        counts = {}
        for item in self.audit:
            counts[item["decision"]] = counts.get(item["decision"], 0) + 1
        return dict(confirmed_speakers=len(self.prototypes), decisions=counts,
                    unresolved_appearances=sum(item["speaker"] is None for item in self.audit))


class AttributionRouter:
    """Route disjoint, committed PCM intervals to isolated ASR caches."""

    def __init__(self, make_asr, emit, silence_samples=12800,
                 window_pause_samples=None, window_limit_samples=None,
                 window_overlap_samples=0, on_complete=None):
        self.make_asr, self.emit = make_asr, emit
        self.on_complete = on_complete
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
            if self.on_complete:
                self.on_complete(self.completed)

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
        unresolved_short = frame.get("attribution") == "short_unresolved"
        if not slots and not unresolved_short:
            if self.current is not None:
                self.gap = np.concatenate((self.gap, audio))
                if len(self.gap) >= self.silence_samples:
                    self._close("silence_gap")
            return
        speakers = (list(frame["speaker_candidates"]) if unresolved_short else
                    list(frame.get("speaker_labels", ["ABCD"[s] for s in slots])))
        attribution = ("short_unresolved" if unresolved_short else
                       "single_speaker" if len(speakers) == 1 else "overlap_unresolved")
        raw_slots = list(frame.get("raw_speakers", slots))
        reasons = {str(slot): reason for slot, reason in
                   frame.get("stabilization_reasons", {}).items()}
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
                                speaker=speakers[0] if attribution == "single_speaker" else None,
                                speaker_candidates=speakers,
                                attribution=attribution, raw_speaker_slots=raw_slots,
                                stabilization_reasons=reasons,
                                start_sample=start, end_sample=start, text="",
                                timestamp_scope="source_audio_segment_not_word_timestamps",
                                turn_complete=None)
            self.asr = self.make_asr()
            if rollover is not None and len(rollover[0]):
                self.asr.prime(*rollover)
        else:
            self.current["raw_speaker_slots"] = sorted(set(
                self.current["raw_speaker_slots"] + raw_slots))
            self.current["stabilization_reasons"].update(reasons)
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
    stabilization_enabled = getattr(args, "speaker_stabilization", True)
    stabilizer = SpeakerStabilizer(PyannoteVoiceEncoder()) if stabilization_enabled else None
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
    stabilization_config = (stabilizer.config if stabilizer is not None else
                            dict(enabled=False, policy="raw_sortformer_slots"))
    routing = ("committed speaker frames -> voiceprint stabilization -> original PCM -> isolated segment ASR cache"
               if stabilizer is not None else
               "committed raw speaker frames -> original PCM -> isolated segment ASR cache")
    config = dict(diarizer=diarizer.config, asr=asr_config,
                  speaker_stabilization=stabilization_config,
                  warmup_performed=bool(asr_config.get("warmup")),
                  routing=routing,
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
    write_json(session.out / "speech_spans.json", [])
    router = AttributionRouter(
        make_asr, emit, silence_samples=round(args.silence_gap * SR),
        window_pause_samples=round(window_pause * SR) if window_pause is not None else None,
        window_limit_samples=round(window_limit * SR) if window_limit is not None else None,
        window_overlap_samples=round(window_overlap * SR),
        on_complete=lambda records: write_json(
            session.out / "speech_spans.json", merge_asr_windows(records)))
    pending = np.empty(0, dtype=np.float32)
    pending_start = max_pending = 0
    segments, raw_segments = [], []

    def route_frames(items):
        routed = []
        for frame, frame_audio in items:
            for speaker in frame["speakers"]:
                routed.append(dict(start=frame["start_sample"] / SR,
                                   end=frame["end_sample"] / SR,
                                   speaker="speaker{}".format(speaker)))
            router.push(frame, frame_audio)
        segments.extend(routed)
        return routed

    for audio, available, eof in replay(args.audio, 320, session):
        pending = np.concatenate((pending, audio))
        max_pending = max(max_pending, len(pending) + (
            stabilizer.buffered_samples if stabilizer is not None else 0))
        for update in diarizer.push(audio, eof):
            raw_segments.extend(update["segments"])
            session.emit(available, update["committed_sample"] / SR, update["compute_s"],
                         event="diarization_raw", final_flush=eof,
                         segments=update["segments"])
            stable_items = []
            stabilize_start = time.monotonic()
            for frame in update["frames"]:
                a, b = frame["start_sample"] - pending_start, frame["end_sample"] - pending_start
                if not 0 <= a < b <= len(pending):
                    raise RuntimeError("Diarization requested unavailable PCM")
                frame_audio = pending[a:b]
                if stabilizer is None:
                    stable_items.append((frame, frame_audio))
                else:
                    stable_items.extend(stabilizer.push(frame, frame_audio))
                max_pending = max(max_pending, len(pending) + (
                    stabilizer.buffered_samples if stabilizer is not None else 0))
            routed = route_frames(stable_items)
            stabilization_cost = time.monotonic() - stabilize_start
            committed = (stable_items[-1][0]["end_sample"] / SR
                         if stable_items else None)
            session.emit(available, committed, stabilization_cost,
                         event="diarization", final_flush=eof, segments=routed)
            drop = update["committed_sample"] - pending_start
            pending = pending[drop:]
            pending_start = update["committed_sample"]
    if len(pending):
        raise RuntimeError("Unprocessed audio remains at EOF")
    if stabilizer is not None:
        stabilize_start = time.monotonic()
        stable_items = stabilizer.finish()
        routed = route_frames(stable_items)
        session.emit(available,
                     stable_items[-1][0]["end_sample"] / SR if stable_items else None,
                     time.monotonic() - stabilize_start,
                     event="diarization", final_flush=True, segments=routed)
        write_json(session.out / "speaker_stabilization.json", dict(
            config=stabilizer.config, summary=stabilizer.stats(),
            appearances=stabilizer.audit))
        write_json(session.out / "raw_segments.json",
                   merge_segments(raw_segments, session.duration))
    if router.processed != round(session.duration * SR):
        raise RuntimeError("Stabilized audio remains unprocessed at EOF")
    router.finish()
    write_json(session.out / "utterances.json", router.completed)
    speech_spans = merge_asr_windows(router.completed)
    write_json(session.out / "speech_spans.json", speech_spans)
    session.finish(segments=segments, text="\n".join(format_segment(r) for r in speech_spans),
                   extra=dict(first_transcript_output_seconds=first_text,
                              utterance_count=len(router.completed),
                              speech_span_count=len(speech_spans),
                              unresolved_overlap_segments=sum(
                                  r["attribution"] == "overlap_unresolved"
                                  for r in router.completed),
                              unresolved_short_segments=sum(
                                  r["attribution"] == "short_unresolved"
                                  for r in router.completed),
                              speaker_stabilization=(
                                  stabilizer.stats() if stabilizer is not None else
                                  dict(enabled=False)),
                              max_pending_audio_seconds=max_pending / SR,
                              output_delay_scope="diarization watermark only; transcript delay not word-aligned"))
