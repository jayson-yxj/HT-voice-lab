"""Checks that speaker routing neither leaks PCM across identities nor invents turn completion."""
import numpy as np

from run_asr import SegmentASR, text_delta
from run_conversation import (AttributionRouter, SpeakerStabilizer,
                              join_window_text, merge_asr_windows)
from qwen_client import QwenSegmentASR, remove_overlap_prefix


def check_conversation():
    assert remove_overlap_prefix("affordable too, especially when it.",
                                 "Especially when it comes to cost.") == "comes to cost."
    assert remove_overlap_prefix("serve a more digitized and a more automated.",
                                 "Digitise and a more automated production system.") == "production system."
    assert remove_overlap_prefix("日本早稻田开始做人形机器人。",
                                 "人形机器人现在会走路。") == "现在会走路。"
    assert remove_overlap_prefix("完全不同", "新的内容") == "新的内容"
    assert join_window_text("especially when it.", "comes to cost.") == "especially when it comes to cost."
    assert join_window_text("second condition.", "The third one") == "second condition. The third one"
    assert text_delta("we need", "to work") == " to work"
    assert text_delta("人形", "机器人") == "机器人"
    assert text_delta("hello", "，world") == "，world"
    assert text_delta("", "hello") == "hello"
    assert text_delta("hello", "") == ""
    created, events = [], []

    class CaptureASR:
        def __init__(self):
            self.audio = []
            created.append(self)

        def push(self, audio):
            self.audio.append(audio.copy())
            return [dict(text="增量", compute_s=0)]

        def finish(self):
            return dict(text="尾字", compute_s=0)

        def prime(self, audio, previous_text):
            self.primed = (audio.copy(), previous_text)

    completions = []
    router = AttributionRouter(CaptureASR, lambda record, cost: events.append(record),
                               silence_samples=7680,
                               on_complete=lambda records: completions.append(len(records)))

    def feed(start, end, speakers, value, **metadata):
        frame = dict(start_sample=start, end_sample=end, speakers=speakers)
        frame.update(metadata)
        router.push(frame,
                    np.full(end - start, value, dtype=np.float32))

    feed(0, 3200, [0], 1)
    feed(3200, 4800, [], 0)
    feed(4800, 8000, [0], 1)
    feed(8000, 12800, [1], 2)
    feed(12800, 20480, [], 0)
    feed(20480, 22400, [0], 1)
    router.finish()
    assert [r["speaker"] for r in router.completed] == ["A", "B", "A"]
    assert [(r["start_sample"], r["end_sample"]) for r in router.completed] == [(0, 8000), (8000, 12800), (20480, 22400)]
    assert [r["close_reason"] for r in router.completed] == ["speaker_change", "silence_gap", "eof"]
    assert completions == [1, 2, 3]
    assert all(r["turn_complete"] is None for r in events)
    assert all(r["text"].endswith("尾字") for r in router.completed)
    np.testing.assert_array_equal(np.concatenate(created[0].audio), np.r_[np.ones(3200), np.zeros(1600), np.ones(3200)])
    assert np.all(np.concatenate(created[1].audio) == 2)
    assert np.all(np.concatenate(created[2].audio) == 1)
    for segment in router.completed:
        revisions = [e for e in events if e["id"] == segment["id"]]
        assert [r["revision"] for r in revisions] == list(range(1, len(revisions) + 1))
        assert sum(r["status"] == "final" for r in revisions) == 1
        assert "".join(r["text_delta"] for r in revisions) == segment["text"]

    router = AttributionRouter(CaptureASR, lambda record, cost: None)
    feed(0, 1600, [0], 1)
    feed(1600, 3200, [0, 1], 3)
    feed(3200, 4800, [1], 2)
    router.finish()
    overlap = router.completed[1]
    assert overlap["speaker"] is None and overlap["speaker_candidates"] == ["A", "B"]
    assert overlap["attribution"] == "overlap_unresolved"
    assert len(router.completed) == 3
    router = AttributionRouter(CaptureASR, lambda record, cost: None)
    feed(0, 1600, [0], 1)
    feed(1600, 12800, [], 0)  # A natural 0.7s pause must keep this ASR context.
    feed(12800, 14400, [0], 1)
    assert not router.completed
    router.finish()
    assert len(router.completed) == 1 and router.completed[0]["end_sample"] == 14400
    router = AttributionRouter(CaptureASR, lambda record, cost: None)
    feed(0, 320, [], 0)
    router.finish()
    assert router.completed == []
    try:
        feed(320, 640, [0], 1)
        raise AssertionError("Accepted audio after EOF")
    except ValueError:
        pass

    # Provisional slots need enough clean speech to create a person. A reused raw
    # slot can match an existing voice, while a short fragment remains auditable.
    def fake_embedding(audio):
        index = int(round(float(np.mean(audio)))) - 1
        result = np.zeros(3, dtype=np.float32)
        result[index] = 1
        return result

    stabilizer = SpeakerStabilizer(
        fake_embedding, confirm_samples=4, max_hold_samples=8,
        prototype_samples=8, match_threshold=0.3, match_margin=0.15,
        appearance_gap_samples=0)
    stabilized = []

    def stabilize(start, end, speakers, value):
        stabilized.extend(stabilizer.push(
            dict(start_sample=start, end_sample=end, speakers=speakers),
            np.full(end - start, value, dtype=np.float32)))

    stabilize(0, 4, [0], 1)       # Confirm A.
    stabilize(4, 5, [], 0)
    stabilize(5, 9, [1], 2)       # Confirm B.
    stabilize(9, 10, [], 0)
    stabilize(10, 14, [2], 2)     # Raw C is actually B.
    stabilize(14, 15, [], 0)
    stabilize(15, 17, [2], 1)     # Too short to assign safely.
    stabilize(17, 18, [], 0)
    stabilize(18, 22, [3], 3)     # Confirm a real third person.
    stabilized.extend(stabilizer.finish())
    frames = [frame for frame, _ in stabilized if frame["raw_speakers"]]
    assert [frame["speakers"] for frame in frames] == [[0], [1], [1], [], [2]]
    assert frames[2]["stabilization_reasons"][2] == "voiceprint_match"
    assert frames[3]["attribution"] == "short_unresolved"
    assert stabilizer.stats()["confirmed_speakers"] == 3
    assert stabilizer.stats()["unresolved_appearances"] == 1

    # Borderline voiceprints only merge short Sortformer fragments. A raw slot
    # that persists to the hold limit becomes a new person unless the match is
    # strong; this prevents two similar real speakers from collapsing together.
    def borderline_embedding(audio):
        if np.mean(audio) < 1.5:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.32, np.sqrt(1 - 0.32 ** 2)], dtype=np.float32)

    borderline = SpeakerStabilizer(
        borderline_embedding, confirm_samples=2, recheck_samples=1,
        max_hold_samples=4, prototype_samples=8, appearance_gap_samples=0)
    borderline.push(dict(start_sample=0, end_sample=2, speakers=[0]),
                    np.ones(2, dtype=np.float32))
    borderline.push(dict(start_sample=2, end_sample=3, speakers=[]),
                    np.zeros(1, dtype=np.float32))
    borderline.push(dict(start_sample=3, end_sample=6, speakers=[1]),
                    np.full(3, 2, dtype=np.float32))
    borderline.push(dict(start_sample=6, end_sample=7, speakers=[]),
                    np.zeros(1, dtype=np.float32))
    assert borderline.audit[-1]["decision"] == "voiceprint_match"
    borderline.push(dict(start_sample=7, end_sample=9, speakers=[2]),
                    np.full(2, 2, dtype=np.float32))
    persistent = borderline.push(dict(start_sample=9, end_sample=11, speakers=[2]),
                                 np.full(2, 2, dtype=np.float32))
    assert [frame["speakers"] for frame, _ in persistent
            if frame["raw_speakers"]] == [[1], [1]]
    borderline.finish()
    assert borderline.audit[-1]["decision"] == "confirmed_new_speaker"
    assert borderline.stats()["confirmed_speakers"] == 2

    # A provisional slot that misses at 1.5s must be rechecked before it is
    # allowed to create a new identity.
    def improving_embedding(audio):
        if np.mean(audio) < 1.5 or len(audio) >= 8:
            return np.array([1, 0], dtype=np.float32)
        return np.array([0, 1], dtype=np.float32)

    retrying = SpeakerStabilizer(
        improving_embedding, confirm_samples=4, recheck_samples=4,
        max_hold_samples=8, prototype_samples=8, appearance_gap_samples=2)
    retrying.push(dict(start_sample=0, end_sample=4, speakers=[0]),
                  np.ones(4, dtype=np.float32))
    retrying.push(dict(start_sample=4, end_sample=5, speakers=[]),
                  np.zeros(1, dtype=np.float32))
    assert retrying.push(dict(start_sample=5, end_sample=7, speakers=[1]),
                         np.full(2, 2, dtype=np.float32)) == []
    assert retrying.push(dict(start_sample=7, end_sample=8, speakers=[]),
                         np.zeros(1, dtype=np.float32)) == []
    assert retrying.push(dict(start_sample=8, end_sample=10, speakers=[1]),
                         np.full(2, 2, dtype=np.float32)) == []
    retried = retrying.push(dict(start_sample=10, end_sample=14, speakers=[1]),
                            np.full(4, 2, dtype=np.float32))
    assert [frame["speakers"] for frame, _ in retried] == [[0], [], [0], [0]]
    retrying.finish()
    assert retrying.stats()["confirmed_speakers"] == 1
    assert retrying.audit[-1]["decision"] == "voiceprint_match"

    router = AttributionRouter(CaptureASR, lambda record, cost: None)
    feed(0, 1600, [], 3, attribution="short_unresolved",
         speaker_candidates=["A", "B", "未确认"], raw_speakers=[2],
         stabilization_reasons={2: "short_unresolved"})
    router.finish()
    unresolved = router.completed[0]
    assert unresolved["speaker"] is None
    assert unresolved["attribution"] == "short_unresolved"
    assert unresolved["raw_speaker_slots"] == [2]

    # Bound acoustic ASR state without claiming that the speaker's semantic turn ended.
    limited_events = []
    router = AttributionRouter(CaptureASR, lambda record, cost: limited_events.append(record),
                               silence_samples=8000, window_pause_samples=3200,
                               window_limit_samples=4800, window_overlap_samples=800)
    feed(0, 1600, [0], 1)
    feed(1600, 3200, [0], 1)
    feed(3200, 4000, [], 0)
    feed(4000, 4800, [0], 1)
    feed(4800, 9600, [0], 1)
    router.finish()
    assert [r["speaker"] for r in router.completed] == ["A", "A", "A"]
    assert [r["close_reason"] for r in router.completed] == [
        "asr_window_pause", "asr_window_limit", "eof"]
    assert all(r["turn_complete"] is None for r in limited_events)
    assert [(r["start_sample"], r["end_sample"]) for r in router.completed] == [
        (0, 3200), (4000, 4800), (4800, 9600)]
    np.testing.assert_array_equal(created[-2].primed[0], np.ones(800))
    assert created[-2].primed[1].endswith("尾字")
    spans = merge_asr_windows(router.completed)
    assert len(spans) == 1 and spans[0]["source_segment_ids"] == [
        "segment-000001", "segment-000002", "segment-000003"]
    assert spans[0]["turn_complete"] is None and spans[0]["acoustic_window_count"] == 3
    router = AttributionRouter(CaptureASR, lambda record, cost: None)
    try:
        feed(320, 640, [0], 1)
        raise AssertionError("Accepted a gap in PCM coverage")
    except ValueError:
        pass

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, input, cache, is_final, **kwargs):
            self.calls.append((input.copy(), cache, is_final))
            cache["calls"] = cache.get("calls", 0) + 1
            return [dict(text="尾" if is_final else "字")]

    model = FakeModel()
    first = SegmentASR(model, "cpu")
    waveform = np.arange(9601, dtype=np.float32)
    updates = first.push(waveform[:7000]) + first.push(waveform[7000:])
    updates.append(first.finish())
    assert "".join(u["text"] for u in updates) == "字尾"
    assert updates[-1]["padding_samples"] == 959
    np.testing.assert_array_equal(model.calls[0][0], waveform[:9600])
    assert model.calls[1][0][0] == waveform[-1]
    assert np.all(model.calls[1][0][1:] == 0)
    assert model.calls[0][1] is model.calls[1][1]
    second = SegmentASR(model, "cpu")
    second.push(np.ones(9600, dtype=np.float32))
    second.finish()
    assert model.calls[2][1] is not model.calls[0][1]
    assert len(model.calls[3][0]) == 0 and model.calls[3][2]
    short = SegmentASR(model, "cpu")
    short.push(np.ones(17, dtype=np.float32))
    tail = short.finish()
    assert tail["padding_samples"] == 943 and len(model.calls[-1][0]) == 960
    model.generate = lambda **kwargs: [dict(text="world" if kwargs["is_final"] else "hello")]
    english = SegmentASR(model, "cpu")
    delta = english.push(np.ones(9600, dtype=np.float32))[0]["text"]
    assert delta + english.finish()["text"] == "hello world"

    # Qwen revises words and punctuation; a revised snapshot must replace, not append.
    class SnapshotASR:
        def push(self, audio):
            return [dict(text="人性", text_mode="snapshot", language="Chinese", compute_s=0),
                    dict(text="人形机器人，", text_mode="snapshot", language="Chinese", compute_s=0)]

        def finish(self):
            return dict(text="人形机器人。", text_mode="snapshot", language="Chinese", compute_s=0)

    snapshots = []
    router = AttributionRouter(SnapshotASR, lambda record, cost: snapshots.append(record))
    feed(0, 1280, [0], 1)
    router.finish()
    assert router.completed[0]["text"] == "人形机器人。"
    assert all(r["text_delta"] is None and r["text_update"] == "replace" for r in snapshots)
    assert all(r["turn_complete"] is None for r in snapshots)
    assert [r["revision"] for r in snapshots] == [1, 2, 3]

    class FakeWorker:
        chunk_samples = 4

        def __init__(self):
            self.calls = []

        def request(self, op, audio=None):
            self.calls.append((op, None if audio is None else audio.copy()))
            return dict(text="test", text_mode="snapshot", compute_s=0)

    worker = FakeWorker()
    segment = QwenSegmentASR(worker)
    assert segment.push(np.arange(3, dtype=np.float32)) == []
    assert len(segment.push(np.arange(3, 9, dtype=np.float32))) == 2
    segment.finish()
    assert [op for op, _ in worker.calls] == ["start", "push", "push", "finish"]
    np.testing.assert_array_equal(np.concatenate([a for _, a in worker.calls[1:]]), np.arange(9))
    # Exact chunk boundary still gets a final snapshot, without adding synthetic PCM.
    segment = QwenSegmentASR(worker)
    segment.push(np.ones(4, dtype=np.float32))
    segment.finish()
    assert worker.calls[-1][0] == "finish" and len(worker.calls[-1][1]) == 0
    print("PASS: speaker/overlap PCM isolation, returning identity, silence/EOF semantics, revisions, ASR short tails/cache isolation")


if __name__ == "__main__":
    check_conversation()
