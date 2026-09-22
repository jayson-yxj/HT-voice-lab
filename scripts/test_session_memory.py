"""Checks evidence-preserving compression of completed speaker turns."""
import json
from pathlib import Path
import tempfile

from dialogue_state import DialogueStateTracker, analyze
from live_demo import LiveMemoryUpdater
from memory_graph import project
from session_memory import GroqMemoryCompressor, build, ensure_open_tail_question


def _span(identifier, speaker, start, end, text):
    return dict(id=identifier, speaker=speaker, start_s=start, end_s=end,
                text=text, close_reason="speaker_change")


class _Judge:
    name = "test-judge"

    def assess(self, **item):
        complete = item["text"] in {"B完整意见", "成本太高"}
        return dict(turn_complete=complete, confidence=1.0,
                    reason="fixture", decision_source=self.name)


class _CompleteJudge:
    name = "complete-test-judge"

    def assess(self, **_item):
        return dict(turn_complete=True, confidence=1.0,
                    reason="fixture", decision_source=self.name)


class _LiveCompressor:
    name = "test-live-memory"
    model = "test-model"

    def __init__(self):
        self.calls = []

    def compress(self, turns):
        self.calls.append([item["id"] for item in turns])
        ids = self.calls[-1]
        return dict(
            session_summary="实时演示摘要",
            topics=[dict(name="实时演示", summary="逐步更新。",
                         evidence_turn_ids=ids)],
            open_questions=[], decisions=[], action_items=[],
            turn_memories=[dict(
                turn_id=item["id"], summary=item["text"], speech_act="statement",
                intent="表达观点", stance="neutral", topic_names=["实时演示"], entities=[])
                for item in turns]), dict(latency_s=0.0)


def check_session_memory():
    dialogue = analyze([
        _span("s1", "A", 0, 2, "A认为主要问题是"),
        _span("s2", "B", 2, 4, "B完整意见"),
        _span("s3", "A", 4, 6, "成本太高"),
        _span("s4", "C", 6, 8, "C还有一点"),
    ], semantic_judge=_Judge())

    def transport(payload):
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["max_completion_tokens"] == 950
        content = {
            "session_summary": "B表达意见，A认为成本太高。",
            "topics": [{"name": "方案成本", "summary": "A关注成本。",
                        "evidence_turn_ids": ["turn-000002"]}],
            "open_questions": [], "decisions": [], "action_items": [],
            "turn_memories": [
                {"turn_id": "turn-000001", "summary": "B表达完整意见。",
                 "speech_act": "statement", "intent": "表达意见", "stance": "neutral",
                 "topic_names": [], "entities": []},
                {"turn_id": "turn-000002", "summary": "A认为方案成本太高。",
                 "speech_act": "statement", "intent": "指出风险", "stance": "oppose",
                 "topic_names": ["方案成本"], "entities": []},
            ],
        }
        return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

    compressor = GroqMemoryCompressor(transport=transport)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "dialogue_state.json"
        source.write_text(json.dumps(dialogue, ensure_ascii=False))
        memory = build(dialogue, source, "test-session", compressor)
    assert [item["speaker"] for item in memory["turns"]] == ["B", "A"]
    assert memory["turns"][1]["contribution_ids"] == [
        "contribution-000001", "contribution-000003"]
    assert "text" not in memory["turns"][1]
    assert memory["working_memory"][0]["speaker"] == "C"
    assert memory["working_memory"][0]["text"] == "C还有一点"
    assert memory["scene"]["topics"][0]["evidence_turn_ids"] == ["turn-000002"]
    assert memory["stats"]["completed_turns"] == 2

    calls = []
    def batch_transport(payload):
        request = json.loads(payload["messages"][1]["content"])
        calls.append(request)
        if "turns" in request:
            ids = [item["id"] for item in request["turns"]]
            content = {
                "session_summary": "分批摘要", "topics": [], "open_questions": [],
                "decisions": [], "action_items": [],
                "turn_memories": [{
                    "turn_id": identifier, "summary": identifier,
                    "speech_act": "statement", "intent": "说明",
                    "stance": "neutral", "topic_names": ["旧话题"], "entities": []}
                    for identifier in ids],
            }
        else:
            content = {
                "session_summary": "整场摘要",
                "topics": [{"name": "统一话题", "summary": "整场共同讨论。",
                            "evidence_turn_ids": ["turn-000001", "turn-000005"]}],
                "open_questions": [], "decisions": [], "action_items": [],
            }
        return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}

    batched, _ = GroqMemoryCompressor(transport=batch_transport).compress([
        dict(id="turn-{:06d}".format(index), speaker="A", start_s=index,
             end_s=index + 1, text="测试") for index in range(1, 6)])
    assert len(calls) == 3
    assert len(batched["turn_memories"]) == 5
    assert all(item["topic_names"] == ["统一话题"]
               for item in batched["turn_memories"])
    assert batched["topics"][0]["evidence_turn_ids"] == [
        "turn-000001", "turn-000002", "turn-000003", "turn-000004", "turn-000005"]
    graph = project(memory)
    assert graph["counts"] == {
        "speakers": 3, "topics": 2, "opinions": 2, "intents": 2, "pending": 1}
    assert {item["type"] for item in graph["nodes"]} == {
        "speaker", "topic", "opinion", "intent"}
    assert [(item["type"], item["turn_id"]) for item in graph["edges"]].count(
        ("EXPRESSES", "turn-000002")) == 1
    assert not any(item["type"] == "opinion" and "C还有一点" in item["label"]
                   for item in graph["nodes"])
    memory["working_memory"].append(dict(
        speaker="unresolved", start_s=8, end_s=9, text="重叠片段"))
    assert project(memory)["counts"]["speakers"] == 3
    tail_scene = {"open_questions": []}
    tail_turns = [{"id": "turn-000001", "speaker": "A"}]
    tail_analyses = {"turn-000001": {"speech_act": "question", "summary": "A询问下一步怎么做。"}}
    ensure_open_tail_question(tail_scene, tail_turns, tail_analyses)
    assert tail_scene["open_questions"] == [{
        "question": "A询问下一步怎么做。", "raised_by": "A",
        "evidence_turn_ids": ["turn-000001"]}]

    compressor = _LiveCompressor()
    tracker = DialogueStateTracker(("小派",), semantic_judge=_CompleteJudge())
    with tempfile.TemporaryDirectory() as temporary:
        updater = LiveMemoryUpdater(tracker, compressor, temporary, "live-test")
        updater.initialize()
        initial = json.loads((Path(temporary) / "session_memory.json").read_text())
        assert initial["turns"] == []
        first = dict(id="span-000001", speaker="A", start_s=0, end_s=2,
                     text="这是第一条观点。", status="final",
                     close_reason="asr_window_limit")
        assert updater.sync([first]) == 0
        first["close_reason"] = "speaker_change"
        assert updater.sync([first]) == 1
        second = dict(id="span-000002", speaker="B", start_s=2, end_s=4,
                      text="这是第二条观点。", status="final", close_reason="silence_gap")
        assert updater.sync([first, second]) == 1
        live = json.loads((Path(temporary) / "session_memory.json").read_text())
        assert [item["speaker"] for item in live["turns"]] == ["A", "B"]
        assert compressor.calls == [["turn-000001"], ["turn-000001", "turn-000002"]]
        assert not (Path(temporary) / "session_memory.json.tmp").exists()
    print("PASS: completed-turn grouping, evidence links, working memory, structured compression")


if __name__ == "__main__":
    check_session_memory()
