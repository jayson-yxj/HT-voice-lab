"""Checks evidence-preserving compression of completed speaker turns."""
import json
from pathlib import Path
import tempfile

from dialogue_state import analyze
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
    tail_scene = {"open_questions": []}
    tail_turns = [{"id": "turn-000001", "speaker": "A"}]
    tail_analyses = {"turn-000001": {"speech_act": "question", "summary": "A询问下一步怎么做。"}}
    ensure_open_tail_question(tail_scene, tail_turns, tail_analyses)
    assert tail_scene["open_questions"] == [{
        "question": "A询问下一步怎么做。", "raised_by": "A",
        "evidence_turn_ids": ["turn-000001"]}]
    print("PASS: completed-turn grouping, evidence links, working memory, structured compression")


if __name__ == "__main__":
    check_session_memory()
