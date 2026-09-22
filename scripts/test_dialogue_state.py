"""Checks independent speaker state and explicit assistant reply gating."""
from io import StringIO
from unittest.mock import patch
from urllib.error import URLError

from dialogue_state import analyze
from groq_semantic import GroqSemanticJudge, _open_json, _retry_after_seconds


def span(identifier, speaker, start, end, text, reason="speaker_change"):
    return dict(id=identifier, speaker=speaker, start_s=start, end_s=end,
                text=text, close_reason=reason)


def check_dialogue_state():
    assert _retry_after_seconds({}, "Please try again in 750ms", 0) == 0.75
    assert _retry_after_seconds({}, "Please try again in 1.5s", 0) == 1.5
    class FlakyOpener:
        calls = 0

        def open(self, call, timeout):
            self.calls += 1
            if self.calls < 3:
                raise URLError(TimeoutError("TLS handshake timed out"))
            return StringIO('{"status":"ok"}')

    flaky = FlakyOpener()
    with patch("groq_semantic.time.sleep") as sleep:
        assert _open_json(flaky, object(), 10) == {"status": "ok"}
    assert flaky.calls == 3
    assert [item.args[0] for item in sleep.call_args_list] == [0.5, 1.0]
    result = analyze([
        span("s1", "A", 0, 2, "我先说第一点，这里还没有说完"),
        span("s2", "B", 2, 4, "我补充另一个观点。"),
        span("s3", "A", 4, 6, "刚才那一点还需要考虑成本。"),
        span("s4", "C", 6, 8, "小派参与了这个项目。"),
        span("s5", "D", 8, 10, "大家觉得小派怎么样？"),
        span("s6", "C", 10, 12, "小派，你觉得这个"),
        span("s7", "B", 12, 13, "稍等一下。"),
        span("s8", "C", 13, 14, "方案怎么样？", "silence_gap"),
    ])
    assert result["speaker_states"]["A"]["resumption_count"] == 1
    assert result["speaker_states"]["A"]["semantic_state"] == "unknown"
    assert result["contributions"][1]["turn_complete"] is None
    assert not result["contributions"][3]["reply_requested"]
    assert not result["contributions"][4]["reply_requested"]
    assert result["contributions"][5]["request_candidate"]
    assert not result["contributions"][5]["reply_requested"]
    assert result["contributions"][7]["reply_requested"]
    assert result["contributions"][7]["turn_complete"] is True
    assert result["reply_requests"][0]["text"] == "小派，你觉得这个方案怎么样？"
    assert result["reply_allowed"]
    quiet = analyze([span("s1", "A", 0, 1, "小派？")])
    assert quiet["contributions"][0]["addressed"]
    assert not quiet["reply_allowed"]
    class FakeSemanticJudge:
        name = "fake-semantic-judge"

        def assess(self, **item):
            assert item["context"][-1]["text"] == item["text"]
            return dict(turn_complete=item["text"].endswith("我讲完了。"), confidence=1.0,
                        reason="test_marker", decision_source=self.name)

    judged = analyze([
        span("s1", "A", 0, 1, "这是我的观点，我讲完了。"),
        span("s2", "B", 1, 2, "我还没有说完"),
    ], semantic_judge=FakeSemanticJudge())
    assert judged["contributions"][0]["turn_complete"] is True
    assert judged["contributions"][0]["completed_contribution_ids"] == ["contribution-000001"]
    assert judged["contributions"][1]["turn_complete"] is False
    assert not judged["reply_allowed"]
    class PunctuationlessRequestJudge:
        name = "punctuationless-request-judge"

        def assess(self, **item):
            complete = item["text"].endswith("应该怎么改")
            return dict(turn_complete=complete, confidence=1.0,
                        reason="test_semantic_completion", decision_source=self.name)

    punctuationless = analyze([
        span("s1", "A", 0, 2, "小派你觉得这个方案"),
        span("s2", "B", 2, 3, "我先补充一句"),
        span("s3", "A", 3, 5, "应该怎么改"),
    ], semantic_judge=PunctuationlessRequestJudge())
    assert not punctuationless["contributions"][0]["reply_requested"]
    assert punctuationless["contributions"][2]["reply_requested"]
    assert punctuationless["contributions"][2]["completed_contribution_ids"] == [
        "contribution-000001", "contribution-000003"]
    assert punctuationless["reply_requests"][0]["text"] == "小派你觉得这个方案应该怎么改"
    assert punctuationless["reply_requests"][0]["decision_source"] == "explicit-address+semantic-model-v1"
    class AlwaysCompleteJudge:
        name = "always-complete-judge"

        def assess(self, **item):
            return dict(turn_complete=True, confidence=1.0,
                        reason="test_complete", decision_source=self.name)

    mention_then_question = analyze([
        span("s1", "A", 0, 2, "小派参与了昨天的测试"),
        span("s2", "B", 2, 3, "我补充一句"),
        span("s3", "A", 3, 5, "你觉得呢"),
    ], semantic_judge=AlwaysCompleteJudge())
    assert mention_then_question["contributions"][0]["decision"] == "addressed_without_request"
    assert not mention_then_question["reply_allowed"]
    captured = []
    def fake_groq(payload):
        captured.append(payload)
        return {"choices": [{"message": {"content":
                '{"status":"incomplete","confidence":0.91,"reason":"列举尚未结束"}'}}]}
    groq = GroqSemanticJudge(transport=fake_groq)
    decision = groq.assess("A", ["c1"], "第一点是", "silence_gap",
                           [dict(id="c1", text="第一点是", start_s=0, end_s=1)])
    assert decision["turn_complete"] is False and decision["confidence"] == 0.91
    assert captured[0]["response_format"]["json_schema"]["strict"] is True
    assert captured[0]["max_completion_tokens"] == 160
    assert captured[0]["model"] == "qwen/qwen3.8-27b"
    print("PASS: independent speaker state, interruption/resumption, explicit reply gating")


if __name__ == "__main__":
    check_dialogue_state()
