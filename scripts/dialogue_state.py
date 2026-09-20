"""Per-speaker dialogue state and conservative assistant reply gating."""
import json
import math
from pathlib import Path
import re


TERMINAL = re.compile(r"[。！？.!?]\s*$")
QUESTION_END = re.compile(r"(?:呢|吗|么|吧)[。！？.!?]?\s*$")
REQUEST_CUE = re.compile(
    r"(?:你(?:是)?怎么(?:看|想)(?:的)?|你觉得(?:呢)?|你认为|你有什么看法|"
    r"请(?:回答|说|说说|讲|讲讲|分析|解释|总结|评价)|"
    r"帮我|告诉我|回答一下|说说看|在吗|听得到吗|听到了吗)", re.IGNORECASE)
IMPERATIVE_CUE = re.compile(
    r"(?:请(?:回答|说|说说|讲|讲讲|分析|解释|总结|评价)|帮我|告诉我|回答一下|说说看|在吗|听得到吗|听到了吗)",
    re.IGNORECASE)


def _join_text(previous, current):
    if not previous:
        return current.strip()
    if previous[-1:].isascii() and previous[-1:].isalnum() and current[:1].isascii() and current[:1].isalnum():
        return previous.rstrip() + " " + current.lstrip()
    return previous.rstrip() + current.lstrip()


class ExplicitReplyGate:
    """Require an assistant alias and a complete request before allowing a reply."""

    def __init__(self, aliases=("小派",), context_seconds=30.0):
        self.aliases = tuple(dict.fromkeys(x.strip() for x in aliases if x.strip()))
        if not self.aliases:
            raise ValueError("At least one assistant alias is required")
        self.context_seconds = context_seconds
        self.pending = {}

    def _address(self, text):
        matches = [(text.casefold().find(alias.casefold()), alias) for alias in self.aliases]
        matches = [(index, alias) for index, alias in matches if index >= 0]
        return min(matches, default=(None, None))

    def inspect(self, speaker, text, start_s, end_s, contribution_id):
        index, alias = self._address(text)
        pending = self.pending.get(speaker)
        if pending and start_s - pending["last_end_s"] > self.context_seconds:
            pending = None
            self.pending.pop(speaker, None)
        if index is not None:
            pending = dict(alias=alias, text=text[index:], started_s=start_s,
                           last_end_s=end_s, contribution_ids=[contribution_id])
            self.pending[speaker] = pending
        elif pending:
            pending["text"] = _join_text(pending["text"], text)
            pending["last_end_s"] = end_s
            pending["contribution_ids"].append(contribution_id)
        if not pending:
            return dict(addressed=False, request_candidate=False, reply_requested=False,
                        decision="no_explicit_address")

        addressed = True
        request_candidate = bool(REQUEST_CUE.search(pending["text"]))
        complete = bool("?" in pending["text"] or "？" in pending["text"]
                        or QUESTION_END.search(pending["text"])
                        or IMPERATIVE_CUE.search(pending["text"]) and TERMINAL.search(pending["text"]))
        reply_requested = request_candidate and complete
        result = dict(addressed=addressed, assistant_alias=pending["alias"],
                      request_candidate=request_candidate, reply_requested=reply_requested,
                      request_text=pending["text"],
                      request_started_s=pending["started_s"], request_end_s=pending["last_end_s"],
                      request_contribution_ids=list(pending["contribution_ids"]),
                      decision="explicit_complete_request" if reply_requested else "awaiting_complete_request")
        if reply_requested or TERMINAL.search(pending["text"]) and not request_candidate:
            self.pending.pop(speaker, None)
        return result


class DialogueStateTracker:
    """Incremental state tracker; another speaker never completes someone else's turn."""

    def __init__(self, aliases=("小派",), context_seconds=30.0, semantic_judge=None):
        self.gate = ExplicitReplyGate(aliases, context_seconds)
        self.semantic_judge = semantic_judge
        self.speakers = {}
        self.contributions = []
        self.reply_requests = []
        self.previous_speaker = None

    def _speaker(self, label):
        return self.speakers.setdefault(label, dict(
            semantic_state="unknown", acoustic_state="idle", reply_state="none",
            contribution_ids=[], pending_contribution_ids=[], interruption_count=0,
            resumption_count=0, last_completed_contribution_ids=[], last_transition="created"))

    def _accept_reply_request(self, record, state, decision, source):
        record["turn_complete"] = True
        record["semantic_state"] = "complete"
        record["reply_requested"] = True
        record["decision"] = "explicit_complete_request"
        state["semantic_state"] = "complete"
        state["reply_state"] = "requested"
        state["last_transition"] = "explicit_reply_requested"
        completed_ids = list(state["pending_contribution_ids"])
        record["completed_contribution_ids"] = completed_ids
        state["last_completed_contribution_ids"] = completed_ids
        state["pending_contribution_ids"] = []
        self.gate.pending.pop(record["speaker"], None)
        self.reply_requests.append(dict(
            id="reply-request-{:06d}".format(len(self.reply_requests) + 1),
            speaker=record["speaker"], assistant_alias=decision["assistant_alias"],
            text=decision["request_text"], start_s=decision["request_started_s"],
            end_s=decision["request_end_s"],
            contribution_ids=decision["request_contribution_ids"],
            ready=True, decision_source=source))

    def push(self, span):
        contribution_id = "contribution-{:06d}".format(len(self.contributions) + 1)
        speaker = span.get("speaker")
        record = dict(id=contribution_id, source_span_id=span.get("id"), speaker=speaker,
                      start_s=float(span["start_s"]), end_s=float(span["end_s"]),
                      text=span.get("text", ""), close_reason=span.get("close_reason"),
                      turn_complete=None, completed_contribution_ids=[],
                      reply_requested=False)
        if speaker is None:
            record.update(decision="unresolved_overlap", semantic_state="unknown")
            self.contributions.append(record)
            self.previous_speaker = None
            return record

        state = self._speaker(speaker)
        if self.previous_speaker and self.previous_speaker != speaker:
            previous = self._speaker(self.previous_speaker)
            if previous["semantic_state"] == "unknown":
                previous["acoustic_state"] = "suspended"
                previous["interruption_count"] += 1
                previous["last_transition"] = "suspended_by_other_speaker"
        if state["acoustic_state"] == "suspended" and state["semantic_state"] == "unknown":
            state["resumption_count"] += 1
        if state["semantic_state"] == "complete":
            state["semantic_state"] = "unknown"
            state["pending_contribution_ids"] = []
        state["acoustic_state"] = "paused"
        state["last_transition"] = "contribution_received"
        state["contribution_ids"].append(contribution_id)
        state["pending_contribution_ids"].append(contribution_id)

        decision = self.gate.inspect(speaker, record["text"], record["start_s"],
                                     record["end_s"], contribution_id)
        record.update(decision)
        if decision["reply_requested"]:
            self._accept_reply_request(
                record, state, decision, "explicit-address-rules-v1")
        else:
            pending_ids = list(state["pending_contribution_ids"])
            context = [dict(id=item["id"], text=item["text"], start_s=item["start_s"],
                            end_s=item["end_s"])
                       for item in self.contributions if item["id"] in pending_ids]
            context.append(dict(id=contribution_id, text=record["text"],
                                start_s=record["start_s"], end_s=record["end_s"]))
            if self.semantic_judge:
                try:
                    semantic = self.semantic_judge.assess(
                        speaker=speaker, contribution_ids=pending_ids, text=record["text"],
                        close_reason=record["close_reason"], context=context)
                except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    semantic = dict(turn_complete=None, confidence=None,
                                    reason="{}: {}".format(type(exc).__name__, exc),
                                    decision_source="semantic-model-error")
            else:
                semantic = dict(turn_complete=None, confidence=None,
                                reason="semantic_model_not_configured",
                                decision_source="pending-model")
            record["semantic_decision"] = semantic
            record["turn_complete"] = semantic.get("turn_complete")
            if record["turn_complete"] is True:
                if decision["request_candidate"]:
                    self._accept_reply_request(
                        record, state, decision, "explicit-address+semantic-model-v1")
                else:
                    record["semantic_state"] = "complete"
                    state["semantic_state"] = "complete"
                    completed_ids = list(state["pending_contribution_ids"])
                    record["completed_contribution_ids"] = completed_ids
                    state["last_completed_contribution_ids"] = completed_ids
                    state["pending_contribution_ids"] = []
                    state["last_transition"] = "semantic_turn_complete"
                    if decision["addressed"]:
                        self.gate.pending.pop(speaker, None)
                        record["decision"] = "addressed_without_request"
            else:
                record["semantic_state"] = "unknown"
        self.contributions.append(record)
        self.previous_speaker = speaker
        return record

    def result(self):
        return dict(schema_version=1, assistant_aliases=list(self.gate.aliases),
                    policy=dict(reply_requires_explicit_address=True,
                                other_speaker_does_not_complete_turn=True,
                                silence_does_not_complete_turn=True,
                                general_semantic_completion=(
                                    getattr(self.semantic_judge, "name", "pending_model"))),
                    contributions=self.contributions, speaker_states=self.speakers,
                    reply_requests=self.reply_requests,
                    reply_allowed=bool(self.reply_requests))


def analyze(spans, aliases=("小派",), context_seconds=30.0, semantic_judge=None):
    tracker = DialogueStateTracker(aliases, context_seconds, semantic_judge)
    previous_end = -1.0
    for span in spans:
        if not all(key in span for key in ("start_s", "end_s", "text", "speaker")):
            raise ValueError("Each speech span requires start_s, end_s, text and speaker")
        if float(span["start_s"]) < previous_end:
            raise ValueError("Speech spans must be chronological and non-overlapping")
        if float(span["end_s"]) <= float(span["start_s"]):
            raise ValueError("Speech span end must be after start")
        tracker.push(span)
        previous_end = float(span["end_s"])
    return tracker.result()


def run(args):
    source = Path(args.speech_spans).resolve(strict=True)
    spans = json.loads(source.read_text())
    if not isinstance(spans, list):
        raise ValueError("Speech spans input must be a JSON list")
    aliases = tuple(args.assistant_alias or ["小派"])
    if not math.isfinite(args.invocation_context_seconds) or args.invocation_context_seconds <= 0:
        raise ValueError("Invocation context must be a positive number of seconds")
    if not math.isfinite(args.semantic_timeout) or args.semantic_timeout <= 0:
        raise ValueError("Semantic timeout must be a positive number of seconds")
    semantic_judge = None
    if args.semantic_provider == "groq":
        from groq_semantic import GroqSemanticJudge
        semantic_judge = GroqSemanticJudge(
            args.semantic_model, args.semantic_timeout, proxy=args.semantic_proxy,
            minimum_interval=args.semantic_minimum_interval)
    result = analyze(spans, aliases, args.invocation_context_seconds, semantic_judge)
    target = Path(args.output) if args.output else source.with_name("dialogue_state.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(target)
    return 0
