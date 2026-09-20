"""Session-scoped compressed memory built from completed per-speaker turns."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from groq_semantic import chat_completion


SPEECH_ACTS = [
    "statement", "question", "proposal", "agreement", "disagreement",
    "commitment", "request", "other",
]
STANCES = ["support", "oppose", "neutral", "unclear"]

STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
EVIDENCE_ARRAY = {"type": "array", "items": {"type": "string"}}
SESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "session_summary": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_turn_ids": EVIDENCE_ARRAY,
                },
                "required": ["name", "summary", "evidence_turn_ids"],
                "additionalProperties": False,
            },
        },
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "raised_by": {"type": "string"},
                    "evidence_turn_ids": EVIDENCE_ARRAY,
                },
                "required": ["question", "raised_by", "evidence_turn_ids"],
                "additionalProperties": False,
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "speakers": STRING_ARRAY,
                    "evidence_turn_ids": EVIDENCE_ARRAY,
                },
                "required": ["decision", "speakers", "evidence_turn_ids"],
                "additionalProperties": False,
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": "string"},
                    "status": {"type": "string", "enum": ["proposed", "committed", "unclear"]},
                    "evidence_turn_ids": EVIDENCE_ARRAY,
                },
                "required": ["action", "owner", "status", "evidence_turn_ids"],
                "additionalProperties": False,
            },
        },
        "turn_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "speech_act": {"type": "string", "enum": SPEECH_ACTS},
                    "intent": {"type": "string"},
                    "stance": {"type": "string", "enum": STANCES},
                    "topic_names": STRING_ARRAY,
                    "entities": STRING_ARRAY,
                },
                "required": ["turn_id", "summary", "speech_act", "intent", "stance",
                             "topic_names", "entities"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["session_summary", "topics", "open_questions", "decisions",
                 "action_items", "turn_memories"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """你负责把多人现场对话压缩为可追溯的当前会话记忆。
只能使用输入中明确出现的信息，不得补全常识或猜测姓名。每个人的观点、意图和立场必须分开保存，不能把主持人的问题当成其观点，也不能把不同人的意见合并成共识。
只记录明确作出的决定和行动项；尚未解决的问题放入 open_questions。每个结论都用 evidence_turn_ids 指回来源。
为每个输入 turn_id 返回且只返回一条 turn_memories。每条摘要限一句，intent 使用极短短语，每个话题摘要限一句；避免在多个字段重复同一句话。摘要使用简洁中文，英文专有名词可以保留。"""


def _join_text(previous, current):
    if not previous:
        return current.strip()
    if previous[-1:].isascii() and previous[-1:].isalnum() and current[:1].isascii() and current[:1].isalnum():
        return previous.rstrip() + " " + current.lstrip()
    return previous.rstrip() + current.lstrip()


def completed_turns(dialogue):
    contributions = dialogue.get("contributions")
    if not isinstance(contributions, list):
        raise ValueError("Dialogue state requires a contributions list")
    by_id = {item.get("id"): item for item in contributions}
    turns = []
    used = set()
    for closing in contributions:
        if closing.get("turn_complete") is not True:
            continue
        ids = closing.get("completed_contribution_ids") or [closing.get("id")]
        if any(identifier not in by_id for identifier in ids):
            raise ValueError("Completed turn references an unknown contribution")
        if any(identifier in used for identifier in ids):
            raise ValueError("A contribution belongs to more than one completed turn")
        parts = [by_id[identifier] for identifier in ids]
        speakers = {item.get("speaker") for item in parts}
        if len(speakers) != 1 or None in speakers:
            raise ValueError("A completed turn must belong to exactly one resolved speaker")
        text = ""
        for item in parts:
            text = _join_text(text, item.get("text", ""))
        turns.append(dict(
            id="turn-{:06d}".format(len(turns) + 1), speaker=parts[0]["speaker"],
            start_s=min(float(item["start_s"]) for item in parts),
            end_s=max(float(item["end_s"]) for item in parts), text=text,
            contribution_ids=list(ids), reply_requested=bool(closing.get("reply_requested")),
            completion_source=(closing.get("semantic_decision") or {}).get("decision_source")
                              or closing.get("decision")))
        used.update(ids)
    return turns, used


def working_memory(dialogue, completed_ids):
    grouped = {}
    for item in dialogue.get("contributions", []):
        if item.get("id") in completed_ids:
            continue
        speaker = item.get("speaker") or "unresolved"
        group = grouped.setdefault(speaker, dict(
            speaker=speaker, contribution_ids=[], start_s=float(item["start_s"]),
            end_s=float(item["end_s"]), text=""))
        group["contribution_ids"].append(item.get("id"))
        group["end_s"] = max(group["end_s"], float(item["end_s"]))
        group["text"] = _join_text(group["text"], item.get("text", ""))
    return sorted(grouped.values(), key=lambda item: item["start_s"])


class LocalMemoryCompressor:
    name = "local-extractive-v1"
    model = None

    def compress(self, turns):
        turn_memories = [dict(
            turn_id=item["id"], summary=item["text"], speech_act="other", intent="",
            stance="unclear", topic_names=[], entities=[])
            for item in turns]
        summary = "；".join("{}：{}".format(item["speaker"], item["text"]) for item in turns)
        return dict(session_summary=summary, topics=[], open_questions=[], decisions=[],
                    action_items=[], turn_memories=turn_memories), dict(latency_s=0.0)


class GroqMemoryCompressor:
    name = "groq-session-memory-v1"

    def __init__(self, model="qwen/qwen3.8-27b", timeout=20.0, proxy=None, transport=None):
        if timeout <= 0:
            raise ValueError("Memory timeout must be positive")
        self.model = model
        self.timeout = timeout
        self.proxy = proxy
        self.transport = transport

    def compress(self, turns):
        turn_ids = [item["id"] for item in turns]
        schema = json.loads(json.dumps(SESSION_SCHEMA))
        turn_array = schema["properties"]["turn_memories"]
        turn_array["minItems"] = len(turns)
        turn_array["maxItems"] = len(turns)
        turn_array["items"]["properties"]["turn_id"]["enum"] = turn_ids
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"turns": [
                    {key: item[key] for key in ("id", "speaker", "start_s", "end_s", "text")}
                    for item in turns]}, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": 950,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "session_memory", "strict": True,
                                "schema": schema},
            },
        }
        started = time.monotonic()
        response = (self.transport(payload) if self.transport else
                    chat_completion(payload, self.timeout, proxy=self.proxy))
        elapsed = time.monotonic() - started
        result = json.loads(response["choices"][0]["message"]["content"])
        expected = {item["id"] for item in turns}
        actual = [item.get("turn_id") for item in result.get("turn_memories", [])]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Memory response turn IDs do not match completed turns: expected={} actual={}"
                             .format(sorted(expected), actual))
        _validate_evidence(result, expected)
        return result, dict(latency_s=elapsed)


def _validate_evidence(value, valid_turn_ids):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_turn_ids":
                if not isinstance(item, list) or any(x not in valid_turn_ids for x in item):
                    raise ValueError("Memory response contains invalid evidence turn IDs")
            else:
                _validate_evidence(item, valid_turn_ids)
    elif isinstance(value, list):
        for item in value:
            _validate_evidence(item, valid_turn_ids)


def _text_size(value):
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_text_size(item) for item in value.values())
    if isinstance(value, list):
        return sum(_text_size(item) for item in value)
    return 0


def ensure_open_tail_question(scene, turns, analyses):
    """A final question has no later turn that could answer it, so keep it open."""
    if not turns:
        return
    last = turns[-1]
    analysis = analyses[last["id"]]
    if analysis["speech_act"] not in {"question", "request"}:
        return
    questions = scene.setdefault("open_questions", [])
    if any(last["id"] in item.get("evidence_turn_ids", []) for item in questions):
        return
    questions.append(dict(question=analysis["summary"], raised_by=last["speaker"],
                          evidence_turn_ids=[last["id"]]))


def build(dialogue, source, session_id, compressor):
    turns, completed_ids = completed_turns(dialogue)
    if not turns:
        raise ValueError("No completed semantic turns are available for compression")
    compressed, metrics = compressor.compress(turns)
    analyses = {item["turn_id"]: item for item in compressed.pop("turn_memories")}
    ensure_open_tail_question(compressed, turns, analyses)
    memory_turns = []
    for turn in turns:
        analysis = analyses[turn["id"]]
        memory_turns.append(dict(
            id=turn["id"], speaker=turn["speaker"],
            start_s=turn["start_s"], end_s=turn["end_s"],
            contribution_ids=turn["contribution_ids"],
            source_text_sha256=hashlib.sha256(turn["text"].encode()).hexdigest(),
            reply_requested=turn["reply_requested"],
            completion_source=turn["completion_source"],
            summary=analysis["summary"], speech_act=analysis["speech_act"],
            intent=analysis["intent"], stance=analysis["stance"],
            topic_names=analysis["topic_names"], entities=analysis["entities"]))
    raw_chars = sum(len(item["text"]) for item in turns)
    compressed_chars = _text_size(compressed) + sum(
        _text_size({key: item[key] for key in
                    ("summary", "intent", "topic_names", "entities")})
        for item in memory_turns)
    source_bytes = source.read_bytes()
    pending = working_memory(dialogue, completed_ids)
    return dict(
        schema_version=1, session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(dialogue_state=str(source), sha256=hashlib.sha256(source_bytes).hexdigest()),
        policy=dict(scope="current_session", completed_turns_only=True,
                    raw_evidence_external=True, preserve_speaker_boundaries=True),
        compressor=dict(provider=compressor.name, model=compressor.model,
                        latency_s=metrics["latency_s"]),
        participants=sorted({item["speaker"] for item in turns}),
        turns=memory_turns, working_memory=pending,
        scene=compressed,
        stats=dict(source_contributions=len(dialogue.get("contributions", [])),
                   completed_turns=len(turns), working_speakers=len(pending),
                   raw_completed_text_chars=raw_chars,
                   compressed_text_chars=compressed_chars,
                   text_ratio=round(compressed_chars / raw_chars, 3) if raw_chars else None))


def run(args):
    source = Path(args.dialogue_state).expanduser().resolve(strict=True)
    dialogue = json.loads(source.read_text())
    if not isinstance(dialogue, dict):
        raise ValueError("Dialogue state input must be a JSON object")
    compressor = (GroqMemoryCompressor(args.memory_model, args.memory_timeout,
                                       args.memory_proxy)
                  if args.memory_provider == "groq" else LocalMemoryCompressor())
    session_id = args.session_id or source.parent.name
    if not session_id.strip():
        raise ValueError("Session ID cannot be empty")
    result = build(dialogue, source, session_id, compressor)
    target = Path(args.output) if args.output else source.with_name("session_memory.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(target)
    return 0
