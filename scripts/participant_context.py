"""Create session-scoped anonymous participants and raw turn evidence."""

import hashlib
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from common import write_json
from groq_semantic import chat_completion, check_model_access, load_api_key


IGNORED_SPEAKERS = {"", "unresolved", "未识别人物"}
OBSERVATION_KINDS = {
    "name", "role", "occupation", "location", "experience", "preference",
    "goal", "relationship", "communication_style", "trait",
}
INFERABLE_KINDS = {"communication_style", "trait"}
PERSONAL_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(OBSERVATION_KINDS)},
                    "value": {"type": "string"},
                    "epistemic_status": {"type": "string", "enum": ["explicit", "inferred"]},
                    "state": {"type": "string", "enum": ["candidate", "superseded"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "durability": {"type": "string", "enum": ["stable", "temporary", "unknown"]},
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["source_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                    "context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["source_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["kind", "value", "epistemic_status", "state",
                             "confidence", "durability", "evidence", "context"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}
PERSONAL_SYSTEM_PROMPT = """你从多人对话中提取 target_speaker 自己的个人信息。
conversation 包含完整对话，只允许把 target_speaker 的回答写成其个人信息；其他人的发言只能用于理解提问上下文，不能成为目标人物的自述。不能把转述、猜测、称呼别人或一般话题观点当成个人信息。没有信息就返回空 claims，禁止补全未知字段。
中文和英文使用相同规则。问答中的简短回答也是明确自述，例如 A 问 “What's your favorite beverage?”、目标人物 B 回答 “Coffee.” 时，应提取 B 的 preference=Coffee：evidence 引用 B 的 “Coffee.”，context 引用 A 的问题。
每个 claim 只表达一个原子事实；同一句中的性格、偏好、职业等必须分开。
kind 只能是：name 姓名；occupation 职业或职位；role 在团队中的职责；location 明确自述的居住地或当前所在地；experience 当前活动、过去经历或明确描述的工作／学习／创作方式；preference 明确回答的最喜欢、喜欢、不喜欢或厌倦事项；goal 尚未完成的计划；relationship “我的同事／老师／家人”等与他人的关系；communication_style 重复体现的交流或决策方式；trait 性格特征。目标人物没有说出姓名不影响提取其他属性。
明确说“今天在某地出差”应提取为 temporary location；目的地、被称呼的地名或“将回到某地”本身不能证明 location。一般工作责任用 role，不要改成 occupation。
本人明确自述用 explicit。只有 communication_style 或 trait 可以用 inferred，且推断必须引用至少两个不同话轮；其他类别禁止推断。
若两个以上话轮重复表达同一种第一人称决策或交流习惯，应提取 inferred communication_style，并引用这些话轮。
当前有效信息用 candidate。说话人明确修正旧信息时，旧值仍输出并标记 superseded，新值标记 candidate。
姓名、职业、居住地、长期偏好通常为 stable；出差、最近学习和带时间范围的目标通常为 temporary。
每条 evidence 只能引用 target_speaker 的话轮，其 quote 必须逐字复制原文子串。简短回答依赖他人的问题才能理解时，把必要的问题逐字放入 context；自述本身完整时 context 返回空数组。value 使用简短、忠实的原文语言概括，不要把上下文修饰词并入值；偏好值必须保留喜欢、不喜欢或厌倦等方向。"""


class GroqPersonalInfoExtractor:
    name = "groq-strict-json-v1"

    def __init__(self, model="qwen/qwen3.8-27b", timeout=30.0, proxy=None,
                 minimum_interval=0.0, api_key=None, transport=None):
        if timeout <= 0:
            raise ValueError("Personal extraction timeout must be positive")
        if minimum_interval < 0:
            raise ValueError("Personal extraction minimum interval cannot be negative")
        self.model = model
        self.timeout = timeout
        self.proxy = proxy or os.environ.get("GROQ_HTTPS_PROXY")
        self.api_key = api_key or load_api_key()
        self.transport = transport or self._post
        self.minimum_interval = minimum_interval
        self.last_request_started = None
        if not self.api_key and transport is None:
            raise ValueError("GROQ_API_KEY is not set; run './lab login-groq'")

    def _post(self, payload):
        return chat_completion(payload, self.timeout, self.api_key, self.proxy)

    @staticmethod
    def _personal_signal(speaker, evidence, conversation):
        if any(re.search(r"\b(?:I|I'm|I've|I'd|my|mine|me)\b|我|本人",
                         item["content"], re.IGNORECASE) for item in evidence):
            return True
        for previous, current in zip(conversation, conversation[1:]):
            if (current["speaker_label"] == speaker
                    and previous["speaker_label"] != speaker
                    and "?" in previous["content"]):
                return True
        return False

    def _request(self, payload):
        if self.last_request_started is not None:
            wait = max(0, self.minimum_interval -
                       (time.monotonic() - self.last_request_started))
            if wait:
                time.sleep(wait)
        self.last_request_started = time.monotonic()
        response = self.transport(payload)
        try:
            result = json.loads(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Groq personal extraction returned invalid JSON") from exc
        if not isinstance(result.get("claims"), list):
            raise ValueError("Groq personal extraction response has no claims array")
        return result["claims"]

    def extract(self, speaker, evidence, conversation=None):
        conversation = conversation or evidence
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PERSONAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "target_speaker": speaker,
                    "target_source_ids": [item["source_id"] for item in evidence],
                    "conversation": [dict(source_id=item["source_id"],
                                          speaker=item["speaker_label"],
                                          text=item["content"])
                                     for item in conversation],
                }, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": 2200,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "personal_information", "strict": True,
                                "schema": PERSONAL_CLAIM_SCHEMA},
            },
        }
        claims = self._request(payload)
        if not claims and self._personal_signal(speaker, evidence, conversation):
            payload["messages"][0]["content"] += (
                "\n目标人物的话轮包含第一人称陈述或对个人问题的回答；请逐条复查，"
                "有符合定义的信息就提取，仍无信息才能返回空数组。")
            claims = self._request(payload)
        return claims


def _read_json(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value, raw


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _resolve_dialogue_path(memory_path, configured):
    if not isinstance(configured, str) or not configured:
        raise ValueError("session_memory source.dialogue_state is required")
    source = Path(configured).expanduser()
    candidates = ([memory_path.parent / source] if not source.is_absolute() else [])
    candidates.extend([source, memory_path.parent / source.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Dialogue state not found: {}".format(configured))


def _join_text(previous, current):
    if not previous:
        return current.strip()
    if (previous[-1:].isascii() and previous[-1:].isalnum()
            and current[:1].isascii() and current[:1].isalnum()):
        return previous.rstrip() + " " + current.lstrip()
    return previous.rstrip() + current.lstrip()


def _speaker_labels(memory):
    labels = set()
    sources = [memory.get("participants", []), memory.get("turns", []),
               memory.get("working_memory", [])]
    for index, values in enumerate(sources):
        if not isinstance(values, list):
            raise ValueError("session_memory participant collections must be arrays")
        for item in values:
            label = item if index == 0 else item.get("speaker") if isinstance(item, dict) else None
            if isinstance(label, str) and label not in IGNORED_SPEAKERS:
                labels.add(label)
    return sorted(labels)


def _participant_id(session_id, speaker):
    return "{}:participant:{}".format(session_id, speaker)


def _evidence_id(session_id, turn_id):
    return "{}:evidence:{}".format(session_id, turn_id)


def build_from_path(session_memory):
    """Build model-independent participant context from verified source files."""
    memory_path = Path(session_memory).expanduser().resolve(strict=True)
    memory, memory_raw = _read_json(memory_path)
    if memory.get("schema_version") != 1:
        raise ValueError("Only session_memory schema_version=1 is supported")
    session_id = memory.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_memory.session_id is required")
    source = memory.get("source")
    if not isinstance(source, dict):
        raise ValueError("session_memory.source is required")
    dialogue_path = _resolve_dialogue_path(memory_path, source.get("dialogue_state"))
    dialogue, dialogue_raw = _read_json(dialogue_path)
    dialogue_hash = _sha256(dialogue_raw)
    if source.get("sha256") != dialogue_hash:
        raise ValueError("dialogue_state SHA-256 does not match session_memory source")

    speakers = _speaker_labels(memory)
    participant_by_speaker = {
        speaker: _participant_id(session_id, speaker) for speaker in speakers}
    participants = [dict(
        participant_id=participant_by_speaker[speaker], speaker_label=speaker,
        person_id=None, display_name=None, identity_status="anonymous",
        observation_ids=[], evidence_ids=[])
        for speaker in speakers]
    participant_by_id = {item["participant_id"]: item for item in participants}

    contributions = dialogue.get("contributions")
    if not isinstance(contributions, list):
        raise ValueError("dialogue_state.contributions must be an array")
    by_contribution = {item.get("id"): item for item in contributions
                       if isinstance(item, dict) and isinstance(item.get("id"), str)}
    if len(by_contribution) != len([item for item in contributions
                                    if isinstance(item, dict) and isinstance(item.get("id"), str)]):
        raise ValueError("dialogue_state contains duplicate contribution IDs")

    evidence = []
    used_contributions = set()
    turns = memory.get("turns")
    if not isinstance(turns, list):
        raise ValueError("session_memory.turns must be an array")
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("session_memory.turns must contain objects")
        turn_id, speaker = turn.get("id"), turn.get("speaker")
        contribution_ids = turn.get("contribution_ids")
        if not isinstance(turn_id, str) or speaker not in participant_by_speaker:
            raise ValueError("Completed turn has an invalid id or speaker")
        if (not isinstance(contribution_ids, list) or not contribution_ids
                or not all(isinstance(item, str) for item in contribution_ids)):
            raise ValueError("Completed turn {} has invalid contribution_ids".format(turn_id))
        if any(identifier in used_contributions for identifier in contribution_ids):
            raise ValueError("A contribution belongs to more than one completed turn")
        try:
            parts = [by_contribution[identifier] for identifier in contribution_ids]
        except KeyError as exc:
            raise ValueError("Completed turn references unknown contribution: {}".format(
                exc.args[0])) from None
        if any(item.get("speaker") != speaker for item in parts):
            raise ValueError("Completed turn {} crosses speaker boundaries".format(turn_id))
        if parts[-1].get("turn_complete") is not True:
            raise ValueError("Completed turn {} is not complete in dialogue_state".format(turn_id))
        text = ""
        for part in parts:
            value = part.get("text")
            if not isinstance(value, str):
                raise ValueError("Contribution in {} has no text".format(turn_id))
            text = _join_text(text, value)
        text_hash = _sha256(text.encode())
        if text_hash != turn.get("source_text_sha256"):
            raise ValueError("Completed turn text hash mismatch: {}".format(turn_id))
        identifier = _evidence_id(session_id, turn_id)
        participant_id = participant_by_speaker[speaker]
        evidence.append(dict(
            evidence_id=identifier, participant_id=participant_id,
            modality="speech_text", source_type="completed_turn", source_id=turn_id,
            speaker_label=speaker, start_s=float(turn["start_s"]),
            end_s=float(turn["end_s"]), content=text, content_sha256=text_hash,
            contribution_ids=list(contribution_ids)))
        participant_by_id[participant_id]["evidence_ids"].append(identifier)
        used_contributions.update(contribution_ids)

    result = dict(
        schema_version=1, session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(
            session_memory=dict(path=str(memory_path), sha256=_sha256(memory_raw)),
            dialogue_state=dict(path=str(dialogue_path), sha256=dialogue_hash)),
        policy=dict(identity_scope="current_session", auto_create_participants=True,
                    manual_binding_required=False, unknown_fields_remain_empty=True,
                    completed_turns_only=True),
        participants=participants, observations=[], evidence=evidence, associations=[],
        extraction=dict(provider="none", model=None, rejected_claims=[]),
        stats=dict(participants=len(participants), completed_turns=len(evidence),
                   observations=0, associations=0, rejected_claims=0))
    validate(result)
    return result


def _claim_to_observation(context, participant, claim, evidence_by_source,
                          all_evidence_by_source=None):
    if not isinstance(claim, dict):
        raise ValueError("claim must be an object")
    required = {"kind", "value", "epistemic_status", "state", "confidence",
                "durability", "evidence", "context"}
    if set(claim) != required:
        raise ValueError("claim fields do not match the extraction protocol")
    kind = claim["kind"]
    value = claim["value"].strip() if isinstance(claim["value"], str) else ""
    epistemic = claim["epistemic_status"]
    state = claim["state"]
    durability = claim["durability"]
    confidence = claim["confidence"]
    if kind not in OBSERVATION_KINDS or not value:
        raise ValueError("claim kind or value is invalid")
    if epistemic not in {"explicit", "inferred"}:
        raise ValueError("claim epistemic_status is invalid")
    if epistemic == "inferred" and kind not in INFERABLE_KINDS:
        raise ValueError("only communication_style and trait may be inferred")
    if state not in {"candidate", "superseded"}:
        raise ValueError("claim state is invalid")
    if durability not in {"stable", "temporary", "unknown"}:
        raise ValueError("claim durability is invalid")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1):
        raise ValueError("claim confidence is invalid")
    cited = claim["evidence"]
    if not isinstance(cited, list) or not cited:
        raise ValueError("claim needs evidence")
    evidence_ids = []
    evidence_quotes = []
    distinct_sources = set()
    for citation in cited:
        if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
            raise ValueError("claim citation fields are invalid")
        source_id, quote = citation["source_id"], citation["quote"]
        source = evidence_by_source.get(source_id)
        if source is None:
            raise ValueError("claim cites another speaker or unknown turn")
        if not isinstance(quote, str) or not quote or quote not in source["content"]:
            raise ValueError("claim quote is not an exact source substring")
        evidence_id = source["evidence_id"]
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
        evidence_quotes.append(dict(evidence_id=evidence_id, quote=quote))
        distinct_sources.add(source_id)
    if epistemic == "inferred" and len(distinct_sources) < 2:
        raise ValueError("inferred claim needs evidence from two completed turns")
    context_ids = []
    context_quotes = []
    all_evidence_by_source = all_evidence_by_source or evidence_by_source
    cited_context = claim["context"]
    if not isinstance(cited_context, list):
        raise ValueError("claim context must be an array")
    for citation in cited_context:
        if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
            raise ValueError("claim context citation fields are invalid")
        source = all_evidence_by_source.get(citation["source_id"])
        quote = citation["quote"]
        if source is None or not isinstance(quote, str) or not quote or quote not in source["content"]:
            raise ValueError("claim context quote is not an exact source substring")
        if source["evidence_id"] not in context_ids:
            context_ids.append(source["evidence_id"])
        context_quotes.append(dict(evidence_id=source["evidence_id"], quote=quote))
    if not context_quotes:
        conversation = list(all_evidence_by_source.values())
        positions = {item["source_id"]: index for index, item in enumerate(conversation)}
        for source_id in distinct_sources:
            index = positions.get(source_id, 0)
            if index:
                previous = conversation[index - 1]
                if (previous["participant_id"] != participant["participant_id"]
                        and "?" in previous["content"]):
                    if previous["evidence_id"] not in context_ids:
                        context_ids.append(previous["evidence_id"])
                    context_quotes.append(dict(
                        evidence_id=previous["evidence_id"], quote=previous["content"]))
    identity = "\x1f".join((participant["participant_id"], kind, value, epistemic,
                             state, durability)).encode()
    return dict(
        observation_id="{}:observation:{}".format(
            context["session_id"], _sha256(identity)[:16]),
        participant_id=participant["participant_id"], modality="speech_text",
        kind=kind, value=value, epistemic_status=epistemic, state=state,
        confidence=float(confidence), durability=durability,
        evidence_ids=evidence_ids, evidence_quotes=evidence_quotes,
        context_evidence_ids=context_ids, context_quotes=context_quotes)


def _extract_participant(context, participant, evidence, conversation, extractor):
    observations = {}
    rejections = []
    claims = extractor.extract(participant["speaker_label"], evidence, conversation)
    if not isinstance(claims, list):
        raise ValueError("Personal extractor must return a claims array")
    by_source = {item["source_id"]: item for item in evidence}
    all_by_source = {item["source_id"]: item for item in conversation}
    for index, claim in enumerate(claims):
        try:
            observation = _claim_to_observation(
                context, participant, claim, by_source, all_by_source)
        except (KeyError, TypeError, ValueError) as exc:
            rejections.append(dict(speaker_label=participant["speaker_label"],
                                   claim_index=index, reason=str(exc)))
            continue
        existing = observations.get(observation["observation_id"])
        if existing is None:
            observations[observation["observation_id"]] = observation
            continue
        existing["confidence"] = max(existing["confidence"], observation["confidence"])
        for item in observation["evidence_ids"]:
            if item not in existing["evidence_ids"]:
                existing["evidence_ids"].append(item)
        for item in observation["evidence_quotes"]:
            if item not in existing["evidence_quotes"]:
                existing["evidence_quotes"].append(item)
        for item in observation["context_evidence_ids"]:
            if item not in existing["context_evidence_ids"]:
                existing["context_evidence_ids"].append(item)
        for item in observation["context_quotes"]:
            if item not in existing["context_quotes"]:
                existing["context_quotes"].append(item)
    return list(observations.values()), rejections


def refresh_with_personal_info(context, extractor, previous=None):
    """Refresh changed speakers and reuse verified observations for unchanged speakers."""
    validate(context)
    if context["observations"] or context["extraction"]["provider"] != "none":
        raise ValueError("participant context was already enriched")
    if previous is not None:
        validate(previous)
        if (previous["session_id"] != context["session_id"]
                or previous["extraction"]["provider"] != extractor.name
                or previous["extraction"]["model"] != extractor.model):
            previous = None
    evidence_by_participant = {}
    for item in context["evidence"]:
        evidence_by_participant.setdefault(item["participant_id"], []).append(item)
    previous_participants = ({item["speaker_label"]: item
                              for item in previous["participants"]}
                             if previous else {})
    previous_evidence = ({item["evidence_id"]: item for item in previous["evidence"]}
                         if previous else {})
    previous_observations = ({item["observation_id"]: item
                              for item in previous["observations"]}
                             if previous else {})
    previous_rejections = {}
    if previous:
        for item in previous["extraction"]["rejected_claims"]:
            previous_rejections.setdefault(item["speaker_label"], []).append(item)
    observations = []
    rejections = []
    changed = []
    for participant in context["participants"]:
        evidence = evidence_by_participant.get(participant["participant_id"], [])
        old_participant = previous_participants.get(participant["speaker_label"])
        old_fingerprints = [
            (previous_evidence[item]["source_id"], previous_evidence[item]["content_sha256"])
            for item in old_participant.get("evidence_ids", [])
            if item in previous_evidence] if old_participant else None
        fingerprints = [(item["source_id"], item["content_sha256"]) for item in evidence]
        if old_participant is not None and old_fingerprints == fingerprints:
            reused = [deepcopy(previous_observations[item])
                      for item in old_participant.get("observation_ids", [])
                      if item in previous_observations]
            observations.extend(reused)
            participant["observation_ids"] = [item["observation_id"] for item in reused]
            participant["display_name"] = old_participant.get("display_name")
            participant["identity_status"] = old_participant.get(
                "identity_status", "anonymous")
            rejections.extend(deepcopy(previous_rejections.get(
                participant["speaker_label"], [])))
            continue
        if not evidence:
            continue
        changed.append(participant["speaker_label"])
        extracted, rejected = _extract_participant(
            context, participant, evidence, context["evidence"], extractor)
        observations.extend(extracted)
        participant["observation_ids"] = [item["observation_id"] for item in extracted]
        rejections.extend(rejected)
    context["observations"] = observations
    for participant in context["participants"]:
        names = {item["value"] for item in context["observations"]
                 if item["participant_id"] == participant["participant_id"]
                 and item["kind"] == "name" and item["epistemic_status"] == "explicit"
                 and item["state"] == "candidate"}
        if len(names) == 1:
            participant["display_name"] = names.pop()
            participant["identity_status"] = "session_identified"
    context["extraction"] = dict(provider=extractor.name, model=extractor.model,
                                 rejected_claims=rejections)
    context["stats"]["observations"] = len(context["observations"])
    context["stats"]["rejected_claims"] = len(rejections)
    validate(context)
    return context, changed


def enrich_with_personal_info(context, extractor):
    """Add model claims after enforcing speaker and exact-quote boundaries."""
    return refresh_with_personal_info(context, extractor)[0]


def validate(context):
    """Check referential integrity without adding a JSON Schema dependency."""
    if context.get("schema_version") != 1:
        raise ValueError("participant_context schema_version must be 1")
    participants = context.get("participants")
    evidence = context.get("evidence")
    observations = context.get("observations")
    associations = context.get("associations")
    extraction = context.get("extraction")
    if not all(isinstance(items, list) for items in
               (participants, evidence, observations, associations)):
        raise ValueError("participant_context collections must be arrays")
    if (not isinstance(extraction, dict)
            or set(extraction) != {"provider", "model", "rejected_claims"}
            or not isinstance(extraction["provider"], str)
            or not extraction["provider"]
            or not isinstance(extraction["rejected_claims"], list)):
        raise ValueError("participant_context extraction metadata is invalid")
    participant_ids = [item.get("participant_id") for item in participants]
    speaker_labels = [item.get("speaker_label") for item in participants]
    if len(set(participant_ids)) != len(participant_ids) or None in participant_ids:
        raise ValueError("participant IDs must be unique and non-empty")
    if len(set(speaker_labels)) != len(speaker_labels) or None in speaker_labels:
        raise ValueError("speaker labels must be unique and non-empty")
    participant_ids = set(participant_ids)
    evidence_by_id = {}
    for item in evidence:
        identifier = item.get("evidence_id")
        if not identifier or identifier in evidence_by_id:
            raise ValueError("evidence IDs must be unique and non-empty")
        if item.get("participant_id") not in participant_ids:
            raise ValueError("Evidence references an unknown participant")
        if _sha256(str(item.get("content", "")).encode()) != item.get("content_sha256"):
            raise ValueError("Evidence content hash mismatch")
        start, end = item.get("start_s"), item.get("end_s")
        if (start is None) != (end is None) or (
                start is not None and end < start):
            raise ValueError("Evidence ends before it starts")
        evidence_by_id[identifier] = item
    observation_ids = set()
    for item in observations:
        identifier = item.get("observation_id")
        if not identifier or identifier in observation_ids:
            raise ValueError("observation IDs must be unique and non-empty")
        if item.get("participant_id") not in participant_ids:
            raise ValueError("Observation references an unknown participant")
        if not item.get("evidence_ids") or any(
                value not in evidence_by_id for value in item["evidence_ids"]):
            raise ValueError("Observation requires known evidence")
        if item.get("kind") not in OBSERVATION_KINDS:
            raise ValueError("Observation kind is invalid")
        if item.get("epistemic_status") not in {"explicit", "inferred"}:
            raise ValueError("Observation epistemic_status is invalid")
        if (item["epistemic_status"] == "inferred"
                and item["kind"] not in INFERABLE_KINDS):
            raise ValueError("Observation kind cannot be inferred")
        quotes = item.get("evidence_quotes")
        if not isinstance(quotes, list) or not quotes:
            raise ValueError("Observation requires exact evidence quotes")
        quoted_evidence = set()
        for citation in quotes:
            if (not isinstance(citation, dict)
                    or set(citation) != {"evidence_id", "quote"}):
                raise ValueError("Observation evidence quote is invalid")
            source = evidence_by_id.get(citation["evidence_id"])
            if source is None or source["participant_id"] != item["participant_id"]:
                raise ValueError("Observation quote crosses participant boundaries")
            if (not isinstance(citation["quote"], str) or not citation["quote"]
                    or citation["quote"] not in source["content"]):
                raise ValueError("Observation quote is not an exact source substring")
            quoted_evidence.add(citation["evidence_id"])
        if quoted_evidence != set(item["evidence_ids"]):
            raise ValueError("Observation evidence IDs and quotes do not match")
        context_ids = item.get("context_evidence_ids")
        context_quotes = item.get("context_quotes")
        if not isinstance(context_ids, list) or not isinstance(context_quotes, list):
            raise ValueError("Observation context evidence must be arrays")
        quoted_context = set()
        for citation in context_quotes:
            if (not isinstance(citation, dict)
                    or set(citation) != {"evidence_id", "quote"}):
                raise ValueError("Observation context quote is invalid")
            source = evidence_by_id.get(citation["evidence_id"])
            if (source is None or not isinstance(citation["quote"], str)
                    or not citation["quote"] or citation["quote"] not in source["content"]):
                raise ValueError("Observation context quote is not an exact source substring")
            quoted_context.add(citation["evidence_id"])
        if quoted_context != set(context_ids):
            raise ValueError("Observation context evidence IDs and quotes do not match")
        if item["epistemic_status"] == "inferred" and len(quoted_evidence) < 2:
            raise ValueError("Inferred observation needs two completed turns")
        observation_ids.add(identifier)
    for participant in participants:
        if any(value not in evidence_by_id for value in participant.get("evidence_ids", [])):
            raise ValueError("Participant references unknown evidence")
        if any(value not in observation_ids for value in participant.get("observation_ids", [])):
            raise ValueError("Participant references unknown observations")
    for item in extraction["rejected_claims"]:
        if (not isinstance(item, dict)
                or set(item) != {"speaker_label", "claim_index", "reason"}
                or item["speaker_label"] not in set(speaker_labels)
                or not isinstance(item["claim_index"], int)
                or item["claim_index"] < 0
                or not isinstance(item["reason"], str) or not item["reason"]):
            raise ValueError("Rejected claim metadata is invalid")
    expected = dict(participants=len(participants), completed_turns=len(evidence),
                    observations=len(observations), associations=len(associations),
                    rejected_claims=len(extraction["rejected_claims"]))
    if context.get("stats") != expected:
        raise ValueError("participant_context stats do not match its contents")
    return context


def run(args):
    source = Path(args.session_memory).expanduser().resolve(strict=True)
    result = build_from_path(source)
    if args.personal_provider == "groq":
        check_model_access([args.personal_model], args.personal_timeout,
                           proxy=args.personal_proxy)
        extractor = GroqPersonalInfoExtractor(
            model=args.personal_model, timeout=args.personal_timeout,
            proxy=args.personal_proxy,
            minimum_interval=args.personal_minimum_interval)
        enrich_with_personal_info(result, extractor)
    target = Path(args.output).expanduser() if args.output else source.with_name(
        "participant_context.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(target)
    print("participants={} completed_turns={} observations={} rejected_claims={}".format(
        result["stats"]["participants"], result["stats"]["completed_turns"],
        result["stats"]["observations"], result["stats"]["rejected_claims"]))
    return 0
