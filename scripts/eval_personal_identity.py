"""Run the fixed personal-information fixture through the real extractor."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import write_json
from groq_semantic import check_model_access
from participant_context import GroqPersonalInfoExtractor, enrich_with_personal_info, validate


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _context(case, fixture_path):
    session_id = case["session_id"]
    speakers = list(dict.fromkeys(turn["speaker"] for turn in case["turns"]))
    participants = [dict(
        participant_id="{}:participant:{}".format(session_id, speaker),
        speaker_label=speaker, person_id=None, display_name=None,
        identity_status="anonymous", observation_ids=[], evidence_ids=[])
        for speaker in speakers]
    by_speaker = {item["speaker_label"]: item for item in participants}
    evidence = []
    for turn in case["turns"]:
        participant = by_speaker[turn["speaker"]]
        evidence_id = "{}:evidence:{}".format(session_id, turn["id"])
        evidence.append(dict(
            evidence_id=evidence_id, participant_id=participant["participant_id"],
            modality="speech_text", source_type="completed_turn",
            source_id=turn["id"], speaker_label=turn["speaker"],
            start_s=None, end_s=None, content=turn["text"],
            content_sha256=_hash(turn["text"]), contribution_ids=[]))
        participant["evidence_ids"].append(evidence_id)
    return validate(dict(
        schema_version=1, session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(
            session_memory=dict(path=str(fixture_path), sha256="0" * 64),
            dialogue_state=dict(path=str(fixture_path), sha256="0" * 64)),
        policy=dict(identity_scope="current_session", auto_create_participants=True,
                    manual_binding_required=False, unknown_fields_remain_empty=True,
                    completed_turns_only=True),
        participants=participants, observations=[], evidence=evidence,
        associations=[],
        extraction=dict(provider="none", model=None, rejected_claims=[]),
        stats=dict(participants=len(participants), completed_turns=len(evidence),
                   observations=0, associations=0, rejected_claims=0)))


def _claim_key(item, subject):
    return (subject, item["kind"], item["value"].strip(),
            item["epistemic_status"], item["state"], item["durability"])


def _slot_key(key):
    return key[:2] + key[3:]


def _score(case, context):
    speaker_by_participant = {item["participant_id"]: item["speaker_label"]
                              for item in context["participants"]}
    expected = {_claim_key(item, item["subject"]): item
                for item in case["expected"]["observations"]}
    actual = {_claim_key(item, speaker_by_participant[item["participant_id"]]): item
              for item in context["observations"]}
    matched = sorted(set(expected) & set(actual))
    expected_slots = Counter(_slot_key(item) for item in expected)
    actual_slots = Counter(_slot_key(item) for item in actual)
    matched_slots = sum((expected_slots & actual_slots).values())
    evidence_by_id = {item["evidence_id"]: item for item in context["evidence"]}
    evidence_expected = sum(len(item["evidence"]) for item in expected.values())
    evidence_matched = 0
    expected_by_slot = {}
    actual_by_slot = {}
    for key in expected:
        expected_by_slot.setdefault(_slot_key(key), []).append(key)
    for key in actual:
        actual_by_slot.setdefault(_slot_key(key), []).append(key)
    for slot, expected_keys in expected_by_slot.items():
        for expected_key, actual_key in zip(sorted(expected_keys),
                                            sorted(actual_by_slot.get(slot, []))):
            wanted = {(item["turn_id"], item["quote"])
                      for item in expected[expected_key]["evidence"]}
            got = [(evidence_by_id[item["evidence_id"]]["source_id"], item["quote"])
                   for item in actual[actual_key]["evidence_quotes"]]
            evidence_matched += sum(any(
                expected_turn == actual_turn
                and (expected_quote in actual_quote or actual_quote in expected_quote)
                for actual_turn, actual_quote in got)
                for expected_turn, expected_quote in wanted)
    forbidden = {(item["subject"], item["kind"], item["value"].strip())
                 for item in case["expected"]["forbidden_observations"]}
    forbidden_hits = sorted(forbidden & {(key[0], key[1], key[2]) for key in actual})
    expected_participants = {item["speaker"]: item
                             for item in case["expected"]["participants"]}
    participant_matches = sum(
        expected_participants[item["speaker_label"]]["display_name"] == item["display_name"]
        and expected_participants[item["speaker_label"]]["identity_status"] == item["identity_status"]
        and item["person_id"] is None for item in context["participants"])
    return dict(
        case_id=case["id"], expected_claims=len(expected), actual_claims=len(actual),
        exact_value_matches=len(matched), matched_slots=matched_slots,
        missing_claims=[list(item) for item in sorted(set(expected) - set(actual))],
        unexpected_claims=[list(item) for item in sorted(set(actual) - set(expected))],
        forbidden_hits=[list(item) for item in forbidden_hits],
        evidence_expected=evidence_expected, evidence_matched=evidence_matched,
        participants_expected=len(expected_participants),
        participants_matched=participant_matches,
        rejected_claims=context["extraction"]["rejected_claims"])


def run(args):
    fixture_path = Path(args.fixture).expanduser().resolve(strict=True)
    fixture = json.loads(fixture_path.read_text())
    check_model_access([args.model], args.timeout, proxy=args.proxy)
    extractor = GroqPersonalInfoExtractor(
        model=args.model, timeout=args.timeout, proxy=args.proxy,
        minimum_interval=args.minimum_interval)
    cases = []
    for case in fixture["cases"]:
        context = enrich_with_personal_info(_context(case, fixture_path), extractor)
        cases.append(_score(case, context))
        print("{}: {}/{} claims, forbidden={}, rejected={}".format(
            case["id"], cases[-1]["matched_slots"], cases[-1]["expected_claims"],
            len(cases[-1]["forbidden_hits"]), len(cases[-1]["rejected_claims"])))
    totals = {name: sum(item[name] for item in cases) for name in (
        "expected_claims", "actual_claims", "exact_value_matches", "matched_slots", "evidence_expected",
        "evidence_matched", "participants_expected", "participants_matched")}
    totals["forbidden_hits"] = sum(len(item["forbidden_hits"]) for item in cases)
    totals["rejected_claims"] = sum(len(item["rejected_claims"]) for item in cases)
    totals["exact_value_precision"] = (totals["exact_value_matches"] / totals["actual_claims"]
                                       if totals["actual_claims"] else 1.0)
    totals["exact_value_recall"] = totals["exact_value_matches"] / totals["expected_claims"]
    totals["slot_precision"] = (totals["matched_slots"] / totals["actual_claims"]
                                if totals["actual_claims"] else 1.0)
    totals["slot_recall"] = totals["matched_slots"] / totals["expected_claims"]
    result = dict(schema_version=1, fixture=str(fixture_path), model=args.model,
                  created_at=datetime.now(timezone.utc).isoformat(), totals=totals,
                  cases=cases)
    target = Path(args.output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(target)
    return 0
