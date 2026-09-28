"""Validate the personal-identity evaluation fixture before model work starts."""

import json
from pathlib import Path

from common import ROOT


KINDS = {
    "name", "role", "occupation", "location", "experience", "preference",
    "goal", "relationship", "communication_style", "trait",
}


def check_personal_identity_fixture():
    fixture = json.loads((ROOT / "fixtures/personal_identity_eval.json").read_text())
    assert fixture["schema_version"] == 1
    assert fixture["policy"] == {
        "scope": "current_session",
        "manual_binding_required": False,
        "unknown_fields_remain_empty": True,
        "same_name_does_not_merge": True,
        "evidence_must_be_exact_quote": True,
    }
    cases = fixture["cases"]
    assert len(cases) == 5
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["session_id"] for case in cases}) == len(cases)

    covered_speakers = set()
    inferred = same_name = forbidden = correction = False
    for case in cases:
        turns = case["turns"]
        by_turn = {item["id"]: item for item in turns}
        assert len(by_turn) == len(turns)
        input_speakers = {item["speaker"] for item in turns}
        covered_speakers.update(input_speakers)
        participants = case["expected"]["participants"]
        assert {item["speaker"] for item in participants} == input_speakers
        assert len({item["speaker"] for item in participants}) == len(participants)
        for participant in participants:
            assert participant["person_id"] is None
            if participant["identity_status"] == "anonymous":
                assert participant["display_name"] is None
            else:
                assert participant["identity_status"] == "session_identified"
                assert isinstance(participant["display_name"], str) and participant["display_name"]

        observations = case["expected"]["observations"]
        keys = set()
        for item in observations:
            assert item["subject"] in input_speakers
            assert item["kind"] in KINDS
            assert item["epistemic_status"] in {"explicit", "inferred"}
            assert item["state"] in {"candidate", "superseded"}
            assert item["durability"] in {"stable", "temporary", "unknown"}
            key = (item["subject"], item["kind"], item["value"], item["state"])
            assert key not in keys
            keys.add(key)
            assert item["evidence"]
            for evidence in item["evidence"]:
                turn = by_turn[evidence["turn_id"]]
                assert turn["speaker"] == item["subject"]
                assert evidence["quote"] in turn["text"]
            inferred |= item["epistemic_status"] == "inferred"
            correction |= item["state"] == "superseded"

        blocked = case["expected"]["forbidden_observations"]
        forbidden |= bool(blocked)
        assert not any(
            (item["subject"], item["kind"], item["value"]) ==
            (blocked_item["subject"], blocked_item["kind"], blocked_item["value"])
            for item in observations for blocked_item in blocked
        )
        names = [item["display_name"] for item in participants if item["display_name"]]
        same_name |= len(names) != len(set(names))

    assert covered_speakers == {"A", "B", "C", "D"}
    assert inferred and same_name and forbidden and correction
    print("PASS: personal identity fixture covers unknown, same-name, attribution, correction and inference rules")


if __name__ == "__main__":
    check_personal_identity_fixture()
