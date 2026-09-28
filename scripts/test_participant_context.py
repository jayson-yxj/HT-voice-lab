"""Checks deterministic anonymous participants and raw evidence recovery."""

import hashlib
import json
from pathlib import Path
import tempfile

from participant_context import (build_from_path, enrich_with_personal_info,
                                 refresh_with_personal_info, validate)


class FakeExtractor:
    name = "fake-strict-json-v1"
    model = "fixture"

    def __init__(self):
        self.called = []

    def extract(self, speaker, evidence, conversation=None):
        self.called.append(speaker)
        if speaker != "A":
            return []
        return [
            dict(kind="name", value="林晓", epistemic_status="explicit",
                 state="candidate", confidence=0.99, durability="stable",
                 evidence=[dict(source_id="turn-000001", quote="我叫林晓")], context=[]),
            dict(kind="location", value="杭州", epistemic_status="explicit",
                 state="candidate", confidence=0.8, durability="stable",
                 evidence=[dict(source_id="turn-000001", quote="我住在杭州")], context=[]),
            dict(kind="communication_style", value="表达简洁",
                 epistemic_status="inferred", state="candidate", confidence=0.7,
                 durability="stable",
                 evidence=[dict(source_id="turn-000001", quote="我叫林晓")], context=[]),
        ]


def _write_fixture(directory):
    contributions = [
        dict(id="contribution-000001", speaker="A", start_s=0, end_s=2,
             text="我叫林晓。", turn_complete=True,
             completed_contribution_ids=["contribution-000001"]),
        dict(id="contribution-000002", speaker="B", start_s=2, end_s=4,
             text="你好。", turn_complete=True,
             completed_contribution_ids=["contribution-000002"]),
        dict(id="contribution-000003", speaker=None, start_s=4, end_s=5,
             text="重叠片段", turn_complete=False),
        dict(id="contribution-000004", speaker="C", start_s=5, end_s=6,
             text="我还没说完", turn_complete=False),
    ]
    dialogue = directory / "dialogue_state.json"
    dialogue.write_text(json.dumps(dict(schema_version=1, contributions=contributions),
                                   ensure_ascii=False))
    turns = []
    for index, contribution in enumerate(contributions[:2], 1):
        turns.append(dict(
            id="turn-{:06d}".format(index), speaker=contribution["speaker"],
            start_s=contribution["start_s"], end_s=contribution["end_s"],
            contribution_ids=[contribution["id"]],
            source_text_sha256=hashlib.sha256(contribution["text"].encode()).hexdigest()))
    memory = directory / "session_memory.json"
    memory.write_text(json.dumps(dict(
        schema_version=1, session_id="fixture-session", created_at="fixture",
        source=dict(dialogue_state=str(dialogue),
                    sha256=hashlib.sha256(dialogue.read_bytes()).hexdigest()),
        participants=["A", "B", "C", "unresolved"], turns=turns,
        working_memory=[dict(speaker="C", contribution_ids=["contribution-000004"],
                            start_s=5, end_s=6, text="我还没说完"),
                        dict(speaker="unresolved", contribution_ids=["contribution-000003"],
                            start_s=4, end_s=5, text="重叠片段")]), ensure_ascii=False))
    return memory, dialogue


def check_participant_context():
    with tempfile.TemporaryDirectory() as temporary:
        memory, dialogue = _write_fixture(Path(temporary))
        first = build_from_path(memory)
        second = build_from_path(memory)
        assert [item["speaker_label"] for item in first["participants"]] == ["A", "B", "C"]
        assert [item["participant_id"] for item in first["participants"]] == [
            "fixture-session:participant:A", "fixture-session:participant:B",
            "fixture-session:participant:C"]
        assert [item["participant_id"] for item in first["participants"]] == [
            item["participant_id"] for item in second["participants"]]
        assert all(item["person_id"] is None and item["display_name"] is None
                   and item["identity_status"] == "anonymous"
                   and item["observation_ids"] == [] for item in first["participants"])
        assert [item["content"] for item in first["evidence"]] == ["我叫林晓。", "你好。"]
        assert first["participants"][2]["evidence_ids"] == []
        assert first["observations"] == [] and first["associations"] == []
        assert first["extraction"] == dict(
            provider="none", model=None, rejected_claims=[])
        assert first["stats"] == dict(
            participants=3, completed_turns=2, observations=0, associations=0,
            rejected_claims=0)
        validate(first)

        extractor = FakeExtractor()
        enrich_with_personal_info(first, extractor)
        assert extractor.called == ["A", "B"]  # C has no completed evidence.
        assert len(first["observations"]) == 1
        observation = first["observations"][0]
        assert observation["kind"] == "name" and observation["value"] == "林晓"
        assert observation["evidence_quotes"] == [dict(
            evidence_id="fixture-session:evidence:turn-000001", quote="我叫林晓")]
        assert first["participants"][0]["display_name"] == "林晓"
        assert first["participants"][0]["identity_status"] == "session_identified"
        assert all(item["display_name"] is None for item in first["participants"][1:])
        assert first["stats"]["rejected_claims"] == 2
        assert {item["reason"] for item in first["extraction"]["rejected_claims"]} == {
            "claim quote is not an exact source substring",
            "inferred claim needs evidence from two completed turns",
        }
        validate(first)

        unchanged_extractor = FakeExtractor()
        unchanged, changed = refresh_with_personal_info(
            build_from_path(memory), unchanged_extractor, first)
        assert changed == [] and unchanged_extractor.called == []
        assert unchanged["observations"] == first["observations"]

        dialogue_data = json.loads(dialogue.read_text())
        dialogue_data["contributions"].append(dict(
            id="contribution-000005", speaker="A", start_s=6, end_s=8,
            text="我喜欢徒步。", turn_complete=True,
            completed_contribution_ids=["contribution-000005"]))
        dialogue.write_text(json.dumps(dialogue_data, ensure_ascii=False))
        memory_data = json.loads(memory.read_text())
        memory_data["source"]["sha256"] = hashlib.sha256(dialogue.read_bytes()).hexdigest()
        memory_data["turns"].append(dict(
            id="turn-000003", speaker="A", start_s=6, end_s=8,
            contribution_ids=["contribution-000005"],
            source_text_sha256=hashlib.sha256("我喜欢徒步。".encode()).hexdigest()))
        memory.write_text(json.dumps(memory_data, ensure_ascii=False))
        changed_extractor = FakeExtractor()
        refreshed, changed = refresh_with_personal_info(
            build_from_path(memory), changed_extractor, first)
        assert changed == ["A"] and changed_extractor.called == ["A"]
        assert refreshed["participants"][1]["observation_ids"] == []
        validate(refreshed)

        dialogue.write_text(dialogue.read_text() + " ")
        try:
            build_from_path(memory)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("tampered dialogue state was accepted")
    print("PASS: deterministic anonymous participants and completed-turn evidence")


if __name__ == "__main__":
    check_participant_context()
