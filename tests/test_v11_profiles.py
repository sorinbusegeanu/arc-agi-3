from __future__ import annotations

import hashlib
import json

import pytest

from v11.profiles import KNOWN_ACTION_SETS, compose_profile_prompt, resolve_action_profile
from v11.tools.memory import (
    enqueue_profile_candidate,
    list_profile_candidates,
    migrate_legacy_global_memory,
    profile_memory_path,
    promote_profile_candidates,
    read_profile_memory,
    reject_profile_candidates,
)


def test_catalog_action_sets_are_order_independent_and_reset_is_ignored():
    for actions, profile_id in KNOWN_ACTION_SETS.items():
        legal = [{"id": action, "parameters": {"type": "object", "properties": {}}} for action in actions]
        expected = resolve_action_profile(legal)
        assert expected.profile_id == profile_id
        assert resolve_action_profile(list(reversed(legal)) + [{"id": "RESET"}]) == expected


def test_unknown_action_set_uses_generic_and_argument_schema_fingerprint_changes():
    first = resolve_action_profile([{"id": "ACTION9", "parameters": {"type": "object", "properties": {"x": {"minimum": 0}}}}])
    changed = resolve_action_profile([{"id": "ACTION9", "parameters": {"type": "object", "properties": {"x": {"minimum": 1}}}}])
    assert first.profile_id == changed.profile_id == "generic"
    assert first.action_signature == changed.action_signature
    assert first.argument_schema_fingerprint != changed.argument_schema_fingerprint


def test_profile_prompt_contains_only_active_capabilities():
    profile = resolve_action_profile([{"id": "ACTION6"}])
    prompt, digest = compose_profile_prompt(profile, "skills/v11/profile_capabilities")
    assert "coordinate" in prompt.casefold() or "click" in prompt.casefold()
    assert "movement" not in prompt.casefold()
    assert "hypothesis.evidence 180 characters" in prompt
    assert "changed_pixels` is a raw full-frame difference" in prompt
    assert "action1" not in prompt.casefold()
    assert digest == compose_profile_prompt(profile, "skills/v11/profile_capabilities")[1]


def test_candidates_are_isolated_and_promotion_requires_independent_evidence_and_heldout_gain(tmp_path):
    state = tmp_path / "state"
    a = enqueue_profile_candidate(state, "a6", game_id="game-a", run_id="r1", claim="Compare outcomes", applicability_conditions=["when testing a target"], evidence_ids=["E1"])
    b = enqueue_profile_candidate(state, "a6", game_id="game-b", run_id="r2", claim="Compare outcomes", applicability_conditions=["when testing a target"], evidence_ids=["E2"])
    assert len(list_profile_candidates(state, "a6", pending_only=True)) == 2
    markdown = "# Approved method\n\n- Compare outcomes after one action."
    revision = hashlib.sha256(markdown.encode()).hexdigest()
    report = {"profile_id": "a6", "improved": True, "candidate_revision": revision,
              "baseline_revision": "baseline-1", "heldout_games": ["game-c"], "metrics": {"success_rate": 0.2}}
    result = promote_profile_candidates(state, "a6", [a["candidate_id"], b["candidate_id"]], markdown, report)
    assert result["event"] == "promoted"
    assert read_profile_memory(state, "a6").startswith("# Approved method")
    assert read_profile_memory(state, "a5") == ""
    assert list_profile_candidates(state, "a6", pending_only=True) == []


def test_profile_promotion_rejects_source_game_as_heldout_and_candidates_can_be_rejected(tmp_path):
    state = tmp_path / "state"
    rows = [enqueue_profile_candidate(state, "a5", game_id=f"g{i}", run_id=f"r{i}", claim="Method", applicability_conditions=[], evidence_ids=[f"E{i}"]) for i in range(2)]
    markdown = "# Method\n\n- Test once."
    report = {"profile_id": "a5", "improved": True,
              "candidate_revision": hashlib.sha256(markdown.encode()).hexdigest(),
              "baseline_revision": "b", "heldout_games": ["g0"], "metrics": {"gain": 1}}
    with pytest.raises(ValueError, match="must not include"):
        promote_profile_candidates(state, "a5", [r["candidate_id"] for r in rows], markdown, report)
    assert reject_profile_candidates(state, "a5", [r["candidate_id"] for r in rows], "unsupported") == 2
    assert not list_profile_candidates(state, "a5", pending_only=True)


def test_legacy_global_memory_is_deleted_once_and_marker_survives(tmp_path):
    legacy = tmp_path / "global" / "MEMORY.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("stale", encoding="utf-8")
    assert migrate_legacy_global_memory(tmp_path) is True
    assert not legacy.exists()
    assert migrate_legacy_global_memory(tmp_path) is False
    assert json.loads((tmp_path / "migrations/v11-profile-memory.json").read_text())["legacy_file_removed"] is True
    assert profile_memory_path(tmp_path, "a6").parent.name == "a6"
