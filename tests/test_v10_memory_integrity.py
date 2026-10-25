from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v10.frames import FrameStore
from v10.tools.history import History
from v10.tools.memory import (
    MAX_GLOBAL_MEMORY_MARKDOWN_CHARS, GameMemory, MemoryValidationError, default_state_dir,
    game_memory_dir, global_memory_path, read_global_memory, save_global_memory, shared_skill_path,
)


def test_default_persistent_memory_and_shared_skill_are_project_local(monkeypatch) -> None:
    monkeypatch.delenv("V10_STATE_DIR", raising=False)
    project_skills = Path(__file__).resolve().parents[1] / "skills"

    assert default_state_dir() == project_skills
    assert game_memory_dir(default_state_dir(), "ez01") == project_skills / "games/ez01"
    assert global_memory_path(default_state_dir()) == project_skills / "global/MEMORY.md"
    assert shared_skill_path() == project_skills / "SKILL.md"
    assert shared_skill_path().is_file()


def test_global_memory_is_bounded_markdown_and_replacement_is_atomic(tmp_path) -> None:
    memory_text = "# Global Memory\n\n- Compare each object before and after an action."

    assert save_global_memory(tmp_path, memory_text) is True
    path = global_memory_path(tmp_path)
    assert path.read_text() == memory_text + "\n"
    assert read_global_memory(tmp_path) == memory_text
    assert save_global_memory(tmp_path, memory_text) is False

    with pytest.raises(ValueError, match="exceeds"):
        save_global_memory(tmp_path, "# Global Memory\n\n" + "x" * MAX_GLOBAL_MEMORY_MARKDOWN_CHARS)
    assert path.read_text() == memory_text + "\n"

    path.write_text("# Global Memory\n\n" + "x" * MAX_GLOBAL_MEMORY_MARKDOWN_CHARS)
    assert len(read_global_memory(tmp_path)) <= MAX_GLOBAL_MEMORY_MARKDOWN_CHARS
    assert "truncated for prompt budget" in read_global_memory(tmp_path)


def test_model_cannot_author_prediction_verdict_or_evidence_metadata(tmp_path) -> None:
    memory = GameMemory(tmp_path)
    result = memory.propose_update([
        {"op": "upsert", "id": "PR1", "type": "prediction", "claim": "ACTION1 changes one pixel", "status": "pending"},
        {"op": "verdict", "id": "PR1", "status": "matched"},
        {"op": "upsert", "id": "M1", "type": "mechanic", "claim": "ACTION1 changes one pixel", "verified_by": ["PR1"]},
    ])

    assert [row["id"] for row in result["accepted"]] == ["PR1"]
    assert [row["reason"] for row in result["rejected"]] == [
        "prediction_verdict_is_harness_owned",
        "unknown_field: verified_by",
    ]


def test_only_matching_harness_audit_can_record_prediction_verdict(tmp_path) -> None:
    history = History(tmp_path)
    history.append({
        "type": "audit", "kind": "prediction_result", "prediction_id": "PR1",
        "verdict": "matched", "details": {"verdict": "matched"},
    })
    audit_id = history.ids()[0]
    memory = GameMemory(tmp_path)
    memory.propose_update([
        {"op": "upsert", "id": "PR1", "type": "prediction", "claim": "ACTION1 changes one pixel", "status": "pending"},
    ], current_step=1)

    with pytest.raises(MemoryValidationError, match="does not match"):
        memory.record_prediction_verdict("PR1", "failed", audit_id, step=2)
    memory.record_prediction_verdict("PR1", "matched", audit_id, step=2)

    assert memory.get("PR1", through_step=1)["status"] == "pending"
    assert memory.get("PR1", through_step=2)["status"] == "matched"
    assert "verified_by" not in memory.get("PR1")
    assert "evidence_ids" not in memory.get("PR1")


def test_pattern_updates_do_not_store_prediction_evidence(tmp_path) -> None:
    history = History(tmp_path)
    audit = history.append({
        "type": "audit", "kind": "prediction_result", "prediction_id": "PR1",
        "verdict": "matched", "details": {"verdict": "matched"},
    })
    memory = GameMemory(tmp_path)
    memory.propose_update([
        {"op": "upsert", "id": "PR1", "type": "prediction", "claim": "ACTION1 changes one pixel", "status": "pending"},
    ])
    memory.record_prediction_verdict("PR1", "matched", audit["evidence_id"])

    result = memory.propose_update([
        {"op": "upsert", "id": "M1", "type": "mechanic", "claim": "ACTION1 changes one pixel", "status": "confirmed"},
    ])

    assert result["accepted"][0]["id"] == "M1"
    assert "verified_by" not in result["accepted"][0]


def test_step_bounded_read_does_not_reveal_future_revision_or_entries(tmp_path) -> None:
    memory = GameMemory(tmp_path)
    memory.propose_update([
        {"op": "upsert", "id": "H1", "type": "hypothesis", "claim": "first"},
    ], current_step=1)
    memory.propose_update([
        {"op": "upsert", "id": "H2", "type": "hypothesis", "claim": "later"},
    ], current_step=10)

    old_view = memory.read(through_step=1)
    assert old_view["revision"] == 1
    assert [entry["id"] for entry in old_view["entries"]] == ["H1"]


def test_frame_store_round_trips_values_without_int16_truncation(tmp_path) -> None:
    frame = np.array([[70000]], dtype=np.int64)
    store = FrameStore(tmp_path)

    frame_id = store.put(frame)

    assert np.array_equal(store.get(frame_id), frame)


def test_memory_persists_between_runs_but_isolated_by_game(tmp_path) -> None:
    state_dir = tmp_path / "state"
    game_a = game_memory_dir(state_dir, "game/a")
    first = GameMemory(tmp_path / "run-one", store_dir=game_a, run_id="run-one")
    first.propose_update([
        {
            "op": "upsert", "id": "H1", "type": "hypothesis",
            "claim": "A persistent fact. (evidence: T1)",
        },
    ], current_step=1)

    next_run = GameMemory(tmp_path / "run-two", store_dir=game_a, run_id="run-two")
    other_game = GameMemory(tmp_path / "other", store_dir=game_memory_dir(state_dir, "game-b"), run_id="run-three")

    assert next_run.read()["entries"][0]["id"] == "run-one:H1"
    assert "evidence_ids" not in next_run.read()["entries"][0]
    assert next_run.read()["entries"][0]["claim"] == "A persistent fact."
    assert "evidence:" not in (game_a / "MEMORY.md").read_text().casefold()
    assert "evidence_ids" not in next_run.compact()["entries"][0]
    assert other_game.read()["entries"] == []
    assert (game_a / "MEMORY.md").exists()


def test_review_summary_replaces_view_and_is_the_only_compacted_memory(tmp_path) -> None:
    store_dir = game_memory_dir(tmp_path / "state", "ez01")
    first = GameMemory(tmp_path / "run-one", store_dir=store_dir, run_id="run-one")
    first.propose_update([
        {"op": "upsert", "id": "P1", "type": "pattern", "claim": "Prior observation."},
    ], current_step=1)
    markdown = "## Patterns\nA recurring effect. (evidence: T1)\n\n## Uncertain ideas\n- none"
    clean_markdown = "## Patterns\nA recurring effect.\n\n## Uncertain ideas\n- none"

    saved = first.save_review_summary(markdown, current_step=2)

    assert saved["id"] == "review:latest"
    assert saved["revision"] == 2
    assert first.view_path.read_text() == clean_markdown + "\n"
    compact = first.compact(max_entries=5, through_step=2)
    assert len(compact["entries"]) == 1
    assert compact["entries"][0]["type"] == "review"
    assert compact["entries"][0]["claim"] == clean_markdown
    assert "evidence_ids" not in compact["entries"][0]
    assert compact["entries"][0]["id"] == "review:latest"
    assert "evidence_ids" not in first.read()["entries"][-1]

    next_run = GameMemory(tmp_path / "run-two", store_dir=store_dir, run_id="run-two")
    assert next_run.compact(max_entries=5, through_step=0)["entries"][0]["claim"] == clean_markdown


def test_empty_review_preserves_existing_memory_and_review_entries_are_harness_owned(tmp_path) -> None:
    memory = GameMemory(tmp_path)
    memory.propose_update([
        {"op": "upsert", "id": "P1", "type": "pattern", "claim": "Prior observation."},
    ], current_step=1)
    before_view = memory.view_path.read_text()
    before_ledger = memory.path.read_text()

    with pytest.raises(MemoryValidationError, match="non-empty"):
        memory.save_review_summary("  \n", current_step=2)
    result = memory.propose_update([
        {"op": "upsert", "id": "R1", "type": "review", "claim": "Model-authored review"},
    ], current_step=2)

    assert memory.view_path.read_text() == before_view
    assert memory.path.read_text() == before_ledger
    assert result["rejected"][0]["reason"] == "review_entries_are_harness_owned"


def test_failed_review_view_save_rolls_back_latest_summary(tmp_path, monkeypatch) -> None:
    memory = GameMemory(tmp_path, run_id="run-one")
    memory.save_review_summary("## Patterns\nPrevious summary.", current_step=1)
    before_view = memory.view_path.read_text()
    monkeypatch.setattr(memory, "write_view", lambda: (_ for _ in ()).throw(OSError("view is read-only")))

    with pytest.raises(OSError, match="view is read-only"):
        memory.save_review_summary("## Patterns\nReplacement summary.", current_step=2)

    assert memory.view_path.read_text() == before_view
    assert memory.compact()["entries"][0]["claim"] == "## Patterns\nPrevious summary."


def test_pattern_updates_are_independent_incremental_and_idempotent(tmp_path) -> None:
    state_dir = game_memory_dir(tmp_path / "state", "ez01")
    memory = GameMemory(tmp_path / "run-one", store_dir=state_dir, run_id="run-one", game_id="ez01")
    first = memory.apply_review_changes([
        {
            "operation": "upsert", "pattern_type": "mechanic", "status": "confirmed",
            "claim": "Up closes the gap by moving blue upward.",
            "applicability_conditions": ["The squares are vertically separated."],
            "state_selector": {"level": 0},
            "expected_observations": [{
                "action": "up", "state_selector": {"level": 0},
                "measurable_prediction": {"all": [
                    {"path": "measurements.changed_pixels", "op": "eq", "value": 128},
                ]},
            }],
        },
        {
            "operation": "upsert", "pattern_type": "strategy", "status": "confirmed",
            "claim": "Stack, then separate to finish a level.",
            "applicability_conditions": ["A level uses the two-square mechanic."],
            "goal": "Complete the current level.", "preconditions": ["Two squares are visible."],
            "action_policy": "Press up until stacked, then press up once more.",
            "success_conditions": ["levels_completed increases."],
        },
    ], game_id="ez01", current_step=2, covered_event_count=2)

    assert first["added"] == 2
    assert first["committed"] is True
    mechanic, strategy = memory.patterns("ez01")
    assert mechanic["status"] == "confirmed"
    assert "confidence" not in mechanic
    assert "supporting_evidence" not in mechanic
    assert strategy["pattern_type"] == "strategy"
    assert strategy["goal"] == "Complete the current level."
    view_text = memory.view_path.read_text()
    assert "Refs:" not in view_text
    assert "confidence" not in view_text
    assert "T2" not in view_text

    prior_strategy = dict(strategy)
    row_count = len(memory.ledger())
    updated = memory.apply_review_changes([{
        "operation": "upsert", "pattern_id": mechanic["id"],
        "status": "candidate",
        "applicability_conditions": ["The squares are vertically separated, not stacked."],
    }], game_id="ez01", current_step=4, covered_event_count=4)
    assert updated["updated"] == 1
    current = {entry["id"]: entry for entry in memory.patterns("ez01")}
    assert current[mechanic["id"]]["status"] == "candidate"
    assert current[strategy["id"]] == prior_strategy
    history = [row for row in memory.ledger() if row.get("id") == mechanic["id"]]
    assert len(history) == 2
    assert history[-1]["pattern_revision"] == 2

    before_repeat = len(memory.ledger())
    duplicate = memory.apply_review_changes([{
        "operation": "upsert", "pattern_id": mechanic["id"], "status": "candidate",
        "applicability_conditions": ["The squares are vertically separated, not stacked."],
    }], game_id="ez01", current_step=4, covered_event_count=4)
    assert duplicate["duplicate_review"] is True
    assert len(memory.ledger()) == before_repeat
    assert row_count < before_repeat


def test_run_state_is_owned_by_its_run_and_second_run_reuses_patterns(tmp_path) -> None:
    state_dir = game_memory_dir(tmp_path / "state", "ez01")
    first = GameMemory(tmp_path / "run-one", store_dir=state_dir, run_id="run-one", game_id="ez01")
    first.record_run_state("ez01", {
        "level": 5, "levels_completed": 5, "win_levels": 5, "game_state": "WIN", "terminal": True,
    }, step=25)
    first.apply_review_changes([{
        "operation": "upsert", "pattern_type": "mechanic", "status": "confirmed", "claim": "Up moves blue upward by eight pixels.",
        "applicability_conditions": ["Squares are vertically separated."],
        "state_selector": {},
    }], game_id="ez01", current_step=25, covered_event_count=2)

    second = GameMemory(tmp_path / "run-two", store_dir=state_dir, run_id="run-two", game_id="ez01")
    second.record_run_state("ez01", {
        "level": 0, "levels_completed": 0, "win_levels": 5, "game_state": "NOT_FINISHED", "terminal": False,
    }, step=0)
    current_state = {"level": 0, "levels_completed": 0, "game_state": "NOT_FINISHED", "terminal": False}
    prompt_memory = second.compact(game_id="ez01", current_state=current_state)

    assert [entry["pattern_type"] for entry in prompt_memory["entries"]] == ["mechanic"]
    assert prompt_memory["entries"][0]["id"] == first.patterns("ez01")[0]["id"]
    assert all(entry.get("run_id") != "run-one" for entry in prompt_memory["entries"] if entry.get("type") == "state")
    assert "level 0" in second.view_path.read_text()
    assert "From run" not in second.view_path.read_text()


def test_legacy_run_ledger_migrates_before_run_dir_reuse(tmp_path) -> None:
    from v10.runlog import write_json

    legacy = tmp_path / "run"
    legacy.mkdir()
    memory = GameMemory(legacy)
    memory.propose_update([
        {"op": "upsert", "id": "H1", "type": "hypothesis", "claim": "Older knowledge"},
    ], current_step=1)
    write_json(legacy / "run_manifest.json", {"run_id": "old-run", "game_id": "fixture"})

    imported = GameMemory.migrate_run_ledger(legacy, tmp_path / "persistent", "fixture")

    assert imported == 1
    assert GameMemory(legacy, store_dir=tmp_path / "persistent").read()["entries"][0]["id"] == "old-run:H1"
    assert GameMemory.migrate_run_ledger(legacy, tmp_path / "persistent", "fixture") == 0


def test_review_rejects_unusable_state_selectors_without_persisting_evidence_refs(tmp_path) -> None:
    memory = GameMemory(tmp_path / "run", store_dir=tmp_path / "state", run_id="run-a", game_id="ez01")
    result = memory.apply_review_changes([
        {
            "operation": "upsert", "pattern_type": "mechanic", "status": "confirmed", "claim": "A measured action changes the board.",
            "applicability_conditions": [], "state_selector": {"unknown_field": True},
        },
        {
            "operation": "upsert", "pattern_type": "mechanic", "status": "confirmed", "claim": "A measured action changes the board.",
            "applicability_conditions": [], "state_selector": {"level": 0},
        },
    ], game_id="ez01", current_step=2, covered_event_count=2)

    assert result["added"] == 1
    assert result["rejected"] == [{"index": 0, "pattern_id": None, "reason": "invalid_state_selector"}]
    assert "supporting_evidence" not in memory.patterns("ez01")[0]


def test_retirement_is_a_revision_and_does_not_erase_history(tmp_path) -> None:
    memory = GameMemory(tmp_path / "run", store_dir=tmp_path / "state", run_id="run-a", game_id="ez01")
    memory.apply_review_changes([{
        "operation": "upsert", "pattern_type": "hypothesis", "claim": "The cue indicates a reset.",
    }], game_id="ez01", current_step=1, covered_event_count=1)
    pattern_id = memory.patterns("ez01")[0]["id"]

    result = memory.apply_review_changes([{
        "operation": "retire", "pattern_id": pattern_id,
    }], game_id="ez01", current_step=2, covered_event_count=2)

    revisions = [row for row in memory.ledger() if row.get("id") == pattern_id]
    assert result["retired"] == 1
    assert [row["pattern_revision"] for row in revisions] == [1, 2]
    assert revisions[-1]["status"] == "retired"
    assert memory.patterns("ez01") == []
    assert memory.patterns("ez01", include_retired=True)[0]["claim"] == "The cue indicates a reset."


def test_review_checkpoints_store_event_count_without_observation_ids(tmp_path) -> None:
    memory = GameMemory(tmp_path / "run", store_dir=tmp_path / "state", run_id="run-a", game_id="ez01")
    memory.apply_review_changes([], game_id="ez01", current_step=1, covered_event_count=1)

    result = memory.apply_review_changes([{
        "operation": "upsert", "pattern_type": "hypothesis", "claim": "A later cue may signal a reset.",
    }], game_id="ez01", current_step=2, covered_event_count=2)

    assert result["added"] == 1
    assert memory.reviewed_event_count("ez01") == 2
    assert memory.has_review_coverage("ez01", 2)
    assert "T1" not in memory.path.read_text()
    assert "T2" not in memory.path.read_text()


def test_partial_review_rejections_do_not_checkpoint_unreviewed_evidence(tmp_path) -> None:
    memory = GameMemory(tmp_path / "run", store_dir=tmp_path / "state", run_id="run-a", game_id="ez01")
    result = memory.apply_review_changes([
        {
            "operation": "upsert", "pattern_type": "mechanic", "claim": "Up moves the player.",
            "applicability_conditions": [], "state_selector": {},
        },
        {
            "operation": "upsert", "pattern_type": "strategy", "claim": "Move to the goal.",
            "goal": "Complete the level.", "preconditions": [], "action_policy": "Move to the goal.",
            "success_conditions": ["levels_completed increases."],
            "expected_observations": [{"action": "up", "prediction": "pixels change"}],
        },
    ], game_id="ez01", current_step=3, covered_event_count=2)

    assert result["added"] == 1
    assert result["checkpointed"] is False
    assert result["rejected"][0]["reason"] == "invalid_expected_observations"
    assert not memory.has_review_coverage("ez01", 2)
    assert memory.reviewed_event_count("ez01") == 0

    repaired = memory.apply_review_changes([{
        "operation": "upsert", "pattern_type": "strategy", "claim": "Move to the goal.",
        "goal": "Complete the level.", "preconditions": [], "action_policy": "Move to the goal.",
        "success_conditions": ["levels_completed increases."],
        "expected_observations": [{
            "action": "up", "measurable_prediction": {"all": [
                {"path": "measurements.changed_pixels", "op": "gt", "value": 0},
            ]},
        }],
    }], game_id="ez01", current_step=3, covered_event_count=2)

    assert repaired["added"] == 1
    assert repaired["checkpointed"] is True
    assert memory.has_review_coverage("ez01", 2)
    assert len(memory.patterns("ez01")) == 2
