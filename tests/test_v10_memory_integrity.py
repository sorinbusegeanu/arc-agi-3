from __future__ import annotations

import numpy as np
import pytest

from v10.frames import FrameStore
from v10.tools.history import History
from v10.tools.memory import GameMemory, MemoryValidationError, game_memory_dir


def test_model_cannot_author_prediction_verdict_or_self_promote(tmp_path) -> None:
    memory = GameMemory(tmp_path)
    result = memory.propose_update([
        {"op": "upsert", "id": "PR1", "type": "prediction", "claim": "ACTION1 changes one pixel", "status": "pending"},
        {"op": "verdict", "id": "PR1", "status": "matched"},
        {"op": "upsert", "id": "M1", "type": "mechanic", "claim": "ACTION1 changes one pixel", "verified_by": ["PR1"]},
    ])

    assert [row["id"] for row in result["accepted"]] == ["PR1"]
    assert [row["reason"] for row in result["rejected"]] == [
        "prediction_verdict_is_harness_owned",
        "promotion_without_matched_prediction: PR1",
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
    assert memory.get("PR1")["verified_by"] == audit_id
    assert audit_id in memory.get("PR1")["evidence_ids"]


def test_harness_matched_prediction_can_support_mechanic_promotion(tmp_path) -> None:
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
        {"op": "upsert", "id": "M1", "type": "mechanic", "claim": "ACTION1 changes one pixel", "verified_by": ["PR1"]},
    ])

    assert result["accepted"][0]["id"] == "M1"


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
        {"op": "upsert", "id": "H1", "type": "hypothesis", "claim": "A persistent fact"},
    ], known_evidence_ids={"T1"}, current_step=1)

    next_run = GameMemory(tmp_path / "run-two", store_dir=game_a, run_id="run-two")
    other_game = GameMemory(tmp_path / "other", store_dir=game_memory_dir(state_dir, "game-b"), run_id="run-three")

    assert next_run.read()["entries"][0]["id"] == "run-one:H1"
    assert next_run.read()["entries"][0]["evidence_ids"] == ["run-one:T1"]
    assert other_game.read()["entries"] == []
    assert (game_a / "MEMORY.md").exists()


def test_legacy_run_ledger_migrates_before_run_dir_reuse(tmp_path) -> None:
    from v10.runlog import write_json

    legacy = tmp_path / "run"
    legacy.mkdir()
    memory = GameMemory(legacy)
    memory.propose_update([
        {"op": "upsert", "id": "H1", "type": "hypothesis", "claim": "Older knowledge", "evidence_ids": ["T1"]},
    ], known_evidence_ids={"T1"}, current_step=1)
    write_json(legacy / "run_manifest.json", {"run_id": "old-run", "game_id": "fixture"})

    imported = GameMemory.migrate_run_ledger(legacy, tmp_path / "persistent", "fixture")

    assert imported == 1
    assert GameMemory(legacy, store_dir=tmp_path / "persistent").read()["entries"][0]["id"] == "old-run:H1"
    assert GameMemory.migrate_run_ledger(legacy, tmp_path / "persistent", "fixture") == 0
