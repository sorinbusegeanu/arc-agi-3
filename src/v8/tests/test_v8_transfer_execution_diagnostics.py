from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v8 import information_flow_diagnostics as flow
from v8 import learning_fixes_v088 as learning
from v8.arena import NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    ValidationState,
    stable_u64,
)
from v8.publication import _StrategyRow
from v8.transfer import TransferCandidate, TransferValidator


def _memory(
    index: int,
    level: MemoryLevel,
    memory_type: MemoryType,
    key_parts: tuple[int, ...] | None = None,
) -> NodeRecord:
    key = key_parts or (index,)
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=index,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=4,
        significance_sum=1.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=1.0,
        explanatory_sum=1.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=index,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
    )


def _candidate(row: NodeRecord) -> TransferCandidate:
    return TransferCandidate(
        row.uid,
        1,
        1.0,
        (stable_u64("source", person=b"v8-game"),),
        MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (999,)),
        (stable_u64("correspondence", person=b"v8-game"),),
    )


def _empty_inventory() -> dict[str, object]:
    return {
        "identity_index": "GAME_PROVENANCE target UID low word",
        "identities_actually_present_count": 0,
        "memory_counts_by_level": {},
        "m3_exists": False,
        "m4_exists": False,
        "executable_lower_level_memory_exists": False,
    }


def test_target_memory_exists_and_is_executable() -> None:
    ancestor = _memory(1, MemoryLevel.M3, MemoryType.ROLE)
    outcome = _memory(2, MemoryLevel.M6, MemoryType.OUTCOME)
    strategy = _memory(
        3,
        MemoryLevel.M7,
        MemoryType.STRATEGY,
        (1, outcome.uid.hi, outcome.uid.lo, 77),
    )
    cache_row = _StrategyRow(1, outcome.uid, strategy.uid, 4, 0.8, 2.0, 77, False, True)
    view = SimpleNamespace(
        _node_by_uid={row.uid: row for row in (ancestor, outcome, strategy)},
        _parents={strategy.uid: {ancestor.uid}},
        _strategy_by_context={77: (cache_row,)},
        _strategy_fallback=(cache_row,),
    )

    snapshot = learning._candidate_execution_snapshot(
        view, (ancestor, outcome, strategy), ancestor.uid
    )
    event = learning._target_resolution_event(
        read_view=view,
        nodes=(ancestor, outcome, strategy),
        candidate=_candidate(ancestor),
        game_id="target",
        target_hash=stable_u64("target", person=b"v8-game"),
        probe_diagnostic={"available_action_ids": [1], "observed_context_buckets": [77]},
        candidate_snapshot=snapshot,
        target_inventory=_empty_inventory(),
        trajectory={"successful_trajectory_exists": False},
        executable_reference=learning._executable_reference(view, (ancestor, outcome, strategy)),
        used=1,
    )

    assert snapshot["cached_executable_descendant_count"] == 1
    assert event["executable_predicate_result"] is True
    assert event["exact_executable_predicate_failure_reason"] is None


def test_target_memory_exists_but_fails_executable_predicate() -> None:
    ancestor = _memory(10, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={ancestor.uid: ancestor},
        _parents={},
        _strategy_by_context={},
        _strategy_fallback=(),
    )
    snapshot = learning._candidate_execution_snapshot(view, (ancestor,), ancestor.uid)

    assert snapshot["lookup_result_status"] == "found_structural_ancestor"
    assert snapshot["executable_predicate_failure_reason"] == (
        "no_m7_strategy_descendant_for_required_ancestor"
    )
    assert "M7 strategy descendant" in snapshot["missing_or_invalid_executable_fields"]


def test_target_memory_lookup_returns_no_memory() -> None:
    missing = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (404,))
    view = SimpleNamespace(
        _node_by_uid={}, _parents={}, _strategy_by_context={}, _strategy_fallback=()
    )
    snapshot = learning._candidate_execution_snapshot(view, (), missing)

    assert snapshot["lookup_result_status"] == "not_found"
    assert snapshot["required_ancestor_memory"] is None
    assert snapshot["executable_predicate_failure_reason"] == (
        "required_ancestor_lookup_returned_none"
    )


def test_world_provenance_identity_mismatch_is_visible() -> None:
    source_hash = stable_u64("source", person=b"v8-game")
    target_hash = stable_u64("target", person=b"v8-game")
    ancestor = _memory(20, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={ancestor.uid: ancestor},
        _parents={},
        _strategy_by_context={},
        _strategy_fallback=(),
        _v839_direct_games={ancestor.uid: {source_hash}},
    )
    inventories = learning._direct_target_memory_inventories(
        view, (ancestor,), (target_hash,)
    )
    snapshot = learning._candidate_execution_snapshot(view, (ancestor,), ancestor.uid)
    event = learning._target_resolution_event(
        read_view=view,
        nodes=(ancestor,),
        candidate=_candidate(ancestor),
        game_id="target",
        target_hash=target_hash,
        probe_diagnostic={},
        candidate_snapshot=snapshot,
        target_inventory=inventories[target_hash],
        trajectory={"successful_trajectory_exists": False},
        executable_reference={},
        used=0,
    )

    assert event["requested_lookup_identity"]["provenance_world_hash"] == target_hash
    assert source_hash in event["candidate_memory_identities_actually_present"][
        "formation_world_hashes"
    ]
    assert event["target_specific_memory_lookup_performed"] is False
    assert event["target_world_memory"]["identities_actually_present_count"] == 0


def test_m3_present_m4_absent_is_diagnosed_explicitly() -> None:
    target_hash = stable_u64("target", person=b"v8-game")
    m3 = _memory(30, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={m3.uid: m3},
        _v839_direct_games={m3.uid: {target_hash}},
    )
    inventory = learning._direct_target_memory_inventories(
        view, (m3,), (target_hash,)
    )[target_hash]

    assert inventory["m3_exists"] is True
    assert inventory["m4_exists"] is False
    assert inventory["memory_counts_by_level"] == {"M3": 1}


def test_successful_trajectory_exists_but_is_not_linked_to_target_memory() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        optimizer_root = Path(raw_root) / "trajectory_optimizer"
        optimizer_root.mkdir(parents=True)
        (optimizer_root / "best_successful.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "games": {
                        "target": {
                            "trajectory_id": "target-win",
                            "successes": 1,
                            "levels": [{"level": 0, "actions": [1, 2]}],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        row = learning._trajectory_inventory(("target",))["target"]

    assert row["successful_trajectory_exists"] is True
    assert row["trajectory_id"] == "target-win"
    assert row["executable_representation_available"] is True
    assert row["linked_memory_id"] is None


def test_resolution_instrumentation_does_not_change_scheduler_result() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        ancestor = _memory(40, MemoryLevel.M3, MemoryType.ROLE)
        candidate = _candidate(ancestor)
        transfer = TransferValidator()
        transfer.candidates = lambda *_args, **_kwargs: (candidate,)
        peers = SimpleNamespace(
            transfer=transfer,
            record_transfer_trial=transfer.record_trial,
            _append_evidence=lambda *_args, **_kwargs: None,
        )
        view = SimpleNamespace(
            node_records=lambda: (ancestor,),
            source_games=lambda _uid: frozenset(),
            _node_by_uid={ancestor.uid: ancestor},
            _parents={},
            _strategy_by_context={},
            _strategy_fallback=(),
            _v839_direct_games={},
        )
        runtime = SimpleNamespace(peers=peers, read_view=view)

        with patch.object(learning, "_held_out_games", return_value=("target",)), patch.object(
            learning, "_probe_policy_v088", return_value=(0.0, 0)
        ):
            result = learning._run_automatic_transfer_experiments_v088(
                runtime,
                games=("source",),
                env_root=None,
                seed=1,
                steps_per_trial=2,
                max_trials=1,
            )

        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]
        detail = [row for row in records if row["stage"] == "target_memory_resolution"]

    assert (result.attempted, result.completed, result.passed) == (0, 0, 0)
    assert detail[0]["exact_executable_predicate_failure_reason"] == (
        "no_m7_strategy_descendant_for_required_ancestor"
    )
    assert detail[0]["target_memory_level"] == "M3"


def test_target_resolution_detail_collection_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        for index in range(100):
            flow.emit_bounded(
                "transfer",
                "target_memory_resolution",
                input_count=1,
                output_count=0,
                rejection_counts={"not_executable": 1},
                examples=({"target_world": f"target-{index}"},),
            )
        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]

    assert len(records) == flow.MAX_EXAMPLES

