from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v8 import information_flow_diagnostics as flow
from v8 import learning_fixes_v088 as learning
from v8 import trajectory_optimizer_v818 as v818
from v8 import trajectory_target_minimization_v820 as v820
from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.trajectory_optimizer_v814 import (
    ReplayAnchor,
    SuccessfulTrajectory,
    TrajectoryCandidate,
    TrajectoryOptimizationService,
    TrajectoryTarget,
    _trajectory_id,
)
from v8.transfer import TransferCandidate, TransferValidator


def _node(index: int, level: MemoryLevel = MemoryLevel.M3) -> NodeRecord:
    return NodeRecord(
        MemoryUid.from_key(level, MemoryType.ROLE, (index,)), index, int(level),
        int(MemoryType.ROLE), (index,), 4, 1.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 1.0, index, 0, int(CognitiveState.ACTIVE),
        int(ValidationState.STRUCTURAL),
    )


def _edge(left: NodeRecord, right: NodeRecord, score: float = 1.0) -> EdgeRecord:
    return EdgeRecord(
        left.uid, int(RelationType.TRANSFER_CORRESPONDENCE), right.uid,
        1, 1, float(score), 1.0, 1, 1,
    )


def _source(actions=(1, 2, 3), *, round_index: int = 0) -> SuccessfulTrajectory:
    anchor = ReplayAnchor("diagnostic-world", 0, (), None)
    target = TrajectoryTarget(1, "LEVEL")
    values = tuple(actions)
    return SuccessfulTrajectory(
        _trajectory_id(anchor, target, values), anchor, target, values,
        MemoryUid.zero(), MemoryUid.zero(), int(round_index),
    )


def _records(root: Path, *, stage: str | None = None) -> list[dict[str, object]]:
    path = root / flow.LOG_NAME
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows if stage is None else [row for row in rows if row["stage"] == stage]


def test_transfer_candidate_rejections_are_exact_and_reconcile() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        validator = TransferValidator()

        def run(left, right, score, games):
            return validator.candidates(
                (left, right), (_edge(left, right, score),),
                provenance=lambda uid: frozenset(games.get(uid, ())),
            )

        m3 = _node(1)
        m2 = _node(2, MemoryLevel.M2)
        run(m3, m2, 1.0, {m3.uid: {1}, m2.uid: {2}})
        a, b = _node(3), _node(4)
        run(a, b, 0.0, {a.uid: {1}, b.uid: {2}})
        run(a, b, 1.0, {a.uid: set(), b.uid: {2}})
        run(a, b, 1.0, {a.uid: {1}, b.uid: set()})
        run(a, b, 1.0, {a.uid: {1}, b.uid: {1}})
        run(a, b, 1.0, {a.uid: {1, 2}, b.uid: {2}})

        c = _node(5)
        validator.candidates(
            (a, b, c), (_edge(a, b, 0.5), _edge(a, c, 0.9)),
            provenance=lambda uid: frozenset({a.uid: {1}, b.uid: {2}, c.uid: {3}}[uid]),
        )

        rows = _records(Path(raw_root), stage="candidate_selection")
        reasons = {
            reason
            for row in rows
            for reason in row["rejection_counts"]
        }
        assert {
            "source_or_target_not_m3_or_m4",
            "nonpositive_correspondence_score",
            "source_provenance_missing",
            "target_provenance_missing",
            "provenance_not_distinct",
            "no_new_target_world",
            "superseded_by_stronger_correspondence",
        } <= reasons
        for row in rows:
            assert row["input_count"] == row["output_count"] + sum(row["rejection_counts"].values())


def test_accepted_transfer_candidate_is_counted_without_scheduler_change() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        row = _node(10)
        candidate = TransferCandidate(row.uid, 1, 0.9, (), _node(11).uid, (22,))
        transfer = TransferValidator()
        transfer.candidates = lambda *_args, **_kwargs: (candidate,)
        peers = SimpleNamespace(
            transfer=transfer,
            record_transfer_trial=transfer.record_trial,
            _append_evidence=lambda *_args, **_kwargs: None,
        )
        view = SimpleNamespace(node_records=lambda: (row,), source_games=lambda _uid: frozenset())
        runtime = SimpleNamespace(peers=peers, read_view=view, generation=1)

        with patch.object(learning, "_held_out_games", return_value=("held-out",)), patch.object(
            learning, "_probe_policy_v088", side_effect=((1.0, 1), (0.0, 0))
        ) as probe:
            result = learning._run_automatic_transfer_experiments_v088(
                runtime, games=("train",), env_root=None, seed=1,
                steps_per_trial=4, max_trials=1,
            )

        assert (result.attempted, result.completed, result.passed) == (1, 1, 1)
        assert probe.call_count == 2
        summary = _records(Path(raw_root), stage="pipeline_summary")[-1]["counters"]
        assert summary["admissible_candidates"] == 1
        assert summary["eligible_target_worlds"] == 1
        assert summary["scheduled_trials"] == summary["completed_trials"] == summary["passed_trials"] == 1
        assert summary["failed_trials"] == 0


def test_optimizer_boundary_exposes_trajectories_seen_bypass_by_id() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        service = TrajectoryOptimizationService(
            Path(raw_root) / "trajectory_optimizer", validator=lambda _candidate: None
        )
        service._v819_lock = threading.RLock()
        service._v819_source_seen = set()
        service._v819_source_pending = {}
        service._v819_source_kind = {}
        row = _source()
        with patch.object(v818, "_route_candidate", return_value=True):
            accepted = solve_fix._BASE_SERVICE_SUBMIT_V819(service, row)

        assert accepted is True
        assert service._trajectories_seen == 0
        record = _records(Path(raw_root), stage="optimizer_submission")[-1]
        example = record["examples"][0]
        assert example["trajectory_id"] == row.trajectory_id
        assert example["optimizer_received"] is True
        assert example["counted_in_trajectories_seen"] is False
        assert example["counter_path"] == "source_validation_bypass"


def test_optimizer_generation_validation_and_acceptance_reconcile() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        service = TrajectoryOptimizationService(
            Path(raw_root) / "trajectory_optimizer", validator=lambda _candidate: None
        )
        source = _source((1, 2, 3))
        candidates = (
            TrajectoryCandidate("candidate-a", source, "DELETE_ACTION", (1, 2), 2, 1),
            TrajectoryCandidate("candidate-b", source, "DELETE_ACTION", (1, 3), 1, 1),
        )
        service._sources.put(source)
        routed = []

        def route(_service, candidate):
            routed.append(candidate)
            if len(routed) == len(candidates):
                service._stop.set()
            return True

        with patch.object(v818, "_restore_pending_sources"), patch.object(
            v818, "_ingest_inbox_v818"
        ), patch.object(v818, "_start_waiting_validators"), patch.object(
            v820, "_BASE_GENERATE_V818", return_value=candidates
        ), patch.object(v818, "_route_candidate", side_effect=route):
            v818._optimizer_loop_v818(service)

        service._stop.clear()
        q = service._v818_game_queues.setdefault("diagnostic-world", queue.Queue())
        q.put(candidates[0])

        class Validator:
            def __init__(self, *_args):
                pass

            def validate(self, _candidate):
                service._stop.set()
                return v818.V818ValidationResult(
                    True, 2, "validated", "LEVEL", 1, 2, 2,
                )

        with patch.object(v818, "_GameReplayValidator", Validator), patch.object(
            v818, "_retire_game_validator"
        ), patch.object(service, "submit_trajectory", return_value=False):
            v818._game_validator_loop(service, "diagnostic-world")

        metrics = service.metrics()
        assert metrics.candidates_generated == 2
        assert metrics.validations == 2
        assert metrics.validation_successes == 2
        assert metrics.validated_variants == 1
        generated = _records(Path(raw_root), stage="optimizer_candidate_generation")[-1]
        validation = _records(Path(raw_root), stage="optimizer_validation")[-1]
        assert generated["output_count"] == 2
        assert validation["examples"][0]["trajectory_id"] == source.trajectory_id
        assert validation["examples"][0]["accepted_variant_id"] == "candidate-a"
        summary = _records(Path(raw_root), stage="pipeline_summary")[-1]["counters"]
        assert summary["candidates_generated"] == 2
        assert summary["validations"] == 2
        assert summary["validation_successes"] == 2
        assert summary["accepted_variants"] == 1


def test_diagnostics_do_not_change_candidate_results() -> None:
    left, right = _node(20), _node(21)
    edge = _edge(left, right, 0.75)
    provenance = lambda uid: frozenset({left.uid: {1}, right.uid: {2}}[uid])
    with patch.dict(os.environ, {}, clear=True):
        expected = TransferValidator().candidates((left, right), (edge,), provenance=provenance)
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        actual = TransferValidator().candidates((left, right), (edge,), provenance=provenance)
    assert actual == expected


def test_example_and_detail_logging_are_bounded() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        flow.emit(
            "transfer", "bounded_batch", input_count=100, output_count=0,
            examples=({"candidate_uid": index} for index in range(100)),
        )
        for index in range(100):
            flow.emit_bounded(
                "trajectory_optimizer", "bounded_detail",
                input_count=1, output_count=1,
                examples=({"trajectory_id": str(index)},),
            )
        rows = _records(Path(raw_root))
        batch = [row for row in rows if row["stage"] == "bounded_batch"][0]
        details = [row for row in rows if row["stage"] == "bounded_detail"]
        assert len(batch["examples"]) == flow.MAX_EXAMPLES
        assert len(details) == flow.MAX_EXAMPLES
