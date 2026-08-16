from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import v8
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.trajectory_optimizer_v814 import (
    ReplayAnchor,
    SuccessfulTrajectory,
    TrajectoryOptimizationService,
    TrajectoryOptimizerConfig,
    TrajectoryTarget,
    ValidatedTrajectory,
    generate_optimization_candidates,
    select_validated_variant,
)
from v8.trajectory_validation_v814 import ValidationResult, validate_candidate


class FakeReplayAdapter:
    def __init__(self, accepted: tuple[int, ...]) -> None:
        self.accepted = tuple(accepted)

    def validate(self, candidate) -> ValidationResult:
        success = tuple(candidate.actions) == self.accepted
        return ValidationResult(
            success,
            len(candidate.actions),
            "target_preserved" if success else "target_not_reached",
        )


def source(actions, *, prefix=(), trajectory_id="source", round_index=0):
    return SuccessfulTrajectory(
        trajectory_id,
        ReplayAnchor("generic-world", 7, tuple(prefix), None),
        TrajectoryTarget(1, "LEVEL"),
        tuple(actions),
        MemoryUid.zero(),
        MemoryUid.zero(),
        round_index,
    )


class TrajectoryOptimizerV814Tests(unittest.TestCase):
    def test_reduces_repeated_pairs_and_all_copy_counts(self) -> None:
        row = source((1, 2, 1, 2, 1, 2, 1, 2, 9))
        candidates = generate_optimization_candidates(
            row,
            TrajectoryOptimizerConfig(max_candidates_per_round=128),
        )
        repeats = {
            candidate.actions
            for candidate in candidates
            if candidate.edit_kind == "REDUCE_REPEAT"
        }
        self.assertIn((1, 2, 9), repeats)
        self.assertIn((1, 2, 1, 2, 9), repeats)
        self.assertIn((1, 2, 1, 2, 1, 2, 9), repeats)

    def test_reduces_repeated_triples_and_four_sets(self) -> None:
        row = source((1, 2, 3) * 4 + (8,))
        candidates = generate_optimization_candidates(
            row,
            TrajectoryOptimizerConfig(max_candidates_per_round=128),
        )
        repeats = {
            candidate.actions
            for candidate in candidates
            if candidate.edit_kind == "REDUCE_REPEAT"
        }
        self.assertIn((1, 2, 3, 8), repeats)
        self.assertIn((1, 2, 3, 1, 2, 3, 8), repeats)
        self.assertIn((1, 2, 3, 1, 2, 3, 1, 2, 3, 8), repeats)

    def test_generates_single_and_segment_deletions_without_state_comparison(self) -> None:
        row = source((10, 20, 30, 40, 50))
        candidates = generate_optimization_candidates(
            row,
            TrajectoryOptimizerConfig(
                max_candidates_per_round=128,
                max_segment_delete=3,
            ),
        )
        by_kind = {}
        for candidate in candidates:
            by_kind.setdefault(candidate.edit_kind, set()).add(candidate.actions)
        self.assertIn((10, 30, 40, 50), by_kind["DELETE_ACTION"])
        self.assertIn((10, 40, 50), by_kind["DELETE_SEGMENT"])

    def test_variant_selection_uses_action_prefix_not_observation_identity(self) -> None:
        uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        row = ValidatedTrajectory(
            "variant",
            ReplayAnchor("generic-world", 7, (4, 5, 6), None),
            TrajectoryTarget(2, "LEVEL"),
            (7, 8),
            uid,
            MemoryUid.zero(),
            MemoryUid.zero(),
            5,
            "REDUCE_REPEAT",
        )
        self.assertIsNone(
            select_validated_variant(
                (row,),
                source_id="generic-world",
                seed=7,
                action_history=(4, 5),
            )
        )
        selected = select_validated_variant(
            (row,),
            source_id="generic-world",
            seed=7,
            action_history=(4, 5, 6),
        )
        self.assertIs(selected, row)

    def test_generic_validation_adapter_receives_only_edited_trajectory(self) -> None:
        row = source((1, 2, 1, 2, 3))
        candidate = next(
            candidate
            for candidate in generate_optimization_candidates(
                row,
                TrajectoryOptimizerConfig(max_candidates_per_round=128),
            )
            if candidate.actions == (1, 2, 3)
        )
        result = validate_candidate(candidate, FakeReplayAdapter((1, 2, 3)))
        self.assertTrue(result.success)

    def test_service_iteratively_accepts_shorter_validated_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            adapter = FakeReplayAdapter((1, 2, 3))
            service = TrajectoryOptimizationService(
                Path(root),
                validator=adapter.validate,
                config=TrajectoryOptimizerConfig(
                    max_candidates_per_round=128,
                    max_optimization_rounds=3,
                    poll_interval_seconds=0.01,
                ),
            )
            service.start()
            try:
                self.assertTrue(service.submit_trajectory(source((1, 2, 1, 2, 3))))
                self.assertTrue(service.drain(timeout=3.0))
                state = service.state_dict()
                validated = state["validated"]
                self.assertEqual(len(validated), 1)
                self.assertEqual(tuple(validated[0]["actions"]), (1, 2, 3))
                self.assertEqual(validated[0]["edit_kind"], "REDUCE_REPEAT")
                published = json.loads(
                    (Path(root) / "validated.json").read_text(encoding="utf-8")
                )
                self.assertEqual(tuple(published["validated"][0]["actions"]), (1, 2, 3))
                metrics = service.metrics()
                self.assertGreater(metrics.candidates_generated, 0)
                self.assertGreater(metrics.validations, 0)
                self.assertGreater(metrics.validation_successes, 0)
            finally:
                service.stop(drain=False, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
