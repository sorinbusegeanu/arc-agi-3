from __future__ import annotations

import unittest

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningCoordinator,
    FrontierCandidate,
    FrontierScope,
    FrontierSource,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState


def m6(value: int) -> MemoryUid:
    return MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (value, 2, 3))


def m7(value: int) -> MemoryUid:
    return MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (value, 4, 5))


def row(value: int, *, cost: int = 20) -> FrontierCandidate:
    return FrontierCandidate(
        m7(value),
        f"trajectory-{value}",
        value * 100,
        cost,
        2,
        2,
        int(ValidationState.TESTED),
        FrontierSource.SAMPLER,
        value,
    )


class FrontierCoordinationV819Tests(unittest.TestCase):
    def test_level_frontier_version_is_monotonic_across_distinct_scopes(self) -> None:
        coordinator = AdaptiveLearningCoordinator()
        first_outcome = m6(1)
        second_outcome = m6(2)
        first_scope = FrontierScope("world", 3, 100, first_outcome.hi, first_outcome.lo)
        second_scope = FrontierScope("world", 3, 200, second_outcome.hi, second_outcome.lo)

        coordinator.observe_frontier_candidate(
            first_scope,
            row(1, cost=30),
            terminal_state="LEVEL",
            generation=1,
        )
        first_version = coordinator._record("world", 3).frontier_version
        coordinator.observe_frontier_candidate(
            first_scope,
            row(2, cost=20),
            terminal_state="LEVEL",
            generation=2,
        )
        second_version = coordinator._record("world", 3).frontier_version
        coordinator.observe_frontier_candidate(
            second_scope,
            row(3, cost=25),
            terminal_state="LEVEL",
            generation=3,
        )
        third_version = coordinator._record("world", 3).frontier_version

        self.assertLess(first_version, second_version)
        self.assertLess(second_version, third_version)

    def test_source_validation_uses_isolated_cost_frontier_key(self) -> None:
        target_uid = m6(7)
        anchor = optimizer.ReplayAnchor("world", 0, (), None)
        target = optimizer.TrajectoryTarget(2, "LEVEL")
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (1, 2, 3)),
            anchor,
            target,
            (1, 2, 3),
            MemoryUid.zero(),
            target_uid,
            0,
        )
        task = v819._source_validation_candidate(optimizer, source)

        self.assertTrue(task.source.target_outcome_uid.is_zero)
        self.assertNotEqual(v818._target_key(task.source), v818._target_key(source))
        self.assertEqual(task.actions, source.actions)
        self.assertEqual(task.edit_kind, "VALIDATE_SOURCE")


if __name__ == "__main__":
    unittest.main()
