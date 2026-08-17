from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import trajectory_optimizer_v814 as optimizer
from v8.adaptive_learning_allocation_v819 import FrontierSource
from v8.model import MemoryLevel, MemoryType, MemoryUid


class _Coordinator:
    def __init__(self) -> None:
        self.observed = []

    def observe_frontier_candidate(self, scope, candidate, **kwargs):
        self.observed.append((scope, candidate, kwargs))
        return True


class SamplerFrontierPublicationV819Tests(unittest.TestCase):
    def test_sampler_source_publishes_canonical_m7_without_overwriting_optimized_sidecar(self) -> None:
        anchor = optimizer.ReplayAnchor("world", 0, (), None)
        target = optimizer.TrajectoryTarget(2, "LEVEL")
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (1, 2, 3)),
            anchor,
            target,
            (1, 2, 3),
            MemoryUid.zero(),
            MemoryUid.zero(),
            0,
        )
        candidate = optimizer.TrajectoryCandidate(
            f"source-{source.trajectory_id}",
            source,
            "VALIDATE_SOURCE",
            source.actions,
            0,
            0,
        )
        result = SimpleNamespace(
            success=True,
            attempts=2,
            successes=2,
            terminal_context=123,
            terminal_action=3,
            outcome_signature=456,
        )
        target_uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        preserved = {"optimized": object()}
        service = SimpleNamespace(
            _v819_source_kind={source.trajectory_id: FrontierSource.SAMPLER.value},
            _v819_source_pending={source.trajectory_id: source},
            _v819_lock=threading.RLock(),
            _validated=preserved,
        )
        coordinator = _Coordinator()
        runtime = SimpleNamespace(
            _v814_trajectory_optimizer=service,
            _v819_adaptive_learning=coordinator,
            generation=9,
        )

        with patch.object(optimizer, "_runtime_validation_callback") as canonical, patch.object(
            v819,
            "_BASE_SERVICE_SUBMIT",
            return_value=True,
        ) as optimize:
            v819._publish_validated_source(runtime, candidate, result, target_uid)

        self.assertIs(service._validated, preserved)
        self.assertEqual(service._validated, preserved)
        canonical.assert_called_once()
        optimize.assert_called_once()
        self.assertEqual(len(coordinator.observed), 1)
        frontier_candidate = coordinator.observed[0][1]
        self.assertEqual(frontier_candidate.source, FrontierSource.SAMPLER)
        self.assertEqual(frontier_candidate.cost, 3)
        self.assertEqual(frontier_candidate.successes, 2)


if __name__ == "__main__":
    unittest.main()
