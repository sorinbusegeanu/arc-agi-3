from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import trajectory_optimizer_v814 as optimizer
from v8.adaptive_learning_allocation_v819 import FrontierSource, SamplingMode
from v8.model import MemoryUid


def source(*, game="world", actions=(1, 2, 3), level=1):
    anchor = optimizer.ReplayAnchor(game, 77, (), None)
    target = optimizer.TrajectoryTarget(level, "LEVEL")
    return optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        tuple(actions),
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )


class CaptureSourceTests(unittest.TestCase):
    def test_transfer_sampling_marks_captured_trajectory_as_transfer_source(self) -> None:
        row = source()
        prior = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = SamplingMode.TRANSFER.value
            raw = row.to_dict()
        finally:
            if prior is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior
        self.assertEqual(raw["frontier_source"], FrontierSource.TRANSFER.value)
        self.assertEqual(raw["sampling_mode"], SamplingMode.TRANSFER.value)
        self.assertNotIn("seed", raw["anchor"])

    def test_normal_sampling_marks_sampler_source(self) -> None:
        row = source()
        prior = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = SamplingMode.ALTERNATIVE.value
            raw = row.to_dict()
        finally:
            if prior is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior
        self.assertEqual(raw["frontier_source"], FrontierSource.SAMPLER.value)


class ServiceCompatibilityTests(unittest.TestCase):
    def test_standalone_optimizer_keeps_v818_submit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda _candidate: SimpleNamespace(success=False, reason="no"),
            )
            row = source()
            self.assertTrue(service.submit_trajectory(row))
            self.assertEqual(service._sources.unfinished_tasks, 1)
            self.assertFalse(hasattr(service, "_v819_runtime"))

    def test_runtime_owned_source_is_validated_before_optimizer_queue(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda _candidate: SimpleNamespace(success=True, reason="ok"),
            )
            coordinator = v819.AdaptiveLearningCoordinator()
            runtime = SimpleNamespace(_v819_adaptive_learning=coordinator)
            service._v819_runtime = runtime
            service._v819_lock = __import__("threading").RLock()
            service._v819_source_seen = set()
            service._v819_source_pending = {}
            service._v819_source_kind = {}
            row = source()
            with patch.object(v819, "_BASE_ROUTE_CANDIDATE", return_value=True) as route:
                self.assertTrue(v819._service_submit_v819(service, row))
            self.assertEqual(service._sources.unfinished_tasks, 0)
            routed = route.call_args.args[1]
            self.assertEqual(routed.edit_kind, "VALIDATE_SOURCE")
            self.assertEqual(tuple(routed.actions), tuple(row.actions))


if __name__ == "__main__":
    unittest.main()
