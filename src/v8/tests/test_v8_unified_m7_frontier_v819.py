from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
from v8 import trajectory_optimizer_v814 as optimizer
from v8.adaptive_learning_allocation_v819 import FrontierSource
from v8.model import MemoryUid


def source(*, actions=(1, 2, 3), prefix=(), level=1, terminal="LEVEL"):
    anchor = optimizer.ReplayAnchor("world", 0, tuple(prefix), None)
    target = optimizer.TrajectoryTarget(level, terminal)
    actions = tuple(int(value) for value in actions)
    return optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        actions,
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )


def runtime_owned_service(root: str):
    service = optimizer.TrajectoryOptimizationService(
        Path(root),
        validator=lambda _candidate: SimpleNamespace(success=True, reason="ok"),
    )
    coordinator = v819.AdaptiveLearningCoordinator()
    coordinator.register_games(("world",))
    runtime = SimpleNamespace(_v819_adaptive_learning=coordinator)
    service._v819_runtime = runtime
    service._v819_lock = threading.RLock()
    service._v819_source_seen = set()
    service._v819_source_pending = {}
    service._v819_source_kind = {}
    return service


class CaptureSourceTests(unittest.TestCase):
    def test_normal_sampling_marks_sampler_source(self) -> None:
        prior = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.DISCOVERY.value
            raw = source().to_dict()
        finally:
            if prior is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior
        self.assertEqual(raw["frontier_source"], FrontierSource.SAMPLER.value)

    def test_transfer_sampling_marks_captured_trajectory_as_transfer_source(self) -> None:
        prior = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.TRANSFER.value
            raw = source().to_dict()
        finally:
            if prior is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior
        self.assertEqual(raw["frontier_source"], FrontierSource.TRANSFER.value)


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

    def test_runtime_owned_partial_source_is_validated_cheaply_before_optimizer_queue(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            row = source(level=2, prefix=(9, 8))
            with patch.object(v819, "_BASE_ROUTE_CANDIDATE", return_value=True) as route:
                self.assertTrue(v819._service_submit_v819(service, row))
            route.assert_called_once()
            routed = route.call_args.args[1]
            self.assertEqual(routed.edit_kind, "VALIDATE_SOURCE")
            self.assertEqual(tuple(routed.actions), tuple(row.actions))
            self.assertEqual(service._sources.unfinished_tasks, 0)
            self.assertEqual(
                service._v819_pre_win_sources["world"][2].trajectory_id,
                row.trajectory_id,
            )

    def test_runtime_owned_win_source_is_validated_before_optimizer_queue(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            row = source(level=3, terminal="WIN", prefix=(9, 8))
            with patch.object(v819, "_BASE_ROUTE_CANDIDATE", return_value=True) as route:
                self.assertTrue(v819._service_submit_v819(service, row))
            self.assertEqual(service._sources.unfinished_tasks, 0)
            routed = route.call_args.args[1]
            self.assertEqual(routed.edit_kind, "VALIDATE_SOURCE")
            self.assertEqual(tuple(routed.actions), tuple(row.actions))

    def test_pre_win_storage_keeps_shortest_source_without_repeated_validation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            longer = source(actions=(1, 2, 3, 4), level=2, prefix=(9, 8, 7))
            shorter = source(actions=(1, 2), level=2, prefix=(9,))
            with patch.object(v819, "_BASE_ROUTE_CANDIDATE", return_value=True) as route:
                self.assertTrue(v819._service_submit_v819(service, longer))
                self.assertTrue(v819._service_submit_v819(service, shorter))
            self.assertEqual(route.call_count, 1)
            stored = service._v819_pre_win_sources["world"][2]
            self.assertEqual(stored.trajectory_id, shorter.trajectory_id)

    def test_validated_win_can_release_deferred_levels_to_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            rows = (
                source(actions=(1, 2), level=1),
                source(actions=(3, 4), level=2, prefix=(1, 2)),
            )
            with patch.object(v819, "_BASE_ROUTE_CANDIDATE", return_value=True):
                for row in rows:
                    self.assertTrue(v819._service_submit_v819(service, row))
            with patch.object(v819, "_BASE_SERVICE_SUBMIT", return_value=True) as submit:
                released = solve_fix._release_pre_win_sources(service, "world")
            self.assertEqual(released, 2)
            self.assertEqual(submit.call_count, 2)
            self.assertNotIn("world", service._v819_pre_win_sources)


if __name__ == "__main__":
    unittest.main()
