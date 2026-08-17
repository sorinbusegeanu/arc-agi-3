from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import v8
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
from v8 import progressive_level_learning_v820 as progressive
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_target_minimization_v820 as v820
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


class ProgressiveAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        progressive._PREWIN_CHEAP_TRAJECTORIES.clear()

    def test_best_pre_win_level_source_is_validated_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            row = source(actions=(1, 2, 3), prefix=(9, 8), level=2)
            with patch.object(
                solve_fix,
                "_BASE_SERVICE_SUBMIT_V819",
                return_value=True,
            ) as submit:
                self.assertTrue(v819._service_submit_v819(service, row))
            submit.assert_called_once_with(service, row)
            self.assertEqual(
                service._v819_pre_win_sources["world"][2].trajectory_id,
                row.trajectory_id,
            )
            self.assertIn(row.trajectory_id, progressive._PREWIN_CHEAP_TRAJECTORIES)

    def test_only_first_pre_win_source_is_validated_while_storage_keeps_improving(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            first = source(actions=(1, 2, 3), prefix=(9, 8), level=2)
            worse = source(actions=(1, 2, 3, 4), prefix=(9, 8), level=2)
            better = source(actions=(1,), prefix=(9,), level=2)
            with patch.object(
                solve_fix,
                "_BASE_SERVICE_SUBMIT_V819",
                return_value=True,
            ) as submit:
                self.assertTrue(v819._service_submit_v819(service, first))
                self.assertTrue(v819._service_submit_v819(service, worse))
                self.assertTrue(v819._service_submit_v819(service, better))
            self.assertEqual(submit.call_count, 1)
            self.assertEqual(
                service._v819_pre_win_sources["world"][2].trajectory_id,
                better.trajectory_id,
            )

    def test_win_and_already_solved_paths_keep_existing_submit_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = runtime_owned_service(root)
            win = source(actions=(1,), level=3, terminal="WIN")
            with patch.object(progressive, "_BASE_SERVICE_SUBMIT", return_value=True) as submit:
                self.assertTrue(v819._service_submit_v819(service, win))
            submit.assert_called_once_with(service, win)


class CheapMinimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        progressive._PREWIN_CHEAP_TRAJECTORIES.clear()

    def test_pre_win_source_gets_target_minimize_without_ddmin_candidates(self) -> None:
        row = source(actions=tuple(range(1, 20)))
        progressive._PREWIN_CHEAP_TRAJECTORIES.add(row.trajectory_id)
        self.assertEqual(
            progressive._generate_progressive_v820(
                row,
                optimizer.TrajectoryOptimizerConfig(max_candidates_per_round=16),
            ),
            (),
        )
        self.assertNotIn(row.trajectory_id, progressive._PREWIN_CHEAP_TRAJECTORIES)
        self.assertTrue(
            progressive._generate_progressive_v820(
                row,
                optimizer.TrajectoryOptimizerConfig(max_candidates_per_round=16),
            )
        )

    def test_successful_pre_win_minimization_does_not_recurse_into_ddmin(self) -> None:
        fake = Mock()
        coordinator = SimpleNamespace(
            game_state=lambda _game: v819.GameLearningState.UNSOLVED
        )
        service = SimpleNamespace(
            _v819_runtime=SimpleNamespace(_v819_adaptive_learning=coordinator)
        )
        candidate = SimpleNamespace(
            source=SimpleNamespace(
                anchor=SimpleNamespace(source_id="world"),
                target=SimpleNamespace(terminal_state="LEVEL"),
            )
        )
        with patch.object(progressive, "_BASE_SUBMIT_NEXT_SOURCE", fake):
            progressive._submit_next_source_progressive(service, candidate, object())
        fake.assert_not_called()

    def test_post_win_minimization_keeps_recursive_optimizer(self) -> None:
        fake = Mock()
        coordinator = SimpleNamespace(
            game_state=lambda _game: v819.GameLearningState.SOLVED_OPTIMIZING
        )
        service = SimpleNamespace(
            _v819_runtime=SimpleNamespace(_v819_adaptive_learning=coordinator)
        )
        candidate = SimpleNamespace(
            source=SimpleNamespace(
                anchor=SimpleNamespace(source_id="world"),
                target=SimpleNamespace(terminal_state="LEVEL"),
            )
        )
        validated = object()
        with patch.object(progressive, "_BASE_SUBMIT_NEXT_SOURCE", fake):
            progressive._submit_next_source_progressive(service, candidate, validated)
        fake.assert_called_once_with(service, candidate, validated)


class BatchedLearningIngestionTests(unittest.TestCase):
    def test_parent_learning_updates_are_merged_in_small_bursts(self) -> None:
        calls = []
        runtime = SimpleNamespace(
            record_actor_results=lambda rows: calls.append(tuple(rows))
        )
        proxy = progressive._BatchingRuntimeProxy(runtime, batch_size=4)
        for actor_id in range(1, 5):
            proxy.record_actor_results(
                (actor_module.ActorLearningBatch(actor_id, f"g{actor_id}"),)
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual({row.actor_id for row in calls[0]}, {1, 2, 3, 4})

    def test_flush_merges_repeated_actor_batches(self) -> None:
        calls = []
        runtime = SimpleNamespace(
            record_actor_results=lambda rows: calls.append(tuple(rows))
        )
        proxy = progressive._BatchingRuntimeProxy(runtime, batch_size=8)
        proxy.record_actor_results((actor_module.ActorLearningBatch(1, "world"),))
        proxy.record_actor_results((actor_module.ActorLearningBatch(1, "world"),))
        self.assertEqual(calls, [])
        proxy.flush()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 1)
        self.assertEqual(calls[0][0].actor_id, 1)


if __name__ == "__main__":
    unittest.main()
