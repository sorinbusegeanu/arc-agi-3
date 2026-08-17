from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import v8  # noqa: F401 - installs the chronological runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import complete_win_trajectory_repair_v825 as repair
from v8 import runtime_win_optimization_v834 as v834
from v8 import solved_game_recovery_v821 as recovery
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818


class RuntimeWinOptimizationV834Tests(unittest.TestCase):
    def test_final_hooks_are_installed(self):
        self.assertIs(v819._ResultAdapter.put, v834._result_adapter_put_v834)
        self.assertIs(v819.AdaptiveLearningCoordinator.game_state, v834._game_state_v834)
        self.assertIs(recovery._publish_runtime_levels, repair._publish_runtime_levels_v825)
        self.assertIs(repair._BASE_PUBLISH_RUNTIME_LEVELS, v834._publish_runtime_levels_v834)
        self.assertIs(v818._prefix_for, v834._prefix_for_v834)

    def test_observed_win_immediately_promotes_allocator_state(self):
        with tempfile.TemporaryDirectory() as root:
            trajectory_root = os.path.join(root, "trajectory_optimizer")
            with mock.patch.dict(
                os.environ,
                {optimizer._TRAJECTORY_ROOT_ENV: trajectory_root},
                clear=False,
            ):
                coordinator = v819.AdaptiveLearningCoordinator()
                coordinator.register_games(("ez01",))
                coordinator._v834_runtime = SimpleNamespace(generation=1234)
                self.assertEqual(
                    coordinator.game_state("ez01"),
                    v819.GameLearningState.UNSOLVED,
                )

                v834._write_runtime_win_marker(
                    "ez01",
                    SimpleNamespace(wins=1, levels_completed=5, steps=1270),
                )

                self.assertEqual(
                    coordinator.game_state("ez01"),
                    v819.GameLearningState.SOLVED_OPTIMIZING,
                )
                self.assertTrue(coordinator._game_won["ez01"])
                record = coordinator._record("ez01", 5)
                self.assertEqual(record.state, v819.GameLearningState.SOLVED_OPTIMIZING)
                self.assertEqual(record.last_success_generation, 1234)
                self.assertEqual(record.optimizer_exhausted_version, -1)
                self.assertEqual(coordinator.choose_mode("ez01"), v819.SamplingMode.VERIFY)

    def test_complete_win_becomes_full_optimizer_source(self):
        prior_capture = list(optimizer._CAPTURED_FOR_TESTS)
        prior_seed = optimizer._CAPTURE_SEED
        prior_root = optimizer._CAPTURE_ENV_ROOT
        optimizer._CAPTURED_FOR_TESTS.clear()
        optimizer._CAPTURE_SEED = 17
        optimizer._CAPTURE_ENV_ROOT = "/tmp/arc-games"
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
                v834._publish_complete_optimizer_source(
                    "ez01",
                    ((1, 1), (2, 2, 2), (3,), (4, 4), (5,)),
                )
            self.assertEqual(len(optimizer._CAPTURED_FOR_TESTS), 1)
            row = optimizer._CAPTURED_FOR_TESTS[0]
            self.assertEqual(row.anchor.source_id, "ez01")
            self.assertEqual(row.anchor.seed, 17)
            self.assertEqual(row.anchor.prefix_actions, ())
            self.assertEqual(row.anchor.env_root, "/tmp/arc-games")
            self.assertEqual(row.target.levels_completed, 5)
            self.assertEqual(row.target.terminal_state, "WIN")
            self.assertEqual(row.actions, (1, 1, 2, 2, 2, 3, 4, 4, 5))
            self.assertEqual(row.cost, 9)
        finally:
            optimizer._CAPTURED_FOR_TESTS[:] = prior_capture
            optimizer._CAPTURE_SEED = prior_seed
            optimizer._CAPTURE_ENV_ROOT = prior_root

    def test_full_game_win_validation_never_prepends_level_prefix(self):
        source = optimizer.SuccessfulTrajectory(
            "complete-win",
            optimizer.ReplayAnchor("ez01", 0, (), None),
            optimizer.TrajectoryTarget(5, "WIN"),
            (1, 1, 1, 1, 1),
        )
        candidate = optimizer.TrajectoryCandidate(
            "validate-complete-win",
            source,
            "VALIDATE_SOURCE",
            source.actions,
            0,
            0,
        )
        service = SimpleNamespace()
        self.assertEqual(v818._prefix_for(service, candidate), ())


if __name__ == "__main__":
    unittest.main()
