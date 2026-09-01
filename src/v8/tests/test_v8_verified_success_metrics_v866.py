from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v8 import learning_effectiveness_report_v850 as report
from v8 import trajectory_optimizer_v814 as trajectory
from v8 import verified_success_metrics_v866 as verified
from v8.actor import ActorProgress
from v8.adaptive_learning_allocation_v819 import AdaptiveLearningCoordinator
from v8.mixed_environment_v859 import MIX_GAME_IDS
from v8.trajectory_optimizer_v814 import (
    ReplayAnchor,
    SuccessfulTrajectory,
    TrajectoryTarget,
)
from v8.verified_success_metrics_v866 import (
    SUCCESS_ROOT_ENV,
    _ARC_CAPTURE,
    _VerifiedAdapterProxy,
    record_verified_success_v866,
    verified_success_snapshot_v866,
)


class _Runtime:
    def __init__(self, root: Path, games):
        self.root = root
        self.generation = 1
        self.watermark = 0
        self._v814_trajectory_optimizer = None
        self._v850_total_step_budget = 100
        self._v866_total_step_budget = 100
        self._v866_selected_games = tuple(games)
        self._v866_success_root = str(root / "verified")
        Path(self._v866_success_root).mkdir(parents=True, exist_ok=True)


class _Boundary:
    continuation = False
    primary_valence = 1


class _PositiveAdapter:
    def __init__(self):
        self.closed = False

    def step(self, action):
        return int(action)

    def cognitive_boundary_event(self):
        return _Boundary()

    def reset(self):
        return None

    def close(self):
        self.closed = True


class VerifiedSuccessMetricsV866Tests(unittest.TestCase):
    def setUp(self):
        self.previous_root = os.environ.get(SUCCESS_ROOT_ENV)
        os.environ.pop(SUCCESS_ROOT_ENV, None)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop(SUCCESS_ROOT_ENV, None)
        else:
            os.environ[SUCCESS_ROOT_ENV] = self.previous_root

    def test_actor_counters_cannot_fake_level_or_game_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _Runtime(Path(tmp), ("ic01",))
            coordinator = AdaptiveLearningCoordinator()
            coordinator.register_games(("ic01",))
            completed = {
                "ic01": {
                    "steps": 100,
                    "wins": 4,
                    "levels_completed": 30,
                    "max_level_reached": 5,
                    "first_win_step": 7,
                }
            }
            payload = report.learning_effectiveness_snapshot_v850(
                runtime, coordinator, completed, {}, {}
            )
            outcome = payload["effectiveness"]["outcome_effectiveness"]
            efficiency = payload["effectiveness"]["efficiency"]
            self.assertEqual(outcome["current_run_levels_solved"], 0)
            self.assertEqual(outcome["current_run_games_won"], 0)
            self.assertEqual(outcome["level_solve_rate_pct"], 0.0)
            self.assertEqual(outcome["game_solve_rate_pct"], 0.0)
            self.assertEqual(outcome["successful_trajectories"], 0)
            self.assertEqual(outcome["actor_reported_level_advance_events"], 30)
            self.assertEqual(outcome["actor_reported_games_won"], 1)
            self.assertIsNone(efficiency["steps_per_solved_level"])
            self.assertIsNone(efficiency["mean_first_win_step"])

    def test_arc_environment_uses_verified_capture_step_wrapper(self):
        from v7.environment.arc_adapter import ArcGridEnvironment
        from v8 import learning_transfer_correctness_v854 as transfer
        from v8 import runtime_repair_v822 as runtime_repair

        self.assertIs(ArcGridEnvironment.step, runtime_repair._runtime_env_step)
        self.assertIs(transfer._BASE_RESTART_STEP, trajectory._capture_env_step)
        self.assertIs(
            trajectory._BASE_ENV_STEP,
            verified._capture_env_step_v866,
        )

    def test_composed_arc_capture_advances_verified_step(self):
        from v8 import learning_transfer_correctness_v854 as transfer

        prior = (
            trajectory._CAPTURE_ACTIVE,
            trajectory._CAPTURE_PREFIX,
            trajectory._CAPTURE_SEGMENT,
            trajectory._ACTOR_ACTION_HISTORY,
            trajectory._ACTOR_RESET_EPOCH,
        )
        sentinel = object()
        try:
            trajectory._CAPTURE_ACTIVE = True
            trajectory._CAPTURE_PREFIX = []
            trajectory._CAPTURE_SEGMENT = []
            trajectory._ACTOR_ACTION_HISTORY = []
            _ARC_CAPTURE.step = 0
            with patch.object(
                verified,
                "_BASE_TRAJECTORY_CAPTURE_STEP",
                return_value=sentinel,
            ) as lower:
                result = transfer._BASE_RESTART_STEP(object(), 7)
            self.assertIs(result, sentinel)
            self.assertEqual(_ARC_CAPTURE.step, 1)
            lower.assert_called_once()
        finally:
            (
                trajectory._CAPTURE_ACTIVE,
                trajectory._CAPTURE_PREFIX,
                trajectory._CAPTURE_SEGMENT,
                trajectory._ACTOR_ACTION_HISTORY,
                trajectory._ACTOR_RESET_EPOCH,
            ) = prior

    def test_verified_arc_level_and_win_drive_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "verified"
            record_verified_success_v866(
                game_id="ic01",
                seed=1,
                terminal_state="LEVEL",
                levels_completed=2,
                actions=(1, 2),
                capture_step=12,
                root=root,
            )
            level_only = verified_success_snapshot_v866(root, ("ic01",))
            self.assertEqual(level_only["current_run_levels_solved"], 2)
            self.assertEqual(level_only["current_run_games_won"], 0)
            self.assertEqual(level_only["level_solve_rate_pct"], 40.0)
            self.assertIsNone(level_only["mean_first_win_step"])

            record_verified_success_v866(
                game_id="ic01",
                seed=1,
                terminal_state="WIN",
                levels_completed=5,
                actions=(3, 4),
                capture_step=37,
                root=root,
            )
            won = verified_success_snapshot_v866(root, ("ic01",))
            self.assertEqual(won["current_run_levels_solved"], 5)
            self.assertEqual(won["current_run_games_won"], 1)
            self.assertEqual(won["level_solve_rate_pct"], 100.0)
            self.assertEqual(won["game_solve_rate_pct"], 100.0)
            self.assertEqual(won["mean_first_win_step"], 37.0)

    def test_mixed_denominator_is_thirteen_units_and_five_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "verified"
            record_verified_success_v866(
                game_id="FrozenLake-v1",
                seed=1,
                terminal_state="WIN",
                levels_completed=1,
                actions=(1, 2, 3),
                capture_step=3,
                root=root,
            )
            payload = verified_success_snapshot_v866(root, MIX_GAME_IDS)
            self.assertEqual(payload["level_target_count"], 13)
            self.assertEqual(payload["game_target_count"], 5)
            self.assertEqual(payload["current_run_levels_solved"], 1)
            self.assertEqual(payload["current_run_games_won"], 1)
            self.assertAlmostEqual(payload["level_solve_rate_pct"], 100.0 / 13.0)
            self.assertAlmostEqual(payload["game_solve_rate_pct"], 20.0)

    def test_generic_positive_terminal_persists_actual_action_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "verified"
            os.environ[SUCCESS_ROOT_ENV] = str(root)
            inner = _PositiveAdapter()
            proxy = _VerifiedAdapterProxy(inner, "FrozenLake-v1", 9)
            proxy.step(2)
            rows = verified_success_snapshot_v866(root, ("FrozenLake-v1",))
            self.assertEqual(rows["successful_trajectories"], 1)
            self.assertEqual(rows["current_run_games_won"], 1)
            event_file = next((root / "events").glob("*.json"))
            self.assertIn('"actions":[2]', event_file.read_text(encoding="utf-8"))
            proxy.reset()
            proxy.close()
            self.assertTrue(inner.closed)

    def test_arc_optimizer_capture_is_mirrored_to_verified_run_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "verified"
            os.environ[SUCCESS_ROOT_ENV] = str(root)
            _ARC_CAPTURE.step = 41
            row = SuccessfulTrajectory(
                "verified-test",
                ReplayAnchor("ic01", 3, (), None),
                TrajectoryTarget(5, "WIN"),
                (1, 2, 3),
            )
            trajectory._write_successful_trajectory(row)
            payload = verified_success_snapshot_v866(root, ("ic01",))
            self.assertEqual(payload["successful_trajectories"], 1)
            self.assertEqual(payload["current_run_games_won"], 1)
            self.assertEqual(payload["mean_first_win_step"], 41.0)

    def test_periodic_actor_wins_are_zero_without_verified_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "verified"
            root.mkdir(parents=True)
            os.environ[SUCCESS_ROOT_ENV] = str(root)
            rows = (
                ActorProgress(1, "ic01", 50, 3, 0, 12, planned_steps=10),
            )
            from v8 import reporter

            line = reporter.format_periodic_progress_line(rows, 100)
            self.assertIn("L=0.0%", line)
            self.assertIn("G=0.0%", line)
            self.assertIn("step/L=-", line)
            self.assertIn("firstWin=-", line)

    def test_inner_mixed_effectiveness_reporter_is_disabled_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _Runtime(Path(tmp), ("ic01", "FrozenLake-v1"))
            runtime._v866_mixed_scope_active = True

            with patch.object(
                verified,
                "learning_effectiveness_snapshot_v866",
            ) as snapshot, patch.object(report, "_emit_effectiveness_stdout") as emit:
                verified._write_learning_effectiveness_log_v866(
                    runtime,
                    object(),
                    {},
                    {},
                    {},
                )

            snapshot.assert_not_called()
            emit.assert_not_called()
            self.assertFalse((Path(tmp) / report._REPORT_FILE).exists())

    def test_pure_arc_effectiveness_stdout_remains_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _Runtime(Path(tmp), ("ic01",))
            runtime._v866_mixed_scope_active = False
            payload = {"scope": {"steps": 50}, "effectiveness": {}}

            with patch.object(
                verified,
                "learning_effectiveness_snapshot_v866",
                return_value=payload,
            ), patch.object(report, "_emit_effectiveness_stdout") as emit:
                verified._write_learning_effectiveness_log_v866(
                    runtime,
                    object(),
                    {},
                    {},
                    {},
                )

            emit.assert_called_once_with(payload, budget_consumed_pct=50.0)


if __name__ == "__main__":
    unittest.main()
