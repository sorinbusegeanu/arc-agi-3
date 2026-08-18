from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import v8  # noqa: F401 - installs the chronological runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import optimizer_budget_control_v830 as v830
from v8 import runtime_win_optimization_v834 as v834
from v8 import runtime_win_scope_v835 as v835
from v8 import trajectory_optimizer_v814 as optimizer


def _candidate(*, game: str, cost: int, terminal: str = "LEVEL", prefix=()):
    actions = tuple(1 for _ in range(max(2, int(cost))))
    anchor = optimizer.ReplayAnchor(game, 0, tuple(prefix), None)
    target = optimizer.TrajectoryTarget(5, terminal)
    source = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        actions,
    )
    return optimizer.TrajectoryCandidate(
        optimizer._candidate_id(source, "DELETE_ACTION", actions[:-1]),
        source,
        "DELETE_ACTION",
        actions[:-1],
        0,
        1,
    )


class RuntimeWinScopeV835Tests(unittest.TestCase):
    def test_stale_runtime_win_marker_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            trajectory_root = os.path.join(root, "trajectory_optimizer")
            with mock.patch.dict(
                os.environ,
                {
                    optimizer._TRAJECTORY_ROOT_ENV: trajectory_root,
                    v835._RUN_SESSION_ENV: "current-run",
                },
                clear=False,
            ):
                path = v834._marker_path("ez01")
                self.assertIsNotNone(path)
                optimizer._atomic_json(
                    path,
                    {
                        "game_id": "ez01",
                        "run_session": "previous-run",
                        "observed_levels": 5,
                        "steps": 1270,
                        "time_ns": 1,
                    },
                )
                coordinator = v819.AdaptiveLearningCoordinator()
                coordinator.register_games(("ez01",))
                coordinator._v834_runtime = SimpleNamespace(generation=10)
                self.assertEqual(
                    coordinator.game_state("ez01"),
                    v819.GameLearningState.UNSOLVED,
                )
                self.assertFalse(coordinator._game_won["ez01"])

    def test_runtime_win_uses_game_scope_not_cumulative_level_count(self):
        events = []
        with tempfile.TemporaryDirectory() as root:
            trajectory_root = os.path.join(root, "trajectory_optimizer")
            with mock.patch.dict(
                os.environ,
                {
                    optimizer._TRAJECTORY_ROOT_ENV: trajectory_root,
                    v835._RUN_SESSION_ENV: "current-run",
                },
                clear=False,
            ):
                coordinator = v819.AdaptiveLearningCoordinator(event_sink=events.append)
                coordinator.register_games(("ez01",))
                coordinator._v834_runtime = SimpleNamespace(generation=55)
                v834._write_runtime_win_marker(
                    "ez01",
                    SimpleNamespace(wins=1, levels_completed=11, steps=1270),
                )
                self.assertEqual(
                    coordinator.game_state("ez01"),
                    v819.GameLearningState.SOLVED_OPTIMIZING,
                )
                self.assertIn(("ez01", v835._FULL_WIN_SCOPE_LEVEL), coordinator._records)
                self.assertNotIn(("ez01", 11), coordinator._records)
                self.assertTrue(any("target=EPISODE:+1" in row for row in events))
                self.assertFalse(any("level=11" in row for row in events))

    def test_runtime_init_creates_fresh_session_before_base_init(self):
        seen = []

        def base_init(instance, *args, **kwargs):
            del args, kwargs
            seen.append(os.environ.get(v835._RUN_SESSION_ENV, ""))
            instance.base_called = True

        with mock.patch.object(v835, "_BASE_RUNTIME_INIT", base_init):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(v835._RUN_SESSION_ENV, None)
                instance = SimpleNamespace()
                v835._runtime_init_base_v835(instance)
                self.assertTrue(instance.base_called)
                self.assertTrue(seen[0])
                self.assertEqual(instance._v835_run_session, seen[0])
                self.assertEqual(os.environ[v835._RUN_SESSION_ENV], seen[0])

    def test_full_win_has_separate_optimizer_scope_and_status(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        full = _candidate(game="ez01", cost=1270, terminal="WIN")
        level = _candidate(game="ez01", cost=10, terminal="LEVEL")

        full_game, full_scope, full_cost = v830._candidate_scope(full)
        level_game, level_scope, level_cost = v830._candidate_scope(level)
        self.assertEqual((full_game, full_scope, full_cost), ("ez01", v835._FULL_WIN_SCOPE_LEVEL, 1270))
        self.assertEqual((level_game, level_scope, level_cost), ("ez01", 5, 10))

        v830._register_candidate(coordinator, full)
        v830._register_candidate(coordinator, level)
        self.assertEqual(
            v830._stats_for(coordinator, "ez01", v835._FULL_WIN_SCOPE_LEVEL).source_cost,
            1270,
        )
        self.assertEqual(v830._stats_for(coordinator, "ez01", 5).source_cost, 10)
        message = v830._status_message(
            coordinator,
            "ez01",
            v835._FULL_WIN_SCOPE_LEVEL,
            "START",
        )
        self.assertIn("target=EPISODE:+1", message)
        self.assertIn("cost=1270", message)
        self.assertNotIn(f"level={v835._FULL_WIN_SCOPE_LEVEL}", message)

    def test_full_win_precheck_and_consumption_never_touch_level_five_budget(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        full = _candidate(game="ez01", cost=1270, terminal="WIN")
        v830._register_candidate(coordinator, full)
        key = ("ez01", v835._FULL_WIN_SCOPE_LEVEL)

        prior, present = v830._context_set(v830._BUDGET_CONTEXT, mode="precheck", key=key)
        try:
            self.assertTrue(
                coordinator.reserve_optimization(game_id="ez01", level=5, attempts=2)
            )
        finally:
            v830._context_restore(v830._BUDGET_CONTEXT, prior, present)
        self.assertEqual(
            coordinator._record("ez01", v835._FULL_WIN_SCOPE_LEVEL).consumed_optimization_budget,
            0,
        )

        prior, present = v830._context_set(v830._BUDGET_CONTEXT, mode="consume", key=key)
        try:
            self.assertTrue(
                coordinator.reserve_optimization(game_id="ez01", level=5, attempts=2)
            )
        finally:
            v830._context_restore(v830._BUDGET_CONTEXT, prior, present)
        self.assertEqual(
            coordinator._record("ez01", v835._FULL_WIN_SCOPE_LEVEL).consumed_optimization_budget,
            2,
        )
        self.assertEqual(coordinator._record("ez01", 5).consumed_optimization_budget, 0)

    def test_full_win_validation_updates_full_win_stats_not_level_fragment(self):
        events = []
        coordinator = v819.AdaptiveLearningCoordinator(event_sink=events.append)
        full = _candidate(game="ez01", cost=1270, terminal="WIN")
        v830._register_candidate(coordinator, full)

        process_prior, process_present = v830._context_set(
            v830._PROCESS_CONTEXT,
            source_cost=1270,
            key=("ez01", 5),
        )
        full_present = hasattr(v835._FULL_WIN_CONTEXT, "key")
        full_prior = getattr(v835._FULL_WIN_CONTEXT, "key", None)
        v835._FULL_WIN_CONTEXT.key = ("ez01", v835._FULL_WIN_SCOPE_LEVEL)
        try:
            coordinator.record_optimizer_validation(
                game_id="ez01",
                level=5,
                attempts=2,
                successes=2,
                saved_actions=100,
                improved=True,
                generation=100,
            )
        finally:
            v830._context_restore(v830._PROCESS_CONTEXT, process_prior, process_present)
            if full_present:
                v835._FULL_WIN_CONTEXT.key = full_prior
            else:
                delattr(v835._FULL_WIN_CONTEXT, "key")

        stats = v830._stats_for(coordinator, "ez01", v835._FULL_WIN_SCOPE_LEVEL)
        self.assertEqual(stats.best_cost, 1170)
        self.assertEqual(stats.saved_actions, 100)
        self.assertNotIn(("ez01", 5), coordinator._v830_optimizer_budget_stats)
        self.assertTrue(
            any("target=EPISODE:+1" in row and "status=IMPROVED" in row for row in events)
        )


if __name__ == "__main__":
    unittest.main()
