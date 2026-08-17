from __future__ import annotations

import unittest

import v8
from v8 import actor as actor_module
from v8 import dedicated_lifecycle_v813 as lifecycle
from v8 import progress_runtime_fix_v822 as progress
from v8 import runtime_repair_v822 as repair


class _Queue:
    def __init__(self) -> None:
        self.rows = []

    def put_nowait(self, row) -> None:
        self.rows.append(row)


class RuntimeRepairV822Tests(unittest.TestCase):
    def test_lifecycle_worker_is_delayed_wrapper_with_five_minute_default(self) -> None:
        self.assertEqual(repair._LIFECYCLE_START_DELAY_SECONDS, 300.0)
        self.assertIs(lifecycle._lifecycle_worker, repair._delayed_lifecycle_worker)

    def test_exact_five_level_win_requires_four_internal_boundaries(self) -> None:
        self.assertEqual(
            repair._episode_levels((1, 1, 1, 1, 1), (1, 2, 3, 4), 5),
            ((1,), (1,), (1,), (1,), (1,)),
        )
        self.assertIsNone(
            repair._episode_levels((1, 1, 1, 1, 1), (1, 2, 4), 5)
        )

    def test_progress_publisher_preserves_deepest_level_and_true_win_cost(self) -> None:
        from v8 import episode_progress_reporting_v821 as episode
        from v8 import learning_fixes_v088 as learning

        progress._FIRST_WIN_AT.clear()
        progress._LAST_STEPS.clear()
        progress._LAST_WINS.clear()
        episode._MAX_LEVEL_REACHED.clear()
        episode._MAX_LEVEL_REACHED["ez01"] = 5
        prior_best = learning._BEST_WIN_STEPS
        prior_last = learning._LAST_WIN_STEPS
        try:
            learning._BEST_WIN_STEPS = 5
            learning._LAST_WIN_STEPS = 7
            target = _Queue()
            job = actor_module.ActorJob(3, "ez01", 10000, 0)
            actor_module._publish_progress(
                target,
                job=job,
                steps=4000,
                wins=1,
                failures=0,
                levels_completed=5,
                replans=0,
                planned_steps=0,
            )
            row = target.rows[0]
            self.assertIsInstance(row, progress.V822ActorProgress)
            self.assertEqual(row.steps, 4000)
            self.assertEqual(row.max_level_reached, 5)
            self.assertEqual(row.best_win_steps, 5)
            self.assertEqual(row.last_win_steps, 7)
        finally:
            learning._BEST_WIN_STEPS = prior_best
            learning._LAST_WIN_STEPS = prior_last

    def test_v822_progress_publisher_is_last_installed_authority(self) -> None:
        self.assertIs(actor_module.ActorProgress, progress.V822ActorProgress)
        self.assertIs(actor_module._publish_progress, progress._publish_progress_v822)


if __name__ == "__main__":
    unittest.main()
