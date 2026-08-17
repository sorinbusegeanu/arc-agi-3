from __future__ import annotations

import pickle
import unittest

from v8 import actor as actor_module
from v8.diagnostics import format_game_rate_line, game_summary
from v8 import episode_progress_reporting_v821 as progress_fix


class _Queue:
    def __init__(self) -> None:
        self.rows = []

    def put_nowait(self, row) -> None:
        self.rows.append(row)


class EpisodeProgressReportingV821Tests(unittest.TestCase):
    def setUp(self) -> None:
        progress_fix._MAX_LEVEL_REACHED.clear()
        progress_fix._ACTIVE_GAME_ID = None

    def test_cumulative_level_events_do_not_fake_full_level_completion(self) -> None:
        rows = (
            actor_module.ActorProgress(1, "ez01", 500, 0, 0, 9, 0, 0, 0, 1),
            actor_module.ActorProgress(2, "ez02", 500, 0, 0, 12, 0, 0, 0, 2),
        )
        self.assertEqual(game_summary(rows), (0.0, 30.0, 0, 2))
        self.assertEqual(
            format_game_rate_line(rows),
            "current_run_wins=0.0% current_run_levels_solved=30.0% current_run_solved_games=0/2",
        )

    def test_deepest_level_is_monotonic_across_episode_resets(self) -> None:
        progress_fix._ACTIVE_GAME_ID = "ez01"
        try:
            progress_fix._record_level_progress(1)
            progress_fix._record_level_progress(3)
            progress_fix._record_level_progress(0)
            progress_fix._record_level_progress(2)
        finally:
            progress_fix._ACTIVE_GAME_ID = None
        self.assertEqual(progress_fix._MAX_LEVEL_REACHED["ez01"], 3)

    def test_real_progress_publication_carries_deepest_level_and_first_win(self) -> None:
        job = actor_module.ActorJob(4, "ez01", 100, 0)
        progress_fix._MAX_LEVEL_REACHED["ez01"] = 2
        target = _Queue()
        progress_fix._publish_episode_progress(
            target,
            job=job,
            steps=80,
            wins=0,
            failures=0,
            levels_completed=7,
            replans=0,
            planned_steps=0,
        )
        self.assertEqual(len(target.rows), 1)
        row = target.rows[0]
        self.assertIsInstance(row, actor_module.ActorProgress)
        self.assertEqual(row.levels_completed, 7)
        self.assertEqual(row.max_level_reached, 2)
        self.assertEqual(row.first_win_step, 0)

    def test_progress_without_environment_registration_keeps_legacy_sentinel(self) -> None:
        job = actor_module.ActorJob(5, "tt01", 10, 0)
        target = _Queue()
        progress_fix._publish_episode_progress(
            target,
            job=job,
            steps=7,
            wins=1,
            failures=0,
            levels_completed=2,
            replans=3,
            planned_steps=4,
        )
        self.assertEqual(target.rows[0].max_level_reached, -1)

    def test_progress_row_is_pickle_safe_for_multiprocessing_queue(self) -> None:
        row = actor_module.ActorProgress(1, "ez01", 12, 0, 0, 4, 0, 0, 0, 2)
        restored = pickle.loads(pickle.dumps(row))
        self.assertEqual(restored, row)
        self.assertEqual(restored.max_level_reached, 2)


if __name__ == "__main__":
    unittest.main()
