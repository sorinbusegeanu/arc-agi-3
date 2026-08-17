from __future__ import annotations

import unittest

from v8.actor import ActorProgress
from v8.diagnostics import (
    HYPOTHESIS_IDS,
    format_game_rate_line,
    format_hypothesis_line,
    game_rates,
    game_summary,
    hypothesis_statuses,
    solved_game_ids,
    solved_game_steps,
)


class DiagnosticsTests(unittest.TestCase):
    def test_hypothesis_status_is_single_line_and_complete(self) -> None:
        line = format_hypothesis_line()
        self.assertNotIn("\n", line)
        self.assertTrue(line.startswith("hypotheses "))
        for hypothesis_id in HYPOTHESIS_IDS:
            self.assertIn(
                f"{hypothesis_id}=INSUFFICIENT_EVIDENCE",
                line,
            )

    def test_hypothesis_override_is_validated(self) -> None:
        statuses = hypothesis_statuses({"H13": "PARTIALLY_VALID"})
        self.assertEqual(statuses["H13"], "PARTIALLY_VALID")
        with self.assertRaises(ValueError):
            hypothesis_statuses({"H99": "VALID"})
        with self.assertRaises(ValueError):
            hypothesis_statuses({"H01": "UNKNOWN"})

    def test_game_rates_use_partial_level_progress_not_binary_game_progress(self) -> None:
        rows = (
            ActorProgress(1, "game-a", 100, 0, 0, 0),
            ActorProgress(2, "game-a", 100, 1, 0, 5, 0, 0, 86, 5, 5, 5),
            ActorProgress(3, "game-b", 100, 0, 1, 2),
            ActorProgress(4, "game-c", 100, 0, 0, 1),
        )
        win_rate, level_rate = game_rates(rows)
        self.assertAlmostEqual(win_rate, 100.0 / 3.0)
        self.assertAlmostEqual(level_rate, 100.0 * 8.0 / 15.0)
        self.assertEqual(game_summary(rows), (100.0 / 3.0, 100.0 * 8.0 / 15.0, 1, 3))
        self.assertEqual(solved_game_ids(rows), ("game-a",))
        self.assertEqual(solved_game_steps(rows), (("game-a", 86),))
        self.assertEqual(
            format_game_rate_line(rows),
            "current_run_wins=33.3% current_run_levels_solved=53.3% current_run_solved_games=1/3 (game-a:B=5,L=5)",
        )

    def test_final_level_progress_requires_terminal_win(self) -> None:
        rows = (
            ActorProgress(1, "ez01", 500, 0, 0, 5, 0, 0, 0, 5),
            ActorProgress(2, "ez02", 500, 1, 0, 5, 0, 0, 400, 5, 5, 5),
        )
        self.assertEqual(game_summary(rows), (50.0, 90.0, 1, 2))
        self.assertEqual(
            format_game_rate_line(rows),
            "current_run_wins=50.0% current_run_levels_solved=90.0% current_run_solved_games=1/2 (ez02:B=5,L=5)",
        )

    def test_distinct_solved_games_and_steps_are_not_double_counted_across_actor_lanes(self) -> None:
        rows = (
            ActorProgress(1, "ez02", 120, 1, 0, 5, 0, 0, 101, 5, 8, 8),
            ActorProgress(2, "ez02", 100, 2, 0, 5, 0, 0, 94, 5, 5, 6),
            ActorProgress(3, "ez01", 100, 1, 0, 5, 0, 0, 88, 5, 5, 5),
            ActorProgress(4, "ez03", 100, 0, 0, 2),
        )
        self.assertEqual(game_summary(rows)[2:], (2, 3))
        self.assertEqual(solved_game_ids(rows), ("ez01", "ez02"))
        self.assertEqual(solved_game_steps(rows), (("ez01", 88), ("ez02", 94)))
        self.assertEqual(
            format_game_rate_line(rows),
            "current_run_wins=66.7% current_run_levels_solved=80.0% current_run_solved_games=2/3 (ez01:B=5,L=5; ez02:B=5)",
        )

    def test_progress_line_is_single_dedicated_percentage_line(self) -> None:
        rows = (ActorProgress(1, "ez01", 101, 1, 0, 5, 0, 0, 101, 5, 5, 5),)
        line = format_game_rate_line(rows)
        self.assertNotIn("\n", line)
        self.assertEqual(
            line,
            "current_run_wins=100.0% current_run_levels_solved=100.0% current_run_solved_games=1/1 (ez01:B=5,L=5)",
        )

    def test_partial_progress_for_one_of_ten_games_is_visible(self) -> None:
        rows = tuple(
            ActorProgress(index, f"g{index:02d}", 100, 0, 0, 2 if index == 1 else 0)
            for index in range(1, 11)
        )
        self.assertEqual(game_rates(rows), (0.0, 4.0))

    def test_empty_progress_has_zero_rates(self) -> None:
        self.assertEqual(game_rates(()), (0.0, 0.0))
        self.assertEqual(game_summary(()), (0.0, 0.0, 0, 0))
        self.assertEqual(solved_game_ids(()), ())
        self.assertEqual(solved_game_steps(()), ())
        self.assertEqual(
            format_game_rate_line(()),
            "current_run_wins=0.0% current_run_levels_solved=0.0% current_run_solved_games=0/0",
        )


if __name__ == "__main__":
    unittest.main()
