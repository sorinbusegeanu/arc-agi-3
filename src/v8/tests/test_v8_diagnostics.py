from __future__ import annotations

import unittest

from v8.actor import ActorProgress
from v8.diagnostics import (
    HYPOTHESIS_IDS,
    format_game_rate_line,
    format_hypothesis_line,
    game_rates,
    hypothesis_statuses,
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

    def test_game_rates_are_game_level_not_actor_level(self) -> None:
        rows = (
            ActorProgress(1, "game-a", 100, 0, 0, 0),
            ActorProgress(2, "game-a", 100, 1, 0, 0),
            ActorProgress(3, "game-b", 100, 0, 1, 2),
            ActorProgress(4, "game-c", 100, 0, 0, 1),
        )
        win_rate, level_rate = game_rates(rows)
        self.assertAlmostEqual(win_rate, 100.0 / 3.0)
        self.assertAlmostEqual(level_rate, 200.0 / 3.0)
        self.assertEqual(
            format_game_rate_line(rows),
            "win_rate=33.3% level_rate=66.7%",
        )

    def test_empty_progress_has_zero_rates(self) -> None:
        self.assertEqual(game_rates(()), (0.0, 0.0))
        self.assertEqual(
            format_game_rate_line(()),
            "win_rate=0.0% level_rate=0.0%",
        )


if __name__ == "__main__":
    unittest.main()
