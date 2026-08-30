from __future__ import annotations

import unittest
from unittest.mock import patch

import v8  # noqa: F401 - installs current runtime stack
from v8 import cli as base_cli
from v8.cli_v819 import main as adaptive_cli_main
from v8.mixed_environment_v859 import (
    RESEARCH_1_ARC_GAME_IDS,
    RESEARCH_1_GAME_IDS,
    RESEARCH_1_GENERIC_GAME_IDS,
    RESEARCH_1_SPECS,
    is_mixed_environment_selector,
    is_research_1_selector,
    resolve_mixed_game_selector,
)
from v8.research import researcher_packet


class ResearchPresetV871Tests(unittest.TestCase):
    def test_research_1_has_twelve_heterogeneous_games(self):
        self.assertTrue(is_research_1_selector("research_1"))
        self.assertTrue(is_mixed_environment_selector("research_1"))
        self.assertEqual(resolve_mixed_game_selector("research_1"), RESEARCH_1_GAME_IDS)
        self.assertEqual(len(RESEARCH_1_GAME_IDS), 12)
        self.assertEqual(len(set(RESEARCH_1_GAME_IDS)), 12)
        self.assertEqual(tuple(spec.environment_id for spec in RESEARCH_1_SPECS), RESEARCH_1_GAME_IDS)
        self.assertEqual(len(RESEARCH_1_ARC_GAME_IDS), 9)
        self.assertEqual(
            RESEARCH_1_GENERIC_GAME_IDS,
            {"FrozenLake-v1", "ArcAgi/Chess-v0", "ArcAgi/Sudoku-v0"},
        )

    def test_research_1_contains_related_transfer_pairs_and_distinct_pressures(self):
        expected = {
            "tp01", "tp02",
            "gp01", "gp03",
            "ex01", "ex02",
            "lo01", "mm01", "fi01",
            "FrozenLake-v1", "ArcAgi/Chess-v0", "ArcAgi/Sudoku-v0",
        }
        self.assertEqual(set(RESEARCH_1_GAME_IDS), expected)

    def test_cli_routes_research_1_through_mixed_dispatch(self):
        original_dispatch = base_cli.run_actor_jobs

        def inspect_main(_argv):
            import v7.game_sets as game_sets
            from v8.mixed_environment_v859 import run_mixed_actor_jobs

            self.assertEqual(game_sets.resolve_game_selector("research_1"), RESEARCH_1_GAME_IDS)
            self.assertIs(base_cli.run_actor_jobs, run_mixed_actor_jobs)
            return 0

        with patch.object(base_cli, "main", side_effect=inspect_main):
            self.assertEqual(
                adaptive_cli_main(["continuous-run", "--games", "research_1", "--actors", "12"]),
                0,
            )
        self.assertIs(base_cli.run_actor_jobs, original_dispatch)

    def test_research_packet_labels_research_1_transfer_scope(self):
        packet = researcher_packet.build_packet(
            {
                "games": list(RESEARCH_1_GAME_IDS),
                "actors": [],
                "automatic_transfer_experiments": {"attempted": 0, "completed": 0, "passed": 0},
                "metrics": {"level_counts": {}},
            },
            revision="test",
            argv=["continuous-run", "--games", "research_1"],
            h_report=[],
            reporting_cut={},
            evidence_digest={"available": True, "record_count": 0},
            log_tail="",
        )
        self.assertIn("ARC-only subset of research_1", packet)
        self.assertIn("do not causally test those families", packet)


if __name__ == "__main__":
    unittest.main()
