from __future__ import annotations

import json
import unittest

import v8  # noqa: F401 - installs the production runtime stack
from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.environments.chess_env import ChessAdapter
from v8.environments.gym_adapter import GymDiscreteAdapter
from v8.environments.sudoku_env import SudokuAdapter
from v8 import trajectory_inspection_v819 as inspection
from v8 import verified_success_metrics_v866 as verified
from v8 import verified_trajectory_export_v868 as export
from v8.research import researcher_packet


class VerifiedTrajectoryProvenanceV869Tests(unittest.TestCase):
    def test_generic_adapters_expose_actual_active_episode_seed(self):
        gym = GymDiscreteAdapter("FrozenLake-v1", seed=7, make_kwargs={"is_slippery": False})
        chess = ChessAdapter(seed=11)
        sudoku = SudokuAdapter(seed=13)
        try:
            self.assertEqual(gym.cognitive_episode_seed(), 7)
            self.assertEqual(chess.cognitive_episode_seed(), 11)
            self.assertEqual(sudoku.cognitive_episode_seed(), 13)
            gym.reset()
            chess.reset()
            sudoku.reset()
            self.assertEqual(gym.cognitive_episode_seed(), 8)
            self.assertEqual(chess.cognitive_episode_seed(), 12)
            self.assertEqual(sudoku.cognitive_episode_seed(), 13 + 104729)
        finally:
            gym.close()
            chess.close()
            sudoku.close()

    def test_verified_proxy_records_active_episode_seed(self):
        class FakeInner:
            def step(self, action):
                return action

            def cognitive_boundary_event(self):
                return BoundaryEvent(BoundaryScope.EPISODE, 1, False)

            def cognitive_episode_seed(self):
                return 12345

        captured = []
        original = verified.record_verified_success_v866
        verified.record_verified_success_v866 = lambda **kwargs: captured.append(kwargs) or True
        try:
            proxy = verified._VerifiedAdapterProxy(FakeInner(), "fake", 7)
            proxy.step(3)
        finally:
            verified.record_verified_success_v866 = original
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["seed"], 12345)
        self.assertEqual(captured[0]["actions"], (3,))

    def test_verified_export_and_best_text_preserve_seed(self):
        row = {
            "game_id": "ArcAgi/Chess-v0",
            "trajectory_id": "win",
            "terminal_state": "WIN",
            "seed": 4321,
            "actions": [3330, 1045],
        }
        record = export._verified_export_record(row)
        self.assertEqual(record["seed"], 4321)
        lines = inspection._format_best_trajectory_lines("ArcAgi/Chess-v0", record)
        self.assertIn("seed=4321", lines[0])

    def test_public_verified_proxy_step_is_v869(self):
        self.assertEqual(
            verified._VerifiedAdapterProxy.step.__module__,
            "v8.verified_trajectory_provenance_v869",
        )

    def test_research_packet_does_not_call_carriers_roles(self):
        summary = {
            "games": ["gp03"],
            "actors": [{"game_id": "gp03", "steps": 100}],
            "automatic_transfer_experiments": {"attempted": 0, "completed": 0, "passed": 0},
            "metrics": {
                "level_counts": {"1": 100, "2": 2, "3": 9, "4": 0, "7": 0},
                "watermark": 100,
            },
        }
        packet = researcher_packet.build_packet(
            summary,
            revision="test",
            argv=["continuous-run", "--games", "gp03"],
            h_report=[],
            reporting_cut={},
            evidence_digest={
                "available": True,
                "record_count": 1,
                "evidence_kind_counts": {"carrier_candidate": 1},
            },
            log_tail="",
        )
        start = packet.index("## Deterministic causal-chain diagnostic")
        end = packet.index("## H01-H15 compact status")
        diagnostic = packet[start:end]
        self.assertIn('"edge": "M3_ROLE_FORMATION"', diagnostic)
        self.assertIn('"status": "INSUFFICIENT_EVIDENCE"', diagnostic)
        self.assertIn('"first_unresolved_link": "M3_ROLE_FORMATION"', diagnostic)
        self.assertIn("no role_candidate evidence exists", diagnostic)

    def test_research_packet_passes_m3_role_only_with_role_candidate_evidence(self):
        summary = {
            "games": ["gp03"],
            "actors": [{"game_id": "gp03", "steps": 100}],
            "automatic_transfer_experiments": {"attempted": 0, "completed": 0, "passed": 0},
            "metrics": {
                "level_counts": {"1": 100, "2": 2, "3": 9, "4": 0, "7": 0},
                "watermark": 100,
            },
        }
        packet = researcher_packet.build_packet(
            summary,
            revision="test",
            argv=["continuous-run", "--games", "gp03"],
            h_report=[],
            reporting_cut={},
            evidence_digest={
                "available": True,
                "record_count": 1,
                "evidence_kind_counts": {"role_candidate": 1},
            },
            log_tail="",
        )
        start = packet.index("## Deterministic causal-chain diagnostic")
        end = packet.index("## H01-H15 compact status")
        diagnostic = packet[start:end]
        self.assertIn('"edge": "M3_ROLE_FORMATION"', diagnostic)
        self.assertIn('"evidence_count": 1', diagnostic)


if __name__ == "__main__":
    unittest.main()
