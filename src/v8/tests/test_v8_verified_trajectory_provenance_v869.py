from __future__ import annotations

import unittest

import v8  # noqa: F401 - installs the production runtime stack
from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.environments.chess_env import ChessAdapter
from v8.environments.gym_adapter import GymDiscreteAdapter
from v8.environments.sudoku_env import SudokuAdapter
from v8 import trajectory_inspection_v819 as inspection
from v8 import verified_success_metrics_v866 as verified
from v8 import verified_trajectory_export_v868 as export


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

if __name__ == "__main__":
    unittest.main()
