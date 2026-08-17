from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import v8
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import solved_game_recovery_v821 as recovery
from v8 import trajectory_inspection_v819_fixups as visibility
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryUid


class SolvedGameRecoveryV821Tests(unittest.TestCase):
    def setUp(self) -> None:
        recovery._RECENT_LEVEL_PREFIXES.clear()

    @staticmethod
    def _row(*, level: int, terminal: str, prefix=(), actions=(1,)):
        anchor = optimizer.ReplayAnchor("ic02", 0, tuple(prefix), None)
        target = optimizer.TrajectoryTarget(int(level), str(terminal))
        action_values = tuple(actions)
        return optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, action_values),
            anchor,
            target,
            action_values,
            MemoryUid.zero(),
            MemoryUid.zero(),
            0,
        )

    def test_terminal_win_directly_publishes_complete_solution_with_level_boundaries(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_complete_win(
                    self._row(level=1, terminal="LEVEL", actions=(2, 3))
                )
                recovery._publish_complete_win(
                    self._row(
                        level=2,
                        terminal="WIN",
                        prefix=(2, 3),
                        actions=(4, 5),
                    )
                )
                paths = tuple((root / "solutions_inbox").glob("*.json"))
                self.assertEqual(len(paths), 1)
                payload = json.loads(paths[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["game_id"], "ic02")
                self.assertEqual(payload["terminal_state"], "WIN")
                self.assertEqual(payload["total_cost"], 4)
                self.assertEqual(
                    [row["actions"] for row in payload["levels"]],
                    [[2, 3], [4, 5]],
                )
        finally:
            if prior is None:
                os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
            else:
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = prior

    def test_terminal_win_is_never_dropped_when_exact_boundaries_are_missing(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_complete_win(
                    self._row(level=3, terminal="WIN", actions=(6, 7, 8))
                )
                path = next((root / "solutions_inbox").glob("*.json"))
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["total_cost"], 3)
                self.assertEqual(payload["levels"], [{"level": 0, "actions": [6, 7, 8]}])
        finally:
            if prior is None:
                os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
            else:
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = prior

    def test_show_best_can_recover_durable_validated_win(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optimizer_root = root / "trajectory_optimizer"
            optimizer_root.mkdir(parents=True, exist_ok=True)
            anchor = optimizer.ReplayAnchor("ic02", 0, (), None)
            target = optimizer.TrajectoryTarget(2, "WIN")
            row = optimizer.ValidatedTrajectory(
                "validated-win",
                anchor,
                target,
                (1, 2, 3),
                MemoryUid.zero(),
                MemoryUid.zero(),
                MemoryUid.zero(),
                5,
                "TRUNCATE_SUCCESS_PREFIX",
                2,
                2,
            )
            (optimizer_root / "validated.json").write_text(
                json.dumps({"version": 1, "validated": [row.to_dict()]}),
                encoding="utf-8",
            )
            record = visibility._best_visible_solution(root, "ic02")
            self.assertIsNotNone(record)
            self.assertEqual(record["game_id"], "ic02")
            self.assertEqual(record["source"], "optimized")
            self.assertEqual(record["total_cost"], 3)

    def test_runtime_win_does_not_expire_back_into_unsolved_discovery_priority(self) -> None:
        self.assertTrue(math.isinf(perf._PROVISIONAL_WIN_SECONDS))

    def test_recovery_wrapper_is_installed_after_existing_trajectory_layers(self) -> None:
        self.assertIs(optimizer._write_successful_trajectory, recovery._write_successful_trajectory_v821)


if __name__ == "__main__":
    unittest.main()
