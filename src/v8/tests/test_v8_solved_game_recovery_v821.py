from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import solved_game_recovery_v821 as recovery
from v8 import trajectory_inspection_v819_fixups as visibility
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryUid


class SolvedGameRecoveryV821Tests(unittest.TestCase):
    def setUp(self) -> None:
        recovery._RUNTIME_ACTIONS.clear()
        recovery._RUNTIME_BOUNDARIES.clear()

    def test_runtime_win_publishes_complete_solution_with_level_boundaries(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_runtime_win("ic02", (2, 3, 4, 5), (2, 4))
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

    def test_runtime_win_is_never_dropped_when_exact_boundaries_are_missing(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_runtime_win("ic02", (6, 7, 8), ())
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

    def test_recovery_is_installed_at_real_environment_boundary(self) -> None:
        self.assertIs(ArcGridEnvironment.step, recovery._tracked_env_step)
        self.assertIs(ArcGridEnvironment.reset, recovery._tracked_env_reset)


if __name__ == "__main__":
    unittest.main()
