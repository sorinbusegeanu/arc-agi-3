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
        recovery._RUNTIME_SEGMENT_ACTIONS.clear()
        recovery._RUNTIME_LEVEL_SEGMENTS.clear()
        recovery._RUNTIME_CURRENT_LEVEL.clear()

    def test_runtime_win_publishes_complete_solution_with_level_boundaries(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_runtime_win(
                    "ic02",
                    (2, 3, 4, 5),
                    (2,),
                    expected_levels=2,
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

    def test_known_incomplete_runtime_win_is_not_published(self) -> None:
        prior = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "trajectory_optimizer"
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
                recovery._publish_runtime_win(
                    "ic02",
                    (2, 3),
                    (),
                    expected_levels=5,
                )
                self.assertEqual(tuple((root / "solutions_inbox").glob("*.json")), ())
        finally:
            if prior is None:
                os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
            else:
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = prior

    def test_level_segments_survive_probe_resets_and_keep_shortest_success(self) -> None:
        key = 17
        recovery._store_level_segment(key, 0, (4, 4, 2))
        recovery._store_level_segment(key, 0, (2,))
        recovery._store_level_segment(key, 1, (3, 3))
        recovery._store_level_segment(key, 2, (4,))
        levels = recovery._complete_runtime_levels(key, 3)
        self.assertEqual(levels, ((2,), (3, 3), (4,)))

    def test_show_best_reconstructs_independently_validated_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optimizer_root = root / "trajectory_optimizer"
            optimizer_root.mkdir(parents=True, exist_ok=True)

            rows = []
            for completed, terminal, prefix, actions in (
                (1, "LEVEL", (), (1,)),
                (2, "LEVEL", (9, 9), (2,)),
                (3, "LEVEL", (8,), (3, 3)),
                (4, "LEVEL", (7, 7), (4,)),
                (5, "WIN", (), (2, 3)),
            ):
                anchor = optimizer.ReplayAnchor("ic02", 0, tuple(prefix), None)
                target = optimizer.TrajectoryTarget(completed, terminal)
                rows.append(
                    optimizer.ValidatedTrajectory(
                        f"validated-{completed}",
                        anchor,
                        target,
                        tuple(actions),
                        MemoryUid.zero(),
                        MemoryUid.zero(),
                        MemoryUid.zero(),
                        20,
                        "TRUNCATE_SUCCESS_PREFIX",
                        2,
                        2,
                    )
                )

            (optimizer_root / "validated.json").write_text(
                json.dumps({"version": 1, "validated": [row.to_dict() for row in rows]}),
                encoding="utf-8",
            )
            record = visibility._best_visible_solution(root, "ic02")
            self.assertIsNotNone(record)
            self.assertEqual(record["game_id"], "ic02")
            self.assertEqual(record["source"], "optimized")
            self.assertEqual(record["total_cost"], 6)
            self.assertEqual(
                [level["actions"] for level in record["levels"]],
                [[1], [2], [3, 3], [4], [2, 3]],
            )

    def test_complete_five_level_solution_outranks_stale_two_action_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            optimizer_root = root / "trajectory_optimizer"
            optimizer_root.mkdir(parents=True, exist_ok=True)
            stale = {
                "version": 1,
                "games": {
                    "ic02": {
                        "game_id": "ic02",
                        "trajectory_id": "stale",
                        "source": "observed",
                        "terminal_state": "WIN",
                        "total_cost": 2,
                        "levels": [{"level": 0, "actions": [2, 3]}],
                        "attempts": 1,
                        "successes": 1,
                        "reliability": 1.0,
                    }
                },
            }
            (optimizer_root / "best_successful.json").write_text(
                json.dumps(stale), encoding="utf-8"
            )

            rows = []
            for completed in range(1, 6):
                terminal = "WIN" if completed == 5 else "LEVEL"
                anchor = optimizer.ReplayAnchor("ic02", 0, (), None)
                target = optimizer.TrajectoryTarget(completed, terminal)
                rows.append(
                    optimizer.ValidatedTrajectory(
                        f"v{completed}",
                        anchor,
                        target,
                        (completed,),
                        MemoryUid.zero(),
                        MemoryUid.zero(),
                        MemoryUid.zero(),
                        10,
                        "DIRECT_ACTION",
                        2,
                        2,
                    )
                )
            (optimizer_root / "validated.json").write_text(
                json.dumps({"version": 1, "validated": [row.to_dict() for row in rows]}),
                encoding="utf-8",
            )

            record = visibility._best_visible_solution(root, "ic02")
            self.assertIsNotNone(record)
            self.assertEqual(len(record["levels"]), 5)
            self.assertEqual(record["total_cost"], 5)

    def test_missing_validated_level_does_not_manufacture_complete_solution(self) -> None:
        anchor = optimizer.ReplayAnchor("ic02", 0, (), None)
        rows = (
            optimizer.ValidatedTrajectory(
                "v1",
                anchor,
                optimizer.TrajectoryTarget(1, "LEVEL"),
                (1,),
                MemoryUid.zero(),
                MemoryUid.zero(),
                MemoryUid.zero(),
                4,
                "DIRECT_ACTION",
                2,
                2,
            ),
            optimizer.ValidatedTrajectory(
                "v3",
                anchor,
                optimizer.TrajectoryTarget(3, "WIN"),
                (3,),
                MemoryUid.zero(),
                MemoryUid.zero(),
                MemoryUid.zero(),
                4,
                "DIRECT_ACTION",
                2,
                2,
            ),
        )
        self.assertIsNone(recovery._validated_levels(rows, rows[-1]))

    def test_runtime_win_does_not_expire_back_into_unsolved_discovery_priority(self) -> None:
        self.assertTrue(math.isinf(perf._PROVISIONAL_WIN_SECONDS))

    def test_recovery_is_installed_at_real_environment_boundary(self) -> None:
        self.assertIs(ArcGridEnvironment.step, recovery._tracked_env_step)
        self.assertIs(ArcGridEnvironment.reset, recovery._tracked_env_reset)


if __name__ == "__main__":
    unittest.main()
