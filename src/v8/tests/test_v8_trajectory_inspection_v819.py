from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.model import MemoryUid


class TrajectoryInspectionV819Tests(unittest.TestCase):
    def setUp(self) -> None:
        inspection._reset_observed_capture()
        inspection._CAPTURED_SOLUTIONS_FOR_TESTS.clear()
        self._prior_root = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")

    def tearDown(self) -> None:
        inspection._reset_observed_capture()
        if self._prior_root is None:
            os.environ.pop("ARC_AGI3_V8_TRAJECTORY_ROOT", None)
        else:
            os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = self._prior_root

    @staticmethod
    def row(
        game: str,
        trajectory_id: str,
        *,
        prefix=(),
        actions=(),
        levels_completed: int,
        terminal_state: str,
    ):
        return optimizer.SuccessfulTrajectory(
            trajectory_id,
            optimizer.ReplayAnchor(game, 0, tuple(prefix), None),
            optimizer.TrajectoryTarget(levels_completed, terminal_state),
            tuple(actions),
            MemoryUid.zero(),
            MemoryUid.zero(),
            0,
        )

    @staticmethod
    def solution(
        game: str,
        identifier: str,
        levels,
        *,
        source: str = "observed",
        attempts: int = 1,
        successes: int = 1,
    ) -> dict[str, object]:
        normalized = [tuple(int(value) for value in level) for level in levels]
        payload = {
            "game_id": game,
            "source": source,
            "terminal_state": "WIN",
            "total_cost": sum(len(level) for level in normalized),
            "levels": [
                {"level": index, "actions": list(level)}
                for index, level in enumerate(normalized)
            ],
            "attempts": attempts,
            "successes": successes,
            "reliability": successes / max(1, attempts),
        }
        if source == "optimized":
            payload["variant_id"] = identifier
        else:
            payload["trajectory_id"] = identifier
        return payload

    @staticmethod
    def service(root: Path, *, on_validation=None):
        return optimizer.TrajectoryOptimizationService(
            root,
            validator=lambda _candidate: SimpleNamespace(success=False),
            on_validation=on_validation,
        )

    def _solution_files(self, root: Path):
        return sorted((root / "solutions_inbox").glob("*.json"))

    def test_multi_level_observed_win_capture_preserves_exact_boundaries(self) -> None:
        root = Path(tempfile.mkdtemp())
        os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = str(root)

        optimizer._write_successful_trajectory(
            self.row("g", "l0", actions=(1, 1, 2), levels_completed=1, terminal_state="LEVEL")
        )
        optimizer._write_successful_trajectory(
            self.row(
                "g",
                "l1",
                prefix=(1, 1, 2),
                actions=(3, 4),
                levels_completed=2,
                terminal_state="LEVEL",
            )
        )
        optimizer._write_successful_trajectory(
            self.row(
                "g",
                "win",
                prefix=(1, 1, 2, 3, 4),
                actions=(5, 6),
                levels_completed=3,
                terminal_state="WIN",
            )
        )

        files = self._solution_files(root)
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["source"], "observed")
        self.assertEqual(payload["terminal_state"], "WIN")
        self.assertEqual(payload["total_cost"], 7)
        self.assertEqual(
            [tuple(level["actions"]) for level in payload["levels"]],
            [(1, 1, 2), (3, 4), (5, 6)],
        )

    def test_stale_partial_episode_is_reset_by_new_first_level(self) -> None:
        root = Path(tempfile.mkdtemp())
        os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = str(root)

        optimizer._write_successful_trajectory(
            self.row("g", "old", actions=(9, 9), levels_completed=1, terminal_state="LEVEL")
        )
        optimizer._write_successful_trajectory(
            self.row("g", "new", actions=(1,), levels_completed=1, terminal_state="LEVEL")
        )
        optimizer._write_successful_trajectory(
            self.row(
                "g",
                "win",
                prefix=(1,),
                actions=(2, 3),
                levels_completed=2,
                terminal_state="WIN",
            )
        )

        payload = json.loads(self._solution_files(root)[0].read_text(encoding="utf-8"))
        self.assertEqual(
            [tuple(level["actions"]) for level in payload["levels"]],
            [(1,), (2, 3)],
        )

    def test_observed_prefix_mismatch_never_manufactures_complete_solution(self) -> None:
        root = Path(tempfile.mkdtemp())
        os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = str(root)

        optimizer._write_successful_trajectory(
            self.row("g", "l0", actions=(1,), levels_completed=1, terminal_state="LEVEL")
        )
        optimizer._write_successful_trajectory(
            self.row(
                "g",
                "bad",
                prefix=(8,),
                actions=(2,),
                levels_completed=2,
                terminal_state="LEVEL",
            )
        )
        optimizer._write_successful_trajectory(
            self.row(
                "g",
                "win",
                prefix=(8, 2),
                actions=(3,),
                levels_completed=3,
                terminal_state="WIN",
            )
        )
        self.assertEqual(self._solution_files(root), [])

    def test_shortest_solution_replaces_prior_best(self) -> None:
        root = Path(tempfile.mkdtemp())
        service = self.service(root)
        optimizer._atomic_json(
            service.solutions_inbox / "long.json",
            self.solution("g", "long", ((1, 2, 3), (4, 5))),
        )
        v818._ingest_inbox_v818(service)
        optimizer._atomic_json(
            service.solutions_inbox / "short.json",
            self.solution("g", "short", ((1, 2), (4,))),
        )
        v818._ingest_inbox_v818(service)

        payload = json.loads(service.best_successful_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["games"]["g"]["trajectory_id"], "short")
        self.assertEqual(payload["games"]["g"]["total_cost"], 3)

    def test_equal_cost_prefers_reliability_then_optimized_source(self) -> None:
        root = Path(tempfile.mkdtemp())
        service = self.service(root)
        weak = self.solution("g", "weak", ((1, 2),), attempts=2, successes=1)
        strong = self.solution("g", "strong", ((1, 2),), attempts=2, successes=2)
        optimized = self.solution(
            "g", "opt", ((1, 2),), source="optimized", attempts=2, successes=2
        )
        self.assertTrue(inspection._consider_best_solution(service, weak))
        self.assertTrue(inspection._consider_best_solution(service, strong))
        self.assertTrue(inspection._consider_best_solution(service, optimized))
        best = service._v819_best_successful["g"]
        self.assertEqual(best["source"], "optimized")
        self.assertEqual(best["variant_id"], "opt")

    def test_optimized_win_reconstructs_levels_from_v818_cumulative_prefixes(self) -> None:
        root = Path(tempfile.mkdtemp())
        service = self.service(root, on_validation=lambda *_args: None)
        source = self.row(
            "g",
            "source",
            prefix=(1, 2),
            actions=(3, 4, 5),
            levels_completed=2,
            terminal_state="WIN",
        )
        candidate = optimizer.TrajectoryCandidate(
            "candidate",
            source,
            "DELETE_ACTION",
            (3, 4),
            2,
            1,
        )
        validated = optimizer.ValidatedTrajectory(
            "variant",
            source.anchor,
            source.target,
            candidate.actions,
            MemoryUid.zero(),
            MemoryUid.zero(),
            MemoryUid.zero(),
            source.cost,
            candidate.edit_kind,
            2,
            2,
        )
        service._v818_best_prefixes["g"] = {
            0: (),
            1: (1, 2),
            2: (1, 2, 3, 4),
        }
        result = SimpleNamespace(prefix_actions=(1, 2), attempts=2, successes=2)

        service.on_validation(candidate, result, validated)
        best = service._v819_best_successful["g"]
        self.assertEqual(best["source"], "optimized")
        self.assertEqual(best["variant_id"], "variant")
        self.assertEqual(best["total_cost"], 4)
        self.assertEqual(
            [tuple(level["actions"]) for level in best["levels"]],
            [(1, 2), (3, 4)],
        )

    def test_inconsistent_optimized_prefix_chain_is_rejected(self) -> None:
        root = Path(tempfile.mkdtemp())
        service = self.service(root, on_validation=lambda *_args: None)
        source = self.row(
            "g",
            "source",
            prefix=(1, 2),
            actions=(3, 4, 5),
            levels_completed=2,
            terminal_state="WIN",
        )
        candidate = optimizer.TrajectoryCandidate(
            "candidate",
            source,
            "DELETE_ACTION",
            (3, 4),
            2,
            1,
        )
        validated = optimizer.ValidatedTrajectory(
            "variant",
            source.anchor,
            source.target,
            candidate.actions,
            MemoryUid.zero(),
            MemoryUid.zero(),
            MemoryUid.zero(),
            source.cost,
            candidate.edit_kind,
            2,
            2,
        )
        service._v818_best_prefixes["g"] = {
            0: (),
            1: (1, 9),
            2: (1, 2, 3, 4),
        }
        result = SimpleNamespace(prefix_actions=(1, 2), attempts=2, successes=2)

        service.on_validation(candidate, result, validated)
        self.assertNotIn("g", service._v819_best_successful)

    def test_best_solution_persists_and_reloads(self) -> None:
        root = Path(tempfile.mkdtemp())
        first = self.service(root)
        record = self.solution("g", "persisted", ((1, 2), (3,)))
        self.assertTrue(inspection._consider_best_solution(first, record))

        second = self.service(root)
        self.assertEqual(second._v819_best_successful["g"]["trajectory_id"], "persisted")
        payload = json.loads(second.best_successful_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)

    def test_structural_click_is_formatted_as_native_a6(self) -> None:
        from v8 import action_targeting_v810 as targeting

        root = Path(tempfile.mkdtemp())
        trajectory_root = root / "trajectory_optimizer"
        token = int(targeting._STRUCTURAL_CLICK_MARKER | (123 << 8) | 6)
        optimizer._atomic_json(
            trajectory_root / "best_successful.json",
            {
                "version": 1,
                "games": {
                    "g": self.solution("g", "click", ((1, token, 2),)),
                },
            },
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "g")
        self.assertEqual(code, 0)
        self.assertIn("L0: A1,A6,A2", stream.getvalue())

    def test_missing_game_returns_one(self) -> None:
        root = Path(tempfile.mkdtemp())
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "missing")
        self.assertEqual(code, 1)
        self.assertEqual(
            stream.getvalue().strip(),
            "game=missing no successful trajectory found",
        )

    def test_cli_inspection_does_not_construct_runtime(self) -> None:
        from v8 import cli

        root = Path(tempfile.mkdtemp())
        optimizer._atomic_json(
            root / "trajectory_optimizer" / "best_successful.json",
            {
                "version": 1,
                "games": {
                    "g": self.solution("g", "shown", ((1, 2),)),
                },
            },
        )
        stream = io.StringIO()
        with patch.object(cli, "ContinuousMemoryRuntime", side_effect=AssertionError("runtime constructed")):
            with redirect_stdout(stream):
                code = cli.main(
                    [
                        "continuous-run",
                        "--root",
                        str(root),
                        "--show-best-trajectory",
                        "g",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertIn("game=g cost=2 source=observed reliability=1.000", stream.getvalue())

    def test_normal_continuous_run_still_requires_games(self) -> None:
        from v8 import cli

        with self.assertRaisesRegex(ValueError, "--games is required"):
            cli.main(["continuous-run", "--no-snapshots", "--no-peers"])


if __name__ == "__main__":
    unittest.main()
