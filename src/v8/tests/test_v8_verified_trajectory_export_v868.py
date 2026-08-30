from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import v8
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_inspection_v819 as inspection
from v8 import verified_trajectory_export_v868 as export


class VerifiedTrajectoryExportV868Tests(unittest.TestCase):
    @staticmethod
    def _run(root: Path, name: str, started_ns: int, events) -> Path:
        run = root / "verified_success" / name
        (run / "events").mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps({"schema_version": 1, "started_ns": int(started_ns)}),
            encoding="utf-8",
        )
        for index, event in enumerate(events):
            payload = {
                "schema_version": 1,
                "trajectory_id": f"t-{name}-{index}",
                "game_id": event[0],
                "terminal_state": event[1],
                "levels_completed": event[2],
                "actions": list(event[3]),
                "action_count": len(event[3]),
                "recorded_ns": int(event[4]) if len(event) > 4 else index + 1,
            }
            (run / "events" / f"{index}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return run

    @staticmethod
    def _arc_record(game: str, actions) -> dict[str, object]:
        values = list(actions)
        return {
            "game_id": game,
            "trajectory_id": f"arc-{game}",
            "source": "observed",
            "terminal_state": "WIN",
            "total_cost": len(values),
            "levels": [{"level": 0, "actions": values}],
            "attempts": 1,
            "successes": 1,
            "reliability": 1.0,
        }

    def test_installed_export_authority_is_v868(self) -> None:
        self.assertIs(inspection.save_best_trajectories, export.save_best_trajectories_v868)

    def test_latest_verified_run_defines_exported_solved_games(self) -> None:
        root = Path(tempfile.mkdtemp())
        self._run(
            root,
            "run-old",
            10,
            (("ArcAgi/Chess-v0", "WIN", 1, (99,), 1),),
        )
        self._run(
            root,
            "run-new",
            20,
            (
                ("tp02", "WIN", 5, (1, 2), 10),
                ("FrozenLake-v1", "WIN", 1, (3, 4, 5), 11),
                ("gp03", "LEVEL", 1, (6,), 12),
            ),
        )
        optimizer._atomic_json(
            root / "trajectory_optimizer" / "best_successful.json",
            {
                "version": 1,
                "games": {
                    "tp02": self._arc_record("tp02", (1, 2)),
                    "ArcAgi/Chess-v0": self._arc_record("ArcAgi/Chess-v0", (99,)),
                },
            },
        )
        output = root / "best.txt"
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.save_best_trajectories(root, output)
        text = output.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertIn("game=tp02 cost=2 source=observed reliability=1.000", text)
        self.assertIn(
            "game=FrozenLake-v1 cost=3 source=verified reliability=1.000", text
        )
        self.assertIn("L0: A3,A4,A5", text)
        self.assertNotIn("game=gp03", text)
        self.assertNotIn("game=ArcAgi/Chess-v0", text)
        self.assertIn(f"saved best trajectories games=2 path={output}", stream.getvalue())

    def test_shortest_win_in_latest_run_is_exported_for_generic_game(self) -> None:
        root = Path(tempfile.mkdtemp())
        self._run(
            root,
            "run-current",
            50,
            (
                ("ArcAgi/Sudoku-v0", "WIN", 1, (9, 8, 7, 6), 1),
                ("ArcAgi/Sudoku-v0", "WIN", 1, (5, 4), 2),
            ),
        )
        output = root / "best.txt"
        inspection.save_best_trajectories(root, output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("game=ArcAgi/Sudoku-v0 cost=2 source=verified reliability=1.000", text)
        self.assertIn("L0: A5,A4", text)
        self.assertNotIn("A9,A8,A7,A6", text)

    def test_no_verified_run_preserves_legacy_export(self) -> None:
        root = Path(tempfile.mkdtemp())
        optimizer._atomic_json(
            root / "trajectory_optimizer" / "best_successful.json",
            {
                "version": 1,
                "games": {"tp02": self._arc_record("tp02", (1, 2, 3))},
            },
        )
        output = root / "best.txt"
        inspection.save_best_trajectories(root, output)
        self.assertIn(
            "game=tp02 cost=3 source=observed reliability=1.000",
            output.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
