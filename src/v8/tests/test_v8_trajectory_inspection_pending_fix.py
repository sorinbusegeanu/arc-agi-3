from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import v8
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_v814 as optimizer


class TrajectoryInspectionPendingFixTests(unittest.TestCase):
    @staticmethod
    def solution(game: str, identifier: str, levels) -> dict[str, object]:
        normalized = [tuple(int(value) for value in level) for level in levels]
        return {
            "game_id": game,
            "trajectory_id": identifier,
            "source": "observed",
            "terminal_state": "WIN",
            "total_cost": sum(len(level) for level in normalized),
            "levels": [
                {"level": index, "actions": list(level)}
                for index, level in enumerate(normalized)
            ],
            "attempts": 1,
            "successes": 1,
            "reliability": 1.0,
        }

    def test_show_best_reads_pending_solution_without_persisted_best(self) -> None:
        root = Path(tempfile.mkdtemp())
        inbox = root / "trajectory_optimizer" / "solutions_inbox"
        pending = inbox / "ic02-win.json"
        optimizer._atomic_json(
            pending,
            self.solution("ic02", "pending-win", ((1, 2), (3,))),
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "ic02")

        self.assertEqual(code, 0)
        self.assertIn("game=ic02 cost=3 source=observed reliability=1.000", stream.getvalue())
        self.assertIn("L0: A1,A2", stream.getvalue())
        self.assertIn("L1: A3", stream.getvalue())
        self.assertTrue(pending.exists())

    def test_show_best_selects_shorter_pending_over_persisted_solution(self) -> None:
        root = Path(tempfile.mkdtemp())
        optimizer_root = root / "trajectory_optimizer"
        optimizer._atomic_json(
            optimizer_root / "best_successful.json",
            {
                "version": 1,
                "games": {
                    "ic02": self.solution("ic02", "persisted", ((1, 2, 3), (4, 5))),
                },
            },
        )
        optimizer._atomic_json(
            optimizer_root / "solutions_inbox" / "shorter.json",
            self.solution("ic02", "pending", ((1, 2), (4,))),
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "ic02")

        self.assertEqual(code, 0)
        self.assertIn("game=ic02 cost=3 source=observed reliability=1.000", stream.getvalue())
        self.assertIn("L0: A1,A2", stream.getvalue())
        self.assertIn("L1: A4", stream.getvalue())

    def test_solution_inbox_is_ingested_before_optimizer_inbox(self) -> None:
        order: list[str] = []
        service = object()
        with patch.object(
            inspection,
            "_ingest_solution_inbox",
            side_effect=lambda _service: order.append("solution"),
        ), patch.object(
            inspection,
            "_BASE_INGEST_INBOX_V818",
            side_effect=lambda _service: order.append("optimizer"),
        ):
            inspection._ingest_inbox_v819(service)

        self.assertEqual(order, ["solution", "optimizer"])


if __name__ == "__main__":
    unittest.main()
