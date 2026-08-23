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
from v8 import trajectory_click_audit_v856 as audit
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_v814 as optimizer
from v8.action_targeting_v810 import _STRUCTURAL_CLICK_MARKER
from v8.learning_blockers_v055 import pack_action_choice
from v8.model import MemoryUid


class TrajectoryClickAuditV856Tests(unittest.TestCase):
    def setUp(self) -> None:
        inspection._reset_observed_capture()
        audit._ACTION_AUDIT_HISTORY.clear()
        self._prior_root = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")

    def tearDown(self) -> None:
        inspection._reset_observed_capture()
        audit._ACTION_AUDIT_HISTORY.clear()
        if self._prior_root is None:
            os.environ.pop("ARC_AGI3_V8_TRAJECTORY_ROOT", None)
        else:
            os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = self._prior_root

    @staticmethod
    def _solution(game: str, token: int) -> dict[str, object]:
        return {
            "game_id": game,
            "trajectory_id": "trajectory",
            "source": "observed",
            "terminal_state": "WIN",
            "total_cost": 3,
            "levels": [{"level": 0, "actions": [1, int(token), 2]}],
            "attempts": 1,
            "successes": 1,
            "reliability": 1.0,
        }

    def test_runtime_audit_preserves_structural_token_and_records_resolved_target(self) -> None:
        token = int(_STRUCTURAL_CLICK_MARKER | (555 << 8) | 6)
        env = SimpleNamespace(
            _v810_click_targets={
                token: SimpleNamespace(x=4, y=6, kind="component_center")
            },
            _v810_last_changed=(),
            _last_grid=((0,),),
        )
        prior_capture = optimizer._CAPTURE_ACTIVE
        try:
            optimizer._CAPTURE_ACTIVE = True
            audit._ACTION_AUDIT_HISTORY.clear()
            with patch.object(audit, "_BASE_CLICK_STEP", return_value="ok") as base:
                result = audit._click_step_v856(env, token)
            self.assertEqual(result, "ok")
            base.assert_called_once_with(env, token)
            self.assertEqual(
                audit._ACTION_AUDIT_HISTORY,
                [
                    {
                        "action_token": token,
                        "native_action": 6,
                        "x": 4,
                        "y": 6,
                        "target_kind": "component_center",
                    }
                ],
            )
        finally:
            optimizer._CAPTURE_ACTIVE = prior_capture

    def test_existing_exact_click_token_is_decoded_without_new_audit(self) -> None:
        root = Path(tempfile.mkdtemp())
        token = pack_action_choice(6, 12, 34)
        optimizer._atomic_json(
            root / "trajectory_optimizer" / "best_successful.json",
            {"version": 1, "games": {"g": self._solution("g", token)}},
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "g")

        self.assertEqual(code, 0)
        text = stream.getvalue()
        self.assertIn("L0: A1,A6,A2", text)
        self.assertIn("L0 clicks: 1=A6(12,34)", text)

    def test_new_structural_observed_solution_persists_executed_coordinates(self) -> None:
        root = Path(tempfile.mkdtemp())
        os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = str(root)
        token = int(_STRUCTURAL_CLICK_MARKER | (123 << 8) | 6)
        audit._ACTION_AUDIT_HISTORY[:] = [
            {
                "action_token": token,
                "native_action": 6,
                "x": 2,
                "y": 3,
                "target_kind": "component_center",
            }
        ]
        row = optimizer.SuccessfulTrajectory(
            "trajectory",
            optimizer.ReplayAnchor("g", 0, (), None),
            optimizer.TrajectoryTarget(1, "WIN"),
            (token,),
            MemoryUid.zero(),
            MemoryUid.zero(),
            0,
        )

        inspection._write_complete_observed_solution(row)

        files = sorted((root / "solutions_inbox").glob("*.json"))
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        level = payload["levels"][0]
        self.assertEqual(level["actions"], [token])
        self.assertEqual(
            level["action_audit"],
            [
                {
                    "action_token": token,
                    "native_action": 6,
                    "x": 2,
                    "y": 3,
                    "target_kind": "component_center",
                }
            ],
        )

    def test_structural_click_detail_uses_persisted_execution_audit(self) -> None:
        token = int(_STRUCTURAL_CLICK_MARKER | (321 << 8) | 6)
        record = {
            "game_id": "g",
            "trajectory_id": "trajectory",
            "source": "observed",
            "terminal_state": "WIN",
            "total_cost": 1,
            "levels": [
                {
                    "level": 0,
                    "actions": [token],
                    "action_audit": [
                        {
                            "action_token": token,
                            "native_action": 6,
                            "x": 5,
                            "y": 7,
                            "target_kind": "changed_component",
                        }
                    ],
                }
            ],
            "attempts": 1,
            "successes": 1,
            "reliability": 1.0,
        }

        validated = inspection._validated_solution_record(record)
        self.assertIsNotNone(validated)
        lines = audit._format_best_trajectory_lines_v856("g", validated)
        self.assertIn("L0: A6", lines)
        self.assertIn("L0 clicks: 0=A6(5,7)[changed_component]", lines)


if __name__ == "__main__":
    unittest.main()
