from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v8 import action_learning_report_v849 as action_report
from v8 import learning_effectiveness_report_v850 as report
from v8 import runtime_stack_v88
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningCoordinator,
    FrontierCandidate,
    FrontierScope,
    FrontierSource,
    GameLearningState,
)
from v8.model import MemoryUid, ValidationState


class LearningEffectivenessReportTests(unittest.TestCase):
    def setUp(self):
        action_report._reset_action_learning_state_v849()

    def tearDown(self):
        action_report._reset_action_learning_state_v849()

    @staticmethod
    def _runtime(root="."):
        return type(
            "Runtime",
            (),
            {
                "root": root,
                "generation": 17,
                "watermark": 1234,
                "_v814_trajectory_optimizer": None,
            },
        )()

    def _coordinator(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("transfer-game", "stuck-game"))
        scope = FrontierScope("transfer-game", 3, 0, 0, 0)
        coordinator.frontier.add(
            scope,
            FrontierCandidate(
                MemoryUid(1, 2),
                "transfer-trajectory",
                11,
                8,
                10,
                8,
                int(ValidationState.TESTED),
                FrontierSource.TRANSFER,
                17,
            ),
        )
        with coordinator._lock:
            coordinator._game_won["transfer-game"] = True
            coordinator._records[("transfer-game", 3)] = type(
                "Record", (), {"state": GameLearningState.SOLVED_OPTIMIZING}
            )()
            run = coordinator._run["transfer-game"]
            run.sample_steps = 100
            run.transfer_attempts = 3
            run.alternative_attempts = 2
            run.optimizer_candidates = 4
            run.optimizer_validations = 10
            run.optimizer_successes = 7
            run.optimizer_saved_actions = 13
            coordinator._run["stuck-game"].sample_steps = 50
        return coordinator

    def test_runtime_stack_installs_report_after_v849(self):
        self.assertEqual(runtime_stack_v88._POST_LAYERS[-1], "learning_effectiveness_report_v850")
        self.assertTrue(report._INSTALLED)

    def test_snapshot_reports_learning_application_progress_and_transfer(self):
        coordinator = self._coordinator()
        action_report._RUN["transfer-game"] = action_report._empty_aggregate()
        action_report._RUN["transfer-game"]["movement_actions_executed"] = 100
        action_report._RUN["transfer-game"]["movement_productive"] = 70
        action_report._RUN["stuck-game"] = action_report._empty_aggregate()
        action_report._RUN["stuck-game"]["movement_actions_executed"] = 50
        action_report._RUN["stuck-game"]["movement_productive"] = 10

        completed = {
            "transfer-game": {
                "steps": 100,
                "wins": 1,
                "failures": 0,
                "levels_completed": 3,
                "replans": 2,
                "planned_steps": 60,
                "first_win_step": 80,
                "resets": 1,
            },
            "stuck-game": {
                "steps": 50,
                "wins": 0,
                "failures": 1,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 20,
                "first_win_step": 0,
                "resets": 2,
            },
        }

        payload = report.learning_effectiveness_snapshot_v850(
            self._runtime(), coordinator, completed, {}, {}
        )
        summary = payload["summary"]
        self.assertEqual(summary["sample_steps"], 150)
        self.assertEqual(summary["planned_steps"], 80)
        self.assertAlmostEqual(summary["planned_step_share"], 80.0 / 150.0)
        self.assertEqual(summary["games_with_m7_plan_applied"], 2)
        self.assertEqual(summary["games_with_m7_plan_and_current_run_progress"], 1)
        self.assertEqual(summary["games_with_transfer_frontier"], 1)
        self.assertEqual(summary["games_with_transfer_frontier_and_current_run_progress"], 1)
        self.assertEqual(summary["optimizer_saved_actions"], 13)
        self.assertAlmostEqual(summary["optimizer_success_rate"], 0.7)
        self.assertAlmostEqual(summary["productive_action_rate"], 80.0 / 150.0)
        self.assertFalse(summary["causal_ablation_executed"])

        transfer = next(row for row in payload["games"] if row["game_id"] == "transfer-game")
        self.assertEqual(transfer["known_levels_solved"], 5)
        self.assertEqual(transfer["effectiveness_status"], "M7_APPLIED_WITH_CURRENT_RUN_PROGRESS")
        self.assertEqual(transfer["frontier_source"], "TRANSFER")
        self.assertTrue(transfer["transfer_frontier_with_current_run_progress"])
        self.assertAlmostEqual(transfer["planned_step_share"], 0.6)
        self.assertEqual(transfer["first_win_step"], 80)

        stuck = next(row for row in payload["games"] if row["game_id"] == "stuck-game")
        self.assertEqual(stuck["effectiveness_status"], "M7_APPLIED_NO_CURRENT_RUN_PROGRESS")

    def test_log_is_jsonl_and_keeps_hypothesis_semantics_separate(self):
        coordinator = self._coordinator()
        completed = {
            "transfer-game": {
                "steps": 20,
                "wins": 1,
                "failures": 0,
                "levels_completed": 1,
                "replans": 0,
                "planned_steps": 10,
                "first_win_step": 20,
                "resets": 0,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            report._write_learning_effectiveness_log(
                self._runtime(tmp), coordinator, completed, {}, {}
            )
            path = Path(tmp) / "learning_effectiveness.log"
            payload = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["summary"]["causal_attribution"], "OBSERVATIONAL_NOT_ABLATED")
        self.assertNotIn("hypotheses", payload)


if __name__ == "__main__":
    unittest.main()
