from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v8 import action_learning_report_v849 as action_report
from v8 import learning_effectiveness_report_v850 as report
from v8 import runtime_stack_v88
from v8 import lease_dispatch_lifecycle_v843 as v843
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningCoordinator,
    FrontierCandidate,
    FrontierScope,
    FrontierSource,
    GameLearningState,
    GameLevelLearningRecord,
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
            coordinator._records[("transfer-game", 3)] = GameLevelLearningRecord(
                state=GameLearningState.SOLVED_OPTIMIZING
            )
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

    def test_runtime_stack_installs_report_after_v849_without_replacing_public_authority(self):
        self.assertEqual(runtime_stack_v88._POST_LAYERS[-1], "learning_effectiveness_report_v850")
        self.assertTrue(report._INSTALLED)
        self.assertIs(v843._BASE_WRITE_ALLOCATION_LOG, report._write_allocation_log_v850)
        self.assertIs(report._BASE_WRITE_ALLOCATION_LOG, action_report._write_allocation_log_v849)

    def test_snapshot_contains_summary_effectiveness_metrics_only(self):
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
        self.assertNotIn("games", payload)
        self.assertNotIn("summary", payload)
        metrics = payload["effectiveness"]

        outcome = metrics["outcome_effectiveness"]
        self.assertAlmostEqual(outcome["level_solve_rate_pct"], 50.0)
        self.assertAlmostEqual(outcome["game_solve_rate_pct"], 50.0)
        self.assertEqual(outcome["current_run_level_advances"], 3)
        self.assertEqual(outcome["current_run_games_won"], 1)

        learning = metrics["learning_application_effectiveness"]
        self.assertAlmostEqual(learning["m7_action_share_pct"], 100.0 * 80.0 / 150.0)
        self.assertAlmostEqual(learning["m7_strategy_effectiveness_pct"], 50.0)
        self.assertEqual(learning["games_with_m7_applied"], 2)

        transfer = metrics["transfer_effectiveness"]
        self.assertAlmostEqual(transfer["transfer_effectiveness_pct"], 100.0)
        self.assertEqual(transfer["transfer_attempts"], 3)
        self.assertEqual(transfer["games_with_transfer_frontier"], 1)

        optimizer = metrics["optimizer_effectiveness"]
        self.assertAlmostEqual(optimizer["optimizer_success_rate_pct"], 70.0)
        self.assertEqual(optimizer["optimizer_saved_actions"], 13)

        action = metrics["action_effectiveness"]
        self.assertAlmostEqual(action["productive_action_rate_pct"], 100.0 * 80.0 / 150.0)

        efficiency = metrics["efficiency"]
        self.assertAlmostEqual(efficiency["steps_per_level_advance"], 50.0)
        self.assertAlmostEqual(efficiency["mean_first_win_step"], 80.0)

        causal = metrics["causal_effectiveness"]
        self.assertEqual(causal["status"], "NOT_MEASURED_NO_ABLATION")

    def test_log_is_jsonl_summary_only_and_keeps_hypothesis_semantics_separate(self):
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
        self.assertEqual(payload["schema_version"], 2)
        self.assertIn("effectiveness", payload)
        self.assertNotIn("games", payload)
        self.assertNotIn("summary", payload)
        self.assertNotIn("hypotheses", payload)


if __name__ == "__main__":
    unittest.main()
