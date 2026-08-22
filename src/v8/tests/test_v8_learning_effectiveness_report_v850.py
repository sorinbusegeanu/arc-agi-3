from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from v8 import action_learning_report_v849 as action_report
from v8 import learning_effectiveness_report_v850 as report
from v8 import lease_dispatch_lifecycle_v843 as v843
from v8 import reporter
from v8 import runtime_stack_v88
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

    @staticmethod
    def _completed():
        return {
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

    def test_runtime_stack_installs_report_after_v849_without_replacing_public_authority(self):
        self.assertEqual(runtime_stack_v88._POST_LAYERS[-1], "learning_effectiveness_report_v850")
        self.assertTrue(report._INSTALLED)
        self.assertIs(v843._BASE_WRITE_ALLOCATION_LOG, report._write_allocation_log_v850)
        self.assertIs(report._BASE_WRITE_ALLOCATION_LOG, action_report._write_allocation_log_v849)
        self.assertIs(reporter._emit_line, report._reporter_emit_line_v850)

    def test_snapshot_contains_summary_effectiveness_areas_only(self):
        coordinator = self._coordinator()
        action_report._RUN["transfer-game"] = action_report._empty_aggregate()
        action_report._RUN["transfer-game"]["movement_actions_executed"] = 100
        action_report._RUN["transfer-game"]["movement_productive"] = 70
        action_report._RUN["stuck-game"] = action_report._empty_aggregate()
        action_report._RUN["stuck-game"]["movement_actions_executed"] = 50
        action_report._RUN["stuck-game"]["movement_productive"] = 10

        payload = report.learning_effectiveness_snapshot_v850(
            self._runtime(), coordinator, self._completed(), {}, {}
        )
        self.assertNotIn("games", payload)
        effectiveness = payload["effectiveness"]
        self.assertEqual(
            set(effectiveness),
            {
                "outcome_effectiveness",
                "learning_application_effectiveness",
                "transfer_effectiveness",
                "optimizer_effectiveness",
                "action_effectiveness",
                "efficiency",
                "causal_effectiveness",
            },
        )
        self.assertAlmostEqual(effectiveness["outcome_effectiveness"]["level_solve_rate_pct"], 50.0)
        self.assertAlmostEqual(effectiveness["outcome_effectiveness"]["game_solve_rate_pct"], 50.0)
        self.assertAlmostEqual(
            effectiveness["learning_application_effectiveness"]["m7_action_share_pct"],
            80.0 / 150.0 * 100.0,
        )
        self.assertAlmostEqual(
            effectiveness["learning_application_effectiveness"]["m7_strategy_effectiveness_pct"],
            50.0,
        )
        self.assertAlmostEqual(
            effectiveness["transfer_effectiveness"]["transfer_effectiveness_pct"],
            100.0,
        )
        self.assertAlmostEqual(
            effectiveness["optimizer_effectiveness"]["optimizer_success_rate_pct"],
            70.0,
        )
        self.assertEqual(effectiveness["optimizer_effectiveness"]["optimizer_saved_actions"], 13)
        self.assertAlmostEqual(effectiveness["action_effectiveness"]["productive_action_rate_pct"], 80.0 / 150.0 * 100.0)
        self.assertAlmostEqual(effectiveness["efficiency"]["steps_per_level_advance"], 50.0)
        self.assertEqual(effectiveness["efficiency"]["mean_first_win_step"], 80.0)
        self.assertEqual(
            effectiveness["causal_effectiveness"]["status"],
            "NOT_MEASURED_NO_ABLATION",
        )

    def test_compact_stdout_contains_effectiveness_ratios_only(self):
        coordinator = self._coordinator()
        action_report._RUN["transfer-game"] = action_report._empty_aggregate()
        action_report._RUN["transfer-game"]["movement_actions_executed"] = 100
        action_report._RUN["transfer-game"]["movement_productive"] = 70
        action_report._RUN["stuck-game"] = action_report._empty_aggregate()
        action_report._RUN["stuck-game"]["movement_actions_executed"] = 50
        action_report._RUN["stuck-game"]["movement_productive"] = 10
        payload = report.learning_effectiveness_snapshot_v850(
            self._runtime(), coordinator, self._completed(), {}, {}
        )
        line = report.format_learning_effectiveness_stdout_v850(payload)
        self.assertEqual(
            line,
            "effectiveness L=50.0% G=50.0% M7=53.3% M7eff=50.0% "
            "Xfer=100.0% Opt=70.0% Prod=53.3% step/L=50 firstWin=80",
        )
        self.assertNotIn("current_run_", line)

    def test_old_current_run_stdout_is_suppressed_but_sampling_done_is_preserved(self):
        calls = []
        base = report._BASE_REPORTER_EMIT_LINE
        try:
            report._BASE_REPORTER_EMIT_LINE = lambda message, output_queue: calls.append(message)
            report._reporter_emit_line_v850(
                "100% - current_run_wins=50.0% current_run_levels_solved=60.0% current_run_solved_games=2/4",
                None,
            )
            report._reporter_emit_line_v850("sampling done", None)
        finally:
            report._BASE_REPORTER_EMIT_LINE = base
        self.assertEqual(calls, ["sampling done"])

    def test_log_is_jsonl_summary_only_and_write_emits_one_effectiveness_line(self):
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
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report._write_learning_effectiveness_log(
                    self._runtime(tmp), coordinator, completed, {}, {}
                )
            path = Path(tmp) / "learning_effectiveness.log"
            payload = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["schema_version"], 2)
        self.assertIn("effectiveness", payload)
        self.assertNotIn("games", payload)
        self.assertNotIn("hypotheses", payload)
        stdout = output.getvalue().strip()
        self.assertIn("effectiveness L=", stdout)
        self.assertNotIn("current_run_", stdout)


if __name__ == "__main__":
    unittest.main()
