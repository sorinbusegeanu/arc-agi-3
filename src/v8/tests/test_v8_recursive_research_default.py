from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v8.research.default_analysis import build_default_analysis, run_default_research_analysis
from v8.research.default_cli import run_with_default_research


class DefaultRecursiveResearchTests(unittest.TestCase):
    def _summary(self):
        return {
            "games": ["ez01", "ez02"],
            "actors": [
                {
                    "game_id": "ez01",
                    "steps": 100,
                    "wins": 1,
                    "failures": 0,
                    "levels_completed": 1,
                    "replans": 0,
                    "planned_steps": 10,
                },
                {
                    "game_id": "ez02",
                    "steps": 100,
                    "wins": 0,
                    "failures": 1,
                    "levels_completed": 0,
                    "replans": 0,
                    "planned_steps": 0,
                },
            ],
            "automatic_transfer_experiments": {
                "attempted": 2,
                "completed": 2,
                "passed": 1,
            },
            "hypotheses": {"H01": "VALID"},
            "metrics": {
                "watermark": 200,
                "generation": 4,
                "memories": 20,
                "edges": 30,
                "level_counts": {"1": 10, "2": 4, "3": 3, "4": 2, "7": 1},
                "peers": None,
            },
            "final_snapshot": None,
        }

    def test_full_evidence_chain_can_pass(self):
        analysis = build_default_analysis(self._summary(), revision="abc")
        self.assertTrue(analysis["chain"]["complete"])
        self.assertIsNone(analysis["chain"]["first_unresolved_link"])
        self.assertEqual(analysis["package"]["run"]["revision"], "abc")

    def test_missing_transfer_evidence_remains_insufficient(self):
        summary = self._summary()
        summary["automatic_transfer_experiments"] = {
            "attempted": 0,
            "completed": 0,
            "passed": 0,
        }
        summary["actors"][0]["planned_steps"] = 0
        analysis = build_default_analysis(summary, revision="abc")
        self.assertEqual(
            analysis["chain"]["first_unresolved_link"],
            "M4_RELEVANT_CANDIDATE",
        )
        statuses = {
            item["edge"]: item["status"] for item in analysis["chain"]["edges"]
        }
        self.assertEqual(
            statuses["M4_RELEVANT_CANDIDATE"],
            "INSUFFICIENT_EVIDENCE",
        )

    def test_m7_without_planning_is_action_integration_failure(self):
        summary = self._summary()
        summary["actors"][0]["planned_steps"] = 0
        analysis = build_default_analysis(summary, revision="abc")
        self.assertEqual(
            analysis["chain"]["first_broken_link"],
            "ACTION_INTEGRATION",
        )

    def test_completed_transfer_without_pass_is_behavioral_failure(self):
        summary = self._summary()
        summary["automatic_transfer_experiments"] = {
            "attempted": 2,
            "completed": 2,
            "passed": 0,
        }
        analysis = build_default_analysis(summary, revision="abc")
        self.assertEqual(
            analysis["chain"]["first_broken_link"],
            "BEHAVIORAL_IMPROVEMENT",
        )

    def test_default_analysis_writes_outputs_and_research_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(
                json.dumps(self._summary()),
                encoding="utf-8",
            )
            result = run_default_research_analysis(root)
            self.assertTrue(Path(result["report_path"]).is_file())
            self.assertTrue(
                (root / "research" / "latest_evidence_package.json").is_file()
            )
            self.assertTrue(
                (root / "research" / "causal_chain_report.json").is_file()
            )
            self.assertTrue(
                (root / "research" / "runtime_hypotheses.json").is_file()
            )
            self.assertTrue(
                (root / "research" / "next_experiment.json").is_file()
            )
            self.assertTrue((root / "research" / "research.db").is_file())

    def test_exact_existing_continuous_command_triggers_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_main(_argv):
                (root / "v8_run_summary.json").write_text(
                    json.dumps(self._summary()),
                    encoding="utf-8",
                )
                return 0

            result = run_with_default_research(
                fake_main,
                [
                    "continuous-run",
                    "--root",
                    str(root),
                    "--games",
                    "mix",
                    "--steps-per-game",
                    "1000",
                    "--actors",
                    "10",
                    "--shards",
                    "4",
                    "--stage-workers",
                    "2",
                ],
            )
            self.assertEqual(result, 0)
            self.assertTrue(
                (root / "research" / "research_summary.md").is_file()
            )

    def test_non_continuous_command_does_not_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_with_default_research(
                lambda _argv: 0,
                ["smoke", "--root", str(root)],
            )
            self.assertEqual(result, 0)
            self.assertFalse((root / "research").exists())


if __name__ == "__main__":
    unittest.main()
