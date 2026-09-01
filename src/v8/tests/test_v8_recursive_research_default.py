from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v8 import information_flow_diagnostics as flow
from v8.research.contracts import ChainStatus
from v8.research.default_analysis import derive_chain_evidence
from v8.research.default_cli import run_with_default_research
from v8.research.experiment_artifacts import (
    DECISION_NAME,
    EVIDENCE_NAME,
    _DECISION_BEGIN,
    _DECISION_END,
    _evidence_digest,
    capture_experiment_start,
    write_experiment_evidence,
)


class DefaultRecursiveResearchTests(unittest.TestCase):
    def _summary(self, *, watermark=300, memories=20, evidence_records=7):
        return {
            "games": ["ic01", "gp03", "ArcAgi/Sudoku-v0"],
            "actors": [
                {"game_id": "ic01", "steps": 100, "wins": 1, "failures": 0, "levels_completed": 1, "resets": 1},
                {"game_id": "gp03", "steps": 100, "wins": 0, "failures": 1, "levels_completed": 0, "resets": 2},
                {"game_id": "ArcAgi/Sudoku-v0", "steps": 100, "wins": 0, "failures": 0, "levels_completed": 0, "resets": 3},
            ],
            "automatic_transfer_experiments": {"attempted": 2, "completed": 2, "passed": 1},
            "metrics": {
                "watermark": watermark,
                "memories": memories,
                "edges": 30,
                "evidence_records": evidence_records,
                "level_counts": {"1": 10, "2": 4, "3": 3, "4": 2, "7": 1},
                "formation_telemetry": {"m1n_count": 10, "eligible_m2_groups": 2, "role_candidates": 1},
                "verified_success": {"game_solve_rate_pct": 33.3},
                "trajectory_optimizer": {"candidates_generated": 2, "validation_successes": 1},
                "adaptive_learning": {"states": {"UNSOLVED": 2}, "sample_steps": 200},
            },
        }

    def test_missing_transfer_evidence_remains_insufficient(self):
        summary = self._summary()
        summary["automatic_transfer_experiments"] = {"attempted": 0, "completed": 0, "passed": 0}
        evidence = derive_chain_evidence(summary)
        self.assertEqual(evidence["M4_RELEVANT_CANDIDATE"].status, ChainStatus.INSUFFICIENT_EVIDENCE)

    def test_evidence_digest_can_be_scoped_to_pre_run_byte_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            first = json.dumps({"evidence_kind": "old", "effect_direction": 0}) + "\n"
            path.write_text(first, encoding="utf-8")
            offset = path.stat().st_size
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"evidence_kind": "new", "effect_direction": 1, "hypothesis_id": "H05"}) + "\n")
            digest = _evidence_digest(path, start_offset=offset)
        self.assertEqual(digest["record_count"], 1)
        self.assertEqual(digest["evidence_kind_counts"], {"new": 1})
        self.assertEqual(digest["hypothesis_id_counts"], {"H05": 1})

    def test_start_boundary_captures_reused_memory_and_explicit_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(json.dumps(self._summary()), encoding="utf-8")
            research = root / "research"
            research.mkdir()
            metadata = {"change_id": "R001-T1", "change_type": "TELEMETRY", "target_hypothesis": "H05"}
            (research / DECISION_NAME).write_text(
                f"# RESEARCH_DECISION\n{_DECISION_BEGIN}\n{json.dumps(metadata)}\n{_DECISION_END}\n",
                encoding="utf-8",
            )
            boundary_path = capture_experiment_start(root, argv=["continuous-run", "--games", "research_1"])
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        self.assertEqual(boundary["start_state"]["watermark"], 300)
        self.assertEqual(boundary["decision_metadata"]["change_id"], "R001-T1")
        self.assertIn("memory is intentionally reused", boundary["start_state_source"])

    def test_evidence_reports_start_end_delta_local_ledger_and_formation_funnel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(json.dumps(self._summary(watermark=100, memories=10, evidence_records=2)), encoding="utf-8")
            (root / "evidence").mkdir()
            ledger = root / "evidence" / "v8_evidence.jsonl"
            ledger.write_text(json.dumps({"evidence_kind": "old"}) + "\n", encoding="utf-8")
            capture_experiment_start(root, argv=["continuous-run", "--games", "research_1"])
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"evidence_kind": "role_candidate", "hypothesis_id": "H05", "effect_direction": 1}) + "\n")
            (root / "v8_run_summary.json").write_text(json.dumps(self._summary(watermark=150, memories=14, evidence_records=3)), encoding="utf-8")
            (root / "reports").mkdir()
            (root / "reports" / "h01_h15.json").write_text(json.dumps([{"hypothesis_id": "H05", "final_decision": "PARTIALLY_VALID"}]), encoding="utf-8")
            (root / "reports" / "reporting_cut.json").write_text(json.dumps({"watermark": 150}), encoding="utf-8")
            path = write_experiment_evidence(root, exit_code=0)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, EVIDENCE_NAME)
        self.assertIn("## Start state", text)
        self.assertIn("## End state", text)
        self.assertIn("## Experiment deltas", text)
        self.assertIn('"watermark": 50', text)
        self.assertIn('"role_candidate": 1', text)
        self.assertIn('"formation_telemetry"', text)
        self.assertIn("Cumulative state — context only", text)

    def test_normal_continuous_run_generates_evidence_not_legacy_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_main(_argv):
                (root / "v8_run_summary.json").write_text(json.dumps(self._summary()), encoding="utf-8")
                return 0

            result = run_with_default_research(fake_main, [
                "continuous-run", "--root", str(root), "--games", "research_1",
                "--steps-per-game", "20000", "--actors", "30", "--shards", "4", "--stage-workers", "2",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((root / "research" / EVIDENCE_NAME).is_file())
            self.assertFalse((root / "research" / "LLM_RESEARCH_PACKET.md").exists())

    def test_normal_run_replaces_prior_evidence_and_information_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            research = root / "research"
            research.mkdir(parents=True)
            evidence_path = research / EVIDENCE_NAME
            evidence_path.write_text(
                "# prior evidence\n- experiment_id: `prior-experiment`\nOLD_EVIDENCE\n",
                encoding="utf-8",
            )
            information_path = root / flow.LOG_NAME
            information_path.write_text(
                '{"stage":"OLD_INFORMATION"}\n', encoding="utf-8"
            )

            def fake_main(_argv):
                self.assertEqual(evidence_path.read_text(encoding="utf-8"), "")
                self.assertEqual(information_path.read_text(encoding="utf-8"), "")
                with patch.dict(
                    os.environ,
                    {"ARC_AGI3_V8_ROOT": str(root)},
                    clear=False,
                ):
                    flow.emit(
                        "transfer",
                        "CURRENT_INFORMATION",
                        input_count=1,
                        output_count=1,
                    )
                (root / "v8_run_summary.json").write_text(
                    json.dumps(self._summary()), encoding="utf-8"
                )
                return 0

            result = run_with_default_research(
                fake_main,
                [
                    "continuous-run",
                    "--root",
                    str(root),
                    "--games",
                    "research_1",
                ],
            )

            evidence = evidence_path.read_text(encoding="utf-8")
            information = information_path.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertNotIn("OLD_EVIDENCE", evidence)
        self.assertIn("parent_experiment_id: `prior-experiment`", evidence)
        self.assertNotIn("OLD_INFORMATION", information)
        self.assertIn("CURRENT_INFORMATION", information)

    def test_non_continuous_command_does_not_create_research_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_with_default_research(lambda _argv: 0, ["smoke", "--root", str(root)]), 0)
            self.assertFalse((root / "research").exists())


if __name__ == "__main__":
    unittest.main()
