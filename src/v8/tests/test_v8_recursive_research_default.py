from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from v8.research.contracts import ChainStatus
from v8.research.default_analysis import derive_chain_evidence
from v8.research.experiment_artifacts import (
    DECISION_NAME,
    EVIDENCE_NAME,
    _DECISION_BEGIN,
    _DECISION_END,
    _evidence_digest,
    capture_experiment_start,
    write_experiment_evidence,
)
from v8.mixed_environment_v859 import RESEARCH_1_GAME_IDS


class DefaultRecursiveResearchTests(unittest.TestCase):
    def _decision(self, *, games="research_1", steps=20000, memory_policy="REUSE"):
        return {
            "change_id": "R100-A1",
            "change_type": "ARCHITECTURE_CHANGE",
            "target_hypothesis": "H05",
            "target_causal_edge": "M2_TO_M3",
            "target_files": ["src/v8/example.py"],
            "target_functions": ["example"],
            "change": ["Run the declared intervention."],
            "must_not_change": ["Scientific thresholds."],
            "games": games,
            "steps_per_game": steps,
            "memory_policy": memory_policy,
            "primary_metric": "first_m3_watermark",
            "predicted_change": "Earlier M3 formation.",
            "minimum_meaningful_effect": "At least one checkpoint earlier.",
            "expected_unchanged_metrics": ["formation thresholds"],
            "falsifier": "M3 formation is not earlier.",
            "decision_rule": ["Reject the explanation if timing does not improve."],
        }

    def _write_decision(self, root: Path, **kwargs):
        research = root / "research"
        research.mkdir(parents=True, exist_ok=True)
        metadata = self._decision(**kwargs)
        (research / DECISION_NAME).write_text(
            f"# RESEARCH_DECISION\n{_DECISION_BEGIN}\n{json.dumps(metadata)}\n{_DECISION_END}\n",
            encoding="utf-8",
        )
        if (
            metadata["memory_policy"] == "REUSE"
            and (root / "v8_run_summary.json").is_file()
        ):
            summary = json.loads((root / "v8_run_summary.json").read_text(encoding="utf-8"))
            snapshot = root / "snapshots" / "snapshot-00000000000000000001"
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "COMPLETE").write_text("", encoding="utf-8")
            manifest_path = snapshot / "manifest.json"
            manifest_path.write_text(json.dumps({
                "snapshot_id": 1,
                "generation": 1,
                "watermark": summary.get("metrics", {}).get("watermark", 0),
                "final": True,
            }), encoding="utf-8")
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            summary["final_snapshot"] = {
                "snapshot_id": 1,
                "generation": 1,
                "watermark": summary.get("metrics", {}).get("watermark", 0),
                "path": str(snapshot),
                "digest": digest,
                "final": True,
            }
            (root / "v8_run_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
        return metadata

    def _summary(self, *, watermark=300, memories=20, evidence_records=7):
        return {
            "games": list(RESEARCH_1_GAME_IDS),
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
            metadata = self._write_decision(root)
            boundary_path = capture_experiment_start(
                root,
                argv=["continuous-run", "--games", "research_1", "--steps-per-game", "20000"],
            )
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        self.assertEqual(boundary["start_state"]["watermark"], 300)
        self.assertEqual(boundary["decision_metadata"]["change_id"], "R100-A1")
        self.assertIn("memory is intentionally reused", boundary["start_state_source"])
        self.assertTrue(boundary["start_state_identity"]["available"])
        self.assertEqual(len(boundary["start_state_identity"]["sha256"]), 64)
        self.assertTrue(boundary["start_snapshot_identity"]["available"])

    def test_evidence_reports_start_end_delta_local_ledger_and_formation_funnel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(json.dumps(self._summary(watermark=100, memories=10, evidence_records=2)), encoding="utf-8")
            (root / "evidence").mkdir()
            ledger = root / "evidence" / "v8_evidence.jsonl"
            ledger.write_text(json.dumps({"evidence_kind": "old"}) + "\n", encoding="utf-8")
            self._write_decision(root)
            capture_experiment_start(
                root,
                argv=["continuous-run", "--games", "research_1", "--steps-per-game", "20000"],
            )
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

    def test_reuse_requires_a_valid_durable_start_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_decision(root)
            with self.assertRaisesRegex(ValueError, "requires a valid durable"):
                capture_experiment_start(
                    root,
                    argv=["continuous-run", "--games", "research_1", "--steps-per-game", "20000"],
                )

    def test_reuse_rejects_incomplete_summary_even_with_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            self._write_decision(root)
            summary = json.loads((root / "v8_run_summary.json").read_text())
            summary.pop("final_snapshot")
            (root / "v8_run_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary.final_snapshot"):
                capture_experiment_start(
                    root,
                    argv=["continuous-run", "--games", "research_1",
                          "--steps-per-game", "20000"],
                )

if __name__ == "__main__":
    unittest.main()
