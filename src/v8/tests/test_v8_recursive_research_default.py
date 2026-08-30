from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v8.research.default_analysis import derive_chain_evidence
from v8.research.researcher_packet import (
    _evidence_digest,
    build_packet,
    write_researcher_packet,
)
from v8.research.default_cli import run_with_default_research
from v8.research.contracts import ChainStatus


class DefaultRecursiveResearchTests(unittest.TestCase):
    def _summary(self):
        return {
            "games": ["ic01", "gp03", "ArcAgi/Sudoku-v0"],
            "actors": [
                {"game_id": "ic01", "steps": 100, "wins": 1, "failures": 0,
                 "levels_completed": 1, "replans": 0, "planned_steps": 10},
                {"game_id": "gp03", "steps": 100, "wins": 0, "failures": 1,
                 "levels_completed": 0, "replans": 0, "planned_steps": 0},
                {"game_id": "ArcAgi/Sudoku-v0", "steps": 100, "wins": 0, "failures": 0,
                 "levels_completed": 0, "replans": 0, "planned_steps": 0},
            ],
            "automatic_transfer_experiments": {"attempted": 2, "completed": 2, "passed": 1},
            "hypotheses": {"H01": "VALID"},
            "metrics": {"watermark": 300, "generation": 4, "memories": 20, "edges": 30,
                        "level_counts": {"1": 10, "2": 4, "3": 3, "4": 2, "7": 1}, "peers": None},
            "final_snapshot": None,
        }

    def test_missing_transfer_evidence_remains_insufficient(self):
        summary = self._summary()
        summary["automatic_transfer_experiments"] = {"attempted": 0, "completed": 0, "passed": 0}
        evidence = derive_chain_evidence(summary)
        self.assertEqual(evidence["M4_RELEVANT_CANDIDATE"].status, ChainStatus.INSUFFICIENT_EVIDENCE)

    def test_packet_delegates_hypothesis_and_next_experiment_to_llm(self):
        packet = build_packet(
            self._summary(), revision="abc", argv=["continuous-run", "--games", "mix"],
            h_report=[{"hypothesis_id": "H01", "final_decision": "VALID"}],
            reporting_cut={"watermark": 300},
            evidence_digest={"available": True, "record_count": 4, "samples": []},
            log_tail="sample log",
        )
        self.assertIn("has **not selected a research hypothesis or next experiment**", packet)
        self.assertIn("Competing hypotheses", packet)
        self.assertIn("Most informative next experiment", packet)
        self.assertIn("--games mix", packet)

    def test_mix_packet_labels_arc_only_intervention_scope(self):
        packet = build_packet(
            self._summary(), revision="abc", argv=["continuous-run", "--games", "mix"],
            h_report=[], reporting_cut={},
            evidence_digest={"available": True, "record_count": 0, "samples": []},
            log_tail="",
        )
        self.assertIn("ARC-only subset of mix", packet)
        self.assertIn("does not causally test those families", packet)
        self.assertIn("restored historical evidence", packet)

    def test_evidence_digest_samples_recent_records_not_oldest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            rows = []
            for watermark in range(1, 6):
                rows.append({
                    "evidence_id": f"row-{watermark}",
                    "evidence_kind": "transfer_trial_pass",
                    "decision_watermark": watermark,
                    "evidence_available_watermark": watermark,
                    "effect_direction": 1,
                    "source_game_hash": 1,
                    "target_game_hash": 2,
                })
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = _evidence_digest(path)
        self.assertEqual(digest["record_count"], 5)
        self.assertEqual(
            [row["decision_watermark"] for row in digest["samples"]],
            [5, 4, 3],
        )

    def test_writer_creates_single_llm_handoff_file_and_removes_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v8_run_summary.json").write_text(json.dumps(self._summary()), encoding="utf-8")
            (root / "reports").mkdir()
            (root / "reports" / "h01_h15.json").write_text("[]", encoding="utf-8")
            (root / "reports" / "reporting_cut.json").write_text("{}", encoding="utf-8")
            research = root / "research"
            (research / "evidence_packages").mkdir(parents=True)
            (research / "evidence_packages" / "stale.json").write_text("{}", encoding="utf-8")
            (research / "next_experiment.json").write_text("{}", encoding="utf-8")
            (research / "runtime_hypotheses.json").write_text("[]", encoding="utf-8")
            path = write_researcher_packet(root, argv=["continuous-run", "--games", "mix"])
            self.assertEqual(path.name, "LLM_RESEARCH_PACKET.md")
            self.assertEqual([p.name for p in research.iterdir()], ["LLM_RESEARCH_PACKET.md"])

    def test_exact_existing_continuous_command_triggers_packet_not_local_research_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_main(_argv):
                (root / "v8_run_summary.json").write_text(json.dumps(self._summary()), encoding="utf-8")
                return 0

            result = run_with_default_research(fake_main, [
                "continuous-run", "--root", str(root), "--games", "mix",
                "--steps-per-game", "1000", "--actors", "10", "--shards", "4", "--stage-workers", "2",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((root / "research" / "LLM_RESEARCH_PACKET.md").is_file())
            self.assertFalse((root / "research" / "next_experiment.json").exists())
            self.assertFalse((root / "research" / "runtime_hypotheses.json").exists())

    def test_non_continuous_command_does_not_create_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_with_default_research(lambda _argv: 0, ["smoke", "--root", str(root)]), 0)
            self.assertFalse((root / "research").exists())


if __name__ == "__main__":
    unittest.main()
