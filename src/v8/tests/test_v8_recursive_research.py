from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v8.research import (
    ChainEdgeEvidence,
    ChainStatus,
    ExperimentPrediction,
    ExperimentPurpose,
    ExperimentSpec,
    MetricResult,
    PredictionDirection,
    PredictionOutcome,
    ResearchStore,
    audit_chain,
    build_evidence_package,
    evaluate_prediction,
    write_evidence_package,
)


class RecursiveResearchTests(unittest.TestCase):
    def test_chain_audit_returns_first_failed_link(self):
        evidence = {
            "M1_FORMATION": ChainEdgeEvidence("M1_FORMATION", ChainStatus.PASS, 4),
            "M2_ABSTRACTION": ChainEdgeEvidence("M2_ABSTRACTION", ChainStatus.PASS, 3),
            "M3_ROLE_FORMATION": ChainEdgeEvidence("M3_ROLE_FORMATION", ChainStatus.PASS, 2),
            "M4_RELEVANT_CANDIDATE": ChainEdgeEvidence("M4_RELEVANT_CANDIDATE", ChainStatus.PASS, 2),
            "CROSS_WORLD_RETRIEVAL": ChainEdgeEvidence("CROSS_WORLD_RETRIEVAL", ChainStatus.PASS, 1),
            "ACTION_INTEGRATION": ChainEdgeEvidence("ACTION_INTEGRATION", ChainStatus.FAIL, 1),
        }
        result = audit_chain(evidence)
        self.assertEqual(result.first_broken_link, "ACTION_INTEGRATION")
        self.assertFalse(result.complete)
        self.assertEqual(result.edges[-1].status, ChainStatus.NOT_REACHED)

    def test_chain_audit_never_guesses_missing_evidence(self):
        result = audit_chain({})
        self.assertIsNone(result.first_broken_link)
        self.assertTrue(all(x.status == ChainStatus.INSUFFICIENT_EVIDENCE for x in result.edges))

    def test_complete_chain_requires_all_pass(self):
        names = (
            "M1_FORMATION", "M2_ABSTRACTION", "M3_ROLE_FORMATION",
            "M4_RELEVANT_CANDIDATE", "CROSS_WORLD_RETRIEVAL",
            "ACTION_INTEGRATION", "BEHAVIORAL_IMPROVEMENT",
        )
        result = audit_chain({name: ChainEdgeEvidence(name, ChainStatus.PASS, 1) for name in names})
        self.assertTrue(result.complete)
        self.assertIsNone(result.first_broken_link)

    def test_prediction_up_confirmed(self):
        prediction = ExperimentPrediction("RH1", "transfer", PredictionDirection.UP, 0.1)
        result = MetricResult("transfer", 0.2, 0.35, sample_count=3)
        self.assertEqual(evaluate_prediction(prediction, result).outcome, PredictionOutcome.CONFIRMED)

    def test_prediction_up_contradicted(self):
        prediction = ExperimentPrediction("RH1", "transfer", PredictionDirection.UP, 0.1)
        result = MetricResult("transfer", 0.4, 0.2, sample_count=3)
        self.assertEqual(evaluate_prediction(prediction, result).outcome, PredictionOutcome.CONTRADICTED)

    def test_prediction_flat_uses_tolerance(self):
        prediction = ExperimentPrediction("RH1", "same_game", PredictionDirection.FLAT, flat_tolerance=0.02)
        result = MetricResult("same_game", 0.50, 0.51, sample_count=3)
        self.assertEqual(evaluate_prediction(prediction, result).outcome, PredictionOutcome.CONFIRMED)

    def test_prediction_requires_minimum_samples(self):
        prediction = ExperimentPrediction("RH1", "transfer", PredictionDirection.UP, 0.1)
        result = MetricResult("transfer", 0.2, 0.5, sample_count=1)
        self.assertEqual(evaluate_prediction(prediction, result).outcome, PredictionOutcome.INCONCLUSIVE)

    def _spec(self):
        return ExperimentSpec(
            experiment_id="EXP001",
            purpose=ExperimentPurpose.DISCRIMINATION,
            code_revision="abc123",
            snapshot_id="snap1",
            games=("ez01",),
            seeds=(0, 1, 2),
            interaction_budget=100,
            conditions={"control": {"mode": "normal"}, "treatment": {"mode": "transfer_off"}},
            predictions=(ExperimentPrediction("RH1", "transfer", PredictionDirection.DOWN, 0.1),),
        )

    def test_experiment_must_have_control(self):
        spec = self._spec()
        invalid = ExperimentSpec(
            experiment_id=spec.experiment_id,
            purpose=spec.purpose,
            code_revision=spec.code_revision,
            snapshot_id=spec.snapshot_id,
            games=spec.games,
            seeds=spec.seeds,
            interaction_budget=spec.interaction_budget,
            conditions={"treatment": {}},
            predictions=spec.predictions,
        )
        with self.assertRaises(ValueError):
            invalid.validate()

    def test_store_rejects_results_before_prediction_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ResearchStore(Path(tmp) / "research.db") as store:
                store.create_experiment(self._spec())
                with self.assertRaises(RuntimeError):
                    store.record_result("EXP001", "control", "transfer", 0.1, 0.0, 3)

    def test_store_accepts_results_after_prediction_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "research.db"
            with ResearchStore(path) as store:
                store.create_experiment(self._spec())
                store.freeze_experiment("EXP001")
                store.record_result("EXP001", "control", "transfer", 0.1, 0.0, 3)
                self.assertEqual(store.result_count("EXP001"), 1)
            with ResearchStore(path) as reopened:
                self.assertTrue(reopened.is_frozen("EXP001"))
                self.assertEqual(reopened.result_count("EXP001"), 1)

    def test_evidence_package_is_validated_and_written_atomically(self):
        package = build_evidence_package(
            run={"revision": "abc", "games": ["ez01"], "steps": 10},
            outcomes={}, development={}, transfer={}, control={}, chain_audit={}, hypotheses={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_evidence_package(Path(tmp) / "run.json", package)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["run"]["revision"], "abc")


if __name__ == "__main__":
    unittest.main()
