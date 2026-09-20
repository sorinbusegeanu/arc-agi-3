from __future__ import annotations

from v9.research.prediction_registry import PredictionResult, ResearchPredictionRegistry


def test_missing_data_is_insufficient_and_failed_threshold_is_violated() -> None:
    registry = ResearchPredictionRegistry()
    definition = registry.definition("F1")
    assert registry.assess("F1", observable_values=None, controls_present=False, threshold_passed=None).result is PredictionResult.INSUFFICIENT_EVIDENCE
    values = {definition.observables[0]: 0.0}
    assert registry.assess("F1", observable_values=values, controls_present=True, threshold_passed=False, evidence_artifacts=("evidence.json",)).result is PredictionResult.VIOLATED
