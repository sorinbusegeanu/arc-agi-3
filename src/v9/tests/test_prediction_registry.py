from __future__ import annotations

from v9.research.prediction_registry import ResearchPredictionRegistry


def test_registry_definitions_are_concrete_and_auditable() -> None:
    registry = ResearchPredictionRegistry()
    for row in registry.definitions:
        assert row.observables
        assert row.controls
        assert row.statistic
        assert row.threshold
        assert row.evidence_artifacts
