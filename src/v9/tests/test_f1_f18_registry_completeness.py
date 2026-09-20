from __future__ import annotations

from v9.research.prediction_registry import ResearchPredictionRegistry


def test_registry_contains_f1_through_f18_and_h19_reject() -> None:
    identifiers = tuple(row.prediction_id for row in ResearchPredictionRegistry().definitions)
    assert identifiers == tuple(f"F{index}" for index in range(1, 19)) + ("H19-REJECT",)
