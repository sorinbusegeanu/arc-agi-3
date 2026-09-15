from __future__ import annotations

import pytest

from v9.hgt.grounding_objectives import (
    GROUNDING_OBJECTIVES,
    GroundingObjectiveEvidence,
    objective_metrics,
    publish_objective_metrics,
)
from v9.research.grounding_h16 import (
    GroundingCondition,
    evaluate_h16,
    publish_h16_report,
    run_synthetic_h16_controls,
)
from v9.telemetry.dashboard import build_primary_dashboard


class _Runtime:
    def __init__(self) -> None:
        self.gauges = {}

    def set_telemetry_gauge(self, key, value) -> None:
        self.gauges[key] = value


def test_grounding_objectives_cover_all_preregistered_hgt_tasks() -> None:
    assert GROUNDING_OBJECTIVES == (
        "symbol_conditioned_interaction_prediction",
        "symbol_conditioned_relevant_memory_retrieval",
        "world_to_symbol_generalization",
        "heldout_symbol_composition",
        "symbol_conditioned_action_ranking",
        "shuffled_alignment_discrimination",
        "grounding_confidence_calibration",
    )


def test_grounding_objective_evidence_enforces_causal_timestamp() -> None:
    with pytest.raises(ValueError, match="future evidence"):
        GroundingObjectiveEvidence(
            GROUNDING_OBJECTIVES[0], 1.0, 0.5, 11, 10, "trial"
        )


def test_grounding_objective_metrics_publish_per_objective() -> None:
    row = GroundingObjectiveEvidence(GROUNDING_OBJECTIVES[0], 1.0, 0.75, 4, 5, "trial")
    runtime = _Runtime()
    metrics = publish_objective_metrics(runtime, (row,))
    assert metrics["hgt_grounding_symbol_conditioned_interaction_prediction_mae"] == 0.25
    assert runtime.gauges == metrics


def test_h16_report_publishes_matched_controls_and_causal_effect() -> None:
    trials = run_synthetic_h16_controls(
        seeds=(1, 2), environment_config_id=7, interaction_budget=24
    )
    report = evaluate_h16(trials)
    runtime = _Runtime()
    publish_h16_report(runtime, report)
    assert runtime.gauges["h16_matched"] == 1
    for condition in GroundingCondition:
        assert f"h16_{condition.value}_score" in runtime.gauges
    assert "h16_heldout_causal_effect" in runtime.gauges


def test_primary_dashboard_contains_grounding_and_h16_metrics() -> None:
    runtime_metrics = {
        "memory_levels": {},
        "grounding_G3_count": 2,
        "grounding_active_count": 2,
        "grounding_mean_confidence": 0.8,
        "symbol_prediction_gain": 0.2,
    }
    diagnostic = {
        "h16_C0_score": 0.1,
        "h16_C1_score": 0.2,
        "h16_C2_score": 0.6,
        "h16_C3_score": 0.15,
        "h16_aligned_advantage": 0.4,
        "h16_heldout_causal_effect": 0.3,
    }
    dashboard = build_primary_dashboard(runtime_metrics, diagnostic)
    assert dashboard["grounding_G3_count"] == 2
    assert dashboard["grounding_mean_confidence"] == 0.8
    assert dashboard["H16_C2_score"] == 0.6
    assert dashboard["H16_heldout_causal_effect"] == 0.3
