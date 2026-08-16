from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v8.evidence import EvidenceRecord


@dataclass(frozen=True, slots=True)
class ScientificTraceabilityRecord:
    hypothesis_id: str
    paper_claim: str
    candidate_evidence: tuple[str, ...]
    required_evidence: tuple[str, ...]
    falsification_evidence: tuple[str, ...]
    ordering_dependencies: tuple[str, ...] = ()


TRACEABILITY: tuple[ScientificTraceabilityRecord, ...] = (
    ScientificTraceabilityRecord("H01", "Contingencies emerge from recurring interaction evidence.", ("contingency_recurrence",), ("contingency_recurrence",), ()),
    ScientificTraceabilityRecord("H02", "Prediction violation is meaningful only after a supported expectation.", ("supported_prediction",), ("prediction_violation",), (), ("H01",)),
    ScientificTraceabilityRecord("H03", "Transformation families compress multiple established contingencies.", ("family_recurrence",), ("family_compression",), (), ("H01",)),
    ScientificTraceabilityRecord("H04", "Carrier hypotheses earn persistence through explanatory/predictive utility.", ("carrier_candidate",), ("carrier_emergence",), ("carrier_utility_fail",), ("H03",)),
    ScientificTraceabilityRecord("H05", "Functional roles recur across distinct carriers or contexts.", ("role_candidate",), ("role_emergence",), ("role_inconsistency",), ("H04",)),
    ScientificTraceabilityRecord("H06", "Structural correspondence predicts held-out role/structural transfer.", ("transfer_structural",), ("transfer_trial_pass",), ("transfer_trial_fail",), ("H05",)),
    ScientificTraceabilityRecord("H07", "Validated concepts require candidate compression/explanation plus held-out transfer.", ("concept_candidate",), ("concept_transfer_pass",), ("concept_transfer_fail",), ("H06",)),
    ScientificTraceabilityRecord("H08", "World-model/consequence structures add explanatory or predictive integration.", ("consequence_structure",), ("world_model_component",), ("world_model_no_gain",), ("H07",)),
    ScientificTraceabilityRecord("H09", "Bounded future-option structure is learned from the discovered interaction graph.", ("future_option_estimate",), ("future_option_estimate",), (), ("H01",)),
    ScientificTraceabilityRecord("H10", "Contradictions are preferentially resolved by context refinement when supported.", ("context_refinement",), ("context_refinement_gain",), ("context_refinement_fail",), ("H02",)),
    ScientificTraceabilityRecord("H11", "Transfer remains positive across multiple distinct held-out targets.", ("transfer_trial_pass",), ("transfer_trial_pass",), ("transfer_trial_fail",), ("H06",)),
    ScientificTraceabilityRecord("H12", "Efficiency is compared only among outcome-comparable strategies.", ("strategy_reuse",), ("strategy_efficiency",), ("strategy_efficiency_invalid_comparison",), ("H13",)),
    ScientificTraceabilityRecord("H13", "Persistent outcome-equivalence classes are stable and consequence-consistent.", ("outcome_equivalence", "outcome_merge"), ("outcome_consistency_holdout",), ("outcome_consistency_fail",), ("H08",)),
    ScientificTraceabilityRecord("H14", "Replanning preserves outcome identity after deliberate strategy ablation.", ("alternative_strategy", "replanning_observed"), ("replanning_recovery_trial",), ("replanning_recovery_fail",), ("H13",)),
    ScientificTraceabilityRecord("H15", "Target-like preference is learned separately from outcome equivalence.", ("preference_probe",), ("stable_preference_probe",), ("preference_instability",), ("H13",)),
)


_MILESTONE_KIND = {
    "stable_contingency": ("contingency_recurrence",),
    "prediction_violation": ("prediction_violation",),
    "family": ("family_compression",),
    "carrier": ("carrier_emergence",),
    "role": ("role_emergence",),
    "concept_candidate": ("concept_candidate",),
    "validated_concept": ("concept_transfer_pass",),
    "future_option": ("future_option_estimate",),
    "outcome_equivalence": ("outcome_consistency_holdout", "outcome_merge"),
    "multiple_strategies": ("alternative_strategy",),
    "replanning": ("replanning_recovery_trial",),
    "preference": ("stable_preference_probe",),
}


def developmental_milestones(evidence: Iterable[EvidenceRecord]) -> dict[str, int | None]:
    rows = tuple(evidence)
    result: dict[str, int | None] = {}
    for name, kinds in _MILESTONE_KIND.items():
        watermarks = [
            int(row.evidence_available_watermark)
            for row in rows
            if row.evidence_kind in kinds
        ]
        result[name] = min(watermarks) if watermarks else None
    return result


def ordering_gates(evidence: Iterable[EvidenceRecord]) -> dict[str, str]:
    milestone = developmental_milestones(evidence)
    pairs = {
        "contingency_before_prediction": ("stable_contingency", "prediction_violation"),
        "family_before_carrier": ("family", "carrier"),
        "carrier_before_role": ("carrier", "role"),
        "role_before_validated_concept": ("role", "validated_concept"),
        "outcome_before_replanning": ("outcome_equivalence", "replanning"),
        "outcome_before_preference": ("outcome_equivalence", "preference"),
    }
    result: dict[str, str] = {}
    for name, (before, after) in pairs.items():
        left, right = milestone.get(before), milestone.get(after)
        if right is None:
            result[name] = "NOT_REACHED"
        elif left is None:
            result[name] = "FAIL"
        else:
            result[name] = "PASS" if int(left) <= int(right) else "FAIL"
    return result
