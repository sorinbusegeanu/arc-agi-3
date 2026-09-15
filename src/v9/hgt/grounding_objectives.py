from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


GROUNDING_OBJECTIVES = (
    "symbol_conditioned_interaction_prediction",
    "symbol_conditioned_relevant_memory_retrieval",
    "world_to_symbol_generalization",
    "heldout_symbol_composition",
    "symbol_conditioned_action_ranking",
    "shuffled_alignment_discrimination",
    "grounding_confidence_calibration",
)


@dataclass(frozen=True, slots=True)
class GroundingObjectiveEvidence:
    objective: str
    target: float
    prediction: float
    causal_watermark: int
    evaluation_watermark: int
    provenance_id: str
    control: str = "aligned"

    def __post_init__(self) -> None:
        if self.objective not in GROUNDING_OBJECTIVES:
            raise ValueError(f"unknown grounding objective: {self.objective}")
        if int(self.causal_watermark) > int(self.evaluation_watermark):
            raise ValueError("grounding target leaks future evidence")


def objective_metrics(rows: Iterable[GroundingObjectiveEvidence]) -> dict[str, float]:
    grouped: dict[str, list[GroundingObjectiveEvidence]] = {name: [] for name in GROUNDING_OBJECTIVES}
    for row in rows:
        grouped[row.objective].append(row)
    result: dict[str, float] = {}
    for objective, evidence in grouped.items():
        if not evidence:
            continue
        mae = sum(abs(float(row.prediction) - float(row.target)) for row in evidence) / len(evidence)
        result[f"hgt_grounding_{objective}_mae"] = mae
        result[f"hgt_grounding_{objective}_accuracy"] = (
            sum(int((row.prediction >= 0.5) == (row.target >= 0.5)) for row in evidence) / len(evidence)
        )
        result[f"hgt_grounding_{objective}_examples"] = float(len(evidence))
    return result


def publish_objective_metrics(runtime: object, rows: Iterable[GroundingObjectiveEvidence]) -> dict[str, float]:
    metrics = objective_metrics(rows)
    setter = getattr(runtime, "set_telemetry_gauge")
    for key, value in metrics.items():
        setter(key, value)
    return metrics


def validate_objective_evidence(rows: Iterable[GroundingObjectiveEvidence]) -> dict[str, int]:
    rows = tuple(rows)
    counts = {name: [0, 0] for name in GROUNDING_OBJECTIVES}
    controls: set[str] = set()
    for row in rows:
        counts[row.objective][int(float(row.target) >= 0.5)] += 1
        controls.add(str(row.control))
    for objective, (negative, positive) in counts.items():
        if negative + positive and (negative == 0 or positive == 0):
            raise ValueError(f"{objective} requires positive and negative evidence")
    return {"examples": len(rows), "positive_examples": sum(v[1] for v in counts.values()), "negative_examples": sum(v[0] for v in counts.values()), "control_groups": len(controls)}
