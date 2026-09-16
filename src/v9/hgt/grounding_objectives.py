from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


GROUNDING_OBJECTIVES = (
    "symbol_conditioned_relevant_memory_retrieval",
    "cross_modal_correspondence_prediction",
    "symbol_conditioned_interaction_consequence_prediction",
    "world_to_symbol_generalization",
    "heldout_cross_modal_composition",
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


def _targets(payload: Mapping[str, Any]) -> dict[str, tuple[float, bool]]:
    control = str(payload.get("cross_modal_control", "aligned"))
    shuffled = control == "shuffled" or float(payload.get("contradiction", 0.0)) > float(payload.get("support", 0.0))
    relation = str(payload.get("symbol_relation", ""))
    active = bool(payload.get("grounding_active", False))
    explicit = bool(payload.get("symbol_identity") is not None and (relation or active or control == "shuffled"))
    correspondence = relation in {
        "CROSS_MODAL_CORRESPONDENCE",
        "SYMBOL_INTERACTION_ALIGNMENT",
        "SYMBOL_TO_INTERACTION_PREDICTION",
        "INTERACTION_TO_SYMBOL_GENERALIZATION",
        "CROSS_MODAL_HELDOUT_TRANSFER",
        "CROSS_MODAL_COMPOSITION",
    } or active
    prospective = bool(payload.get("prospective_prediction", False)) or relation in {
        "SYMBOL_PRECEDES_ACTION",
        "SYMBOL_TO_INTERACTION_PREDICTION",
    }
    generalization = bool(payload.get("world_to_symbol_generalization", False)) or relation == "INTERACTION_TO_SYMBOL_GENERALIZATION" or (active and bool(payload.get("heldout_transfer", False)))
    heldout = bool(payload.get("heldout_transfer", False))
    composition = bool(payload.get("novel_composition", False)) or relation == "CROSS_MODAL_COMPOSITION"
    confidence = max(0.0, min(1.0, float(payload.get("grounding_confidence", 0.0 if shuffled else (1.0 if active else 0.5)))))
    return {
        "symbol_conditioned_relevant_memory_retrieval": (0.0 if shuffled else 1.0, explicit),
        "cross_modal_correspondence_prediction": (0.0 if shuffled else 1.0, bool(shuffled or correspondence)),
        "symbol_conditioned_interaction_consequence_prediction": (0.0 if shuffled else 1.0, bool(shuffled or prospective)),
        "world_to_symbol_generalization": (0.0 if shuffled else 1.0, bool(shuffled or generalization)),
        "heldout_cross_modal_composition": (1.0 if composition and heldout and not shuffled else 0.0, bool(shuffled or heldout or composition)),
        "shuffled_alignment_discrimination": (0.0 if shuffled else 1.0, explicit),
        "grounding_confidence_calibration": (confidence, explicit),
    }


def _causal_bounds(payload: Mapping[str, Any], node: Any) -> tuple[int, int]:
    relation_causal = int(payload.get("causal_watermark", payload.get("symbol_causal_watermark", node.created_watermark)))
    validation_causal = int(payload.get("validation_causal_watermark", relation_causal))
    evaluation = int(payload.get("grounding_evaluation_watermark", max(int(node.created_watermark), validation_causal)))
    return max(relation_causal, validation_causal), evaluation


def build_objective_evidence(read_view: Any, *, prediction: float = 0.5) -> tuple[GroundingObjectiveEvidence, ...]:
    rows: list[GroundingObjectiveEvidence] = []
    for uid, node in read_view.nodes.items():
        payload = read_view.payloads.get(uid, {})
        if payload.get("symbol_identity") is None:
            continue
        causal, evaluation = _causal_bounds(payload, node)
        if causal > evaluation:
            continue
        control = str(payload.get("cross_modal_control", "aligned"))
        for objective, (target, eligible) in _targets(payload).items():
            if eligible:
                rows.append(GroundingObjectiveEvidence(objective, float(target), float(prediction), causal, evaluation, uid.hex(), control))
    return tuple(rows)


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
        result[f"hgt_grounding_{objective}_accuracy"] = sum(int((row.prediction >= 0.5) == (row.target >= 0.5)) for row in evidence) / len(evidence)
        result[f"hgt_grounding_{objective}_examples"] = float(len(evidence))
        result[f"hgt_grounding_{objective}_positive"] = float(sum(row.target >= 0.5 for row in evidence))
        result[f"hgt_grounding_{objective}_negative"] = float(sum(row.target < 0.5 for row in evidence))
    return result


def publish_objective_metrics(runtime: object, rows: Iterable[GroundingObjectiveEvidence]) -> dict[str, float]:
    metrics = objective_metrics(rows)
    setter = getattr(runtime, "set_telemetry_gauge")
    for key, value in metrics.items():
        setter(key, value)
    return metrics


def validate_objective_evidence(rows: Iterable[GroundingObjectiveEvidence], *, require_complete: bool = False) -> dict[str, int]:
    rows = tuple(rows)
    counts = {name: [0, 0] for name in GROUNDING_OBJECTIVES}
    controls: set[str] = set()
    for row in rows:
        counts[row.objective][int(float(row.target) >= 0.5)] += 1
        controls.add(str(row.control))
    for objective, (negative, positive) in counts.items():
        if require_complete and negative + positive == 0:
            raise ValueError(f"{objective} has no eligible evidence")
        if negative + positive and (negative == 0 or positive == 0):
            raise ValueError(f"{objective} requires positive and negative evidence")
    if require_complete and "shuffled" not in controls:
        raise ValueError("complete grounding supervision requires shuffled controls")
    return {
        "examples": len(rows),
        "positive_examples": sum(v[1] for v in counts.values()),
        "negative_examples": sum(v[0] for v in counts.values()),
        "control_groups": len(controls),
        "objectives": sum(int(sum(v) > 0) for v in counts.values()),
    }
