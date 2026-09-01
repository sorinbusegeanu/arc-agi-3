from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum


class GroundingMaturity(IntEnum):
    G0 = 0
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5


@dataclass(frozen=True, slots=True)
class GroundingEvidence:
    source_symbol_structure: int
    target_interaction_structure: int
    environment_instance_id: int
    context_scope_id: int
    lineage_uid: int
    watermark: int
    recurrent_symbolic: bool = False
    cross_modal_temporal: bool = False
    predictive: bool = False
    causal_intervention: bool = False
    held_out: bool = False
    positive: bool = True


@dataclass(frozen=True, slots=True)
class GroundingState:
    source_symbol_structure: int
    target_interaction_structure: int
    environment_instance_id: int
    context_scope_id: int
    lineage_uid: int
    maturity: GroundingMaturity = GroundingMaturity.G0
    historical_peak: GroundingMaturity = GroundingMaturity.G0
    positive_evidence: int = 0
    negative_evidence: int = 0
    held_out_evidence: int = 0
    validation_watermark: int = 0
    suspended: bool = False

    @property
    def grounds_relation_active(self) -> bool:
        return bool(not self.suspended and self.maturity >= GroundingMaturity.G4)


class GroundingRegistry:
    STATE_VERSION = 1

    def __init__(self) -> None:
        self.states: dict[tuple[int, int, int, int, int], GroundingState] = {}

    @staticmethod
    def _key(evidence: GroundingEvidence) -> tuple[int, int, int, int, int]:
        return (int(evidence.source_symbol_structure), int(evidence.target_interaction_structure), int(evidence.environment_instance_id), int(evidence.context_scope_id), int(evidence.lineage_uid))

    @staticmethod
    def _maturity(evidence: GroundingEvidence) -> GroundingMaturity:
        maturity = GroundingMaturity.G0
        if evidence.recurrent_symbolic: maturity = GroundingMaturity.G1
        if evidence.cross_modal_temporal: maturity = GroundingMaturity.G2
        if evidence.predictive: maturity = GroundingMaturity.G3
        if evidence.causal_intervention and evidence.positive: maturity = GroundingMaturity.G4
        if evidence.causal_intervention and evidence.held_out and evidence.positive: maturity = GroundingMaturity.G5
        return maturity

    def observe(self, evidence: GroundingEvidence) -> GroundingState:
        key = self._key(evidence); current = self.states.get(key, GroundingState(*key)); candidate = self._maturity(evidence)
        positive = int(current.positive_evidence) + (1 if evidence.positive else 0); negative = int(current.negative_evidence) + (0 if evidence.positive else 1); held_out = int(current.held_out_evidence) + (1 if evidence.held_out and evidence.positive else 0)
        suspended = bool(not evidence.positive and evidence.causal_intervention); maturity = candidate if evidence.positive else min(current.maturity, GroundingMaturity.G3)
        historical_peak = max(current.historical_peak, current.maturity, candidate if evidence.positive else GroundingMaturity.G0)
        state = replace(current, maturity=maturity, historical_peak=historical_peak, positive_evidence=positive, negative_evidence=negative, held_out_evidence=held_out, validation_watermark=max(int(current.validation_watermark), int(evidence.watermark)), suspended=suspended)
        self.states[key] = state; return state

    def resolve(self, source: int, target: int, environment: int, context: int, lineage: int = 0) -> GroundingState:
        return self.states.get((int(source), int(target), int(environment), int(context), int(lineage)), GroundingState(int(source), int(target), int(environment), int(context), int(lineage)))

    def state_dict(self) -> dict[str, object]:
        return {"version": self.STATE_VERSION, "states": [{**asdict(row), "maturity": int(row.maturity), "historical_peak": int(row.historical_peak)} for row in sorted(self.states.values(), key=lambda x: (x.source_symbol_structure, x.target_interaction_structure, x.environment_instance_id, x.context_scope_id, x.lineage_uid))]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GroundingRegistry":
        if int(state.get("version", 0)) != cls.STATE_VERSION: raise ValueError("unsupported grounding state")
        obj = cls()
        for raw in state.get("states", []):
            if not isinstance(raw, dict): continue
            data = dict(raw); data["maturity"] = GroundingMaturity(int(data["maturity"])); data["historical_peak"] = GroundingMaturity(int(data["historical_peak"])); row = GroundingState(**data)
            obj.states[(row.source_symbol_structure, row.target_interaction_structure, row.environment_instance_id, row.context_scope_id, row.lineage_uid)] = row
        return obj
