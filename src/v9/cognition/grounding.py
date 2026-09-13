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
    symbol_structure_uid: int
    interaction_structure_uid: int
    environment_instance_id: int
    context_scope_id: int
    lineage_uid: int
    watermark: int
    recurrent_symbol: bool = False
    cross_modal_association: bool = False
    prospective_prediction: bool = False
    causal_intervention: bool = False
    novel_composition: bool = False
    unexperienced_interaction: bool = False
    positive: bool = True


@dataclass(frozen=True, slots=True)
class GroundingState:
    maturity: GroundingMaturity = GroundingMaturity.G0
    historical_peak: GroundingMaturity = GroundingMaturity.G0
    positive_evidence: int = 0
    negative_evidence: int = 0
    suspended: bool = False

    @property
    def active(self) -> bool:
        return not self.suspended and self.maturity >= GroundingMaturity.G4


class GroundingRegistry:
    def __init__(self) -> None:
        self.states: dict[tuple[int, int, int, int, int], GroundingState] = {}

    def observe(self, evidence: GroundingEvidence) -> GroundingState:
        key = (evidence.symbol_structure_uid, evidence.interaction_structure_uid, evidence.environment_instance_id, evidence.context_scope_id, evidence.lineage_uid)
        current = self.states.get(key, GroundingState())
        maturity = GroundingMaturity.G0
        if evidence.recurrent_symbol:
            maturity = GroundingMaturity.G1
        if evidence.cross_modal_association:
            maturity = GroundingMaturity.G2
        if evidence.prospective_prediction:
            maturity = GroundingMaturity.G3
        if evidence.causal_intervention and evidence.positive:
            maturity = GroundingMaturity.G4
        if evidence.novel_composition and evidence.causal_intervention and evidence.positive:
            maturity = GroundingMaturity.G4
        if evidence.unexperienced_interaction and evidence.causal_intervention and evidence.positive:
            maturity = GroundingMaturity.G5
        if not evidence.positive:
            maturity = min(current.maturity, GroundingMaturity.G3)
        row = replace(current, maturity=maturity, historical_peak=max(current.historical_peak, maturity), positive_evidence=current.positive_evidence + int(evidence.positive), negative_evidence=current.negative_evidence + int(not evidence.positive), suspended=bool(evidence.causal_intervention and not evidence.positive))
        self.states[key] = row
        return row

    def state_dict(self) -> dict[str, object]:
        return {"states": [{"key": list(key), **asdict(row), "maturity": int(row.maturity), "historical_peak": int(row.historical_peak)} for key, row in sorted(self.states.items())]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GroundingRegistry":
        result = cls()
        for raw in state.get("states", []):
            key = tuple(int(value) for value in raw["key"])
            row = GroundingState(GroundingMaturity(int(raw["maturity"])), GroundingMaturity(int(raw["historical_peak"])), int(raw["positive_evidence"]), int(raw["negative_evidence"]), bool(raw["suspended"]))
            result.states[key] = row
        return result
