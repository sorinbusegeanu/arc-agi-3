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
    heldout_transfer: bool = False
    novel_composition: bool = False
    unexperienced_interaction: bool = False
    symbol_mediated_learning: bool = False
    validation_trial_id: str | None = None
    support: float = 1.0
    contradiction: float = 0.0
    positive: bool = True


@dataclass(frozen=True, slots=True)
class GroundingState:
    maturity: GroundingMaturity = GroundingMaturity.G0
    historical_peak: GroundingMaturity = GroundingMaturity.G0
    positive_evidence: int = 0
    negative_evidence: int = 0
    support: float = 0.0
    contradiction: float = 0.0
    last_causal_watermark: int = 0
    validation_trial_ids: tuple[str, ...] = ()
    suspended: bool = False

    @property
    def behavior_eligible(self) -> bool:
        return not self.suspended and self.maturity >= GroundingMaturity.G3 and self.support > self.contradiction

    @property
    def active(self) -> bool:
        return self.behavior_eligible


class GroundingRegistry:
    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.states: dict[tuple[int, int, int, int, int], GroundingState] = {}

    @staticmethod
    def _candidate_maturity(evidence: GroundingEvidence) -> GroundingMaturity:
        maturity = GroundingMaturity.G0
        if evidence.recurrent_symbol:
            maturity = GroundingMaturity.G1
        if evidence.cross_modal_association:
            maturity = GroundingMaturity.G2
        if evidence.prospective_prediction:
            maturity = GroundingMaturity.G3
        if (evidence.heldout_transfer or evidence.causal_intervention) and evidence.positive:
            maturity = GroundingMaturity.G4
        if (
            evidence.symbol_mediated_learning
            or evidence.unexperienced_interaction
            or (evidence.novel_composition and evidence.heldout_transfer)
        ) and evidence.positive:
            maturity = GroundingMaturity.G5
        return maturity

    def observe(self, evidence: GroundingEvidence) -> GroundingState:
        key = (
            evidence.symbol_structure_uid,
            evidence.interaction_structure_uid,
            evidence.environment_instance_id,
            evidence.context_scope_id,
            evidence.lineage_uid,
        )
        current = self.states.get(key, GroundingState())
        candidate = self._candidate_maturity(evidence)

        support_delta = max(0.0, float(evidence.support)) if evidence.positive else 0.0
        contradiction_delta = max(
            0.0,
            float(evidence.contradiction)
            if evidence.contradiction > 0.0
            else (float(evidence.support) if not evidence.positive else 0.0),
        )
        support = current.support + support_delta
        contradiction = current.contradiction + contradiction_delta
        suspended = contradiction >= support and contradiction > 0.0

        if evidence.positive:
            maturity = max(current.maturity, candidate)
        else:
            maturity = min(current.maturity, GroundingMaturity.G2 if suspended else GroundingMaturity.G3)

        validation_trial_ids = current.validation_trial_ids
        if evidence.validation_trial_id and evidence.validation_trial_id not in validation_trial_ids:
            validation_trial_ids = validation_trial_ids + (str(evidence.validation_trial_id),)

        row = replace(
            current,
            maturity=maturity,
            historical_peak=max(current.historical_peak, candidate),
            positive_evidence=current.positive_evidence + int(evidence.positive),
            negative_evidence=current.negative_evidence + int(not evidence.positive),
            support=support,
            contradiction=contradiction,
            last_causal_watermark=max(current.last_causal_watermark, int(evidence.watermark)),
            validation_trial_ids=validation_trial_ids,
            suspended=suspended,
        )
        self.states[key] = row
        return row

    def eligible_states(
        self,
        *,
        environment_instance_id: int | None = None,
        minimum_maturity: GroundingMaturity = GroundingMaturity.G3,
    ) -> tuple[tuple[tuple[int, int, int, int, int], GroundingState], ...]:
        return tuple(
            (key, row)
            for key, row in self.states.items()
            if row.behavior_eligible
            and row.maturity >= minimum_maturity
            and (environment_instance_id is None or key[2] == int(environment_instance_id))
        )

    def authority(
        self,
        symbol_structure_uid: int,
        interaction_structure_uid: int,
        environment_instance_id: int,
        context_scope_id: int = 0,
        lineage_uid: int = 0,
    ) -> float:
        row = self.states.get(
            (
                int(symbol_structure_uid),
                int(interaction_structure_uid),
                int(environment_instance_id),
                int(context_scope_id),
                int(lineage_uid),
            )
        )
        if row is None or not row.behavior_eligible:
            return 0.0
        confidence = row.support / max(1e-9, row.support + row.contradiction)
        maturity_scale = (int(row.maturity) - int(GroundingMaturity.G2)) / 3.0
        return max(0.0, min(1.0, confidence * maturity_scale))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "states": [
                {
                    "key": list(key),
                    **asdict(row),
                    "maturity": int(row.maturity),
                    "historical_peak": int(row.historical_peak),
                    "validation_trial_ids": list(row.validation_trial_ids),
                }
                for key, row in sorted(self.states.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GroundingRegistry":
        result = cls()
        for raw in state.get("states", []):
            key = tuple(int(value) for value in raw["key"])
            row = GroundingState(
                maturity=GroundingMaturity(int(raw["maturity"])),
                historical_peak=GroundingMaturity(int(raw["historical_peak"])),
                positive_evidence=int(raw["positive_evidence"]),
                negative_evidence=int(raw["negative_evidence"]),
                support=float(raw.get("support", raw.get("positive_evidence", 0))),
                contradiction=float(raw.get("contradiction", raw.get("negative_evidence", 0))),
                last_causal_watermark=int(raw.get("last_causal_watermark", 0)),
                validation_trial_ids=tuple(str(value) for value in raw.get("validation_trial_ids", ())),
                suspended=bool(raw["suspended"]),
            )
            result.states[key] = row
        return result
