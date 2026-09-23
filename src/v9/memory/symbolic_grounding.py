from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from .identity import MemoryUid


class GroundingMaturity(IntEnum):
    G0_INTERACTION = 0
    G1_CONCURRENT_SYMBOLS = 1
    G2_DESCRIPTIVE = 2
    G3_PROSPECTIVE = 3
    G4_NOVEL_COMPOSITION = 4
    G5_SYMBOL_MEDIATED_LEARNING = 5


class SymbolicRelation:
    SYMBOL_PRECEDES_SYMBOL = "SYMBOL_PRECEDES_SYMBOL"
    SYMBOL_FOLLOWS_SYMBOL = "SYMBOL_FOLLOWS_SYMBOL"
    SYMBOL_RECURS_WITHIN_WINDOW = "SYMBOL_RECURS_WITHIN_WINDOW"
    SYMBOL_PRECEDES_ACTION = "SYMBOL_PRECEDES_ACTION"
    SYMBOL_FOLLOWS_ACTION = "SYMBOL_FOLLOWS_ACTION"
    SYMBOL_PRECEDES_NORMALIZED_CHANGE = "SYMBOL_PRECEDES_NORMALIZED_CHANGE"
    SYMBOL_FOLLOWS_NORMALIZED_CHANGE = "SYMBOL_FOLLOWS_NORMALIZED_CHANGE"
    SYMBOL_NEAR_BOUNDARY = "SYMBOL_NEAR_BOUNDARY"
    SYMBOL_COINCIDENT_WITH_PROGRESS = "SYMBOL_COINCIDENT_WITH_PROGRESS"
    SYMBOL_COINCIDENT_WITH_OUTCOME = "SYMBOL_COINCIDENT_WITH_OUTCOME"
    CROSS_MODAL_CORRESPONDENCE = "CROSS_MODAL_CORRESPONDENCE"
    SYMBOL_INTERACTION_ALIGNMENT = "SYMBOL_INTERACTION_ALIGNMENT"
    SYMBOL_TO_INTERACTION_PREDICTION = "SYMBOL_TO_INTERACTION_PREDICTION"
    INTERACTION_TO_SYMBOL_GENERALIZATION = "INTERACTION_TO_SYMBOL_GENERALIZATION"
    CROSS_MODAL_HELDOUT_TRANSFER = "CROSS_MODAL_HELDOUT_TRANSFER"
    CROSS_MODAL_COMPOSITION = "CROSS_MODAL_COMPOSITION"


@dataclass(frozen=True, slots=True)
class GroundingEvidence:
    symbol_lineage: tuple[MemoryUid, ...]
    interaction_lineage: tuple[MemoryUid, ...]
    grounded_lineage: tuple[MemoryUid, ...]
    support: float
    contradiction: float
    validation_trial_ids: tuple[str, ...] = ()
    heldout_transfer: bool = False
    novel_composition: bool = False
    symbol_mediated_learning: bool = False
    interaction_only_support: float = 0.0
    symbol_only_support: float = 0.0
    aligned_cross_modal_support: float = 0.0
    heldout_transfer_support: float = 0.0
    causal_watermark: int = 0
    temporal_offsets: tuple[int, ...] = ()

    @property
    def support_decomposition(self) -> dict[str, float]:
        return {"interaction_only": float(self.interaction_only_support), "symbol_only": float(self.symbol_only_support), "aligned_cross_modal": float(self.aligned_cross_modal_support), "heldout_transfer": float(self.heldout_transfer_support)}

    @property
    def maturity(self) -> GroundingMaturity:
        if self.symbol_mediated_learning:
            return GroundingMaturity.G5_SYMBOL_MEDIATED_LEARNING
        if self.novel_composition:
            return GroundingMaturity.G4_NOVEL_COMPOSITION
        if self.heldout_transfer:
            return GroundingMaturity.G3_PROSPECTIVE
        if self.validation_trial_ids and self.support > self.contradiction:
            return GroundingMaturity.G2_DESCRIPTIVE
        if self.symbol_lineage and self.interaction_lineage:
            return GroundingMaturity.G1_CONCURRENT_SYMBOLS
        return GroundingMaturity.G0_INTERACTION

    @property
    def behavior_eligible(self) -> bool:
        return self.maturity >= GroundingMaturity.G3_PROSPECTIVE and self.support > self.contradiction


def modality_support(channels: Iterable[str]) -> dict[str, int]:
    result = {"interaction": 0, "symbol": 0, "cross_modal": 0}
    for channel in channels:
        key = {"WORLD": "interaction", "SYMBOL": "symbol", "CROSS_MODAL": "cross_modal"}.get(str(channel))
        if key:
            result[key] += 1
    return result
