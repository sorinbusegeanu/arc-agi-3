from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v9.memory.identity import EpisodeId, EventUid, SymbolId, SymbolStreamId, SymbolVocabularyId


SYMBOL_SCHEMA_VERSION = 3


class SymbolTemporalPhase(str, Enum):
    BEFORE_ACTION = "BEFORE_ACTION"
    BETWEEN_ACTIONS = "BETWEEN_ACTIONS"
    AFTER_ACTION = "AFTER_ACTION"
    AFTER_OUTCOME = "AFTER_OUTCOME"
    COINCIDENT = "COINCIDENT"


@dataclass(frozen=True, slots=True)
class SymbolOccurrence:
    """Canonical occurrence-level symbolic sensory evidence."""

    occurrence_id: EventUid
    symbol_id: SymbolId
    position: int
    stream_id: SymbolStreamId
    vocabulary_id: SymbolVocabularyId
    codec_id: str
    causal_watermark: int
    macro_step: int
    micro_step: int
    environment_instance_id: int
    episode_id: EpisodeId
    provenance_id: EventUid
    modality_id: int = 0
    temporal_phase: str = SymbolTemporalPhase.COINCIDENT.value
    source_sequence: int = 0

    def __post_init__(self) -> None:
        if self.position < 0 or self.macro_step < 0 or self.micro_step < 0:
            raise ValueError("symbol position/time cannot be negative")
        if not self.codec_id:
            raise ValueError("codec_id is required")
        try:
            SymbolTemporalPhase(str(self.temporal_phase))
        except ValueError as exc:
            raise ValueError(f"unsupported symbolic temporal phase: {self.temporal_phase}") from exc
        if self.source_sequence < 0:
            raise ValueError("source_sequence cannot be negative")
