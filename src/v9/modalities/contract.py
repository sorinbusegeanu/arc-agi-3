from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from v9.memory.identity import EpisodeId, EventUid, ModalityId
from v9.memory.model import ExperienceEvent


WORLD_MODALITY = ModalityId(1)
SYMBOL_MODALITY = ModalityId(2)


class TimelineEventKind(str, Enum):
    INTERACTION = "INTERACTION"
    PASSIVE_WORLD = "PASSIVE_WORLD"
    PASSIVE_SYMBOL = "PASSIVE_SYMBOL"


@dataclass(frozen=True, slots=True)
class TimelineIdentity:
    event_id: EventUid
    causal_watermark: int
    producer_id: int
    producer_sequence: int
    environment_instance_id: int
    episode_id: EpisodeId
    modality_id: ModalityId

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return self.causal_watermark, self.producer_id, self.producer_sequence


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    identity: TimelineIdentity
    experience: ExperienceEvent

    def __post_init__(self) -> None:
        if self.identity.modality_id != WORLD_MODALITY:
            raise ValueError("interaction events must use world modality")
        if self.identity.event_id != self.experience.event_id:
            raise ValueError("interaction and timeline event IDs differ")


@dataclass(frozen=True, slots=True)
class PassiveWorldEvent:
    identity: TimelineIdentity
    observation_schema_id: int
    observation_signature: int
    payload: Any = None


@dataclass(frozen=True, slots=True)
class PassiveSymbolEvent:
    identity: TimelineIdentity
    vocabulary_id: object
    stream_id: object
    symbol_id: object
    position: int

    def __post_init__(self) -> None:
        if self.identity.modality_id != SYMBOL_MODALITY:
            raise ValueError("symbol events must use symbol modality")
        if self.position < 0:
            raise ValueError("symbol position cannot be negative")


TimelineEvent = InteractionEvent | PassiveWorldEvent | PassiveSymbolEvent

