from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .identity import EventUid, MemoryUid


class MemoryLevel(IntEnum):
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M7 = 7


class MemoryType(IntEnum):
    EPISODE = 1
    GROUNDED_CONTINGENCY = 100
    NORMALIZED_RELATION = 150
    FAMILY = 200
    ROLE = 300
    CONCEPT = 400
    CONSEQUENCE = 500
    OUTCOME = 600
    STRATEGY = 700


class CognitiveState(IntEnum):
    CANDIDATE = 0
    PROBATION = 1
    ACTIVE = 2
    VALIDATED = 3
    QUARANTINED = 4
    RETIRE_PENDING = 5
    RETIRED = 6
    REACTIVATED = 7


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    event_id: EventUid
    watermark: int
    producer_id: int
    producer_sequence: int
    environment_instance_id: int
    global_step: int
    context_signature: int
    action_id: int
    outcome_signature: int
    family_signature: int = 0
    carrier_signature: int = 0
    future_option_delta: float = 0.0
    changed_cells: int = 0
    primary_valence: int = 0
    trajectory_signature: int = 0
    next_context_signature: int = 0
    prediction_error: float = 0.0

    def __post_init__(self) -> None:
        if min(self.watermark, self.producer_id, self.producer_sequence, self.global_step) < 0:
            raise ValueError("watermarks, producer values and steps must be non-negative")
        if self.primary_valence not in {-1, 0, 1}:
            raise ValueError("primary_valence must be -1, 0 or +1")
        if self.prediction_error < 0:
            raise ValueError("prediction_error cannot be negative")


@dataclass(frozen=True, slots=True)
class CanonicalNode:
    uid: MemoryUid
    level: MemoryLevel
    memory_type: MemoryType
    structural_key: tuple[int, ...]
    created_watermark: int

    @classmethod
    def build(
        cls,
        level: MemoryLevel,
        memory_type: MemoryType,
        structural_key: tuple[int, ...],
        created_watermark: int,
    ) -> "CanonicalNode":
        key = tuple(int(value) for value in structural_key)
        return cls(MemoryUid.from_key(level, memory_type, key), level, memory_type, key, int(created_watermark))

