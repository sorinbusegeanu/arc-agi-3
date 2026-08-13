from __future__ import annotations

from dataclasses import dataclass

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel


@dataclass(frozen=True, slots=True)
class MemoryNode:
    memory_id: MemoryId
    level: MemoryLevel
    type_id: int
    created_generation: GenerationId
    updated_generation: GenerationId
    status_flags: int = 0
    support_count: int = 0


@dataclass(frozen=True, slots=True)
class MemoryScore:
    memory_id: MemoryId
    significance: float = 0.0
    prediction_error: float = 0.0
    learning_value: float = 0.0
    transfer_prior: float = 0.0
    explanatory_potential: float = 0.0
    future_option_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class EdgeState:
    source_id: MemoryId
    relation_type: int
    target_id: MemoryId
    support_count: int


@dataclass(frozen=True, slots=True)
class NodeMutation:
    memory_id: MemoryId
    level: MemoryLevel
    type_id: int
    support_delta: int = 0
    status_flags: int | None = None


@dataclass(frozen=True, slots=True)
class EdgeMutation:
    source_id: MemoryId
    relation_type: int
    target_id: MemoryId
    support_delta: int = 1


@dataclass(frozen=True, slots=True)
class ScoreMutation:
    memory_id: MemoryId
    significance: float | None = None
    prediction_error: float | None = None
    learning_value: float | None = None
    transfer_prior: float | None = None
    explanatory_potential: float | None = None
    future_option_delta: float | None = None
