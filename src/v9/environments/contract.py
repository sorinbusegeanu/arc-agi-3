from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .schemas import ActionSchema, EnvironmentIdentity, ObservationSchema


class BoundaryScope(str, Enum):
    NONE = "NONE"
    SUBEPISODE = "SUBEPISODE"
    EPISODE = "EPISODE"


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    scope: BoundaryScope = BoundaryScope.NONE
    primary_valence: int = 0
    continuation: bool = True

    def __post_init__(self) -> None:
        if int(self.primary_valence) not in {-1, 0, 1}:
            raise ValueError("primary_valence must be -1, 0 or +1")


@dataclass(frozen=True, slots=True)
class TaskProgress:
    game_id: str = ""
    level_id: str | None = None
    level_index: int = 0
    levels_completed: int = 0
    terminal: bool = False
    success: bool = False
    failure: bool = False
    truncated: bool = False
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.success and self.failure:
            raise ValueError("task progress cannot be both success and failure")


@dataclass(frozen=True, slots=True)
class WithinActionFrame:
    observation: Any
    ordinal: int


@dataclass(frozen=True, slots=True)
class WithinActionTrace:
    initial_observation: Any
    frames: tuple[WithinActionFrame, ...]
    settled_observation: Any

    def __post_init__(self) -> None:
        if any(frame.ordinal != index for index, frame in enumerate(self.frames)):
            raise ValueError("within-action frame ordinals must be contiguous")


@dataclass(frozen=True, slots=True)
class EnvironmentTransition:
    before_observation: Any
    after_observation: Any
    native_action: Any
    encoded_action: int
    available_actions_before: tuple[int, ...]
    available_actions_after: tuple[int, ...]
    structural_delta: Any
    boundary: BoundaryEvent = BoundaryEvent()
    before_context: int = 0
    after_context: int = 0
    within_action_trace: WithinActionTrace | None = None


@runtime_checkable
class EnvironmentCognitionAdapter(Protocol):
    def identity(self) -> EnvironmentIdentity: ...
    def reset(self) -> Any: ...
    def observe(self) -> Any: ...
    def step(self, native_action: Any) -> Any: ...
    def available_actions(self) -> tuple[int, ...]: ...
    def observation_schema(self) -> ObservationSchema: ...
    def action_schema(self) -> ActionSchema: ...
    def encode_observation(self, observation: Any) -> int: ...
    def encode_action(self, action: Any) -> int: ...
    def semantic_observation(self, observation: Any) -> tuple[tuple[int, int, int, int, float], ...]: ...
    def semantic_action(self, action: Any) -> tuple[tuple[int, int, int, int, float], ...]: ...
    def semantic_delta(self, before: tuple[tuple[int, int, int, int, float], ...], after: tuple[tuple[int, int, int, int, float], ...]) -> tuple[tuple[int, int, int, int, float], ...]: ...
    def transition(self, before: Any, after: Any, action: Any) -> EnvironmentTransition: ...
    def boundary_event(self) -> BoundaryEvent: ...
    def task_progress(self) -> TaskProgress: ...
    def optional_micro_trace(self) -> WithinActionTrace | None: ...
    def optional_symbol_stream(self) -> tuple[object, ...]: ...

