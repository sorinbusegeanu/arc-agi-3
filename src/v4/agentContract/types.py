from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class V4ContractVersion(str, Enum):
    V4_0_0 = "v4.0.0"


@dataclass(frozen=True)
class V4Observation:
    raw_object_name: str
    raw_payload: dict[str, Any]
    game_id: str
    frame: tuple[tuple[tuple[Any, ...], ...], ...]
    state: str
    levels_completed: int
    win_levels: int
    action_input: dict[str, Any]
    guid: str | None
    full_reset: bool
    available_actions: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4Action:
    action_id: int
    action_name: str
    payload: dict[str, Any] | None = None
    reasoning: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4AuthoritativeState:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    full_reset: bool
    available_actions: tuple[int, ...]
    guid: str | None = None
    title: str | None = None
    description: str | None = None
    action_space: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4TerminalSignal:
    status: str
    raw_state: str
    is_terminal: bool
    reset_required: bool
    full_reset: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4TransitionRecord:
    pre_observation: V4Observation
    action: V4Action
    post_observation: V4Observation
    action_legal: bool
    execution_status: str
    terminal_signal: V4TerminalSignal
    step_index: int | None = None
    timestamp_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4StepResult:
    action: V4Action
    action_legal: bool
    terminal_signal: V4TerminalSignal
    raw_state_before: str
    raw_state_after: str
    levels_completed_delta: int | None = None
    win_levels_delta: int | None = None
    reset_required: bool = False
    coordinate_payload: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
