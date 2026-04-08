from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.agentContract.types import V4Action, V4Observation, V4StepResult, V4TransitionRecord
from v4_5.contracts.gameControlProfile import GameControlProfile


@dataclass(frozen=True)
class LiveObservationSnapshot:
    game_id: str
    level_id: str
    step_index: int
    observation: V4Observation
    parsed_state: Any | None
    game_control_profile: GameControlProfile | None = None
    levels_completed: int = 0
    win_levels: int = 0
    terminal_status: str = "non_terminal"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observation"] = self.observation.to_dict()
        return payload


@dataclass(frozen=True)
class ExecutedPrefixResult:
    prefix: tuple[str, ...]
    primitive_actions: tuple[V4Action, ...]
    transitions: tuple[V4TransitionRecord, ...]
    step_results: tuple[V4StepResult, ...]
    steps_executed: int
    pre_levels_completed: int
    post_levels_completed: int
    levels_completed_delta: int
    terminal_status: str
    terminal_success: bool
    terminal_failure: bool
    level_transition: bool
    game_completion: bool
    observed_effects: tuple[str, ...] = ()
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primitive_actions"] = [item.to_dict() for item in self.primitive_actions]
        payload["transitions"] = [item.to_dict() for item in self.transitions]
        payload["step_results"] = [item.to_dict() for item in self.step_results]
        return payload


@dataclass(frozen=True)
class LiveStepRecord:
    round_id: str
    step_index: int
    selected_prefix: tuple[str, ...]
    executed_prefix: tuple[str, ...]
    action_executed: bool
    action_count: int
    pre_levels_completed: int
    post_levels_completed: int
    levels_completed_delta: int
    terminal_status: str
    stop_reason: str | None = None
    failure_reason: str | None = None
    action_legal: bool = True
    observed_effects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveLevelSummary:
    level_index: int
    started_step_index: int
    ended_step_index: int
    completed: bool
    terminal_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveGameResult:
    game_id: str
    attempted: bool
    stop_reason: str
    steps_executed: int
    failure_reason: str | None
    levels_completed_start: int
    levels_completed_end: int
    win_levels: int
    step_records: tuple[LiveStepRecord, ...] = ()
    level_summaries: tuple[LiveLevelSummary, ...] = ()
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["step_records"] = [item.to_dict() for item in self.step_records]
        payload["level_summaries"] = [item.to_dict() for item in self.level_summaries]
        return payload
