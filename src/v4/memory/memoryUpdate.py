from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionMemoryRecordV4:
    action_id: int
    action_name: str
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepResultMemoryRecordV4:
    raw_state_before: str
    raw_state_after: str
    terminal_status: str
    reset_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestedActionOutcomeFactV4:
    state_hash: str
    action_id: int
    action_name: str
    outcome_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationNoteV4:
    state_hash: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalMemoryUpdateV4:
    transition_refs: tuple[str, ...] = ()
    recent_actions: tuple[ActionMemoryRecordV4, ...] = ()
    recent_step_results: tuple[StepResultMemoryRecordV4, ...] = ()
    visited_state_hashes: tuple[str, ...] = ()
    retry_count_increments: dict[str, int] = field(default_factory=dict)
    cooldown_markers: dict[str, int] = field(default_factory=dict)
    tested_action_outcomes: tuple[TestedActionOutcomeFactV4, ...] = ()
    revealed_cells: tuple[tuple[int, int], ...] = ()
    unknown_cells: tuple[tuple[int, int], ...] = ()
    observation_notes: tuple[ObservationNoteV4, ...] = ()

    def __post_init__(self) -> None:
        for key, value in self.retry_count_increments.items():
            if not isinstance(key, str):
                raise ValueError("retry_count_increments keys must be strings")
            if not isinstance(value, int):
                raise ValueError("retry_count_increments values must be ints")
        for key, value in self.cooldown_markers.items():
            if not isinstance(key, str):
                raise ValueError("cooldown_markers keys must be strings")
            if not isinstance(value, int):
                raise ValueError("cooldown_markers values must be ints")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
