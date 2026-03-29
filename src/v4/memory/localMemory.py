from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .memoryUpdate import LocalMemoryUpdateV4


@dataclass(frozen=True)
class LocalMemoryStateV4:
    revision: int = 0
    recent_transition_refs: tuple[str, ...] = ()
    recent_actions: tuple[dict[str, Any], ...] = ()
    recent_step_results: tuple[dict[str, Any], ...] = ()
    visited_state_hashes: tuple[str, ...] = ()
    retry_counts: dict[str, int] = field(default_factory=dict)
    cooldown_markers: dict[str, int] = field(default_factory=dict)
    tested_action_outcomes: tuple[dict[str, Any], ...] = ()
    revealed_cells: tuple[tuple[int, int], ...] = ()
    unknown_cells: tuple[tuple[int, int], ...] = ()
    observation_notes: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalMemoryV4:
    def __init__(
        self,
        *,
        max_recent_items: int = 16,
        max_visited_states: int = 64,
        max_notes: int = 16,
    ) -> None:
        self.max_recent_items = int(max_recent_items)
        self.max_visited_states = int(max_visited_states)
        self.max_notes = int(max_notes)
        self._state = LocalMemoryStateV4()

    def reset(self) -> None:
        self._state = LocalMemoryStateV4()

    def snapshot(self) -> LocalMemoryStateV4:
        return self._state

    def apply_update(self, update: LocalMemoryUpdateV4) -> LocalMemoryStateV4:
        if not isinstance(update, LocalMemoryUpdateV4):
            raise ValueError("update must be LocalMemoryUpdateV4")
        retry_counts = dict(self._state.retry_counts)
        for key, value in update.retry_count_increments.items():
            retry_counts[key] = retry_counts.get(key, 0) + int(value)
        cooldown_markers = dict(self._state.cooldown_markers)
        for key, value in update.cooldown_markers.items():
            cooldown_markers[key] = int(value)
        recent_transition_refs = self._bounded_tuple(self._state.recent_transition_refs + tuple(update.transition_refs), self.max_recent_items)
        recent_actions = self._bounded_tuple(
            self._state.recent_actions + tuple(record.to_dict() for record in update.recent_actions),
            self.max_recent_items,
        )
        recent_step_results = self._bounded_tuple(
            self._state.recent_step_results + tuple(record.to_dict() for record in update.recent_step_results),
            self.max_recent_items,
        )
        visited_state_hashes = self._bounded_unique(
            self._state.visited_state_hashes + tuple(update.visited_state_hashes),
            self.max_visited_states,
        )
        tested_action_outcomes = self._bounded_tuple(
            self._state.tested_action_outcomes + tuple(record.to_dict() for record in update.tested_action_outcomes),
            self.max_recent_items,
        )
        revealed_cells = self._bounded_unique(self._state.revealed_cells + tuple(update.revealed_cells), self.max_visited_states)
        unknown_cells = self._bounded_unique(self._state.unknown_cells + tuple(update.unknown_cells), self.max_visited_states)
        observation_notes = self._bounded_tuple(
            self._state.observation_notes + tuple(record.to_dict() for record in update.observation_notes),
            self.max_notes,
        )
        self._state = LocalMemoryStateV4(
            revision=self._state.revision + 1,
            recent_transition_refs=recent_transition_refs,
            recent_actions=recent_actions,
            recent_step_results=recent_step_results,
            visited_state_hashes=visited_state_hashes,
            retry_counts=retry_counts,
            cooldown_markers=cooldown_markers,
            tested_action_outcomes=tested_action_outcomes,
            revealed_cells=revealed_cells,
            unknown_cells=unknown_cells,
            observation_notes=observation_notes,
        )
        return self._state

    def to_dict(self) -> dict[str, Any]:
        return self._state.to_dict()

    @staticmethod
    def _bounded_tuple(values: tuple[Any, ...], limit: int) -> tuple[Any, ...]:
        return tuple(values[-limit:]) if limit > 0 else ()

    @staticmethod
    def _bounded_unique(values: tuple[Any, ...], limit: int) -> tuple[Any, ...]:
        ordered: list[Any] = []
        for value in values:
            if value in ordered:
                ordered.remove(value)
            ordered.append(value)
        if limit > 0:
            ordered = ordered[-limit:]
        return tuple(ordered)
