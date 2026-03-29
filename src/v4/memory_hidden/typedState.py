from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GridPos = tuple[int, int]
CountCell = tuple[GridPos, int]
ConsistencyFact = tuple[str, GridPos, tuple[GridPos, ...], int]


def _validate_pos(field_name: str, value: GridPos) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name}: must be a 2-tuple")
    if not all(isinstance(part, int) for part in value):
        raise ValueError(f"{field_name}: coordinates must be ints")


def _validate_pos_tuple(field_name: str, values: tuple[GridPos, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name}: must be a tuple")
    for index, value in enumerate(values):
        _validate_pos(f"{field_name}[{index}]", value)


@dataclass(frozen=True)
class MemoryHiddenCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    avatar_position: GridPos
    traversable_safe_cells: tuple[GridPos, ...]
    current_legal_actions: tuple[int, ...]
    terminal_status: str
    step_depth: int
    static_bounds: GridPos
    blocked_cells: tuple[GridPos, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.game_family, str) or not self.game_family:
            raise ValueError("game_family: must be a non-empty string")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id: must be a non-empty string")
        if not isinstance(self.level_index, int) or self.level_index < 0:
            raise ValueError("level_index: must be a non-negative int")
        _validate_pos("avatar_position", self.avatar_position)
        _validate_pos("static_bounds", self.static_bounds)
        _validate_pos_tuple("traversable_safe_cells", self.traversable_safe_cells)
        _validate_pos_tuple("blocked_cells", self.blocked_cells)
        if not isinstance(self.current_legal_actions, tuple):
            raise ValueError("current_legal_actions: must be a tuple")
        for index, action_id in enumerate(self.current_legal_actions):
            if not isinstance(action_id, int):
                raise ValueError(f"current_legal_actions[{index}]: must be an int")
        if self.terminal_status not in {"not_played", "non_terminal", "success", "failure"}:
            raise ValueError("terminal_status: unsupported value")
        if not isinstance(self.step_depth, int) or self.step_depth < 0:
            raise ValueError("step_depth: must be a non-negative int")


@dataclass(frozen=True)
class MemoryHiddenFamilyFieldsV4:
    revealed_safe_cells: tuple[GridPos, ...] = ()
    visible_number_cells: tuple[CountCell, ...] = ()
    unrevealed_frontier_cells: tuple[GridPos, ...] = ()
    known_mines: tuple[GridPos, ...] = ()
    forbidden_cells: tuple[GridPos, ...] = ()
    goal_cell: GridPos | None = None
    local_consistency_facts: tuple[ConsistencyFact, ...] = ()

    def __post_init__(self) -> None:
        _validate_pos_tuple("revealed_safe_cells", self.revealed_safe_cells)
        if not isinstance(self.visible_number_cells, tuple):
            raise ValueError("visible_number_cells: must be a tuple")
        for index, item in enumerate(self.visible_number_cells):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"visible_number_cells[{index}]: must contain position and count")
            _validate_pos(f"visible_number_cells[{index}][0]", item[0])
            if not isinstance(item[1], int) or item[1] < 0:
                raise ValueError(f"visible_number_cells[{index}][1]: must be a non-negative int")
        _validate_pos_tuple("unrevealed_frontier_cells", self.unrevealed_frontier_cells)
        _validate_pos_tuple("known_mines", self.known_mines)
        _validate_pos_tuple("forbidden_cells", self.forbidden_cells)
        if self.goal_cell is not None:
            _validate_pos("goal_cell", self.goal_cell)
        if not isinstance(self.local_consistency_facts, tuple):
            raise ValueError("local_consistency_facts: must be a tuple")
        for index, fact in enumerate(self.local_consistency_facts):
            if not isinstance(fact, tuple) or len(fact) != 4:
                raise ValueError(f"local_consistency_facts[{index}]: must contain kind, source, cells, value")
            if not isinstance(fact[0], str) or not fact[0]:
                raise ValueError(f"local_consistency_facts[{index}][0]: must be a non-empty string")
            _validate_pos(f"local_consistency_facts[{index}][1]", fact[1])
            _validate_pos_tuple(f"local_consistency_facts[{index}][2]", fact[2])
            if not isinstance(fact[3], int) or fact[3] < 0:
                raise ValueError(f"local_consistency_facts[{index}][3]: must be a non-negative int")


@dataclass(frozen=True)
class MemoryHiddenTypedStateV4:
    common: MemoryHiddenCommonFieldsV4
    family: MemoryHiddenFamilyFieldsV4 = field(default_factory=MemoryHiddenFamilyFieldsV4)
    layout_evidence_source: str | None = None

    def __post_init__(self) -> None:
        if self.layout_evidence_source is not None and self.layout_evidence_source not in {
            "direct_observation",
            "environment_metadata",
            "local_memory",
        }:
            raise ValueError("layout_evidence_source: unsupported value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryHiddenTypedStateV4:
        return cls(
            common=MemoryHiddenCommonFieldsV4(**payload["common"]),
            family=MemoryHiddenFamilyFieldsV4(**payload.get("family", {})),
            layout_evidence_source=payload.get("layout_evidence_source"),
        )

    def to_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
