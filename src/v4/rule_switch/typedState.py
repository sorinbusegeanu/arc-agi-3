from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GridPos = tuple[int, int]
ColoredPositions = tuple[tuple[int, tuple[GridPos, ...]], ...]


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


def _validate_colored_positions(field_name: str, values: ColoredPositions) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name}: must be a tuple")
    for index, item in enumerate(values):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field_name}[{index}]: must contain color and positions")
        if not isinstance(item[0], int):
            raise ValueError(f"{field_name}[{index}][0]: must be an int")
        _validate_pos_tuple(f"{field_name}[{index}][1]", item[1])


@dataclass(frozen=True)
class RuleSwitchCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    avatar_position: GridPos
    walkable_cells: tuple[GridPos, ...]
    target_cells: tuple[GridPos, ...]
    goal_cells: tuple[GridPos, ...]
    legal_action_ids: tuple[int, ...]
    terminal_status: str
    step_depth: int
    static_bounds: GridPos
    wall_cells: tuple[GridPos, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.game_family, str) or not self.game_family:
            raise ValueError("game_family: must be a non-empty string")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id: must be a non-empty string")
        if not isinstance(self.level_index, int) or self.level_index < 0:
            raise ValueError("level_index: must be a non-negative int")
        _validate_pos("avatar_position", self.avatar_position)
        _validate_pos("static_bounds", self.static_bounds)
        _validate_pos_tuple("walkable_cells", self.walkable_cells)
        _validate_pos_tuple("target_cells", self.target_cells)
        _validate_pos_tuple("goal_cells", self.goal_cells)
        _validate_pos_tuple("wall_cells", self.wall_cells)
        if not isinstance(self.legal_action_ids, tuple):
            raise ValueError("legal_action_ids: must be a tuple")
        for index, action_id in enumerate(self.legal_action_ids):
            if not isinstance(action_id, int):
                raise ValueError(f"legal_action_ids[{index}]: must be an int")
        if self.terminal_status not in {"not_played", "non_terminal", "success", "failure"}:
            raise ValueError("terminal_status: unsupported value")
        if not isinstance(self.step_depth, int) or self.step_depth < 0:
            raise ValueError("step_depth: must be a non-negative int")


@dataclass(frozen=True)
class RuleSwitchFamilyFieldsV4:
    target_items_by_color: ColoredPositions = ()
    active_safe_color: int | None = None
    safe_color_cycle: tuple[int, ...] = ()
    collected_targets_by_color: tuple[tuple[int, int], ...] = ()
    remaining_targets_by_color: ColoredPositions = ()
    cycle_interval: int | None = None
    cycle_index: int | None = None
    all_cycled: bool | None = None

    def __post_init__(self) -> None:
        _validate_colored_positions("target_items_by_color", self.target_items_by_color)
        if self.active_safe_color is not None and not isinstance(self.active_safe_color, int):
            raise ValueError("active_safe_color: must be an int or null")
        if not isinstance(self.safe_color_cycle, tuple):
            raise ValueError("safe_color_cycle: must be a tuple")
        for index, value in enumerate(self.safe_color_cycle):
            if not isinstance(value, int):
                raise ValueError(f"safe_color_cycle[{index}]: must be an int")
        if not isinstance(self.collected_targets_by_color, tuple):
            raise ValueError("collected_targets_by_color: must be a tuple")
        for index, item in enumerate(self.collected_targets_by_color):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"collected_targets_by_color[{index}]: must contain color and count")
            if not isinstance(item[0], int) or not isinstance(item[1], int) or item[1] < 0:
                raise ValueError(f"collected_targets_by_color[{index}]: invalid entry")
        _validate_colored_positions("remaining_targets_by_color", self.remaining_targets_by_color)
        if self.cycle_interval is not None and (not isinstance(self.cycle_interval, int) or self.cycle_interval <= 0):
            raise ValueError("cycle_interval: must be a positive int or null")
        if self.cycle_index is not None and (not isinstance(self.cycle_index, int) or self.cycle_index < 0):
            raise ValueError("cycle_index: must be a non-negative int or null")
        if self.all_cycled is not None and not isinstance(self.all_cycled, bool):
            raise ValueError("all_cycled: must be a bool or null")


@dataclass(frozen=True)
class RuleSwitchTypedStateV4:
    common: RuleSwitchCommonFieldsV4
    family: RuleSwitchFamilyFieldsV4 = field(default_factory=RuleSwitchFamilyFieldsV4)
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
    def from_dict(cls, payload: dict[str, Any]) -> RuleSwitchTypedStateV4:
        return cls(
            common=RuleSwitchCommonFieldsV4(**payload["common"]),
            family=RuleSwitchFamilyFieldsV4(**payload.get("family", {})),
            layout_evidence_source=payload.get("layout_evidence_source"),
        )

    def to_key(self) -> str:
        payload = self.to_dict()
        common = payload.get("common", {})
        if isinstance(common, dict):
            common.pop("step_depth", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
