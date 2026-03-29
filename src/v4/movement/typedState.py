from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GridPos = tuple[int, int]


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
class MovementCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    avatar_position: GridPos
    traversable_cells: tuple[GridPos, ...]
    current_legal_actions: tuple[int, ...]
    terminal_status: str
    step_depth: int
    static_bounds: GridPos
    blocked_cells: tuple[GridPos, ...] = ()
    target_cells: tuple[GridPos, ...] = ()
    hazard_positions: tuple[GridPos, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.game_family, str) or not self.game_family:
            raise ValueError("game_family: must be a non-empty string")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id: must be a non-empty string")
        if not isinstance(self.level_index, int) or self.level_index < 0:
            raise ValueError("level_index: must be a non-negative int")
        _validate_pos("avatar_position", self.avatar_position)
        _validate_pos("static_bounds", self.static_bounds)
        if self.static_bounds[0] <= 0 or self.static_bounds[1] <= 0:
            raise ValueError("static_bounds: must be positive")
        _validate_pos_tuple("traversable_cells", self.traversable_cells)
        _validate_pos_tuple("blocked_cells", self.blocked_cells)
        _validate_pos_tuple("target_cells", self.target_cells)
        _validate_pos_tuple("hazard_positions", self.hazard_positions)
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
class MovementFamilyFieldsV4:
    key_inventory_bits: int | None = None
    key_positions: tuple[GridPos, ...] = ()
    door_positions: tuple[GridPos, ...] = ()
    door_open: bool | None = None
    switch_positions: tuple[GridPos, ...] = ()
    activated_switch_bits: int | None = None
    door_state_bits: int | None = None
    teleporter_endpoint_positions: tuple[GridPos, ...] = ()
    teleporter_pairs: tuple[tuple[GridPos, GridPos], ...] = ()
    teleporter_pair_map: tuple[tuple[GridPos, GridPos], ...] = ()
    slide_mode: str | None = None
    ice_cell_positions: tuple[GridPos, ...] = ()
    coverage_eligible_cells: tuple[GridPos, ...] = ()
    coverage_mask: tuple[GridPos, ...] = ()
    pushable_block_positions: tuple[GridPos, ...] = ()
    push_target_cells: tuple[GridPos, ...] = ()
    step_limit: int | None = None

    def __post_init__(self) -> None:
        if self.key_inventory_bits is not None and (not isinstance(self.key_inventory_bits, int) or self.key_inventory_bits < 0):
            raise ValueError("key_inventory_bits: must be a non-negative int or null")
        _validate_pos_tuple("key_positions", self.key_positions)
        _validate_pos_tuple("door_positions", self.door_positions)
        if self.door_open is not None and not isinstance(self.door_open, bool):
            raise ValueError("door_open: must be a bool or null")
        _validate_pos_tuple("switch_positions", self.switch_positions)
        if self.activated_switch_bits is not None and (
            not isinstance(self.activated_switch_bits, int) or self.activated_switch_bits < 0
        ):
            raise ValueError("activated_switch_bits: must be a non-negative int or null")
        if self.door_state_bits is not None and (not isinstance(self.door_state_bits, int) or self.door_state_bits < 0):
            raise ValueError("door_state_bits: must be a non-negative int or null")
        if self.switch_positions and self.activated_switch_bits is not None:
            max_mask = (1 << len(self.switch_positions)) - 1
            if self.activated_switch_bits > max_mask:
                raise ValueError("activated_switch_bits: exceeds declared switch positions")
        if self.door_state_bits is not None and self.door_open is not None:
            expected_open = self.door_state_bits > 0
            if self.door_open != expected_open:
                raise ValueError("door_state_bits: inconsistent with door_open")
        _validate_pos_tuple("teleporter_endpoint_positions", self.teleporter_endpoint_positions)
        if not isinstance(self.teleporter_pairs, tuple):
            raise ValueError("teleporter_pairs: must be a tuple")
        for index, pair in enumerate(self.teleporter_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"teleporter_pairs[{index}]: must contain exactly two positions")
            _validate_pos(f"teleporter_pairs[{index}][0]", pair[0])
            _validate_pos(f"teleporter_pairs[{index}][1]", pair[1])
        if not isinstance(self.teleporter_pair_map, tuple):
            raise ValueError("teleporter_pair_map: must be a tuple")
        for index, pair in enumerate(self.teleporter_pair_map):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"teleporter_pair_map[{index}]: must contain exactly one directional mapping")
            _validate_pos(f"teleporter_pair_map[{index}][0]", pair[0])
            _validate_pos(f"teleporter_pair_map[{index}][1]", pair[1])
        if self.slide_mode is not None and not isinstance(self.slide_mode, str):
            raise ValueError("slide_mode: must be a string or null")
        _validate_pos_tuple("ice_cell_positions", self.ice_cell_positions)
        _validate_pos_tuple("coverage_eligible_cells", self.coverage_eligible_cells)
        _validate_pos_tuple("coverage_mask", self.coverage_mask)
        _validate_pos_tuple("pushable_block_positions", self.pushable_block_positions)
        _validate_pos_tuple("push_target_cells", self.push_target_cells)
        if self.step_limit is not None and (not isinstance(self.step_limit, int) or self.step_limit <= 0):
            raise ValueError("step_limit: must be a positive int or null")


@dataclass(frozen=True)
class MovementTypedStateV4:
    common: MovementCommonFieldsV4
    family: MovementFamilyFieldsV4 = field(default_factory=MovementFamilyFieldsV4)
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
    def from_dict(cls, payload: dict[str, Any]) -> MovementTypedStateV4:
        common = MovementCommonFieldsV4(**payload["common"])
        family = MovementFamilyFieldsV4(**payload.get("family", {}))
        return cls(common=common, family=family, layout_evidence_source=payload.get("layout_evidence_source"))

    def to_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
