from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GridPos = tuple[int, int]
TypedTile = tuple[GridPos, str, int]
PairMapping = tuple[GridPos, GridPos]
RegionCells = tuple[GridPos, ...]
ColoredCell = tuple[str, GridPos]
SlotState = tuple[int, int]


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


def _validate_region_tuple(field_name: str, values: tuple[RegionCells, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name}: must be a tuple")
    for index, region in enumerate(values):
        _validate_pos_tuple(f"{field_name}[{index}]", region)


@dataclass(frozen=True)
class ClickCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    static_bounds: GridPos
    clickable_cells: tuple[GridPos, ...]
    legal_action_ids: tuple[int, ...]
    terminal_status: str
    step_depth: int
    visual_grid: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.game_family, str) or not self.game_family:
            raise ValueError("game_family: must be a non-empty string")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id: must be a non-empty string")
        if not isinstance(self.level_index, int) or self.level_index < 0:
            raise ValueError("level_index: must be a non-negative int")
        _validate_pos("static_bounds", self.static_bounds)
        if self.static_bounds[0] <= 0 or self.static_bounds[1] <= 0:
            raise ValueError("static_bounds: must be positive")
        _validate_pos_tuple("clickable_cells", self.clickable_cells)
        if not isinstance(self.legal_action_ids, tuple):
            raise ValueError("legal_action_ids: must be a tuple")
        for index, action_id in enumerate(self.legal_action_ids):
            if not isinstance(action_id, int):
                raise ValueError(f"legal_action_ids[{index}]: must be an int")
        if self.terminal_status not in {"not_played", "non_terminal", "success", "failure"}:
            raise ValueError("terminal_status: unsupported value")
        if not isinstance(self.step_depth, int) or self.step_depth < 0:
            raise ValueError("step_depth: must be a non-negative int")
        if not isinstance(self.visual_grid, tuple):
            raise ValueError("visual_grid: must be a tuple")
        for row_index, row in enumerate(self.visual_grid):
            if not isinstance(row, tuple):
                raise ValueError(f"visual_grid[{row_index}]: must be a tuple")
            for col_index, value in enumerate(row):
                if not isinstance(value, int):
                    raise ValueError(f"visual_grid[{row_index}][{col_index}]: must be an int")


@dataclass(frozen=True)
class ClickFamilyFieldsV4:
    rotation_tiles: tuple[TypedTile, ...] = ()
    target_rotations_by_type: tuple[tuple[str, int], ...] = ()
    reflection_axis_x: int | None = None
    reflection_pairs: tuple[PairMapping, ...] = ()
    mirror_source_cells: tuple[GridPos, ...] = ()
    mirror_target_cells: tuple[GridPos, ...] = ()
    placed_mirror_cells: tuple[GridPos, ...] = ()
    fill_regions: tuple[RegionCells, ...] = ()
    filled_region_indexes: tuple[int, ...] = ()
    sequence_order: tuple[str, ...] = ()
    sequence_progress: int | None = None
    clickable_color_cells: tuple[ColoredCell, ...] = ()
    active_mole_cells: tuple[GridPos, ...] = ()
    mole_click_radius: int | None = None
    memory_slot_colors: tuple[int, ...] = ()
    hidden_slots: tuple[SlotState, ...] = ()
    revealed_slots: tuple[SlotState, ...] = ()
    matched_slots: tuple[int, ...] = ()
    slot_geometry: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rotation_tiles, tuple):
            raise ValueError("rotation_tiles: must be a tuple")
        for index, tile in enumerate(self.rotation_tiles):
            if not isinstance(tile, tuple) or len(tile) != 3:
                raise ValueError(f"rotation_tiles[{index}]: must contain position, type, and rotation")
            _validate_pos(f"rotation_tiles[{index}][0]", tile[0])
            if not isinstance(tile[1], str) or not tile[1]:
                raise ValueError(f"rotation_tiles[{index}][1]: must be a non-empty string")
            if tile[2] not in {0, 90, 180, 270}:
                raise ValueError(f"rotation_tiles[{index}][2]: unsupported rotation")
        if not isinstance(self.target_rotations_by_type, tuple):
            raise ValueError("target_rotations_by_type: must be a tuple")
        for index, item in enumerate(self.target_rotations_by_type):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"target_rotations_by_type[{index}]: must contain type and rotation")
            if not isinstance(item[0], str) or not item[0]:
                raise ValueError(f"target_rotations_by_type[{index}][0]: must be a non-empty string")
            if item[1] not in {0, 90, 180, 270}:
                raise ValueError(f"target_rotations_by_type[{index}][1]: unsupported rotation")
        if self.reflection_axis_x is not None and not isinstance(self.reflection_axis_x, int):
            raise ValueError("reflection_axis_x: must be an int or null")
        if not isinstance(self.reflection_pairs, tuple):
            raise ValueError("reflection_pairs: must be a tuple")
        for index, pair in enumerate(self.reflection_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"reflection_pairs[{index}]: must contain source and mirrored position")
            _validate_pos(f"reflection_pairs[{index}][0]", pair[0])
            _validate_pos(f"reflection_pairs[{index}][1]", pair[1])
        _validate_pos_tuple("mirror_source_cells", self.mirror_source_cells)
        _validate_pos_tuple("mirror_target_cells", self.mirror_target_cells)
        _validate_pos_tuple("placed_mirror_cells", self.placed_mirror_cells)
        _validate_region_tuple("fill_regions", self.fill_regions)
        if not isinstance(self.filled_region_indexes, tuple):
            raise ValueError("filled_region_indexes: must be a tuple")
        for index, value in enumerate(self.filled_region_indexes):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"filled_region_indexes[{index}]: must be a non-negative int")
        if not isinstance(self.sequence_order, tuple):
            raise ValueError("sequence_order: must be a tuple")
        for index, color_name in enumerate(self.sequence_order):
            if not isinstance(color_name, str) or not color_name:
                raise ValueError(f"sequence_order[{index}]: must be a non-empty string")
        if self.sequence_progress is not None and (not isinstance(self.sequence_progress, int) or self.sequence_progress < 0):
            raise ValueError("sequence_progress: must be a non-negative int or null")
        if not isinstance(self.clickable_color_cells, tuple):
            raise ValueError("clickable_color_cells: must be a tuple")
        for index, item in enumerate(self.clickable_color_cells):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"clickable_color_cells[{index}]: must contain color and position")
            if not isinstance(item[0], str) or not item[0]:
                raise ValueError(f"clickable_color_cells[{index}][0]: must be a non-empty string")
            _validate_pos(f"clickable_color_cells[{index}][1]", item[1])
        _validate_pos_tuple("active_mole_cells", self.active_mole_cells)
        if self.mole_click_radius is not None and (not isinstance(self.mole_click_radius, int) or self.mole_click_radius < 0):
            raise ValueError("mole_click_radius: must be a non-negative int or null")
        if not isinstance(self.memory_slot_colors, tuple):
            raise ValueError("memory_slot_colors: must be a tuple")
        for index, value in enumerate(self.memory_slot_colors):
            if not isinstance(value, int):
                raise ValueError(f"memory_slot_colors[{index}]: must be an int")
        if not isinstance(self.hidden_slots, tuple):
            raise ValueError("hidden_slots: must be a tuple")
        for index, item in enumerate(self.hidden_slots):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"hidden_slots[{index}]: must contain slot index and color")
            if not isinstance(item[0], int) or item[0] < 0:
                raise ValueError(f"hidden_slots[{index}][0]: must be a non-negative int")
            if not isinstance(item[1], int):
                raise ValueError(f"hidden_slots[{index}][1]: must be an int")
        if not isinstance(self.revealed_slots, tuple):
            raise ValueError("revealed_slots: must be a tuple")
        for index, item in enumerate(self.revealed_slots):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"revealed_slots[{index}]: must contain slot index and color")
            if not isinstance(item[0], int) or item[0] < 0:
                raise ValueError(f"revealed_slots[{index}][0]: must be a non-negative int")
            if not isinstance(item[1], int):
                raise ValueError(f"revealed_slots[{index}][1]: must be an int")
        if not isinstance(self.matched_slots, tuple):
            raise ValueError("matched_slots: must be a tuple")
        for index, value in enumerate(self.matched_slots):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"matched_slots[{index}]: must be a non-negative int")
        if self.slot_geometry is not None:
            if not isinstance(self.slot_geometry, tuple) or len(self.slot_geometry) != 4:
                raise ValueError("slot_geometry: must be a 4-tuple or null")
            for index, value in enumerate(self.slot_geometry):
                if not isinstance(value, int):
                    raise ValueError(f"slot_geometry[{index}]: must be an int")


@dataclass(frozen=True)
class ClickTypedStateV4:
    common: ClickCommonFieldsV4
    family: ClickFamilyFieldsV4 = field(default_factory=ClickFamilyFieldsV4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClickTypedStateV4:
        return cls(
            common=ClickCommonFieldsV4(**payload["common"]),
            family=ClickFamilyFieldsV4(**payload.get("family", {})),
        )

    def to_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
