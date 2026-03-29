from __future__ import annotations

from dataclasses import replace

from v4.agentContract.types import V4Action

from .typedState import ClickCommonFieldsV4, ClickFamilyFieldsV4, ClickTypedStateV4, GridPos


def _payload_to_grid(action: V4Action, bounds: GridPos) -> GridPos:
    if action.payload is None or "x" not in action.payload or "y" not in action.payload:
        raise ValueError("click action requires x/y payload")
    px = int(action.payload["x"])
    py = int(action.payload["y"])
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    gx = max(0, min(width - 1, (px - x_pad) // max(scale, 1)))
    gy = max(0, min(height - 1, (py - y_pad) // max(scale, 1)))
    return int(gx), int(gy)


def _grid_to_payload(bounds: GridPos, gx: int, gy: int) -> GridPos:
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return gx * scale + scale // 2 + x_pad, gy * scale + scale // 2 + y_pad


def _replace_visual_cell(grid: tuple[tuple[int, ...], ...], pos: GridPos, value: int) -> tuple[tuple[int, ...], ...]:
    rows = [list(row) for row in grid]
    rows[pos[1]][pos[0]] = int(value)
    return tuple(tuple(row) for row in rows)


class ClickTransitionModelV4:
    def apply(self, state: ClickTypedStateV4, action: V4Action) -> tuple[ClickTypedStateV4, dict[str, object]]:
        if action.action_id != 6:
            raise ValueError("click transition model only supports ACTION6")
        clicked_cell = _payload_to_grid(action, state.common.static_bounds)
        family = state.common.game_family
        if family == "pt01":
            return self._apply_pt01(state, clicked_cell)
        if family == "sy01":
            return self._apply_sy01(state, clicked_cell)
        if family == "ff01":
            return self._apply_ff01(state, clicked_cell)
        if family == "sq01":
            return self._apply_sq01(state, clicked_cell)
        if family == "wm01":
            return self._apply_wm01(state, clicked_cell)
        if family == "mm01":
            return self._apply_mm01(state, clicked_cell)
        raise ValueError(f"unsupported click family: {family}")

    def _apply_pt01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        tiles = list(state.family.rotation_tiles)
        changed = None
        for index, (pos, sprite_type, rotation) in enumerate(tiles):
            if pos[0] <= clicked_cell[0] < pos[0] + 3 and pos[1] <= clicked_cell[1] < pos[1] + 3:
                tiles[index] = (pos, sprite_type, (rotation + 90) % 360)
                changed = pos
                break
        target_by_type = dict(state.family.target_rotations_by_type)
        success = bool(tiles) and all(target_by_type.get(sprite_type) == rotation for _, sprite_type, rotation in tiles)
        next_state = ClickTypedStateV4(
            common=replace(
                state.common,
                terminal_status="success" if success else state.common.terminal_status,
                step_depth=state.common.step_depth + 1,
            ),
            family=replace(state.family, rotation_tiles=tuple(tiles)),
        )
        return next_state, {"clicked_cell": clicked_cell, "changed_tile": changed}

    def _apply_sy01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        updated_family, valid_click, toggled_cell = apply_sy01_reflection_update(state, clicked_cell)
        success = updated_family.placed_mirror_cells == updated_family.mirror_target_cells
        return (
            ClickTypedStateV4(
                common=replace(
                    state.common,
                    terminal_status="success" if success else state.common.terminal_status,
                    step_depth=state.common.step_depth + 1,
                ),
                family=updated_family,
            ),
            {"clicked_cell": clicked_cell, "valid_click": valid_click, "toggled_cell": toggled_cell},
        )

    def _apply_ff01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        filled = set(state.family.filled_region_indexes)
        for index, region in enumerate(state.family.fill_regions):
            if clicked_cell in region:
                filled.add(index)
                break
        success = len(filled) == len(state.family.fill_regions) and bool(state.family.fill_regions)
        return (
            ClickTypedStateV4(
                common=replace(state.common, terminal_status="success" if success else state.common.terminal_status, step_depth=state.common.step_depth + 1),
                family=replace(state.family, filled_region_indexes=tuple(sorted(filled))),
            ),
            {"clicked_cell": clicked_cell, "filled_regions": tuple(sorted(filled))},
        )

    def _apply_sq01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        progress = int(state.family.sequence_progress or 0)
        entries = list(state.family.clickable_color_cells)
        clicked_color = None
        clicked_index = None
        for index, (color_name, payload_cell) in enumerate(entries):
            grid_cell = _payload_to_grid(V4Action(action_id=6, action_name="ACTION6", payload={"x": payload_cell[0], "y": payload_cell[1]}), state.common.static_bounds)
            if grid_cell == clicked_cell:
                clicked_color = color_name
                clicked_index = index
                break
        if clicked_color is None:
            return (
                ClickTypedStateV4(common=replace(state.common, step_depth=state.common.step_depth + 1), family=replace(state.family, sequence_progress=0)),
                {"clicked_cell": clicked_cell, "advance": False},
            )
        expected = state.family.sequence_order[progress]
        if clicked_color != expected:
            return (
                ClickTypedStateV4(common=replace(state.common, step_depth=state.common.step_depth + 1), family=replace(state.family, sequence_progress=0)),
                {"clicked_cell": clicked_cell, "advance": False},
            )
        assert clicked_index is not None
        del entries[clicked_index]
        next_progress = progress + 1
        success = next_progress >= len(state.family.sequence_order)
        return (
            ClickTypedStateV4(
                common=replace(state.common, terminal_status="success" if success else state.common.terminal_status, step_depth=state.common.step_depth + 1),
                family=replace(state.family, sequence_progress=next_progress, clickable_color_cells=tuple(entries)),
            ),
            {"clicked_cell": clicked_cell, "advance": True},
        )

    def _apply_wm01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        radius = int(state.family.mole_click_radius or 0)
        hit = any(abs(clicked_cell[0] - mx) <= radius and abs(clicked_cell[1] - my) <= radius for mx, my in state.family.active_mole_cells)
        next_active = () if hit else state.family.active_mole_cells
        return (
            ClickTypedStateV4(common=replace(state.common, step_depth=state.common.step_depth + 1), family=replace(state.family, active_mole_cells=next_active)),
            {"clicked_cell": clicked_cell, "hit": hit},
        )

    def _apply_mm01(self, state: ClickTypedStateV4, clicked_cell: GridPos) -> tuple[ClickTypedStateV4, dict[str, object]]:
        if state.family.slot_geometry is None:
            raise ValueError("mm01 typed state missing slot geometry")
        rows, cols, tile_size, offset_x = state.family.slot_geometry
        del rows
        offset_y = int((64 - ((len(state.family.memory_slot_colors) // cols + (1 if len(state.family.memory_slot_colors) % cols else 0)) * tile_size)) // 2)
        clicked_slot = None
        for slot_index, color in state.family.hidden_slots:
            row = slot_index // cols
            col = slot_index % cols
            gx = offset_x + col * tile_size
            gy = offset_y + row * tile_size
            if gx <= clicked_cell[0] < gx + tile_size and gy <= clicked_cell[1] < gy + tile_size:
                clicked_slot = (slot_index, color)
                break
        if clicked_slot is None:
            return ClickTypedStateV4(common=replace(state.common, step_depth=state.common.step_depth + 1), family=state.family), {"clicked_cell": clicked_cell, "revealed": None}
        hidden = list(state.family.hidden_slots)
        hidden.remove(clicked_slot)
        revealed = list(state.family.revealed_slots) + [clicked_slot]
        matched = set(state.family.matched_slots)
        if len(revealed) >= 2 and revealed[-1][1] == revealed[-2][1]:
            matched.update([revealed[-1][0], revealed[-2][0]])
        success = len(matched) == len(state.family.memory_slot_colors)
        return (
            ClickTypedStateV4(
                common=replace(state.common, terminal_status="success" if success else state.common.terminal_status, step_depth=state.common.step_depth + 1),
                family=replace(state.family, hidden_slots=tuple(hidden), revealed_slots=tuple(revealed[-2:]), matched_slots=tuple(sorted(matched))),
            ),
            {"clicked_cell": clicked_cell, "revealed": clicked_slot[0]},
        )


def apply_sy01_reflection_update(
    state: ClickTypedStateV4,
    clicked_cell: GridPos,
) -> tuple[ClickFamilyFieldsV4, bool, GridPos | None]:
    if state.family.reflection_axis_x is None:
        raise ValueError("sy01 transition requires explicit reflection_axis_x")
    if not state.family.reflection_pairs:
        raise ValueError("sy01 transition requires explicit reflection_pairs")
    axis_x = int(state.family.reflection_axis_x)
    width, _ = state.common.static_bounds
    if not (axis_x < clicked_cell[0] < width):
        return state.family, False, None
    placed = set(state.family.placed_mirror_cells)
    if clicked_cell in placed:
        placed.remove(clicked_cell)
    else:
        placed.add(clicked_cell)
    return replace(state.family, placed_mirror_cells=tuple(sorted(placed))), True, clicked_cell
