from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from v4.state.parsedState import ParsedStateV4

from .typedState import ClickCommonFieldsV4, ClickFamilyFieldsV4, ClickTypedStateV4, GridPos


ROTATION_BY_TAG = {"rot_0": 0, "rot_90": 90, "rot_180": 180, "rot_270": 270}


def _load_game_module(parsed_state: ParsedStateV4, family: str):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError(f"{family} config unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / f"{family}.py"
    spec = importlib.util.spec_from_file_location(f"v4_click_{family}", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{family} config unavailable: failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_level(parsed_state: ParsedStateV4, family: str):
    module = _load_game_module(parsed_state, family)
    levels = getattr(module, "levels", None)
    level_index = parsed_state.current_observation.levels_completed
    if isinstance(levels, list) and 0 <= level_index < len(levels):
        return module, levels[level_index]
    if family == "mm01" and hasattr(module, "create_level") and hasattr(module, "LEVEL_LAYOUTS"):
        layouts = getattr(module, "LEVEL_LAYOUTS")
        if isinstance(layouts, list) and 0 <= level_index < len(layouts):
            return module, module.create_level(level_index)
    raise ValueError(f"{family} config unavailable: invalid level index {level_index}")


def _sample_grid(parsed_state: ParsedStateV4, bounds: GridPos) -> tuple[tuple[int, ...], ...]:
    plane = parsed_state.current_observation.frame[0]
    pixel_h = len(plane)
    pixel_w = len(plane[0]) if pixel_h else 0
    if pixel_h <= 0 or pixel_w <= 0:
        raise ValueError("frame: cannot build click visual grid from empty frame")
    rows = []
    for y in range(bounds[1]):
        row = []
        for x in range(bounds[0]):
            px, py = _grid_to_payload(bounds, x, y)
            px = min(pixel_w - 1, max(0, px))
            py = min(pixel_h - 1, max(0, py))
            row.append(int(plane[py][px]))
        rows.append(tuple(row))
    return tuple(rows)


def _cell_patch_contains_color(parsed_state: ParsedStateV4, bounds: GridPos, gx: int, gy: int, color: int) -> bool:
    plane = parsed_state.current_observation.frame[0]
    pixel_h = len(plane)
    pixel_w = len(plane[0]) if pixel_h else 0
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    start_x = max(0, x_pad + gx * scale)
    end_x = min(pixel_w, start_x + max(scale, 1))
    start_y = max(0, y_pad + gy * scale)
    end_y = min(pixel_h, start_y + max(scale, 1))
    for py in range(start_y, end_y):
        for px in range(start_x, end_x):
            if int(plane[py][px]) == color:
                return True
    return False


def _grid_to_payload(bounds: GridPos, gx: int, gy: int) -> GridPos:
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    if scale <= 0:
        raise ValueError("bounds: cannot map click payload from non-positive scale")
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return gx * scale + scale // 2 + x_pad, gy * scale + scale // 2 + y_pad


def _build_common(parsed_state: ParsedStateV4, family: str, bounds: GridPos, clickable_cells: tuple[GridPos, ...]) -> ClickCommonFieldsV4:
    return ClickCommonFieldsV4(
        game_family=family,
        game_id=parsed_state.current_observation.game_id,
        level_index=parsed_state.current_observation.levels_completed,
        static_bounds=bounds,
        clickable_cells=tuple(sorted(clickable_cells)),
        legal_action_ids=tuple(int(action_id) for action_id in parsed_state.available_actions),
        terminal_status=parsed_state.terminal_signal.status,
        step_depth=parsed_state.step_index,
        visual_grid=_sample_grid(parsed_state, bounds),
    )


def build_pt01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    module, level = _load_level(parsed_state, "pt01")
    bounds = tuple(level.grid_size)
    if bounds != (64, 64):
        raise ValueError("pt01 config unavailable: expected a 64x64 click grid")
    visual_grid = _sample_grid(parsed_state, bounds)
    previous_visual_grid = None
    if parsed_state.previous_observation is not None:
        previous_plane = parsed_state.previous_observation.frame[0]
        previous_visual_grid = tuple(tuple(int(value) for value in row) for row in previous_plane)
    last_action = parsed_state.current_observation.action_input
    last_click = None
    if isinstance(last_action, dict) and int(last_action.get("id", -1)) == 6:
        data = last_action.get("data") or {}
        if isinstance(data, dict) and "x" in data and "y" in data:
            last_click = (int(data["x"]), int(data["y"]))
    rotation_tiles = []
    clickable_cells = []
    sprites = getattr(level, "_sprites", None) or getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        raise ValueError("pt01 config unavailable: level sprites unavailable")
    for sprite in sprites:
        tags = tuple(getattr(sprite, "tags", ()))
        if "rotatable" not in tags:
            continue
        sprite_type = next((tag[5:] for tag in tags if tag.startswith("type_")), None)
        if sprite_type is None:
            raise ValueError("pt01 config unavailable: rotatable missing explicit type tag")
        x = int(sprite.x)
        y = int(sprite.y)
        patch = tuple(tuple(visual_grid[y + dy][x + dx] for dx in range(3)) for dy in range(3))
        matched_rotation = None
        for candidate_rotation in (0, 90, 180, 270):
            candidate = module.create_rotatable(sprite_type, candidate_rotation, 0, 0)
            candidate_pixels = tuple(tuple(int(value) for value in row) for row in candidate.pixels)
            if candidate_pixels == patch:
                matched_rotation = candidate_rotation
                break
        if matched_rotation is None and previous_visual_grid is not None:
            previous_patch = tuple(tuple(previous_visual_grid[y + dy][x + dx] for dx in range(3)) for dy in range(3))
            previous_rotation = None
            for candidate_rotation in (0, 90, 180, 270):
                candidate = module.create_rotatable(sprite_type, candidate_rotation, 0, 0)
                candidate_pixels = tuple(tuple(int(value) for value in row) for row in candidate.pixels)
                if candidate_pixels == previous_patch:
                    previous_rotation = candidate_rotation
                    break
            if previous_rotation is not None and last_click is not None and x <= last_click[0] < x + 3 and y <= last_click[1] < y + 3:
                matched_rotation = (previous_rotation + 90) % 360
            elif previous_rotation is not None:
                matched_rotation = previous_rotation
        if matched_rotation is None:
            raise ValueError(f"pt01 state unavailable: could not match observed rotation for tile {sprite_type} at {(x, y)}")
        rotation_tiles.append(((x, y), sprite_type, matched_rotation))
        clickable_cells.append(_grid_to_payload(bounds, x + 1, y + 1))
    if not rotation_tiles:
        raise ValueError("pt01 state unavailable: no rotatable tiles found")
    target_pattern = level.get_data("target_pattern")
    rotations_by_color = target_pattern["rotations_by_color"]
    target_by_type = []
    for color, rotation in sorted(rotations_by_color.items()):
        sprite_type = module.COLOR_TO_TYPE.get(int(color))
        if sprite_type is None:
            raise ValueError(f"pt01 config unavailable: unmapped target color {color}")
        target_by_type.append((sprite_type, int(rotation)))
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "pt01", bounds, tuple(clickable_cells)),
        family=ClickFamilyFieldsV4(
            rotation_tiles=tuple(rotation_tiles),
            target_rotations_by_type=tuple(target_by_type),
        ),
    )


def build_sy01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    module, level = _load_level(parsed_state, "sy01")
    bounds = tuple(level.grid_size)
    sprites = getattr(level, "_sprites", None) or getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        raise ValueError("sy01 config unavailable: level sprites unavailable")
    divider_positions = {(int(sprite.x), int(sprite.y)) for sprite in sprites if "divider" in getattr(sprite, "tags", ())}
    if not divider_positions:
        raise ValueError("sy01 state unavailable: divider axis not found")
    divider_columns = {x for x, _ in divider_positions}
    if len(divider_columns) != 1:
        raise ValueError("sy01 state unavailable: divider axis is not a single vertical column")
    reflection_axis_x = divider_columns.pop()
    visual_grid = _sample_grid(parsed_state, bounds)
    pattern_positions_raw = level.get_data("pattern_positions")
    if not isinstance(pattern_positions_raw, list):
        raise ValueError("sy01 config unavailable: pattern_positions missing")
    pattern_positions = tuple(sorted((int(pos[0]), int(pos[1])) for pos in pattern_positions_raw))
    reflection_pairs = tuple(
        sorted(
            (
                (x, y),
                (reflection_axis_x + (reflection_axis_x - x), y),
            )
            for x, y in pattern_positions
        )
    )
    target_cells = tuple(sorted(target for _, target in reflection_pairs))
    player_color = int(getattr(module, "PLAYER_COLOR"))
    placed_cells = tuple(
        sorted(
            (x, y)
            for y in range(bounds[1])
            for x in range(reflection_axis_x + 1, bounds[0])
            if _cell_patch_contains_color(parsed_state, bounds, x, y, player_color)
        )
    )
    clickable = tuple(
        _grid_to_payload(bounds, x, y)
        for y in range(bounds[1])
        for x in range(reflection_axis_x + 1, bounds[0])
    )
    if not reflection_pairs or not target_cells:
        raise ValueError("sy01 state unavailable: reflection mapping missing")
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "sy01", bounds, clickable),
        family=ClickFamilyFieldsV4(
            reflection_axis_x=reflection_axis_x,
            reflection_pairs=reflection_pairs,
            mirror_source_cells=pattern_positions,
            mirror_target_cells=target_cells,
            placed_mirror_cells=placed_cells,
        ),
    )


def build_ff01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    module, level = _load_level(parsed_state, "ff01")
    bounds = tuple(level.grid_size)
    shapes = module.get_level_shapes(parsed_state.current_observation.levels_completed + 1)
    regions = tuple(tuple(sorted((int(x), int(y)) for x, y in shape.interior)) for shape in shapes)
    visual_grid = _sample_grid(parsed_state, bounds)
    filled_indexes = []
    clickable = []
    for index, region in enumerate(regions):
        center = region[len(region) // 2]
        clickable.append(_grid_to_payload(bounds, center[0], center[1]))
        if any(visual_grid[y][x] == module.FILL_COLOR for x, y in region):
            filled_indexes.append(index)
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "ff01", bounds, tuple(clickable)),
        family=ClickFamilyFieldsV4(
            fill_regions=regions,
            filled_region_indexes=tuple(filled_indexes),
        ),
    )


def build_sq01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    _, level = _load_level(parsed_state, "sq01")
    bounds = tuple(level.grid_size)
    sequence_order = tuple(str(value) for value in level.get_data("sequence"))
    block_positions = level.get_data("block_positions")
    visual_grid = _sample_grid(parsed_state, bounds)
    clickable_cells = []
    clickable_color_cells = []
    remaining = {}
    for color_name, pos in sorted(block_positions.items()):
        x, y = int(pos[0]), int(pos[1])
        block_present = any(
            0 <= y + dy < bounds[1] and 0 <= x + dx < bounds[0] and visual_grid[y + dy][x + dx] == visual_grid[y][x]
            for dx in (0, 1)
            for dy in (0, 1)
        ) and visual_grid[y][x] != 5
        if block_present:
            remaining[color_name] = (x, y)
            # Sq01 blocks are 2x2 sprites; click the center cell, not the top-left corner.
            payload = _grid_to_payload(bounds, x + 1, y + 1)
            clickable_color_cells.append((color_name, payload))
            clickable_cells.append(payload)
    progress = 0
    for color_name in sequence_order:
        if color_name not in remaining:
            progress += 1
        else:
            break
    if not clickable_cells and progress >= len(sequence_order):
        # After the last correct click, the live env spends a few frames in a
        # pending-advance state with no remaining blocks. ACTION6 is still the
        # legal action surface, and any click advances those frames.
        clickable_cells.append((0, 0))
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "sq01", bounds, tuple(clickable_cells)),
        family=ClickFamilyFieldsV4(
            sequence_order=sequence_order,
            sequence_progress=progress,
            clickable_color_cells=tuple(clickable_color_cells),
        ),
    )


def _detect_active_moles(visual_grid: tuple[tuple[int, ...], ...], hole_positions: tuple[GridPos, ...]) -> tuple[GridPos, ...]:
    active = []
    for hx, hy in hole_positions:
        if any(
            0 <= hy + dy < len(visual_grid)
            and 0 <= hx + dx < len(visual_grid[0])
            and visual_grid[hy + dy][hx + dx] in {8, 12}
            for dx in range(5)
            for dy in range(5)
        ):
            active.append((hx, hy))
    return tuple(sorted(active))


def build_wm01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    _, level = _load_level(parsed_state, "wm01")
    bounds = tuple(level.grid_size)
    sprites = getattr(level, "_sprites", None) or getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        raise ValueError("wm01 config unavailable: level sprites unavailable")
    hole_positions = tuple(sorted((int(sprite.x), int(sprite.y)) for sprite in sprites if "hole" in getattr(sprite, "tags", ())))
    visual_grid = _sample_grid(parsed_state, bounds)
    active_moles = _detect_active_moles(visual_grid, hole_positions)
    clickable = tuple(_grid_to_payload(bounds, x + 2, y + 2) for x, y in (active_moles or hole_positions))
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "wm01", bounds, clickable),
        family=ClickFamilyFieldsV4(
            active_mole_cells=active_moles,
            mole_click_radius=2,
        ),
    )


def build_mm01_click_state(parsed_state: ParsedStateV4) -> ClickTypedStateV4:
    _, level = _load_level(parsed_state, "mm01")
    bounds = tuple(level.grid_size)
    slot_colors = tuple(int(value) for value in level.get_data("slot_colors"))
    rows = int(level.get_data("rows"))
    cols = int(level.get_data("cols"))
    tile_size = int(level.get_data("tile_size"))
    offset_x = int(level.get_data("offset_x"))
    offset_y = int(level.get_data("offset_y"))
    visual_grid = _sample_grid(parsed_state, bounds)
    hidden = []
    revealed = []
    clickable = []
    for slot_index, color in enumerate(slot_colors):
        row = slot_index // cols
        col = slot_index % cols
        gx = offset_x + col * tile_size
        gy = offset_y + row * tile_size
        cx = gx + tile_size // 2
        cy = gy + tile_size // 2
        sample = visual_grid[cy][cx]
        clickable.append(_grid_to_payload(bounds, cx, cy))
        if sample == 3:
            hidden.append((slot_index, color))
        else:
            revealed.append((slot_index, color))
    color_counts: dict[int, int] = {}
    for _, color in revealed:
        color_counts[color] = color_counts.get(color, 0) + 1
    matched = tuple(sorted(slot_index for slot_index, color in revealed if color_counts.get(color, 0) >= 2))
    return ClickTypedStateV4(
        common=_build_common(parsed_state, "mm01", bounds, tuple(clickable)),
        family=ClickFamilyFieldsV4(
            memory_slot_colors=slot_colors,
            hidden_slots=tuple(hidden),
            revealed_slots=tuple(revealed),
            matched_slots=matched,
            slot_geometry=(rows, cols, tile_size, offset_x),
        ),
    )


def build_click_state_for_family(parsed_state: ParsedStateV4, family: str) -> ClickTypedStateV4:
    builders = {
        "pt01": build_pt01_click_state,
        "sy01": build_sy01_click_state,
        "ff01": build_ff01_click_state,
        "sq01": build_sq01_click_state,
        "wm01": build_wm01_click_state,
        "mm01": build_mm01_click_state,
    }
    if family not in builders:
        raise ValueError(f"unsupported click family: {family}")
    return builders[family](parsed_state)
