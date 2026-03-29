from __future__ import annotations

from collections import Counter, deque
import importlib.util
from pathlib import Path

from v4.agentContract.types import V4Observation
from v4.state.parsedState import ParsedStateV4

from .typedState import MovementCommonFieldsV4, MovementFamilyFieldsV4, MovementTypedStateV4


GridPos = tuple[int, int]

_MOVE_DELTAS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


def _find_cells_by_color(grid: tuple[tuple[int, ...], ...], color: int) -> tuple[GridPos, ...]:
    return tuple((x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value == color)


def _most_common_non_avatar_blob_size(observation: V4Observation) -> GridPos:
    plane = observation.frame[0]
    height = len(plane)
    width = len(plane[0]) if height else 0
    if height <= 0 or width <= 0:
        raise ValueError("frame: cannot infer grid bounds from empty observation")
    visited: set[GridPos] = set()
    blobs: list[GridPos] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in visited or int(plane[y][x]) != 9:
                continue
            queue = deque([(x, y)])
            visited.add((x, y))
            cells: list[GridPos] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for dx, dy in _MOVE_DELTAS.values():
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and int(plane[ny][nx]) == 9:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            blobs.append((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    if not blobs:
        raise ValueError("frame: cannot infer grid bounds without directly observed avatar pixels")
    tile_w, tile_h = Counter(blobs).most_common(1)[0][0]
    if width % tile_w != 0 or height % tile_h != 0:
        raise ValueError("frame: inferred tile size does not divide frame dimensions")
    return tile_w, tile_h


def _infer_bounds(parsed_state: ParsedStateV4) -> GridPos:
    observation = parsed_state.current_observation
    metadata = parsed_state.environment_metadata
    if metadata is not None and metadata.coordinate_bounds is not None:
        xmin, ymin, xmax, ymax = metadata.coordinate_bounds
        if xmin == 0 and ymin == 0 and xmax >= 0 and ymax >= 0:
            return xmax + 1, ymax + 1
    try:
        tile_w, tile_h = _most_common_non_avatar_blob_size(observation)
    except ValueError:
        if parsed_state.previous_observation is not None:
            try:
                tile_w, tile_h = _most_common_non_avatar_blob_size(parsed_state.previous_observation)
            except ValueError:
                grid_size = _load_grid_size_from_environment_metadata(parsed_state)
                if grid_size is None:
                    raise
                return grid_size
        else:
            grid_size = _load_grid_size_from_environment_metadata(parsed_state)
            if grid_size is None:
                raise
            return grid_size
    plane = observation.frame[0]
    return len(plane[0]) // tile_w, len(plane) // tile_h


def _sample_grid_from_observation(observation: V4Observation, bounds: GridPos) -> tuple[tuple[int, ...], ...]:
    plane = observation.frame[0]
    pixel_w = len(plane[0])
    pixel_h = len(plane)
    if pixel_w <= 0 or pixel_h <= 0:
        raise ValueError("observation frame cannot be sampled into camera cells")
    grid_rows = []
    for y in range(bounds[1]):
        row = []
        for x in range(bounds[0]):
            px = min(pixel_w - 1, int((x + 0.5) * pixel_w / bounds[0]))
            py = min(pixel_h - 1, int((y + 0.5) * pixel_h / bounds[1]))
            row.append(int(plane[py][px]))
        grid_rows.append(tuple(row))
    return tuple(grid_rows)


def _sample_previous_grid(parsed_state: ParsedStateV4, bounds: GridPos) -> tuple[tuple[int, ...], ...] | None:
    if parsed_state.previous_observation is None:
        return None
    return _sample_grid_from_observation(parsed_state.previous_observation, bounds)


def _find_unique_color(grid: tuple[tuple[int, ...], ...], color: int, field_name: str) -> GridPos:
    found = _find_cells_by_color(grid, color)
    if len(found) != 1:
        raise ValueError(f"{field_name}: expected exactly one cell with color {color}, found {len(found)}")
    return found[0]


def _last_action_id(parsed_state: ParsedStateV4) -> int | None:
    action_id = parsed_state.current_observation.action_input.get("id")
    return int(action_id) if isinstance(action_id, int) else None


def _neighbors(pos: GridPos, bounds: GridPos) -> tuple[GridPos, ...]:
    result: list[GridPos] = []
    for dx, dy in _MOVE_DELTAS.values():
        nxt = (pos[0] + dx, pos[1] + dy)
        if 0 <= nxt[0] < bounds[0] and 0 <= nxt[1] < bounds[1]:
            result.append(nxt)
    return tuple(result)


def _reachable(start: GridPos, traversable: set[GridPos], goal: GridPos) -> bool:
    if start not in traversable or goal not in traversable:
        return False
    queue = deque([start])
    seen = {start}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            return True
        for nxt in _neighbors(pos, (max(x for x, _ in traversable) + 1, max(y for _, y in traversable) + 1)):
            if nxt in traversable and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _base_floor_cells(bounds: GridPos) -> tuple[GridPos, ...]:
    return tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]))


def _avatar_position(parsed_state: ParsedStateV4, bounds: GridPos, family_layout: dict) -> GridPos:
    grid = family_layout["grid"]
    found = _find_cells_by_color(grid, 9)
    if len(found) == 1:
        return found[0]
    previous_observation = parsed_state.previous_observation
    if family_layout["family"] == "tp01" and len(found) == 0 and previous_observation is not None:
        previous_level_index = previous_observation.raw_payload.get("levels_completed")
        current_level_index = parsed_state.current_observation.levels_completed
        if (
            isinstance(previous_level_index, int)
            and current_level_index > previous_level_index
            and "player_start_position" in family_layout
        ):
            return family_layout["player_start_position"]
    if len(found) != 0 or previous_observation is None:
        raise ValueError(f"avatar_position: expected exactly one cell with color 9, found {len(found)}")
    previous_grid = _sample_grid_from_observation(previous_observation, bounds)
    previous_avatar = _find_unique_color(previous_grid, 9, "previous_avatar_position")
    action_id = _last_action_id(parsed_state)
    if action_id not in _MOVE_DELTAS:
        raise ValueError("avatar_position: cannot infer avatar without previous legal movement action")
    delta = _MOVE_DELTAS[action_id]
    candidate = (previous_avatar[0] + delta[0], previous_avatar[1] + delta[1])
    family = family_layout["family"]
    if family == "fs01":
        blocked = set(family_layout["wall_positions"]) | set(family_layout["door_positions"])
        return previous_avatar if candidate in blocked or not (0 <= candidate[0] < bounds[0] and 0 <= candidate[1] < bounds[1]) else candidate
    if family == "tp01":
        blocked = set(family_layout["wall_positions"])
        if candidate in blocked or not (0 <= candidate[0] < bounds[0] and 0 <= candidate[1] < bounds[1]):
            return previous_avatar
        portal_map = dict(family_layout["teleporter_map"])
        return portal_map.get(candidate, candidate)
    if family == "ic01":
        blocked = set(family_layout["wall_positions"]) | set(family_layout["hazard_positions"])
        cursor = previous_avatar
        while True:
            nxt = (cursor[0] + delta[0], cursor[1] + delta[1])
            if not (0 <= nxt[0] < bounds[0] and 0 <= nxt[1] < bounds[1]) or nxt in blocked:
                break
            cursor = nxt
        return cursor
    raise ValueError("avatar_position: cannot infer hidden avatar for this family")


def _choose_fs01_door_and_walls(blocked_cells: tuple[GridPos, ...], bounds: GridPos) -> tuple[tuple[GridPos, ...], tuple[GridPos, ...], bool]:
    blocked_set = set(blocked_cells)
    isolated = []
    for pos in blocked_cells:
        neighbors = [nxt for nxt in _neighbors(pos, bounds) if nxt in blocked_set]
        if not neighbors:
            isolated.append(pos)
    if len(isolated) == 1:
        door_positions = (isolated[0],)
        wall_positions = tuple(sorted(pos for pos in blocked_cells if pos not in door_positions))
        return door_positions, wall_positions, False
    if len(isolated) == 0:
        return (), tuple(sorted(blocked_cells)), True
    raise ValueError("fs01 door state unavailable: multiple isolated blocked cells are ambiguous")


def _load_level_data_from_environment_metadata(parsed_state: ParsedStateV4, family: str) -> dict:
    module = _load_game_module_from_environment_metadata(parsed_state, family)
    level = _load_level_object_from_module(parsed_state, family, module)
    portal_pairs = level.get_data("portal_pairs")
    if portal_pairs is None:
        raise ValueError(f"{family} config unavailable: level data missing portal_pairs")
    player_positions = _extract_sprite_positions(level, "player")
    if len(player_positions) != 1:
        raise ValueError(f"{family} config unavailable: expected exactly one player start, found {len(player_positions)}")
    return {
        "portal_pairs": portal_pairs,
        "player_start_position": player_positions[0],
        "target_positions": _extract_sprite_positions(level, "target"),
        "wall_positions": _extract_sprite_positions(level, "wall"),
    }


def _load_game_module_from_environment_metadata(parsed_state: ParsedStateV4, family: str):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError(f"{family} config unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / f"{family}.py"
    if not module_path.exists():
        raise ValueError(f"{family} config unavailable: missing local game module at {module_path}")
    spec = importlib.util.spec_from_file_location(f"v4_{family}_level_config", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{family} config unavailable: failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_level_object_from_module(parsed_state: ParsedStateV4, family: str, module):
    levels = getattr(module, "levels", None)
    if not isinstance(levels, list):
        raise ValueError(f"{family} config unavailable: module does not expose level list")
    level_index = parsed_state.current_observation.levels_completed
    if not isinstance(level_index, int) or not (0 <= level_index < len(levels)):
        raise ValueError(f"{family} config unavailable: invalid level index {level_index}")
    return levels[level_index]


def _load_grid_size_from_environment_metadata(parsed_state: ParsedStateV4) -> GridPos | None:
    family = parsed_state.current_observation.game_id.split("-", 1)[0]
    try:
        module = _load_game_module_from_environment_metadata(parsed_state, family)
        level = _load_level_object_from_module(parsed_state, family, module)
    except ValueError:
        return None
    if not hasattr(level, "get_data"):
        return None
    grid_size = getattr(level, "grid_size", None)
    if (
        not isinstance(grid_size, tuple)
        or len(grid_size) != 2
        or not all(isinstance(value, int) and value > 0 for value in grid_size)
    ):
        return None
    return grid_size


def _extract_sprite_positions(level, tag: str) -> tuple[GridPos, ...]:
    sprites = getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        sprites = getattr(level, "_sprites", None)
    if not isinstance(sprites, list):
        return ()
    positions: list[GridPos] = []
    for sprite in sprites:
        tags = getattr(sprite, "tags", ())
        if tag not in tags:
            continue
        x = getattr(sprite, "x", None)
        y = getattr(sprite, "y", None)
        if isinstance(x, int) and isinstance(y, int):
            positions.append((x, y))
    return tuple(positions)


def extract_family_layout_from_parsed_state(parsed_state: ParsedStateV4, family: str) -> dict:
    bounds = _infer_bounds(parsed_state)
    grid = _sample_grid_from_observation(parsed_state.current_observation, bounds)
    previous_grid = _sample_previous_grid(parsed_state, bounds)
    layout: dict[str, object] = {
        "family": family,
        "bounds": bounds,
        "grid": grid,
        "layout_evidence_source": "direct_observation",
    }
    if family == "ul01":
        bottom_right = (bounds[0] - 1, bounds[1] - 1)
        key_positions = tuple(sorted(_find_cells_by_color(grid, 11)))
        blocked = tuple(sorted(_find_cells_by_color(grid, 3)))
        if previous_grid is not None and grid[bottom_right[1]][bottom_right[0]] == 11:
            previous_bottom_right = previous_grid[bottom_right[1]][bottom_right[0]]
            previous_non_hud_keys = tuple(
                pos for pos in _find_cells_by_color(previous_grid, 11) if pos != bottom_right
            )
            if previous_bottom_right == 11 or previous_bottom_right in {3, 5} or previous_non_hud_keys:
                key_positions = tuple(pos for pos in key_positions if pos != bottom_right)
            if previous_bottom_right in {3, 11} and bottom_right not in blocked:
                blocked = tuple(sorted(blocked + (bottom_right,)))
        layout.update(
            key_positions=key_positions,
            door_positions=blocked,
            wall_positions=(),
            target_cells=_find_cells_by_color(grid, 14) or blocked,
        )
        return layout
    if family == "fs01":
        switch_positions = tuple(sorted(_find_cells_by_color(grid, 10) + _find_cells_by_color(grid, 11)))
        target_cells = _find_cells_by_color(grid, 14)
        if not switch_positions:
            raise ValueError("fs01 switch state unavailable: no directly observed switch cells")
        if len(target_cells) != 1:
            raise ValueError("fs01 target state unavailable: expected exactly one target cell")
        door_positions, wall_positions, door_open = _choose_fs01_door_and_walls(_find_cells_by_color(grid, 3), bounds)
        layout.update(
            switch_positions=switch_positions,
            target_cells=target_cells,
            door_positions=door_positions,
            wall_positions=wall_positions,
            door_open=door_open,
        )
        return layout
    if family == "tp01":
        level_data = _load_level_data_from_environment_metadata(parsed_state, family)
        teleporter_pairs = tuple(
            (tuple(int(value) for value in pair[0]), tuple(int(value) for value in pair[1]))
            for pair in level_data["portal_pairs"]
        )
        portal_positions = tuple(sorted(pos for pair in teleporter_pairs for pos in pair))
        observed_portals = set(_find_cells_by_color(grid, 7))
        previous_level_index = None
        if parsed_state.previous_observation is not None:
            raw_previous_level_index = parsed_state.previous_observation.raw_payload.get("levels_completed")
            if isinstance(raw_previous_level_index, int):
                previous_level_index = raw_previous_level_index
        level_advanced = previous_level_index is not None and parsed_state.current_observation.levels_completed > previous_level_index
        if not portal_positions:
            raise ValueError("tp01 teleporter state unavailable: no configured teleporter endpoints")
        if any(pos not in observed_portals for pos in portal_positions):
            if not level_advanced:
                raise ValueError("tp01 teleporter state unavailable: configured teleporter endpoints are not directly observed in the frame")
            wall_positions = tuple(sorted(level_data["wall_positions"]))
            target_cells = tuple(sorted(level_data["target_positions"]))
        else:
            wall_positions = tuple(sorted(_find_cells_by_color(grid, 3)))
            target_cells = _find_cells_by_color(grid, 11)
        directional_map = [(a, b) for a, b in teleporter_pairs]
        directional_map.extend((b, a) for a, b in teleporter_pairs)
        teleporter_map = tuple(sorted(directional_map))
        layout.update(
            layout_evidence_source="environment_metadata",
            wall_positions=wall_positions,
            player_start_position=level_data["player_start_position"],
            teleporter_endpoint_positions=portal_positions,
            teleporter_pairs=teleporter_pairs,
            teleporter_map=teleporter_map,
            target_cells=target_cells,
        )
        return layout
    if family == "ic01":
        walls = tuple(sorted(_find_cells_by_color(grid, 3)))
        hazards = tuple(sorted(_find_cells_by_color(grid, 10)))
        traversable = tuple(
            (x, y)
            for y in range(bounds[1])
            for x in range(bounds[0])
            if (x, y) not in set(walls) | set(hazards)
        )
        layout.update(
            wall_positions=walls,
            hazard_positions=hazards,
            ice_cell_positions=traversable,
            target_cells=_find_cells_by_color(grid, 11),
        )
        return layout
    if family == "va01":
        wall_positions = tuple(sorted(_find_cells_by_color(grid, 3)))
        coverage_cells = tuple(sorted(_find_cells_by_color(grid, 12)))
        coverage_eligible_cells = tuple(
            (x, y)
            for y in range(bounds[1])
            for x in range(bounds[0])
            if (x, y) not in set(wall_positions)
        )
        layout.update(
            wall_positions=wall_positions,
            coverage_cells=coverage_cells,
            coverage_eligible_cells=coverage_eligible_cells,
        )
        return layout
    if family == "pb01":
        module = _load_game_module_from_environment_metadata(parsed_state, family)
        level = _load_level_object_from_module(parsed_state, family, module)
        target_positions = tuple(sorted(_extract_sprite_positions(level, "target")))
        wall_positions = tuple(sorted(_extract_sprite_positions(level, "wall")))
        step_limit = level.get_data("step_limit") if hasattr(level, "get_data") else None
        if len(target_positions) != 1:
            raise ValueError(f"pb01 target state unavailable: expected exactly one configured target, found {len(target_positions)}")
        if not isinstance(step_limit, int) or step_limit <= 0:
            raise ValueError("pb01 config unavailable: missing positive step_limit")
        block_positions = tuple(
            sorted(
                pos
                for pos in (_find_cells_by_color(grid, 15) + _find_cells_by_color(grid, 14))
                if pos != (bounds[0] - 1, bounds[1] - 1)
            )
        )
        block_positions = tuple(sorted(set(block_positions) - set(target_positions)))
        if len(block_positions) != 1:
            raise ValueError(f"pushable_block_positions: expected exactly one directly observed block, found {len(block_positions)}")
        layout.update(
            layout_evidence_source="environment_metadata",
            wall_positions=wall_positions,
            target_cells=target_positions,
            pushable_block_positions=block_positions,
            step_limit=step_limit,
        )
        return layout
    raise ValueError(f"unsupported movement family: {family}")


def _base_common(
    parsed_state: ParsedStateV4,
    family: str,
    layout: dict,
    traversable: tuple[GridPos, ...],
    blocked: tuple[GridPos, ...],
    targets: tuple[GridPos, ...],
    hazards: tuple[GridPos, ...],
) -> MovementCommonFieldsV4:
    bounds = layout["bounds"]
    avatar_position = _avatar_position(parsed_state, bounds, layout)
    return MovementCommonFieldsV4(
        game_family=family,
        game_id=parsed_state.current_observation.game_id,
        level_index=parsed_state.current_observation.levels_completed,
        avatar_position=avatar_position,
        traversable_cells=tuple(sorted(traversable)),
        current_legal_actions=tuple(sorted(parsed_state.available_actions)),
        terminal_status=parsed_state.terminal_signal.status,
        step_depth=parsed_state.step_index,
        static_bounds=bounds,
        blocked_cells=tuple(sorted(blocked)),
        target_cells=tuple(sorted(targets)),
        hazard_positions=tuple(sorted(hazards)),
    )


def build_ul01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "ul01")
    bounds = layout["bounds"]
    key_positions = tuple(sorted(layout["key_positions"]))
    door_positions = tuple(sorted(layout["door_positions"]))
    target_cells = tuple(sorted(layout["target_cells"]))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(door_positions))
    common = _base_common(parsed_state, "ul01", layout, traversable, door_positions, target_cells, ())
    family = MovementFamilyFieldsV4(
        key_inventory_bits=0 if key_positions else 1,
        key_positions=key_positions,
        door_positions=door_positions,
        door_open=len(door_positions) == 0,
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))


def build_fs01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "fs01")
    bounds = layout["bounds"]
    avatar = _avatar_position(parsed_state, bounds, layout)
    activated_mask = 0
    for index, pos in enumerate(layout["switch_positions"]):
        color = layout["grid"][pos[1]][pos[0]]
        if color not in {10, 11} and avatar != pos:
            raise ValueError(f"fs01 switch state unavailable at {pos}: observed color {color}")
        if color == 10 or avatar == pos:
            activated_mask |= 1 << index
    door_positions = tuple(sorted(layout["door_positions"]))
    blocked = tuple(sorted(tuple(layout["wall_positions"]) + door_positions))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(blocked))
    common = _base_common(parsed_state, "fs01", layout, traversable, blocked, tuple(layout["target_cells"]), ())
    family = MovementFamilyFieldsV4(
        door_positions=door_positions,
        door_open=bool(layout["door_open"]),
        switch_positions=tuple(sorted(layout["switch_positions"])),
        activated_switch_bits=activated_mask,
        door_state_bits=1 if bool(layout["door_open"]) else 0,
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))


def build_tp01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "tp01")
    bounds = layout["bounds"]
    blocked = tuple(sorted(layout["wall_positions"]))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(blocked))
    common = _base_common(parsed_state, "tp01", layout, traversable, blocked, tuple(layout["target_cells"]), ())
    family = MovementFamilyFieldsV4(
        teleporter_endpoint_positions=tuple(layout["teleporter_endpoint_positions"]),
        teleporter_pairs=tuple(layout["teleporter_pairs"]),
        teleporter_pair_map=tuple(layout["teleporter_map"]),
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))


def build_ic01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "ic01")
    bounds = layout["bounds"]
    blocked = tuple(sorted(tuple(layout["wall_positions"]) + tuple(layout["hazard_positions"])))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(blocked))
    common = _base_common(parsed_state, "ic01", layout, traversable, blocked, tuple(layout["target_cells"]), tuple(layout["hazard_positions"]))
    family = MovementFamilyFieldsV4(
        slide_mode="ice",
        ice_cell_positions=tuple(layout["ice_cell_positions"]),
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))


def build_va01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "va01")
    bounds = layout["bounds"]
    blocked = tuple(sorted(layout["wall_positions"]))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(blocked))
    avatar = _avatar_position(parsed_state, bounds, layout)
    coverage = set(layout["coverage_cells"])
    coverage.add(avatar)
    common = _base_common(parsed_state, "va01", layout, traversable, blocked, (), ())
    family = MovementFamilyFieldsV4(
        coverage_eligible_cells=tuple(layout["coverage_eligible_cells"]),
        coverage_mask=tuple(sorted(coverage)),
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))


def build_pb01_movement_state(parsed_state: ParsedStateV4) -> MovementTypedStateV4:
    layout = extract_family_layout_from_parsed_state(parsed_state, "pb01")
    bounds = layout["bounds"]
    blocked = tuple(sorted(layout["wall_positions"]))
    traversable = tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(blocked))
    common = _base_common(parsed_state, "pb01", layout, traversable, blocked, tuple(layout["target_cells"]), ())
    family = MovementFamilyFieldsV4(
        pushable_block_positions=tuple(layout["pushable_block_positions"]),
        push_target_cells=tuple(layout["target_cells"]),
        step_limit=int(layout["step_limit"]),
    )
    return MovementTypedStateV4(common=common, family=family, layout_evidence_source=str(layout["layout_evidence_source"]))
