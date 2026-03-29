from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

from v4.agentContract.types import V4Observation
from v4.state.parsedState import ParsedStateV4

from .typedState import RuleSwitchCommonFieldsV4, RuleSwitchFamilyFieldsV4, RuleSwitchTypedStateV4


GridPos = tuple[int, int]
_MOVE_ACTIONS = (1, 2, 3, 4)


def _load_level(parsed_state: ParsedStateV4, family: str):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError(f"{family} config unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / f"{family}.py"
    if not module_path.exists():
        raise ValueError(f"{family} config unavailable: missing local game module at {module_path}")
    spec = importlib.util.spec_from_file_location(f"v4_{family}_config", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{family} config unavailable: failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    levels = getattr(module, "levels", None)
    if not isinstance(levels, list):
        raise ValueError(f"{family} config unavailable: module does not expose levels")
    index = parsed_state.current_observation.levels_completed
    if not isinstance(index, int) or not (0 <= index < len(levels)):
        raise ValueError(f"{family} config unavailable: invalid level index {index}")
    return levels[index]


def _iter_level_sprites(level):
    sprites = getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        sprites = getattr(level, "_sprites", None)
    if not isinstance(sprites, list):
        return ()
    return sprites


def _bounds(parsed_state: ParsedStateV4) -> GridPos:
    metadata = parsed_state.environment_metadata
    if metadata is not None and metadata.coordinate_bounds is not None:
        xmin, ymin, xmax, ymax = metadata.coordinate_bounds
        if xmin == 0 and ymin == 0 and xmax >= 0 and ymax >= 0:
            return xmax + 1, ymax + 1
    level = _load_level(parsed_state, "rs01")
    grid_size = getattr(level, "grid_size", None)
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise ValueError("rs01 config unavailable: invalid grid_size")
    return int(grid_size[0]), int(grid_size[1])


def _sample_grid(observation: V4Observation, bounds: GridPos) -> tuple[tuple[int, ...], ...]:
    plane = observation.frame[0]
    pixel_h = len(plane)
    pixel_w = len(plane[0]) if pixel_h else 0
    if pixel_h <= 0 or pixel_w <= 0:
        raise ValueError("observation frame cannot be sampled into grid cells")
    rows = []
    for y in range(bounds[1]):
        row = []
        for x in range(bounds[0]):
            px = min(pixel_w - 1, int((x + 0.5) * pixel_w / bounds[0]))
            py = min(pixel_h - 1, int((y + 0.5) * pixel_h / bounds[1]))
            row.append(int(plane[py][px]))
        rows.append(tuple(row))
    return tuple(rows)


def _cells_by_color(grid: tuple[tuple[int, ...], ...], color: int) -> tuple[GridPos, ...]:
    return tuple((x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value == color)


def _find_avatar(grid: tuple[tuple[int, ...], ...], level) -> GridPos:
    cells = _cells_by_color(grid, 9)
    target = tuple(int(v) for v in next(((sprite.x, sprite.y) for sprite in _iter_level_sprites(level) if "player" in getattr(sprite, "tags", ())), ()))
    if target and target in cells:
        return target  # prefer the known in-bounds player cell over HUD pixels
    if len(cells) != 1:
        raise ValueError(f"avatar_position: expected exactly one avatar cell, found {len(cells)}")
    return cells[0]


def _extract_active_safe_color(observation: V4Observation, safe_colors: tuple[int, ...]) -> int:
    plane = observation.frame[0]
    if not plane:
        raise ValueError("frame unavailable for safe-color extraction")
    search_rows = plane[:4]
    safe_set = set(safe_colors)
    counts = Counter(int(value) for row in search_rows for value in row if int(value) in safe_set)
    if not counts:
        raise ValueError("safe_color: current signpost color not visible in observation")
    return int(counts.most_common(1)[0][0])


def _group_positions_by_color(cells_by_color: dict[int, tuple[GridPos, ...]], *, only_non_empty: bool) -> tuple[tuple[int, tuple[GridPos, ...]], ...]:
    items = []
    for color in sorted(cells_by_color):
        positions = tuple(sorted(cells_by_color[color]))
        if only_non_empty and not positions:
            continue
        items.append((int(color), positions))
    return tuple(items)


def build_rs01_rule_switch_state(parsed_state: ParsedStateV4) -> RuleSwitchTypedStateV4:
    level = _load_level(parsed_state, "rs01")
    bounds = _bounds(parsed_state)
    grid = _sample_grid(parsed_state.current_observation, bounds)
    avatar = _find_avatar(grid, level)
    wall_cells = tuple(sorted(_cells_by_color(grid, 3)))
    safe_colors = tuple(int(value) for value in (level.get_data("safe_colors") or ()))
    if not safe_colors:
        raise ValueError("rs01 state requires explicit safe_colors metadata")
    target_map = {color: tuple(sorted(_cells_by_color(grid, color))) for color in safe_colors}
    target_cells = tuple(sorted(pos for positions in target_map.values() for pos in positions))
    level_target_map: dict[int, list[GridPos]] = {}
    for sprite in _iter_level_sprites(level):
        if "target" not in getattr(sprite, "tags", ()):
            continue
        color = int(sprite.pixels[0][0])
        level_target_map.setdefault(color, []).append((int(sprite.x), int(sprite.y)))
    remaining = _group_positions_by_color(target_map, only_non_empty=False)
    collected = tuple(
        (int(color), len(tuple(level_target_map.get(color, ()))) - len(target_map.get(color, ())))
        for color in sorted(level_target_map)
    )
    active_safe_color = _extract_active_safe_color(parsed_state.current_observation, safe_colors)
    cycle_index = safe_colors.index(active_safe_color) if active_safe_color in safe_colors else None
    walkable = tuple(sorted((x, y) for y in range(bounds[1]) for x in range(bounds[0]) if (x, y) not in set(wall_cells)))
    return RuleSwitchTypedStateV4(
        common=RuleSwitchCommonFieldsV4(
            game_family="rs01",
            game_id=parsed_state.current_observation.game_id,
            level_index=parsed_state.current_observation.levels_completed,
            avatar_position=avatar,
            walkable_cells=walkable,
            target_cells=target_cells,
            goal_cells=(),
            legal_action_ids=tuple(sorted(action for action in parsed_state.available_actions if int(action) in _MOVE_ACTIONS)),
            terminal_status="success" if not target_cells else ("failure" if parsed_state.terminal_signal.status == "failure" else "non_terminal"),
            step_depth=int(parsed_state.step_index),
            static_bounds=bounds,
            wall_cells=wall_cells,
        ),
        family=RuleSwitchFamilyFieldsV4(
            target_items_by_color=_group_positions_by_color({color: tuple(level_target_map.get(color, ())) for color in safe_colors}, only_non_empty=False),
            active_safe_color=active_safe_color,
            safe_color_cycle=safe_colors,
            collected_targets_by_color=tuple((color, max(0, count)) for color, count in collected),
            remaining_targets_by_color=remaining,
            cycle_interval=level.get_data("cycle_interval"),
            cycle_index=cycle_index,
            all_cycled=None,
        ),
        layout_evidence_source="direct_observation",
    )
