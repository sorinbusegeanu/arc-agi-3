from __future__ import annotations

import importlib.util
from pathlib import Path

from v4.agentContract.types import V4Observation
from v4.state.parsedState import ParsedStateV4

from .typedState import GridPos, TimeReactiveCommonFieldsV4, TimeReactiveFamilyFieldsV4, TimeReactiveTypedStateV4

_MOVE_ACTIONS = {1, 2, 3, 4, 5}


def _load_level(parsed_state: ParsedStateV4):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError("sv01 state unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / "sv01.py"
    spec = importlib.util.spec_from_file_location("v4_sv01_module", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"sv01 state unavailable: failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    levels = getattr(module, "levels", None)
    if not isinstance(levels, list):
        raise ValueError("sv01 state unavailable: level list missing")
    level_index = parsed_state.current_observation.levels_completed
    return levels[level_index]


def _grid_from_observation(observation: V4Observation, bounds: GridPos) -> tuple[tuple[int, ...], ...]:
    plane = observation.frame[0]
    rows = []
    for y in range(bounds[1]):
        row = []
        for x in range(bounds[0]):
            px = min(len(plane[0]) - 1, int((x + 0.5) * len(plane[0]) / bounds[0]))
            py = min(len(plane) - 1, int((y + 0.5) * len(plane) / bounds[1]))
            row.append(int(plane[py][px]))
        rows.append(tuple(row))
    return tuple(rows)


def _extract_tag_positions(level, tag: str) -> tuple[GridPos, ...]:
    sprites = getattr(level, "sprites", None)
    if not isinstance(sprites, list):
        sprites = getattr(level, "_sprites", None)
    if not isinstance(sprites, list):
        return ()
    result = []
    for sprite in sprites:
        if tag not in getattr(sprite, "tags", ()):
            continue
        x = getattr(sprite, "x", None)
        y = getattr(sprite, "y", None)
        if isinstance(x, int) and isinstance(y, int):
            result.append((x, y))
    return tuple(sorted(set(result)))


def _parse_bar_value(observation: V4Observation, row: int, color: int, *, scale: int) -> int:
    plane = observation.frame[0]
    count = 0
    for x in range(2, min(len(plane[0]), 22)):
        if int(plane[row][x]) == color:
            count += 1
    return count * scale


def _avatar_position(grid: tuple[tuple[int, ...], ...]) -> GridPos:
    found = tuple((x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value == 9)
    if len(found) != 1:
        raise ValueError(f"avatar_position: expected exactly one cell with color 9, found {len(found)}")
    return found[0]


def build_sv01_time_reactive_state(parsed_state: ParsedStateV4) -> TimeReactiveTypedStateV4:
    level = _load_level(parsed_state)
    bounds = getattr(level, "grid_size", None)
    grid = _grid_from_observation(parsed_state.current_observation, bounds)
    avatar_position = _avatar_position(grid)
    hunger_value = min(100, _parse_bar_value(parsed_state.current_observation, 1, 14, scale=5))
    warmth_value = min(100, _parse_bar_value(parsed_state.current_observation, 2, 12, scale=5))
    survival_timer_remaining = min(60, _parse_bar_value(parsed_state.current_observation, 3, 3, scale=3))
    return TimeReactiveTypedStateV4(
        common=TimeReactiveCommonFieldsV4(
            game_family="sv01",
            game_id=parsed_state.current_observation.game_id,
            level_index=parsed_state.current_observation.levels_completed,
            avatar_position=avatar_position,
            walkable_cells=tuple((x, y) for y in range(bounds[1]) for x in range(bounds[0])),
            current_legal_actions=tuple(int(action_id) for action_id in parsed_state.available_actions if int(action_id) in _MOVE_ACTIONS),
            terminal_status=parsed_state.terminal_signal.status,
            step_depth=parsed_state.step_index,
            static_bounds=bounds,
        ),
        family=TimeReactiveFamilyFieldsV4(
            food_cells=_extract_tag_positions(level, "food"),
            warm_zone_cells=_extract_tag_positions(level, "warm_zone"),
            hunger_value=hunger_value,
            warmth_value=warmth_value,
            survival_timer_remaining=survival_timer_remaining,
            wait_action_id=5 if 5 in parsed_state.available_actions else None,
        ),
        layout_evidence_source="environment_metadata",
    )
