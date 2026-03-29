from __future__ import annotations

import importlib.util
from pathlib import Path

from v4.agentContract.types import V4Observation
from v4.state.parsedState import ParsedStateV4

from .typedState import GridPos, HybridConstructionCommonFieldsV4, HybridConstructionFamilyFieldsV4, HybridConstructionTypedStateV4

_MOVE_ACTIONS = {1, 2, 3, 4}


def _load_level(parsed_state: ParsedStateV4):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError("tb01 state unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / "tb01.py"
    spec = importlib.util.spec_from_file_location("v4_tb01_module", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"tb01 state unavailable: failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "levels")[parsed_state.current_observation.levels_completed]


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


def _find_unique(grid: tuple[tuple[int, ...], ...], color: int) -> GridPos:
    found = tuple((x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value == color)
    if len(found) != 1:
        raise ValueError(f"avatar_position: expected exactly one cell with color {color}, found {len(found)}")
    return found[0]


def build_tb01_hybrid_construction_state(parsed_state: ParsedStateV4) -> HybridConstructionTypedStateV4:
    level = _load_level(parsed_state)
    bounds = getattr(level, "grid_size", None)
    grid = _grid_from_observation(parsed_state.current_observation, bounds)
    avatar_position = _find_unique(grid, 9)
    land_cells = tuple(sorted(set(level.get_data("island_coords")) | set(level.get_data("goal_island_coords"))))
    reef_cells = tuple(sorted(level.get_data("rock_coords") or ()))
    bridge_cells = tuple(sorted((x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value == 12))
    all_cells = {(x, y) for y in range(bounds[1]) for x in range(bounds[0])}
    water_cells = tuple(sorted(all_cells - set(land_cells) - set(reef_cells) - set(bridge_cells)))
    max_bridges = level.get_data("max_bridges")
    step_limit = level.get_data("step_limit")
    bridge_budget_remaining = None if max_bridges is None else max(0, int(max_bridges) - len(bridge_cells))
    step_limit_remaining = None if step_limit is None else max(0, int(step_limit) - int(parsed_state.step_index))
    legal_click_cells = tuple(sorted(set(water_cells) | set(bridge_cells)))
    goal_cells = tuple(sorted(level.get_data("goal_island_coords")))
    goal_cell = goal_cells[0] if goal_cells else None
    return HybridConstructionTypedStateV4(
        common=HybridConstructionCommonFieldsV4(
            game_family="tb01",
            game_id=parsed_state.current_observation.game_id,
            level_index=parsed_state.current_observation.levels_completed,
            avatar_position=avatar_position,
            current_legal_actions=tuple(int(action_id) for action_id in parsed_state.available_actions if int(action_id) in _MOVE_ACTIONS | {6}),
            terminal_status=parsed_state.terminal_signal.status,
            step_depth=parsed_state.step_index,
            static_bounds=bounds,
        ),
        family=HybridConstructionFamilyFieldsV4(
            land_cells=land_cells,
            water_cells=water_cells,
            reef_cells=reef_cells,
            bridge_built_cells=bridge_cells,
            bridge_budget_remaining=bridge_budget_remaining,
            step_limit_remaining=step_limit_remaining,
            goal_cell=goal_cell,
            legal_movement_actions=tuple(sorted(_MOVE_ACTIONS)),
            legal_click_cells=legal_click_cells,
        ),
        layout_evidence_source="environment_metadata",
    )
