from __future__ import annotations

import importlib.util
from pathlib import Path

from v4.state.parsedState import ParsedStateV4

from .typedState import MemoryHiddenCommonFieldsV4, MemoryHiddenFamilyFieldsV4, MemoryHiddenTypedStateV4


GridPos = tuple[int, int]
_MOVE_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_COUNT_BY_COLOR = {1: 0, 8: 1, 11: 2, 14: 3, 15: 4}


def _load_level(parsed_state: ParsedStateV4, family: str):
    metadata = parsed_state.environment_metadata
    if metadata is None or not metadata.local_dir:
        raise ValueError(f"{family} config unavailable: missing environment metadata local_dir")
    module_path = Path(metadata.local_dir) / f"{family}.py"
    spec = importlib.util.spec_from_file_location(f"v4_{family}_config", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{family} config unavailable: failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    levels = getattr(module, "levels", None)
    if not isinstance(levels, list):
        raise ValueError(f"{family} config unavailable: module does not expose levels")
    index = parsed_state.current_observation.levels_completed
    return levels[index]


def _sample_grid(parsed_state: ParsedStateV4, bounds: GridPos) -> tuple[tuple[int, ...], ...]:
    plane = parsed_state.current_observation.frame[0]
    pixel_h = len(plane)
    pixel_w = len(plane[0]) if pixel_h else 0
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


def _neighbors(pos: GridPos, bounds: GridPos) -> tuple[GridPos, ...]:
    result = []
    for dx, dy in _MOVE_DELTAS:
        nxt = (pos[0] + dx, pos[1] + dy)
        if 0 <= nxt[0] < bounds[0] and 0 <= nxt[1] < bounds[1]:
            result.append(nxt)
    return tuple(result)


def _consistency_facts(number_cells: tuple[tuple[GridPos, int], ...], revealed_safe: set[GridPos], frontier: set[GridPos], known_mines: set[GridPos], bounds: GridPos):
    facts = []
    forbidden = set()
    for pos, count in number_cells:
        adjacent = set(_neighbors(pos, bounds))
        adjacent_frontier = tuple(sorted(adjacent & frontier))
        adjacent_known_mines = len(adjacent & known_mines)
        remaining = max(0, count - adjacent_known_mines)
        facts.append(("adjacent_mines_remaining", pos, adjacent_frontier, remaining))
        if remaining == 0:
            for cell in adjacent_frontier:
                forbidden.add(cell)
    return tuple(facts), tuple(sorted(forbidden))


def build_ms01_memory_hidden_state(parsed_state: ParsedStateV4) -> MemoryHiddenTypedStateV4:
    level = _load_level(parsed_state, "ms01")
    bounds = tuple(int(v) for v in level.grid_size)
    grid = _sample_grid(parsed_state, bounds)
    avatar_cells = _cells_by_color(grid, 9)
    if len(avatar_cells) != 1:
        raise ValueError(f"avatar_position: expected exactly one avatar cell, found {len(avatar_cells)}")
    avatar = avatar_cells[0]
    wall_cells = tuple(sorted(_cells_by_color(grid, 3)))
    goal_cells = tuple(sorted(_cells_by_color(grid, 14)))
    goal = goal_cells[0] if goal_cells else None
    visible_number_cells = tuple(sorted(((pos, count) for color, count in _COUNT_BY_COLOR.items() for pos in _cells_by_color(grid, color)), key=lambda item: item[0]))
    revealed_safe = set(pos for pos, _ in visible_number_cells)
    revealed_safe.add(avatar)
    frontier = set()
    blocked = set(wall_cells)
    for pos in sorted(revealed_safe):
        for nxt in _neighbors(pos, bounds):
            if nxt in revealed_safe or nxt in blocked or nxt == goal:
                continue
            frontier.add(nxt)
    local_consistency_facts, forbidden = _consistency_facts(visible_number_cells, revealed_safe, frontier, set(), bounds)
    safe_cells = tuple(sorted(revealed_safe))
    return MemoryHiddenTypedStateV4(
        common=MemoryHiddenCommonFieldsV4(
            game_family="ms01",
            game_id=parsed_state.current_observation.game_id,
            level_index=parsed_state.current_observation.levels_completed,
            avatar_position=avatar,
            traversable_safe_cells=safe_cells,
            current_legal_actions=tuple(sorted(int(value) for value in parsed_state.available_actions if int(value) in {1, 2, 3, 4})),
            terminal_status="success" if parsed_state.terminal_signal.status == "success" else ("failure" if parsed_state.terminal_signal.status == "failure" else "non_terminal"),
            step_depth=int(parsed_state.step_index),
            static_bounds=bounds,
            blocked_cells=wall_cells,
        ),
        family=MemoryHiddenFamilyFieldsV4(
            revealed_safe_cells=safe_cells,
            visible_number_cells=visible_number_cells,
            unrevealed_frontier_cells=tuple(sorted(frontier)),
            known_mines=(),
            forbidden_cells=forbidden,
            goal_cell=goal,
            local_consistency_facts=local_consistency_facts,
        ),
        layout_evidence_source="direct_observation",
    )
