from __future__ import annotations

from .typedState import GridPos, HybridConstructionTypedStateV4


def manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def goal_distance_heuristic(state: HybridConstructionTypedStateV4) -> int:
    if state.family.goal_cell is None:
        return 0
    return manhattan(state.common.avatar_position, state.family.goal_cell)
