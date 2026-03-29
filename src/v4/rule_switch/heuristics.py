from __future__ import annotations

from .typedState import GridPos, RuleSwitchTypedStateV4


def manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_safe_target_heuristic(state: RuleSwitchTypedStateV4) -> int:
    safe_color = state.family.active_safe_color
    if safe_color is None:
        return 0
    for color, positions in state.family.remaining_targets_by_color:
        if color == safe_color and positions:
            return min(manhattan(state.common.avatar_position, pos) for pos in positions)
    return 0
