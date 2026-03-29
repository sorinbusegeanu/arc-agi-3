from __future__ import annotations

from .typedState import MovementTypedStateV4


def zero_heuristic(_: MovementTypedStateV4) -> int:
    return 0


def admissible_target_distance_heuristic(state: MovementTypedStateV4) -> int:
    if not state.common.target_cells:
        return 0
    ax, ay = state.common.avatar_position
    return min(abs(ax - tx) + abs(ay - ty) for tx, ty in state.common.target_cells)


def admissible_remaining_coverage_heuristic(state: MovementTypedStateV4) -> int:
    eligible = set(state.family.coverage_eligible_cells)
    if not eligible:
        return 0
    covered = set(state.family.coverage_mask)
    return max(0, len(eligible - covered))
