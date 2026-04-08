from __future__ import annotations

from itertools import permutations

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


def admissible_push_goal_heuristic(state: MovementTypedStateV4) -> int:
    blocks = tuple(sorted(state.family.pushable_block_positions))
    goals = tuple(sorted(state.family.push_target_cells or state.common.target_cells))
    if not blocks or not goals:
        return 0
    if len(blocks) > len(goals):
        return 0
    best: int | None = None
    for goal_order in permutations(goals, len(blocks)):
        total = sum(abs(block[0] - goal[0]) + abs(block[1] - goal[1]) for block, goal in zip(blocks, goal_order))
        if best is None or total < best:
            best = total
    return 0 if best is None else best
