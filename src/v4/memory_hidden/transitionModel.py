from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .typedState import GridPos, MemoryHiddenTypedStateV4


_DELTAS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


@dataclass(frozen=True)
class MemoryHiddenTransitionAnnotationV4:
    action_id: int
    moved: bool
    blocked: bool
    event: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _add(pos: GridPos, delta: GridPos) -> GridPos:
    return pos[0] + delta[0], pos[1] + delta[1]


def _in_bounds(pos: GridPos, bounds: GridPos) -> bool:
    return 0 <= pos[0] < bounds[0] and 0 <= pos[1] < bounds[1]


class MemoryHiddenTransitionModelV4:
    def apply(self, state: MemoryHiddenTypedStateV4, action_id: int) -> tuple[MemoryHiddenTypedStateV4, MemoryHiddenTransitionAnnotationV4]:
        if action_id not in _DELTAS:
            raise ValueError(f"unsupported primitive memory_hidden action: {action_id}")
        if action_id not in state.common.current_legal_actions:
            raise ValueError(f"action {action_id} is not legal for the current memory_hidden state")
        if state.common.terminal_status in {"success", "failure"}:
            return state, MemoryHiddenTransitionAnnotationV4(action_id, False, True, "terminal")
        next_pos = _add(state.common.avatar_position, _DELTAS[action_id])
        if not _in_bounds(next_pos, state.common.static_bounds):
            successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in set(state.common.blocked_cells):
            successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, False, True, "wall")
        if next_pos in set(state.family.known_mines):
            successor = replace(state, common=replace(state.common, avatar_position=next_pos, terminal_status="failure", step_depth=state.common.step_depth + 1))
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, True, False, "mine")
        if state.family.goal_cell is not None and next_pos == state.family.goal_cell:
            successor = replace(state, common=replace(state.common, avatar_position=next_pos, terminal_status="success", step_depth=state.common.step_depth + 1))
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, True, False, "goal")
        if next_pos in set(state.common.traversable_safe_cells):
            successor = replace(state, common=replace(state.common, avatar_position=next_pos, step_depth=state.common.step_depth + 1))
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, True, False, "move_safe")
        if next_pos in set(state.family.unrevealed_frontier_cells):
            successor = replace(
                state,
                common=replace(
                    state.common,
                    avatar_position=next_pos,
                    traversable_safe_cells=tuple(sorted(set(state.common.traversable_safe_cells) | {next_pos})),
                    step_depth=state.common.step_depth + 1,
                ),
                family=replace(
                    state.family,
                    revealed_safe_cells=tuple(sorted(set(state.family.revealed_safe_cells) | {next_pos})),
                    unrevealed_frontier_cells=tuple(sorted(pos for pos in state.family.unrevealed_frontier_cells if pos != next_pos)),
                ),
            )
            return successor, MemoryHiddenTransitionAnnotationV4(action_id, True, False, "reveal_frontier")
        raise ValueError(f"memory_hidden transition reached unsupported cell {next_pos}")
