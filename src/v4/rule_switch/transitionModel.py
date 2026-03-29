from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .typedState import GridPos, RuleSwitchTypedStateV4

_DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True)
class RuleSwitchTransitionAnnotationV4:
    action_id: int
    moved: bool
    blocked: bool
    event: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _add(pos: GridPos, delta: GridPos) -> GridPos:
    return pos[0] + delta[0], pos[1] + delta[1]


class RuleSwitchTransitionModelV4:
    def apply(self, state: RuleSwitchTypedStateV4, action_id: int) -> tuple[RuleSwitchTypedStateV4, RuleSwitchTransitionAnnotationV4]:
        if action_id not in _DIRECTIONS:
            raise ValueError(f"unsupported primitive rule_switch action: {action_id}")
        if action_id not in state.common.legal_action_ids:
            raise ValueError(f"action {action_id} is not legal for the current rule_switch state")
        if state.common.terminal_status in {"success", "failure"}:
            return state, RuleSwitchTransitionAnnotationV4(action_id, False, True, "terminal")
        next_pos = _add(state.common.avatar_position, _DIRECTIONS[action_id])
        if not (0 <= next_pos[0] < state.common.static_bounds[0] and 0 <= next_pos[1] < state.common.static_bounds[1]):
            successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
            return successor, RuleSwitchTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in set(state.common.wall_cells):
            successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
            return successor, RuleSwitchTransitionAnnotationV4(action_id, False, True, "wall")
        remaining = {color: list(positions) for color, positions in state.family.remaining_targets_by_color}
        event = "move"
        terminal_status = "non_terminal"
        for color, positions in list(remaining.items()):
            if next_pos not in positions:
                continue
            if state.family.active_safe_color is None:
                raise ValueError("rs01 transition requires explicit active_safe_color")
            if color != state.family.active_safe_color:
                terminal_status = "failure"
                event = "wrong_color_target"
            else:
                positions.remove(next_pos)
                remaining[color] = positions
                event = "collect_safe_target"
            break
        grouped_remaining = tuple((color, tuple(sorted(positions))) for color, positions in sorted(remaining.items()))
        remaining_targets = tuple(sorted(pos for positions in remaining.values() for pos in positions))
        if terminal_status != "failure" and not remaining_targets:
            terminal_status = "success"
        base_targets = {int(color): tuple(positions) for color, positions in state.family.target_items_by_color}
        collected = tuple(
            (color, len(base_targets[color]) - len(tuple(remaining.get(color, ()))))
            for color in sorted(base_targets)
        )
        successor = replace(
            state,
            common=replace(
                state.common,
                avatar_position=next_pos,
                target_cells=remaining_targets,
                step_depth=state.common.step_depth + 1,
                terminal_status=terminal_status,
            ),
            family=replace(
                state.family,
                remaining_targets_by_color=grouped_remaining,
                collected_targets_by_color=collected,
            ),
        )
        return successor, RuleSwitchTransitionAnnotationV4(action_id, True, False, event)
