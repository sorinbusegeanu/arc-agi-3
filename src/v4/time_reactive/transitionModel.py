from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .typedState import GridPos, TimeReactiveTypedStateV4

_DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (0, 0)}


@dataclass(frozen=True)
class TimeReactiveTransitionAnnotationV4:
    action_id: int
    moved: bool
    blocked: bool
    event: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TimeReactiveTransitionModelV4:
    def apply(self, state: TimeReactiveTypedStateV4, action_id: int) -> tuple[TimeReactiveTypedStateV4, TimeReactiveTransitionAnnotationV4]:
        if action_id not in _DIRECTIONS:
            raise ValueError(f"unsupported primitive time_reactive action: {action_id}")
        delta = _DIRECTIONS[action_id]
        next_pos = (state.common.avatar_position[0] + delta[0], state.common.avatar_position[1] + delta[1])
        if next_pos not in state.common.walkable_cells:
            next_pos = state.common.avatar_position
        hunger = max(0, state.family.hunger_value - state.family.hunger_decay_per_step)
        warmth = state.family.warmth_value
        event = "wait" if action_id == 5 else "move"
        if next_pos in state.family.food_cells:
            hunger = min(100, hunger + state.family.food_restore_amount)
            event = "eat"
        if next_pos not in state.family.warm_zone_cells:
            warmth = max(0, warmth - state.family.warmth_decay_per_step)
        timer = max(0, state.family.survival_timer_remaining - 1)
        terminal_status = "non_terminal"
        if hunger <= 0 or warmth <= 0:
            terminal_status = "failure"
        elif timer <= 0:
            terminal_status = "success"
        successor = TimeReactiveTypedStateV4(
            common=replace(
                state.common,
                avatar_position=next_pos,
                step_depth=state.common.step_depth + 1,
                terminal_status=terminal_status,
            ),
            family=replace(
                state.family,
                food_cells=tuple(pos for pos in state.family.food_cells if pos != next_pos),
                hunger_value=hunger,
                warmth_value=warmth,
                survival_timer_remaining=timer,
            ),
            layout_evidence_source=state.layout_evidence_source,
        )
        return successor, TimeReactiveTransitionAnnotationV4(action_id, next_pos != state.common.avatar_position, next_pos == state.common.avatar_position and action_id != 5, event)
