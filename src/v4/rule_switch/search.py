from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

from .transitionModel import RuleSwitchTransitionModelV4
from .typedState import RuleSwitchTypedStateV4


@dataclass(frozen=True)
class RuleSwitchSearchOutcomeV4:
    status: str
    plan: tuple[int, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RuleSwitchSearchV4:
    def __init__(self, transition_model: RuleSwitchTransitionModelV4 | None = None) -> None:
        self.transition_model = transition_model if transition_model is not None else RuleSwitchTransitionModelV4()

    def _state_key(self, state: RuleSwitchTypedStateV4) -> str:
        return state.to_key()

    def legal_actions(self, state: RuleSwitchTypedStateV4) -> tuple[int, ...]:
        safe_color = state.family.active_safe_color
        allowed_targets = set(state.common.goal_cells)
        remaining_total = 0
        if safe_color is not None:
            for color, positions in state.family.remaining_targets_by_color:
                remaining_total += len(positions)
                if color == safe_color:
                    allowed_targets.update(positions)
        if remaining_total == 0 and state.common.terminal_status == "success":
            filtered = []
            for action_id in sorted(int(value) for value in state.common.legal_action_ids):
                try:
                    successor, annotation = self.transition_model.apply(state, action_id)
                except ValueError:
                    continue
                if annotation.blocked:
                    continue
                if successor.common.terminal_status == "failure":
                    continue
                filtered.append(action_id)
            return tuple(filtered)
        filtered = []
        for action_id in sorted(int(value) for value in state.common.legal_action_ids):
            try:
                successor, annotation = self.transition_model.apply(state, action_id)
            except ValueError:
                continue
            if annotation.blocked:
                continue
            if successor.common.terminal_status == "failure":
                continue
            if successor.common.avatar_position in state.common.target_cells and successor.common.avatar_position not in allowed_targets:
                continue
            filtered.append(action_id)
        return tuple(filtered)

    def search(self, initial_state: RuleSwitchTypedStateV4, goal_predicate: Callable[[RuleSwitchTypedStateV4], bool], *, max_depth: int | None = None) -> RuleSwitchSearchOutcomeV4:
        if goal_predicate(initial_state):
            return RuleSwitchSearchOutcomeV4(status="found", plan=(), explored_nodes=0)
        queue = deque([(initial_state, ())])
        seen = {self._state_key(initial_state)}
        explored = 0
        while queue:
            state, plan = queue.popleft()
            explored += 1
            if max_depth is not None and len(plan) >= max_depth:
                continue
            for action_id in self.legal_actions(state):
                try:
                    successor, _ = self.transition_model.apply(state, action_id)
                except ValueError:
                    continue
                next_plan = plan + (action_id,)
                if goal_predicate(successor):
                    return RuleSwitchSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = self._state_key(successor)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((successor, next_plan))
        return RuleSwitchSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no safe-color goal state reached")
