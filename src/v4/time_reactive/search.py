from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

from .transitionModel import TimeReactiveTransitionModelV4
from .typedState import TimeReactiveTypedStateV4


@dataclass(frozen=True)
class TimeReactiveSearchOutcomeV4:
    status: str
    plan: tuple[int, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TimeReactiveSearchV4:
    def __init__(self, transition_model: TimeReactiveTransitionModelV4 | None = None) -> None:
        self.transition_model = transition_model if transition_model is not None else TimeReactiveTransitionModelV4()

    def search(self, initial_state: TimeReactiveTypedStateV4, goal_predicate: Callable[[TimeReactiveTypedStateV4], bool], *, max_depth: int) -> TimeReactiveSearchOutcomeV4:
        queue = deque([(initial_state, ())])
        seen = {initial_state.to_key()}
        explored = 0
        while queue:
            state, plan = queue.popleft()
            explored += 1
            if len(plan) >= max_depth:
                continue
            for action_id in sorted(state.common.current_legal_actions):
                successor, _ = self.transition_model.apply(state, action_id)
                next_plan = plan + (action_id,)
                if goal_predicate(successor):
                    return TimeReactiveSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                if successor.common.terminal_status == "failure":
                    continue
                key = successor.to_key()
                if key in seen:
                    continue
                seen.add(key)
                queue.append((successor, next_plan))
        return TimeReactiveSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no goal state reached")
