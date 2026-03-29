from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from v4.agentContract.types import V4Action

from .transitionModel import HybridConstructionTransitionModelV4, _grid_payload
from .typedState import HybridConstructionTypedStateV4


@dataclass(frozen=True)
class HybridConstructionSearchOutcomeV4:
    status: str
    plan: tuple[V4Action, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HybridConstructionSearchV4:
    def __init__(self, transition_model: HybridConstructionTransitionModelV4 | None = None) -> None:
        self.transition_model = transition_model if transition_model is not None else HybridConstructionTransitionModelV4()

    def _candidates(self, state: HybridConstructionTypedStateV4) -> tuple[int | V4Action, ...]:
        actions: list[int | V4Action] = list(state.family.legal_movement_actions)
        for cell in state.family.legal_click_cells:
            px, py = _grid_payload(state.common.static_bounds, cell[0], cell[1])
            actions.append(V4Action(action_id=6, action_name="ACTION6", payload={"x": px, "y": py, "game_id": state.common.game_id}))
        return tuple(actions)

    def search(self, initial_state: HybridConstructionTypedStateV4, *, max_depth: int = 12) -> HybridConstructionSearchOutcomeV4:
        queue = deque([(initial_state, ())])
        seen = {initial_state.to_key()}
        explored = 0
        while queue:
            state, plan = queue.popleft()
            explored += 1
            if len(plan) >= max_depth:
                continue
            for action in self._candidates(state):
                successor, _ = self.transition_model.apply(state, action)
                next_plan = plan + ((action if isinstance(action, V4Action) else V4Action(action_id=action, action_name=f"ACTION{action}")),)
                if successor.common.terminal_status == "success":
                    return HybridConstructionSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = successor.to_key()
                if key in seen:
                    continue
                seen.add(key)
                queue.append((successor, next_plan))
        return HybridConstructionSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no goal state reached")
