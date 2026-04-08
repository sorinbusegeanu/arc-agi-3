from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass
import json
from typing import Callable

from .heuristics import zero_heuristic
from .transitionModel import MovementTransitionModelV4
from .typedState import MovementTypedStateV4


@dataclass(frozen=True)
class MovementSearchOutcomeV4:
    status: str
    plan: tuple[int, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MovementSearchV4:
    def __init__(self, transition_model: MovementTransitionModelV4 | None = None) -> None:
        self.transition_model = transition_model if transition_model is not None else MovementTransitionModelV4()

    def _state_key(self, state: MovementTypedStateV4) -> str:
        payload = state.to_dict()
        common = payload.get("common", {})
        if isinstance(common, dict):
            common.pop("step_depth", None)
        family = payload.get("family", {})
        push_positions = family.get("pushable_block_positions")
        if isinstance(push_positions, (list, tuple)):
            family["pushable_block_positions"] = tuple(sorted(tuple(pos) for pos in push_positions))
        solved_goal_cells = family.get("push_solved_goal_cells")
        if isinstance(solved_goal_cells, (list, tuple)):
            family["push_solved_goal_cells"] = tuple(sorted(tuple(pos) for pos in solved_goal_cells))
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def search(
        self,
        initial_state: MovementTypedStateV4,
        goal_predicate: Callable[[MovementTypedStateV4], bool],
        legal_action_generator: Callable[[MovementTypedStateV4], tuple[int, ...] | list[int]] | None = None,
        *,
        algorithm: str = "bfs",
        max_depth: int | None = None,
        heuristic: Callable[[MovementTypedStateV4], int] | None = None,
    ) -> MovementSearchOutcomeV4:
        if goal_predicate(initial_state):
            return MovementSearchOutcomeV4(status="found", plan=(), explored_nodes=0)
        generator = legal_action_generator or (lambda state: state.common.current_legal_actions)
        if algorithm == "bfs":
            return self._bfs(initial_state, goal_predicate, generator, max_depth=max_depth)
        if algorithm == "astar":
            return self._astar(initial_state, goal_predicate, generator, max_depth=max_depth, heuristic=heuristic or zero_heuristic)
        raise ValueError(f"unsupported search algorithm: {algorithm}")

    def _bfs(
        self,
        initial_state: MovementTypedStateV4,
        goal_predicate: Callable[[MovementTypedStateV4], bool],
        generator: Callable[[MovementTypedStateV4], tuple[int, ...] | list[int]],
        *,
        max_depth: int | None,
    ) -> MovementSearchOutcomeV4:
        queue = deque([(initial_state, ())])
        visited = {self._state_key(initial_state)}
        explored = 0
        bound_hit = False
        while queue:
            state, plan = queue.popleft()
            explored += 1
            if max_depth is not None and len(plan) >= max_depth:
                bound_hit = True
                continue
            for action_id in sorted(int(value) for value in generator(state)):
                successor, _ = self.transition_model.apply(state, action_id)
                next_plan = plan + (action_id,)
                if goal_predicate(successor):
                    return MovementSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = self._state_key(successor)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((successor, next_plan))
        return MovementSearchOutcomeV4(
            status="bound_exhausted" if bound_hit else "exhausted",
            explored_nodes=explored,
            failure_reason="search depth bound exhausted" if bound_hit else "no goal state reached",
        )

    def _astar(
        self,
        initial_state: MovementTypedStateV4,
        goal_predicate: Callable[[MovementTypedStateV4], bool],
        generator: Callable[[MovementTypedStateV4], tuple[int, ...] | list[int]],
        *,
        max_depth: int | None,
        heuristic: Callable[[MovementTypedStateV4], int],
    ) -> MovementSearchOutcomeV4:
        heap: list[tuple[int, int, int, MovementTypedStateV4, tuple[int, ...]]] = []
        counter = 0
        heapq.heappush(heap, (heuristic(initial_state), 0, counter, initial_state, ()))
        best_cost = {self._state_key(initial_state): 0}
        explored = 0
        bound_hit = False
        while heap:
            _, cost, _, state, plan = heapq.heappop(heap)
            explored += 1
            if max_depth is not None and len(plan) >= max_depth:
                bound_hit = True
                continue
            for action_id in sorted(int(value) for value in generator(state)):
                successor, _ = self.transition_model.apply(state, action_id)
                next_plan = plan + (action_id,)
                next_cost = cost + 1
                if goal_predicate(successor):
                    return MovementSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = self._state_key(successor)
                if key in best_cost and best_cost[key] <= next_cost:
                    continue
                best_cost[key] = next_cost
                counter += 1
                heapq.heappush(heap, (next_cost + heuristic(successor), next_cost, counter, successor, next_plan))
        return MovementSearchOutcomeV4(
            status="bound_exhausted" if bound_hit else "exhausted",
            explored_nodes=explored,
            failure_reason="search depth bound exhausted" if bound_hit else "no goal state reached",
        )
