from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

from v4.agentContract.types import V4Action

from .heuristics import zero_heuristic
from .transitionModel import ClickTransitionModelV4
from .typedState import ClickTypedStateV4


def _action_from_click_cell(state: ClickTypedStateV4, click_cell: tuple[int, int]) -> V4Action:
    return V4Action(action_id=6, action_name="ACTION6", payload={"x": click_cell[0], "y": click_cell[1], "game_id": state.common.game_id})


def _action_from_grid_cell(state: ClickTypedStateV4, grid_cell: tuple[int, int]) -> V4Action:
    width, height = state.common.static_bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return V4Action(
        action_id=6,
        action_name="ACTION6",
        payload={
            "x": grid_cell[0] * scale + scale // 2 + x_pad,
            "y": grid_cell[1] * scale + scale // 2 + y_pad,
            "game_id": state.common.game_id,
        },
    )


@dataclass(frozen=True)
class ClickSearchOutcomeV4:
    status: str
    plan: tuple[V4Action, ...] = ()
    explored_nodes: int = 0
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ClickSearchV4:
    def __init__(self, transition_model: ClickTransitionModelV4 | None = None) -> None:
        self.transition_model = transition_model if transition_model is not None else ClickTransitionModelV4()

    def _state_key(self, state: ClickTypedStateV4) -> str:
        return state.to_key()

    def generate_candidates(self, state: ClickTypedStateV4) -> tuple[V4Action, ...]:
        if state.common.game_family == "sy01":
            target_cells = set(state.family.mirror_target_cells)
            placed_cells = set(state.family.placed_mirror_cells)
            delta_cells = tuple(sorted((target_cells - placed_cells) | (placed_cells - target_cells)))
            if delta_cells:
                return tuple(_action_from_grid_cell(state, grid_cell) for grid_cell in delta_cells)
        return tuple(_action_from_click_cell(state, click_cell) for click_cell in state.common.clickable_cells)

    def search(
        self,
        initial_state: ClickTypedStateV4,
        goal_predicate: Callable[[ClickTypedStateV4], bool],
        legal_click_generator: Callable[[ClickTypedStateV4], tuple[V4Action, ...] | list[V4Action]] | None = None,
        *,
        algorithm: str = "bfs",
        max_depth: int | None = None,
        heuristic: Callable[[ClickTypedStateV4], int] | None = None,
    ) -> ClickSearchOutcomeV4:
        if goal_predicate(initial_state):
            return ClickSearchOutcomeV4(status="found", plan=(), explored_nodes=0)
        generator = legal_click_generator or self.generate_candidates
        if algorithm == "greedy":
            candidates = tuple(generator(initial_state))
            if not candidates:
                return ClickSearchOutcomeV4(status="exhausted", failure_reason="no legal click candidates")
            return ClickSearchOutcomeV4(status="found", plan=(candidates[0],), explored_nodes=1)
        if algorithm == "astar":
            return self._astar(initial_state, goal_predicate, generator, max_depth=max_depth, heuristic=heuristic or zero_heuristic)
        return self._bfs(initial_state, goal_predicate, generator, max_depth=max_depth)

    def _bfs(
        self,
        initial_state: ClickTypedStateV4,
        goal_predicate: Callable[[ClickTypedStateV4], bool],
        generator: Callable[[ClickTypedStateV4], tuple[V4Action, ...] | list[V4Action]],
        *,
        max_depth: int | None,
    ) -> ClickSearchOutcomeV4:
        queue = deque([(initial_state, ())])
        visited = {self._state_key(initial_state)}
        explored = 0
        while queue:
            state, plan = queue.popleft()
            explored += 1
            if max_depth is not None and len(plan) >= max_depth:
                continue
            for action in generator(state):
                successor, _ = self.transition_model.apply(state, action)
                next_plan = plan + (action,)
                if goal_predicate(successor):
                    return ClickSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = self._state_key(successor)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((successor, next_plan))
        return ClickSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no goal state reached")

    def _astar(
        self,
        initial_state: ClickTypedStateV4,
        goal_predicate: Callable[[ClickTypedStateV4], bool],
        generator: Callable[[ClickTypedStateV4], tuple[V4Action, ...] | list[V4Action]],
        *,
        max_depth: int | None,
        heuristic: Callable[[ClickTypedStateV4], int],
    ) -> ClickSearchOutcomeV4:
        heap: list[tuple[int, int, int, ClickTypedStateV4, tuple[V4Action, ...]]] = []
        counter = 0
        heapq.heappush(heap, (heuristic(initial_state), 0, counter, initial_state, ()))
        best_cost = {self._state_key(initial_state): 0}
        explored = 0
        while heap:
            _, cost, _, state, plan = heapq.heappop(heap)
            explored += 1
            if max_depth is not None and len(plan) >= max_depth:
                continue
            for action in generator(state):
                successor, _ = self.transition_model.apply(state, action)
                next_plan = plan + (action,)
                next_cost = cost + 1
                if goal_predicate(successor):
                    return ClickSearchOutcomeV4(status="found", plan=next_plan, explored_nodes=explored)
                key = self._state_key(successor)
                if key in best_cost and best_cost[key] <= next_cost:
                    continue
                best_cost[key] = next_cost
                counter += 1
                heapq.heappush(heap, (next_cost + heuristic(successor), next_cost, counter, successor, next_plan))
        return ClickSearchOutcomeV4(status="exhausted", explored_nodes=explored, failure_reason="no goal state reached")
