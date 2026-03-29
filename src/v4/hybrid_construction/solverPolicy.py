from __future__ import annotations

from collections import deque

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .search import HybridConstructionSearchV4
from .stateBuilder import HybridConstructionStateBuilderV4
from .transitionModel import HybridConstructionTransitionModelV4

_MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _movement_path(state) -> tuple[int, ...]:
    if state.family.goal_cell is None:
        return ()
    start = state.common.avatar_position
    goal = state.family.goal_cell
    traversable = set(state.family.land_cells) | set(state.family.bridge_built_cells)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        pos, plan = queue.popleft()
        if pos == goal:
            return plan
        for action_id, delta in _MOVE_DELTAS.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if nxt not in traversable or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, plan + (action_id,)))
    return ()


def _bridge_payload(state, cell):
    width, height = state.common.static_bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return {"x": cell[0] * scale + scale // 2 + x_pad, "y": cell[1] * scale + scale // 2 + y_pad, "game_id": state.common.game_id}


class HybridConstructionSolverPolicyV4(PolicyBaseV4):
    def __init__(
        self,
        *,
        state_builder: HybridConstructionStateBuilderV4 | None = None,
        transition_model: HybridConstructionTransitionModelV4 | None = None,
        search: HybridConstructionSearchV4 | None = None,
        search_bound: int = 12,
    ) -> None:
        self.state_builder = state_builder if state_builder is not None else HybridConstructionStateBuilderV4()
        self.transition_model = transition_model if transition_model is not None else HybridConstructionTransitionModelV4()
        self.search = search if search is not None else HybridConstructionSearchV4(self.transition_model)
        self.search_bound = int(search_bound)

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="tb01")
        move_path = _movement_path(state)
        if move_path:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(move_path[0], parsed_state=parsed_state),
                annotations={"policy": "hybrid_construction_solver", "family": "tb01", "search_status": "movement_path"},
            )
        if state.family.goal_cell is not None and state.family.legal_click_cells:
            ax, ay = state.common.avatar_position
            gx, gy = state.family.goal_cell
            ranked = sorted(
                state.family.legal_click_cells,
                key=lambda cell: (abs(cell[0] - gx) + abs(cell[1] - gy), abs(cell[0] - ax) + abs(cell[1] - ay), cell),
            )
            cell = ranked[0]
            payload = _bridge_payload(state, cell)
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=payload),
                annotations={"policy": "hybrid_construction_solver", "family": "tb01", "search_status": "bridge_greedy", "fallback": "goal_ranked_bridge"},
            )
        return PolicyDecisionV4(
            primitive_action=legal_action_from_id(1, parsed_state=parsed_state),
            annotations={"policy": "hybrid_construction_solver", "family": "tb01", "search_status": "exhausted", "fallback": "lowest_move"},
        )
