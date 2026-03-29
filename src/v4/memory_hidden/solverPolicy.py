from __future__ import annotations

from collections import deque

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .search import MemoryHiddenSearchV4
from .stateBuilder import MemoryHiddenStateBuilderV4


_DELTAS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


class MemoryHiddenSolverPolicyV4(PolicyBaseV4):
    def __init__(self, *, state_builder: MemoryHiddenStateBuilderV4 | None = None, search: MemoryHiddenSearchV4 | None = None) -> None:
        self.state_builder = state_builder if state_builder is not None else MemoryHiddenStateBuilderV4()
        self.search = search if search is not None else MemoryHiddenSearchV4()

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        typed_state = self.state_builder.build(parsed_state, family=family)
        goal = typed_state.family.goal_cell
        if goal is not None and goal in set(typed_state.common.traversable_safe_cells):
            outcome = self.search.path_within_safe_region(typed_state, {goal})
            if outcome.status == "found" and outcome.plan:
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(outcome.plan[0], parsed_state=parsed_state),
                    annotations={"policy": "memory_hidden_solver", "family": family, "search_status": outcome.status, "target": "goal"},
                )
        frontier_action = self._frontier_action(typed_state)
        if frontier_action is not None:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(frontier_action, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "frontier", "target": "frontier"},
            )
        fallback = self._revealed_safe_fallback(typed_state)
        if fallback is not None:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(fallback, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "fallback", "fallback": "revealed_safe_only"},
            )
        raise ValueError("ms01 invalid state: no proven-safe path, no locally consistent frontier move, and no revealed-safe fallback action")

    def _frontier_action(self, state) -> int | None:
        safe = set(state.common.traversable_safe_cells)
        frontier = [cell for cell in state.family.unrevealed_frontier_cells if cell not in set(state.family.forbidden_cells)]
        if not frontier:
            return None
        blocked_targets = set(state.family.forbidden_cells) | set(state.family.known_mines)
        queue = deque([(state.common.avatar_position, ())])
        seen = {state.common.avatar_position}
        while queue:
            pos, plan = queue.popleft()
            for action_id, delta in _DELTAS.items():
                if action_id not in state.common.current_legal_actions:
                    continue
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in frontier and nxt not in blocked_targets:
                    return plan[0] if plan else action_id
                if nxt in seen or nxt not in safe:
                    continue
                seen.add(nxt)
                queue.append((nxt, plan + (action_id,)))
        return None

    def _revealed_safe_fallback(self, state) -> int | None:
        safe = set(state.common.traversable_safe_cells)
        for action_id, delta in _DELTAS.items():
            if action_id not in state.common.current_legal_actions:
                continue
            nxt = (state.common.avatar_position[0] + delta[0], state.common.avatar_position[1] + delta[1])
            if nxt in safe:
                return action_id
        return None
