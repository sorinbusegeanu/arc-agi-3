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
        self._last_state_key: str | None = None
        self._last_action: int | None = None
        self._last_frontier_anchor: tuple[int, int] | None = None
        self._recent_frontier_anchors: deque[tuple[int, int]] = deque(maxlen=4)

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        typed_state = self.state_builder.build(parsed_state, family=family)
        goal = typed_state.family.goal_cell
        if goal is not None and goal in set(typed_state.common.traversable_safe_cells):
            outcome = self.search.path_within_safe_region(typed_state, {goal})
            if outcome.status == "found" and outcome.plan:
                action_id = self._choose_non_repeating(typed_state, [outcome.plan[0]])
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(action_id, parsed_state=parsed_state),
                    annotations={"policy": "memory_hidden_solver", "family": family, "search_status": outcome.status, "target": "goal"},
                )
        anchor_actions = self._frontier_anchor_actions(typed_state)
        if anchor_actions:
            anchor_action = self._choose_non_repeating(typed_state, list(anchor_actions) + [action for action in typed_state.common.current_legal_actions if action not in anchor_actions])
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(anchor_action, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "anchor_frontier", "target": "frontier_anchor"},
            )
        frontier_action = self._frontier_action(typed_state)
        if frontier_action is not None:
            frontier_action = self._choose_non_repeating(typed_state, [frontier_action] + [action for action in typed_state.common.current_legal_actions if action != frontier_action])
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(frontier_action, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "frontier", "target": "frontier"},
            )
        frontier_step = self._frontier_step_action(typed_state)
        if frontier_step is not None:
            frontier_step = self._choose_non_repeating(typed_state, [frontier_step] + [action for action in typed_state.common.current_legal_actions if action != frontier_step])
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(frontier_step, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "frontier_step", "target": "frontier"},
            )
        fallback = self._revealed_safe_fallback(typed_state)
        if fallback is not None:
            fallback = self._choose_non_repeating(typed_state, [fallback] + [action for action in typed_state.common.current_legal_actions if action != fallback])
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(fallback, parsed_state=parsed_state),
                annotations={"policy": "memory_hidden_solver", "family": family, "search_status": "fallback", "fallback": "revealed_safe_only"},
            )
        raise ValueError("ms01 invalid state: no proven-safe path, no locally consistent frontier move, and no revealed-safe fallback action")

    def _frontier_anchor_actions(self, state) -> tuple[int, ...]:
        frontier = [cell for cell in state.family.unrevealed_frontier_cells if cell not in set(state.family.forbidden_cells)]
        if not frontier:
            return ()
        safe = set(state.common.traversable_safe_cells)
        anchors = []
        for cell in frontier:
            for delta in _DELTAS.values():
                anchor = (cell[0] - delta[0], cell[1] - delta[1])
                if anchor in safe:
                    distance = abs(state.common.avatar_position[0] - anchor[0]) + abs(state.common.avatar_position[1] - anchor[1])
                    goal_bonus = 0
                    if state.family.goal_cell is not None:
                        goal_bonus = abs(state.family.goal_cell[0] - anchor[0]) + abs(state.family.goal_cell[1] - anchor[1])
                    anchors.append((goal_bonus, distance, anchor))
        if not anchors:
            return ()
        candidates: list[tuple[tuple[int, int], int]] = []
        for _, _, anchor in sorted(anchors):
            outcome = self.search.path_within_safe_region(state, {anchor})
            if outcome.status == "found" and outcome.plan:
                candidates.append((anchor, outcome.plan[0]))
        if not candidates:
            return ()
        ordered = sorted(
            candidates,
            key=lambda item: (
                1 if item[0] in self._recent_frontier_anchors else 0,
                1 if item[0] == self._last_frontier_anchor else 0,
                item[0],
                item[1],
            ),
        )
        self._last_frontier_anchor = ordered[0][0]
        self._recent_frontier_anchors.append(self._last_frontier_anchor)
        return tuple(action for _, action in ordered)

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

    def _frontier_step_action(self, state) -> int | None:
        frontier = set(state.family.unrevealed_frontier_cells) - set(state.family.forbidden_cells)
        if not frontier:
            return None
        for action_id, delta in _DELTAS.items():
            if action_id not in state.common.current_legal_actions:
                continue
            nxt = (state.common.avatar_position[0] + delta[0], state.common.avatar_position[1] + delta[1])
            if nxt in frontier:
                return action_id
        return None

    def _choose_non_repeating(self, state, candidates: list[int]) -> int:
        filtered = [int(action) for action in candidates if int(action) in state.common.current_legal_actions]
        if not filtered:
            raise ValueError("ms01 invalid state: no legal action candidates remain")
        state_key = state.to_key()
        if state_key == self._last_state_key and self._last_action in filtered and len(filtered) > 1:
            filtered = [action for action in filtered if action != self._last_action] + [self._last_action]
        chosen = filtered[0]
        self._last_state_key = state_key
        self._last_action = chosen
        return chosen
