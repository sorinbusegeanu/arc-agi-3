from __future__ import annotations

from collections import deque

from v4.agentContract.types import V4Action
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
        self._tb01_last_state_key: str | None = None
        self._tb01_last_bridge_cell: tuple[int, int] | None = None
        self._tb01_repeat_streak: int = 0
        self._tb01_recent_bridge_cells: list[tuple[int, int]] = []

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="tb01")
        state_key = self._repeat_key(state)
        if state_key == self._tb01_last_state_key:
            self._tb01_repeat_streak += 1
        else:
            self._tb01_repeat_streak = 0
            self._tb01_last_state_key = state_key
        move_path = _movement_path(state)
        if move_path:
            selected_identity = self._candidate_identity(
                target_locator=state.family.goal_cell,
                bridge_anchor=state.common.avatar_position,
                bridge_target=state.family.goal_cell,
                construction_target=state.family.goal_cell,
                mode_hint="movement_path",
            )
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(move_path[0], parsed_state=parsed_state),
                annotations={
                    "policy": "hybrid_construction_solver",
                    "family": "tb01",
                    "search_status": "movement_path",
                    "primary_target_kind": "enable_construction_path",
                    "target_locator": state.family.goal_cell,
                    "route_or_plan_size": len(move_path),
                    "mode_hint": "movement_path",
                    "bridge_anchor": state.common.avatar_position,
                    "bridge_target": state.family.goal_cell,
                    "construction_target": state.family.goal_cell,
                    "candidate_count_before_filter": 1,
                    "candidate_count_after_filter": 1,
                    "candidate_count_after_ranking": 1,
                    "rejection_reason_counts": {},
                    "selected_candidate_rank": 0,
                    "selected_candidate_score": float(len(move_path)),
                    "candidate_identity_list_before_filter": [selected_identity],
                    "candidate_identity_list_after_filter": [selected_identity],
                    "candidate_identity_list_after_ranking": [selected_identity],
                    "selected_candidate_identity": selected_identity,
                },
            )
        if self._tb01_repeat_streak > 0 and state.family.goal_cell is not None and state.family.legal_movement_actions:
            gx, gy = state.family.goal_cell
            ranked_moves = sorted(
                state.family.legal_movement_actions,
                key=lambda action_id: self._move_rank(state, action_id, gx, gy),
            )
            if ranked_moves:
                ranked_move_identities = [
                    self._candidate_identity(
                        target_locator=state.family.goal_cell,
                        bridge_anchor=state.common.avatar_position,
                        bridge_target=state.family.goal_cell,
                        construction_target=state.family.goal_cell,
                        mode_hint="movement_repeat_break",
                    )
                    for _ in ranked_moves
                ]
                selected_identity = ranked_move_identities[0] if ranked_move_identities else None
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(ranked_moves[0], parsed_state=parsed_state),
                    annotations={
                        "policy": "hybrid_construction_solver",
                        "family": "tb01",
                        "search_status": "movement_repeat_break",
                        "primary_target_kind": "enable_construction_path",
                        "target_locator": state.family.goal_cell,
                        "route_or_plan_size": len(ranked_moves),
                        "mode_hint": "movement_repeat_break",
                        "bridge_anchor": state.common.avatar_position,
                        "bridge_target": state.family.goal_cell,
                        "construction_target": state.family.goal_cell,
                        "candidate_count_before_filter": len(ranked_moves),
                        "candidate_count_after_filter": len(ranked_moves),
                        "candidate_count_after_ranking": len(ranked_moves),
                        "rejection_reason_counts": {},
                        "selected_candidate_rank": 0,
                        "selected_candidate_score": float(len(ranked_moves)),
                        "candidate_identity_list_before_filter": list(ranked_move_identities),
                        "candidate_identity_list_after_filter": list(ranked_move_identities),
                        "candidate_identity_list_after_ranking": list(ranked_move_identities),
                        "selected_candidate_identity": selected_identity,
                    },
                )
        if state.family.goal_cell is not None and state.family.legal_click_cells:
            ax, ay = state.common.avatar_position
            gx, gy = state.family.goal_cell
            ranked = sorted(
                state.family.legal_click_cells,
                key=lambda cell: self._bridge_rank(state, cell, gx, gy, ax, ay),
            )
            before_filter_identities = [
                self._candidate_identity(
                    target_locator=cell,
                    bridge_anchor=state.common.avatar_position,
                    bridge_target=cell,
                    construction_target=cell,
                    mode_hint="goal_ranked_bridge",
                )
                for cell in ranked
            ]
            if self._tb01_repeat_streak > 0 and self._tb01_last_bridge_cell in ranked and len(ranked) > 1:
                ranked = [cell for cell in ranked if cell != self._tb01_last_bridge_cell] + [self._tb01_last_bridge_cell]
            if self._tb01_repeat_streak > 0:
                filtered = [cell for cell in ranked if cell not in self._tb01_recent_bridge_cells]
                if filtered:
                    ranked = filtered + [cell for cell in ranked if cell in self._tb01_recent_bridge_cells]
            after_filter_identities = [
                self._candidate_identity(
                    target_locator=cell,
                    bridge_anchor=state.common.avatar_position,
                    bridge_target=cell,
                    construction_target=cell,
                    mode_hint="goal_ranked_bridge",
                )
                for cell in ranked
            ]
            cell = ranked[0]
            self._tb01_last_bridge_cell = cell
            self._tb01_recent_bridge_cells.append(cell)
            self._tb01_recent_bridge_cells = self._tb01_recent_bridge_cells[-6:]
            payload = _bridge_payload(state, cell)
            selected_identity = self._candidate_identity(
                target_locator=cell,
                bridge_anchor=state.common.avatar_position,
                bridge_target=cell,
                construction_target=cell,
                mode_hint="goal_ranked_bridge",
            )
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=payload),
                annotations={
                    "policy": "hybrid_construction_solver",
                    "family": "tb01",
                    "search_status": "bridge_greedy",
                    "fallback": "goal_ranked_bridge",
                    "primary_target_kind": "enable_construction_path",
                    "target_locator": cell,
                    "route_or_plan_size": len(ranked),
                    "mode_hint": "goal_ranked_bridge",
                    "bridge_anchor": state.common.avatar_position,
                    "bridge_target": cell,
                    "construction_target": cell,
                    "candidate_count_before_filter": len(state.family.legal_click_cells),
                    "candidate_count_after_filter": len(ranked),
                    "candidate_count_after_ranking": len(ranked),
                    "rejection_reason_counts": {},
                    "selected_candidate_rank": 0,
                    "selected_candidate_score": float(len(ranked)),
                    "candidate_identity_list_before_filter": before_filter_identities,
                    "candidate_identity_list_after_filter": after_filter_identities,
                    "candidate_identity_list_after_ranking": list(after_filter_identities),
                    "selected_candidate_identity": selected_identity,
                },
            )
        selected_identity = self._candidate_identity(
            target_locator=state.family.goal_cell,
            bridge_anchor=state.common.avatar_position,
            bridge_target=state.family.goal_cell,
            construction_target=state.family.goal_cell,
            mode_hint="lowest_move",
        )
        return PolicyDecisionV4(
            primitive_action=legal_action_from_id(1, parsed_state=parsed_state),
            annotations={
                "policy": "hybrid_construction_solver",
                "family": "tb01",
                "search_status": "exhausted",
                "fallback": "lowest_move",
                "primary_target_kind": "enable_construction_path",
                "target_locator": state.family.goal_cell,
                "route_or_plan_size": len(state.family.legal_movement_actions),
                "mode_hint": "lowest_move",
                "bridge_anchor": state.common.avatar_position,
                "bridge_target": state.family.goal_cell,
                "construction_target": state.family.goal_cell,
                "candidate_count_before_filter": len(state.family.legal_movement_actions),
                "candidate_count_after_filter": len(state.family.legal_movement_actions),
                "candidate_count_after_ranking": len(state.family.legal_movement_actions),
                "rejection_reason_counts": {},
                "selected_candidate_rank": 0,
                "selected_candidate_score": float(len(state.family.legal_movement_actions)),
                "candidate_identity_list_before_filter": [selected_identity] if state.family.legal_movement_actions else [],
                "candidate_identity_list_after_filter": [selected_identity] if state.family.legal_movement_actions else [],
                "candidate_identity_list_after_ranking": [selected_identity] if state.family.legal_movement_actions else [],
                "selected_candidate_identity": selected_identity if state.family.legal_movement_actions else None,
            },
        )

    @staticmethod
    def _candidate_identity(
        *,
        target_locator,
        bridge_anchor,
        bridge_target,
        construction_target,
        mode_hint,
    ) -> str:
        parts: list[str] = []
        if target_locator not in (None, "", {}):
            parts.append(f"target_locator={target_locator}")
        if bridge_anchor not in (None, "", {}):
            parts.append(f"bridge_anchor={bridge_anchor}")
        if bridge_target not in (None, "", {}):
            parts.append(f"bridge_target={bridge_target}")
        if construction_target not in (None, "", {}):
            parts.append(f"construction_target={construction_target}")
        if mode_hint not in (None, ""):
            parts.append(f"mode_hint={mode_hint}")
        return "|".join(parts)

    def _move_rank(self, state, action_id: int, gx: int, gy: int) -> tuple[int, int]:
        delta = _MOVE_DELTAS.get(int(action_id), (0, 0))
        nxt = (state.common.avatar_position[0] + delta[0], state.common.avatar_position[1] + delta[1])
        traversable = set(state.family.land_cells) | set(state.family.bridge_built_cells)
        blocked = 1 if nxt not in traversable else 0
        distance = abs(nxt[0] - gx) + abs(nxt[1] - gy)
        return (blocked, distance)

    def _bridge_rank(self, state, cell: tuple[int, int], gx: int, gy: int, ax: int, ay: int) -> tuple[int, int, int, tuple[int, int]]:
        payload = _bridge_payload(state, cell)
        action = V4Action(action_id=6, action_name="ACTION6", payload=payload)
        successor, _ = self.transition_model.apply(state, action)
        successor_path = _movement_path(successor)
        creates_path = 0 if successor_path else 1
        goal_distance = abs(cell[0] - gx) + abs(cell[1] - gy)
        avatar_distance = abs(cell[0] - ax) + abs(cell[1] - ay)
        return (creates_path, goal_distance, avatar_distance, cell)

    @staticmethod
    def _repeat_key(state) -> tuple[object, ...]:
        return (
            state.common.avatar_position,
            tuple(sorted(state.family.bridge_built_cells)),
            tuple(sorted(state.family.legal_click_cells)),
            state.family.bridge_budget_remaining,
            state.family.goal_cell,
        )
