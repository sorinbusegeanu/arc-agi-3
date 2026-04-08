from __future__ import annotations

from collections import OrderedDict

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .heuristics import admissible_push_goal_heuristic, admissible_remaining_coverage_heuristic
from .search import MovementSearchV4
from .stateBuilder import MovementStateBuilderV4
from .transitionModel import MovementTransitionModelV4
from .typedState import MovementTypedStateV4


class MovementSolverPolicyV4(PolicyBaseV4):
    def __init__(
        self,
        *,
        state_builder: MovementStateBuilderV4 | None = None,
        transition_model: MovementTransitionModelV4 | None = None,
        search: MovementSearchV4 | None = None,
        max_plan_prefix: int = 4,
        default_algorithm: str = "bfs",
        search_bound: int | None = 24,
    ) -> None:
        self.state_builder = state_builder if state_builder is not None else MovementStateBuilderV4()
        self.transition_model = transition_model if transition_model is not None else MovementTransitionModelV4()
        self.search = search if search is not None else MovementSearchV4(self.transition_model)
        self.max_plan_prefix = int(max_plan_prefix)
        self.default_algorithm = default_algorithm
        self.search_bound = search_bound
        self._episode_cache: OrderedDict[str, MovementTypedStateV4] = OrderedDict()
        self._family_repeat_state: dict[str, str] = {}
        self._family_repeat_action: dict[str, int] = {}
        self._family_repeat_streak: dict[str, int] = {}
        self._family_cached_plan: dict[str, list[int]] = {}
        self._family_cached_plan_states: dict[str, list[str]] = {}
        self._family_last_found_plan: dict[str, list[int]] = {}

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        carry_state = self._episode_cache.get(self._cache_key(parsed_state)) if family in {"fs01", "fs02", "fs03", "pb01", "pb02", "pb03"} else None
        typed_state = self.state_builder.build(parsed_state, family=family, carry_state=carry_state)
        if family in {"fs01", "fs02", "fs03", "pb01", "pb02", "pb03"}:
            self._remember_state(parsed_state, typed_state)
        cached_action_id = self._consume_cached_plan_action(parsed_state, typed_state, family)
        if cached_action_id is not None:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(cached_action_id, parsed_state=parsed_state),
                annotations={"policy": "movement_solver", "family": family, "search_status": "cached_plan", "fallback": "certifying_prefix"},
            )
        if family == "va01":
            action_id = self._fallback_action_id(parsed_state, typed_state, family)
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(action_id, parsed_state=parsed_state),
                annotations={"policy": "movement_solver", "family": family, "search_status": "greedy_coverage", "fallback": "coverage_ranked_move"},
            )
        family_bound = self._search_bound_for_family(family)
        algorithm = "astar" if family in {"va01", "pb01", "pb02", "pb03"} else self.default_algorithm
        heuristic = None
        if family == "va01":
            heuristic = admissible_remaining_coverage_heuristic
        elif family in {"pb01", "pb02", "pb03"}:
            heuristic = admissible_push_goal_heuristic
        outcome = self.search.search(
            typed_state,
            goal_predicate=lambda state: self._goal_reached(state, family),
            legal_action_generator=lambda state: state.common.current_legal_actions,
            algorithm=algorithm,
            max_depth=family_bound,
            heuristic=heuristic,
        )
        if outcome.status == "found" and outcome.plan:
            self._remember_cached_plan(typed_state, family, outcome.plan)
            if family in {"fs02", "ic01"}:
                self._family_last_found_plan[family] = [int(action_id) for action_id in outcome.plan]
            prefix = outcome.plan[: max(1, min(self.max_plan_prefix, 4))]
            if len(prefix) == 1:
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(prefix[0], parsed_state=parsed_state),
                    annotations={"policy": "movement_solver", "family": family, "search_status": outcome.status, "plan_length": len(outcome.plan)},
                )
            return PolicyDecisionV4(
                short_plan=tuple(legal_action_from_id(action_id, parsed_state=parsed_state) for action_id in prefix),
                annotations={"policy": "movement_solver", "family": family, "search_status": outcome.status, "plan_length": len(outcome.plan)},
            )
        if outcome.status == "found":
            if family in {"fs02", "ic01"}:
                cached = self._family_last_found_plan.get(family) or []
                if cached:
                    next_action = int(cached[0])
                    if next_action in [int(action_id) for action_id in parsed_state.available_actions]:
                        if len(cached) > 1:
                            self._family_last_found_plan[family] = cached[1:]
                        else:
                            self._family_last_found_plan.pop(family, None)
                        return PolicyDecisionV4(
                            primitive_action=legal_action_from_id(next_action, parsed_state=parsed_state),
                            annotations={
                                "policy": "movement_solver",
                                "family": family,
                                "search_status": "cached_plan",
                                "failure_reason": "goal_seen_but_prefix_exhausted",
                                "fallback": "last_found_prefix",
                            },
                        )
            fallback_action_id = self._fallback_action_id(parsed_state, typed_state, family)
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(fallback_action_id, parsed_state=parsed_state),
                annotations={
                    "policy": "movement_solver",
                    "family": family,
                    "search_status": "goal_ready",
                    "fallback": "goal_ready_advance",
                    "failure_reason": "goal_state_reached_but_search_returned_empty_plan",
                },
            )
        fallback_action_id = self._fallback_action_id(parsed_state, typed_state, family)
        failure_reason = outcome.failure_reason
        if outcome.status in {"exhausted", "bound_exhausted"} and family in {"fs01", "fs02", "ic01"} and "goal state reached" in str(outcome.failure_reason or ""):
            failure_reason = "goal_seen_but_plan_extraction_failed"
        if family in {"fs02", "ic01"}:
            cached = self._family_last_found_plan.get(family) or []
            if cached:
                next_action = int(cached[0])
                if next_action in [int(action_id) for action_id in parsed_state.available_actions]:
                    if len(cached) > 1:
                        self._family_last_found_plan[family] = cached[1:]
                    else:
                        self._family_last_found_plan.pop(family, None)
                    return PolicyDecisionV4(
                        primitive_action=legal_action_from_id(next_action, parsed_state=parsed_state),
                        annotations={
                            "policy": "movement_solver",
                            "family": family,
                            "search_status": outcome.status,
                            "failure_reason": "certifying_prefix_lost_on_replan" if family == "fs02" else "goal_state_revisited_without_extractable_prefix",
                            "fallback": "last_found_prefix",
                        },
                    )
        return PolicyDecisionV4(
            primitive_action=legal_action_from_id(fallback_action_id, parsed_state=parsed_state),
            annotations={
                "policy": "movement_solver",
                "family": family,
                "search_status": outcome.status,
                "failure_reason": failure_reason,
                "fallback": "lowest_legal_action",
            },
        )

    def _cache_key(self, parsed_state: ParsedStateV4) -> str:
        observation = parsed_state.current_observation
        return f"{observation.game_id}:{observation.levels_completed}"

    def _remember_state(self, parsed_state: ParsedStateV4, typed_state: MovementTypedStateV4) -> None:
        key = self._cache_key(parsed_state)
        self._episode_cache[key] = typed_state
        self._episode_cache.move_to_end(key)
        while len(self._episode_cache) > 8:
            self._episode_cache.popitem(last=False)

    def _search_bound_for_family(self, family: str) -> int | None:
        if self.search_bound is None:
            return None
        if family == "tp01":
            return min(self.search_bound, 24)
        if family == "ic01":
            return min(self.search_bound, 16)
        if family == "va01":
            return min(self.search_bound, 16)
        if family in {"fs01", "fs02", "fs03"}:
            return min(self.search_bound, 32)
        if family in {"pb01", "pb03"}:
            return min(self.search_bound, 16)
        if family == "pb02":
            return min(self.search_bound, 28)
        return self.search_bound

    def _goal_reached(self, state: MovementTypedStateV4, family: str) -> bool:
        if state.common.terminal_status == "success":
            return True
        if family == "ic01" and state.common.target_cells:
            return state.common.avatar_position in set(state.common.target_cells)
        return False

    def _consume_cached_plan_action(self, parsed_state: ParsedStateV4, typed_state: MovementTypedStateV4, family: str) -> int | None:
        if family not in {"fs01", "fs02", "fs03", "ic01"}:
            return None
        state_key = self.search._state_key(typed_state)
        cached = self._family_cached_plan.get(family) or []
        expected_states = self._family_cached_plan_states.get(family) or []
        if not cached:
            return None
        if expected_states:
            if state_key in expected_states[:-1]:
                index = expected_states.index(state_key)
                if index > 0:
                    cached = cached[index:]
                    expected_states = expected_states[index:]
                    self._family_cached_plan[family] = list(cached)
                    self._family_cached_plan_states[family] = list(expected_states)
            elif family == "fs02":
                self._family_cached_plan.pop(family, None)
                self._family_cached_plan_states.pop(family, None)
                return None
        allowed = set(int(action_id) for action_id in parsed_state.available_actions)
        while cached:
            candidate = int(cached[0])
            if candidate not in allowed:
                self._family_cached_plan.pop(family, None)
                self._family_cached_plan_states.pop(family, None)
                return None
            cached.pop(0)
            if expected_states:
                expected_states.pop(0)
            if cached:
                self._family_cached_plan[family] = cached
                if expected_states:
                    self._family_cached_plan_states[family] = expected_states
            else:
                self._family_cached_plan.pop(family, None)
                self._family_cached_plan_states.pop(family, None)
            return candidate
        return None

    def _remember_cached_plan(self, typed_state: MovementTypedStateV4, family: str, plan: tuple[int, ...]) -> None:
        if family not in {"fs01", "fs02", "fs03", "ic01"}:
            return
        cached_plan = [int(action_id) for action_id in plan]
        expected_states = [self.search._state_key(typed_state)]
        cursor = typed_state
        for action_id in cached_plan:
            cursor, _ = self.transition_model.apply(cursor, int(action_id))
            expected_states.append(self.search._state_key(cursor))
        self._family_cached_plan[family] = cached_plan
        self._family_cached_plan_states[family] = expected_states

    def _fallback_action_id(self, parsed_state: ParsedStateV4, typed_state: MovementTypedStateV4, family: str) -> int:
        legal = [int(action_id) for action_id in parsed_state.available_actions]
        if family not in {"fs01", "fs02", "fs03", "ic01", "va01"}:
            return min(legal)
        ranked = sorted(legal, key=lambda action_id: self._movement_fallback_rank(typed_state, action_id, family))
        state_key = self.search._state_key(typed_state)
        repeated_state = self._family_repeat_state.get(family) == state_key
        repeated_action = self._family_repeat_action.get(family)
        if repeated_state:
            self._family_repeat_streak[family] = int(self._family_repeat_streak.get(family, 0)) + 1
        else:
            self._family_repeat_streak[family] = 0
        if repeated_state and repeated_action in ranked and len(ranked) > 1:
            ranked = [action for action in ranked if action != repeated_action] + [repeated_action]
        chosen = ranked[0]
        self._family_repeat_state[family] = state_key
        self._family_repeat_action[family] = chosen
        return chosen

    def _movement_fallback_rank(self, typed_state: MovementTypedStateV4, action_id: int, family: str) -> tuple[int, int, int]:
        successor, annotation = self.transition_model.apply(typed_state, action_id)
        blocked_penalty = 1 if annotation.blocked else 0
        no_change_penalty = 1 if successor.common.avatar_position == typed_state.common.avatar_position else 0
        if family in {"fs01", "fs02", "fs03"}:
            activated_bits = int(successor.family.activated_switch_bits or 0)
            threshold = int(successor.family.switch_group_threshold or len(tuple(successor.family.switch_positions)) or 1)
            activated_count = activated_bits.bit_count()
            repeat_streak = int(self._family_repeat_streak.get(family, 0))
            if successor.family.door_open:
                distance = min(
                    abs(successor.common.avatar_position[0] - x) + abs(successor.common.avatar_position[1] - y)
                    for x, y in successor.common.target_cells
                ) if successor.common.target_cells else 0
                repeat_penalty = 1 if repeat_streak > 0 and int(action_id) == self._family_repeat_action.get(family) else 0
                return (blocked_penalty, no_change_penalty, repeat_penalty, 0, distance)
            switch_targets = [
                pos
                for index, pos in enumerate(successor.family.switch_positions)
                if not (activated_bits & (1 << index))
            ]
            distance = min(
                abs(successor.common.avatar_position[0] - x) + abs(successor.common.avatar_position[1] - y)
                for x, y in (switch_targets or successor.family.switch_positions)
            ) if successor.family.switch_positions else 0
            ready_penalty = 0 if activated_count >= threshold else 1
            repeat_penalty = 1 if repeat_streak > 1 and int(action_id) == self._family_repeat_action.get(family) else 0
            return (blocked_penalty, no_change_penalty, repeat_penalty, ready_penalty, distance)
        if family == "ic01":
            distance = min(
                abs(successor.common.avatar_position[0] - x) + abs(successor.common.avatar_position[1] - y)
                for x, y in successor.common.target_cells
            ) if successor.common.target_cells else 0
            return (blocked_penalty, no_change_penalty, distance, int(action_id))
        if family == "va01":
            coverage_gain = -len(set(successor.family.coverage_mask) - set(typed_state.family.coverage_mask))
            remaining = set(successor.family.coverage_eligible_cells) - set(successor.family.coverage_mask)
            distance = min(
                abs(successor.common.avatar_position[0] - x) + abs(successor.common.avatar_position[1] - y)
                for x, y in remaining
            ) if remaining else 0
            return (blocked_penalty, no_change_penalty, coverage_gain, distance)
        return (blocked_penalty, no_change_penalty, int(action_id))
