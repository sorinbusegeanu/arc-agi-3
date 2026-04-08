from __future__ import annotations

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .search import RuleSwitchSearchV4
from .stateBuilder import RuleSwitchStateBuilderV4
from .transitionModel import RuleSwitchTransitionModelV4


class RuleSwitchSolverPolicyV4(PolicyBaseV4):
    def __init__(
        self,
        *,
        state_builder: RuleSwitchStateBuilderV4 | None = None,
        transition_model: RuleSwitchTransitionModelV4 | None = None,
        search: RuleSwitchSearchV4 | None = None,
        search_bound: int = 24,
    ) -> None:
        self.state_builder = state_builder if state_builder is not None else RuleSwitchStateBuilderV4()
        self.transition_model = transition_model if transition_model is not None else RuleSwitchTransitionModelV4()
        self.search = search if search is not None else RuleSwitchSearchV4(self.transition_model)
        self.search_bound = int(search_bound)
        self._last_state_key: str | None = None
        self._last_action: int | None = None
        self._cached_route: list[int] = []
        self._cached_route_states: list[str] = []

    @staticmethod
    def _annotate_search_failure(exc: Exception, *, missing_field: str) -> Exception:
        setattr(exc, "abort_site", "policy.decide")
        setattr(exc, "missing_field", missing_field)
        return exc

    def _safe_apply(self, state, action_id: int):
        try:
            return self.transition_model.apply(state, int(action_id))
        except ValueError:
            return None

    def _goal_ready_actions(self, state) -> tuple[int, ...]:
        actions = []
        for action_id in sorted(state.common.legal_action_ids):
            applied = self._safe_apply(state, int(action_id))
            if applied is None:
                continue
            successor, annotation = applied
            if annotation.blocked or successor.common.terminal_status == "failure":
                continue
            actions.append(int(action_id))
        return tuple(actions)

    def _nonfailure_actions(self, state) -> tuple[int, ...]:
        actions = list(self._goal_ready_actions(state))
        if actions:
            return tuple(actions)
        return tuple(int(action_id) for action_id in sorted(state.common.legal_action_ids))

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="rs01")
        state_key = state.to_key()
        if self._cached_route and self._cached_route_states:
            if state_key in self._cached_route_states[:-1]:
                index = self._cached_route_states.index(state_key)
                if index > 0:
                    self._cached_route = self._cached_route[index:]
                    self._cached_route_states = self._cached_route_states[index:]
            else:
                self._cached_route = []
                self._cached_route_states = []
        outcome = self.search.search(
            state,
            goal_predicate=self._goal_reached,
            max_depth=self.search_bound,
        )
        if self._cached_route:
            next_action = int(self._cached_route[0])
            if next_action in tuple(int(action_id) for action_id in state.common.legal_action_ids):
                self._cached_route = self._cached_route[1:]
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(next_action, parsed_state=parsed_state),
                    annotations={
                        "policy": "rule_switch_solver",
                        "family": "rs01",
                        "search_status": "cached_plan",
                        "fallback": "cached_safe_color_prefix",
                        "failure_reason": "success_path_found_but_transition_filter_pruned_all",
                    },
                )
        if outcome.status == "found" and outcome.plan:
            self._remember_cached_route(state, outcome.plan)
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(outcome.plan[0], parsed_state=parsed_state),
                annotations={"policy": "rule_switch_solver", "family": "rs01", "search_status": outcome.status},
            )
        if outcome.status == "found":
            legal = self._goal_ready_actions(state)
            if legal:
                chosen = legal[0]
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(chosen, parsed_state=parsed_state),
                    annotations={"policy": "rule_switch_solver", "family": "rs01", "search_status": "goal_ready", "fallback": "goal_ready_advance"},
                )
            if state.common.terminal_status == "success" and state.common.legal_action_ids:
                chosen = min(int(action_id) for action_id in state.common.legal_action_ids)
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(chosen, parsed_state=parsed_state),
                    annotations={
                        "policy": "rule_switch_solver",
                        "family": "rs01",
                        "search_status": "goal_ready",
                        "fallback": "goal_ready_any_legal",
                        "failure_reason": "success_state_reached_but_no_nonfailure_transition",
                    },
                )
        legal = self.search.legal_actions(state)
        if not legal and all(not positions for _, positions in state.family.remaining_targets_by_color):
            legal = self._goal_ready_actions(state)
        if not legal:
            legal = self._nonfailure_actions(state)
        if legal:
            ranked = list(legal)
            active_targets = ()
            for color, positions in state.family.remaining_targets_by_color:
                if color == state.family.active_safe_color:
                    active_targets = positions
                    break
            if active_targets:
                ranked = sorted(ranked, key=lambda action_id: self._action_rank(state, action_id, active_targets))
            state_key = state.to_key()
            if state_key == self._last_state_key and self._last_action in ranked and len(ranked) > 1:
                ranked = [action for action in ranked if action != self._last_action] + [self._last_action]
            chosen = ranked[0]
            self._last_state_key = state_key
            self._last_action = chosen
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(chosen, parsed_state=parsed_state),
                annotations={
                    "policy": "rule_switch_solver",
                    "family": "rs01",
                    "search_status": outcome.status,
                    "fallback": "safe_color_filtered",
                    "failure_reason": "success_path_found_but_transition_filter_pruned_all" if outcome.status == "found" else None,
                },
            )
        if all(not positions for _, positions in state.family.remaining_targets_by_color) and state.common.legal_action_ids:
            chosen = min(int(action_id) for action_id in state.common.legal_action_ids)
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(chosen, parsed_state=parsed_state),
                annotations={
                    "policy": "rule_switch_solver",
                    "family": "rs01",
                    "search_status": outcome.status,
                    "fallback": "success_any_legal",
                    "failure_reason": "success_state_reached_but_no_safe_filtered_action",
                },
            )
        raise self._annotate_search_failure(
            ValueError("rs01 safe-color action unavailable: no safe-color legal action exists"),
            missing_field="safe_color_action_set",
        )

    @staticmethod
    def _goal_reached(state) -> bool:
        if state.common.terminal_status == "success":
            return True
        active_safe = state.family.active_safe_color
        if active_safe is None:
            return False
        for color, positions in state.family.remaining_targets_by_color:
            if color == active_safe:
                return len(tuple(positions)) == 0
        return False

    def _action_rank(self, state, action_id: int, active_targets) -> tuple[int, int]:
        applied = self._safe_apply(state, action_id)
        if applied is None:
            return (10**6, int(action_id))
        successor, _ = applied
        distance = min(
            abs(successor.common.avatar_position[0] - x) + abs(successor.common.avatar_position[1] - y)
            for x, y in active_targets
        )
        return (distance, int(action_id))

    def _remember_cached_route(self, state, plan: tuple[int, ...]) -> None:
        cached_route = [int(action_id) for action_id in plan[1:]]
        route_states: list[str] = []
        cursor = state
        for action_id in plan:
            successor, _ = self.transition_model.apply(cursor, int(action_id))
            route_states.append(successor.to_key())
            cursor = successor
        self._cached_route = cached_route
        self._cached_route_states = route_states
