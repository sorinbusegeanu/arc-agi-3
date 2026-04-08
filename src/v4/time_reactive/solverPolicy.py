from __future__ import annotations

from collections import deque

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .search import TimeReactiveSearchV4
from .stateBuilder import TimeReactiveStateBuilderV4
from .transitionModel import TimeReactiveTransitionModelV4


class TimeReactiveSolverPolicyV4(PolicyBaseV4):
    def __init__(
        self,
        *,
        state_builder: TimeReactiveStateBuilderV4 | None = None,
        transition_model: TimeReactiveTransitionModelV4 | None = None,
        search: TimeReactiveSearchV4 | None = None,
        search_bound: int = 8,
    ) -> None:
        self.state_builder = state_builder if state_builder is not None else TimeReactiveStateBuilderV4()
        self.transition_model = transition_model if transition_model is not None else TimeReactiveTransitionModelV4()
        self.search = search if search is not None else TimeReactiveSearchV4(self.transition_model)
        self.search_bound = int(search_bound)
        self._last_state_key: str | None = None
        self._last_action: int | None = None
        self._recent_successor_keys: deque[str] = deque(maxlen=8)

    @staticmethod
    def _annotate_search_failure(exc: Exception, *, missing_field: str) -> Exception:
        setattr(exc, "abort_site", "policy.decide")
        setattr(exc, "missing_field", missing_field)
        return exc

    @staticmethod
    def _candidate_identity(
        *,
        target_locator,
        primary_target_kind,
        mode_hint,
        hazard_window,
    ) -> str:
        parts: list[str] = []
        if target_locator not in (None, "", {}):
            parts.append(f"target_locator={target_locator}")
        if primary_target_kind not in (None, ""):
            parts.append(f"primary_target_kind={primary_target_kind}")
        if mode_hint not in (None, ""):
            parts.append(f"mode_hint={mode_hint}")
        if hazard_window is not None:
            parts.append(f"hazard_window={hazard_window}")
        return "|".join(parts)

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="sv01")
        outcome = self.search.search(
            state,
            goal_predicate=lambda typed_state: typed_state.common.terminal_status == "success",
            max_depth=self.search_bound,
        )
        if outcome.status == "found" and outcome.plan:
            selected_identity = self._candidate_identity(
                target_locator=None,
                primary_target_kind="preserve_safety_margin",
                mode_hint=outcome.status,
                hazard_window=state.family.survival_timer_remaining,
            )
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(outcome.plan[0], parsed_state=parsed_state),
                annotations={
                    "policy": "time_reactive_solver",
                    "family": "sv01",
                    "search_status": outcome.status,
                    "primary_target_kind": "preserve_safety_margin",
                    "target_locator": None,
                    "route_or_plan_size": len(outcome.plan),
                    "mode_hint": outcome.status,
                    "hazard_window": state.family.survival_timer_remaining,
                    "candidate_count_before_filter": 1,
                    "candidate_count_after_filter": 1,
                    "candidate_count_after_ranking": 1,
                    "rejection_reason_counts": {},
                    "selected_candidate_rank": 0,
                    "selected_candidate_score": float(len(outcome.plan)),
                    "candidate_identity_list_before_filter": [selected_identity],
                    "candidate_identity_list_after_filter": [selected_identity],
                    "candidate_identity_list_after_ranking": [selected_identity],
                    "selected_candidate_identity": selected_identity,
                },
            )
        safe_actions, failure_tag, failure_reason = self._safe_action_candidates(state)
        if safe_actions:
            before_filter_identities = [
                self._candidate_identity(
                    target_locator=self.transition_model.apply(state, action_id)[0].common.avatar_position,
                    primary_target_kind="preserve_safety_margin",
                    mode_hint="bounded_safe",
                    hazard_window=state.family.survival_timer_remaining,
                )
                for action_id in safe_actions
            ]
            ranked = sorted(safe_actions, key=lambda action_id: self._bounded_safe_rank(state, action_id))
            state_key = state.to_key()
            if state_key == self._last_state_key and self._last_action in ranked and len(ranked) > 1:
                ranked = [action for action in ranked if action != self._last_action] + [self._last_action]
            after_filter_identities = [
                self._candidate_identity(
                    target_locator=self.transition_model.apply(state, action_id)[0].common.avatar_position,
                    primary_target_kind="preserve_safety_margin",
                    mode_hint="bounded_safe",
                    hazard_window=state.family.survival_timer_remaining,
                )
                for action_id in ranked
            ]
            ranked = sorted(ranked, key=lambda action_id: self._successor_repeat_penalty(state, action_id))
            after_ranking_identities = [
                self._candidate_identity(
                    target_locator=self.transition_model.apply(state, action_id)[0].common.avatar_position,
                    primary_target_kind="preserve_safety_margin",
                    mode_hint="bounded_safe",
                    hazard_window=state.family.survival_timer_remaining,
                )
                for action_id in ranked
            ]
            preferred = ranked[0]
            self._last_state_key = state_key
            self._last_action = preferred
            successor, _ = self.transition_model.apply(state, preferred)
            self._recent_successor_keys.append(successor.to_key())
            selected_identity = self._candidate_identity(
                target_locator=successor.common.avatar_position,
                primary_target_kind="preserve_safety_margin",
                mode_hint="bounded_safe",
                hazard_window=state.family.survival_timer_remaining,
            )
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(preferred, parsed_state=parsed_state),
                annotations={
                    "policy": "time_reactive_solver",
                    "family": "sv01",
                    "search_status": outcome.status,
                    "fallback": "bounded_safe",
                    "branch_count": len(safe_actions),
                    "primary_target_kind": "preserve_safety_margin",
                    "target_locator": successor.common.avatar_position,
                    "route_or_plan_size": len(safe_actions),
                    "mode_hint": "bounded_safe",
                    "hazard_window": state.family.survival_timer_remaining,
                    "candidate_count_before_filter": len(safe_actions),
                    "candidate_count_after_filter": len(safe_actions),
                    "candidate_count_after_ranking": len(ranked),
                    "rejection_reason_counts": {},
                    "selected_candidate_rank": 0,
                    "selected_candidate_score": float(len(safe_actions)),
                    "candidate_identity_list_before_filter": before_filter_identities,
                    "candidate_identity_list_after_filter": after_filter_identities,
                    "candidate_identity_list_after_ranking": after_ranking_identities,
                    "selected_candidate_identity": selected_identity,
                },
            )
        raise self._annotate_search_failure(
            ValueError(f"sv01 bounded-safe action unavailable: {failure_reason}"),
            missing_field=failure_tag,
        )

    def _bounded_safe_rank(self, state, action_id: int) -> tuple[int, int, int]:
        successor, _ = self.transition_model.apply(state, action_id)
        wait_penalty = 0 if action_id == 5 and state.family.hunger_value < 40 else 1
        survival_penalty = -int(successor.family.survival_timer_remaining)
        resource_penalty = -(int(successor.family.hunger_value) + int(successor.family.warmth_value))
        return (wait_penalty, survival_penalty, resource_penalty)

    def _safe_action_candidates(self, state) -> tuple[list[int], str, str]:
        safe_actions: list[int] = []
        repeat_only_actions: list[int] = []
        legal_actions = [int(action_id) for action_id in state.common.current_legal_actions]
        wait_action_id = state.family.wait_action_id
        saw_nonfailure = False
        for action_id in legal_actions:
            successor, _ = self.transition_model.apply(state, action_id)
            if successor.common.terminal_status == "failure":
                continue
            saw_nonfailure = True
            if successor.to_key() in self._recent_successor_keys:
                repeat_only_actions.append(action_id)
                continue
            safe_actions.append(action_id)
        if safe_actions:
            return safe_actions, "", ""
        if repeat_only_actions:
            return repeat_only_actions, "all_safe_moves_pruned_by_repeat_guard", "all safe moves pruned by repeat guard"
        if wait_action_id is not None and int(wait_action_id) not in legal_actions:
            return [], "wait_action_unavailable", "wait action unavailable in current legal set"
        if saw_nonfailure:
            return [], "all_safe_moves_pruned_by_repeat_guard", "all safe moves pruned by repeat guard"
        if legal_actions:
            return [], "bounded_horizon_infeasible", "bounded horizon infeasible"
        return [], "no_legal_safe_move", "no legal safe move"

    def _successor_repeat_penalty(self, state, action_id: int) -> tuple[int, int]:
        successor, _ = self.transition_model.apply(state, action_id)
        repeat_penalty = 1 if successor.to_key() in self._recent_successor_keys else 0
        return (repeat_penalty, int(action_id))
