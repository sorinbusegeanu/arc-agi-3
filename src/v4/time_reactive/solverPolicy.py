from __future__ import annotations

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

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="sv01")
        outcome = self.search.search(
            state,
            goal_predicate=lambda typed_state: typed_state.common.terminal_status == "success",
            max_depth=self.search_bound,
        )
        if outcome.status == "found" and outcome.plan:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(outcome.plan[0], parsed_state=parsed_state),
                annotations={"policy": "time_reactive_solver", "family": "sv01", "search_status": outcome.status},
            )
        safe_actions = []
        for action_id in state.common.current_legal_actions:
            successor, _ = self.transition_model.apply(state, action_id)
            if successor.common.terminal_status != "failure":
                safe_actions.append(action_id)
        if safe_actions:
            preferred = 5 if 5 in safe_actions and state.family.hunger_value < 40 else safe_actions[0]
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(preferred, parsed_state=parsed_state),
                annotations={"policy": "time_reactive_solver", "family": "sv01", "search_status": outcome.status, "fallback": "bounded_safe"},
            )
        raise ValueError("sv01 invalid-state abort: no bounded-safe action exists")
