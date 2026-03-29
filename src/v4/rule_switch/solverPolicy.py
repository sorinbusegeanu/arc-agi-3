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

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        state = self.state_builder.build(parsed_state, family="rs01")
        outcome = self.search.search(
            state,
            goal_predicate=lambda typed_state: typed_state.common.terminal_status == "success",
            max_depth=self.search_bound,
        )
        if outcome.status == "found" and outcome.plan:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(outcome.plan[0], parsed_state=parsed_state),
                annotations={"policy": "rule_switch_solver", "family": "rs01", "search_status": outcome.status},
            )
        legal = self.search.legal_actions(state)
        if legal:
            return PolicyDecisionV4(
                primitive_action=legal_action_from_id(legal[0], parsed_state=parsed_state),
                annotations={"policy": "rule_switch_solver", "family": "rs01", "search_status": outcome.status, "fallback": "safe_color_filtered"},
            )
        raise ValueError("rs01 invalid-state abort: no safe-color legal action exists")
