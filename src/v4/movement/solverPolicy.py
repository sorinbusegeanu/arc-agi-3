from __future__ import annotations

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .heuristics import admissible_remaining_coverage_heuristic
from .search import MovementSearchV4
from .stateBuilder import MovementStateBuilderV4
from .transitionModel import MovementTransitionModelV4


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

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        typed_state = self.state_builder.build(parsed_state, family=family)
        family_bound = self._search_bound_for_family(family)
        algorithm = "astar" if family == "va01" else self.default_algorithm
        heuristic = admissible_remaining_coverage_heuristic if family == "va01" else None
        outcome = self.search.search(
            typed_state,
            goal_predicate=lambda state: state.common.terminal_status == "success",
            legal_action_generator=lambda state: state.common.current_legal_actions,
            algorithm=algorithm,
            max_depth=family_bound,
            heuristic=heuristic,
        )
        if outcome.status == "found" and outcome.plan:
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
        fallback_action_id = min(int(action_id) for action_id in parsed_state.available_actions)
        return PolicyDecisionV4(
            primitive_action=legal_action_from_id(fallback_action_id, parsed_state=parsed_state),
            annotations={
                "policy": "movement_solver",
                "family": family,
                "search_status": outcome.status,
                "failure_reason": outcome.failure_reason,
                "fallback": "lowest_legal_action",
            },
        )

    def _search_bound_for_family(self, family: str) -> int | None:
        if self.search_bound is None:
            return None
        if family == "tp01":
            return min(self.search_bound, 24)
        if family == "ic01":
            return min(self.search_bound, 16)
        if family == "va01":
            return min(self.search_bound, 16)
        if family == "pb01":
            return min(self.search_bound, 16)
        return self.search_bound
