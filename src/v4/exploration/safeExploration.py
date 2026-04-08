from __future__ import annotations

from v4.planning.planContracts import CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


class SafeExplorationFilterV4:
    def allow(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> bool:
        if not candidate.action_prefix:
            return False
        if parsed_state.belief_reference is None:
            return False
        if parsed_state.belief_reference.unknown_cell_count <= 0:
            return False
        if candidate.goal_kind != "reveal_information":
            return False
        first_action = candidate.action_prefix[0]
        action_key = f"{first_action.action_id}:{first_action.action_name}"
        if parsed_state.derived_control.retry_counts.get(action_key, 0) >= 2:
            return False
        if action_key in parsed_state.derived_control.cooldown_action_keys:
            return False
        return True
