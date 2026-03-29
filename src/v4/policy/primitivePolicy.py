from __future__ import annotations

from .policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4


class PrimitivePolicyV4(PolicyBaseV4):
    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        if not parsed_state.available_actions:
            raise ValueError("no available actions for primitive policy")
        chosen_action_id = min(int(action_id) for action_id in parsed_state.available_actions)
        action = legal_action_from_id(chosen_action_id, parsed_state=parsed_state)
        return PolicyDecisionV4(
            primitive_action=action,
            annotations={"policy": "primitive", "decision_mode": "deterministic_lowest_legal"},
        )
