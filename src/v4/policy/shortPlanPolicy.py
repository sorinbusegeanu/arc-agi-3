from __future__ import annotations

from .policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4


class ShortPlanPolicyV4(PolicyBaseV4):
    def __init__(self, *, max_plan_length: int = 3) -> None:
        resolved = int(max_plan_length)
        if resolved <= 0:
            raise ValueError("max_plan_length must be greater than 0")
        self.max_plan_length = min(resolved, 4)

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        if not parsed_state.available_actions:
            raise ValueError("no available actions for short plan policy")
        action_ids = sorted(int(action_id) for action_id in parsed_state.available_actions)
        plan = tuple(
            legal_action_from_id(action_id, parsed_state=parsed_state)
            for action_id in action_ids[: self.max_plan_length]
        )
        return PolicyDecisionV4(
            short_plan=plan,
            annotations={"policy": "short_plan", "max_plan_length": self.max_plan_length},
        )
