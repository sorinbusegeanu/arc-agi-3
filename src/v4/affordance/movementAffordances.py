from __future__ import annotations

from v4.agentContract.types import V4Action
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4


def build_movement_candidate_actions(parsed_state: ParsedStateV4) -> tuple[V4Action, ...]:
    action_ids = sorted(int(action_id) for action_id in parsed_state.available_actions)[:4]
    return tuple(legal_action_from_id(action_id, parsed_state=parsed_state) for action_id in action_ids)
