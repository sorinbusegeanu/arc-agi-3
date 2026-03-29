from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.agentContract.types import V4Action
from v4.agentContract.validators import validate_v4_action
from v4.state.parsedState import ParsedStateV4

_ACTION_NAME_BY_ID = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


def legal_action_from_id(
    action_id: int,
    *,
    parsed_state: ParsedStateV4,
    payload: dict[str, Any] | None = None,
) -> V4Action:
    action = V4Action(action_id=int(action_id), action_name=_ACTION_NAME_BY_ID[int(action_id)], payload=payload)
    validate_v4_action(action, observation=parsed_state.current_observation)
    return action


@dataclass(frozen=True)
class PolicyDecisionV4:
    primitive_action: V4Action | None = None
    short_plan: tuple[V4Action, ...] = ()
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_primitive = self.primitive_action is not None
        has_plan = len(self.short_plan) > 0
        if has_primitive == has_plan:
            raise ValueError("PolicyDecisionV4 must contain exactly one primitive action or one short plan")
        if has_plan and len(self.short_plan) > 4:
            raise ValueError("short_plan must remain small and interruptible")

    def first_action(self) -> V4Action:
        if self.primitive_action is not None:
            return self.primitive_action
        return self.short_plan[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyBaseV4:
    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        raise NotImplementedError
