from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from v4.agentContract.types import V4StepResult


class StopReasonV4(str, Enum):
    CONTINUE = "continue"
    TERMINAL_WIN = "terminal_win"
    TERMINAL_FAIL = "terminal_fail"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    INVALID_STATE_ABORT = "invalid_state_abort"


@dataclass(frozen=True)
class StopConditionStatusV4:
    should_stop: bool
    reason: StopReasonV4
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason"] = self.reason.value
        return payload


def evaluate_stop_conditions(
    *,
    step_result: V4StepResult | None,
    next_step_index: int,
    max_steps: int,
    invalid_state_abort: bool = False,
) -> StopConditionStatusV4:
    if invalid_state_abort:
        return StopConditionStatusV4(True, StopReasonV4.INVALID_STATE_ABORT)
    if step_result is not None:
        if step_result.terminal_signal.status == "success":
            return StopConditionStatusV4(True, StopReasonV4.TERMINAL_WIN)
        if step_result.terminal_signal.status == "failure":
            return StopConditionStatusV4(True, StopReasonV4.TERMINAL_FAIL)
    if next_step_index >= int(max_steps):
        return StopConditionStatusV4(True, StopReasonV4.STEP_BUDGET_EXHAUSTED)
    return StopConditionStatusV4(False, StopReasonV4.CONTINUE)
