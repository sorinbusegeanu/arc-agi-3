from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.agentContract.types import V4Action, V4Observation, V4StepResult, V4TransitionRecord


@dataclass(frozen=True)
class StepLedgerRecordV4:
    step_index: int
    pre_observation: V4Observation | None
    parsed_state_summary: dict[str, Any]
    decision_summary: dict[str, Any]
    executed_action: V4Action | None
    transition_record: V4TransitionRecord | None
    step_result: V4StepResult | dict[str, Any] | None
    memory_update_summary: dict[str, Any]
    failure_bucket: str | None
    stop_condition_status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionSummaryV4:
    steps_executed: int
    stop_reason: str
    failures: tuple[str, ...] = ()
    step8_trace_rows: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionLedgerV4:
    def __init__(self) -> None:
        self._records: list[StepLedgerRecordV4] = []

    def append(self, record: StepLedgerRecordV4) -> None:
        if not isinstance(record, StepLedgerRecordV4):
            raise ValueError("record must be StepLedgerRecordV4")
        self._records.append(record)

    def records(self) -> tuple[StepLedgerRecordV4, ...]:
        return tuple(self._records)

    def build_summary(self) -> SessionSummaryV4:
        failures = tuple(record.failure_bucket for record in self._records if record.failure_bucket)
        stop_reason = "continue"
        if self._records:
            stop_reason = str(self._records[-1].stop_condition_status.get("reason", "continue"))
        return SessionSummaryV4(
            steps_executed=sum(1 for record in self._records if record.executed_action is not None),
            stop_reason=stop_reason,
            failures=failures,
        )
