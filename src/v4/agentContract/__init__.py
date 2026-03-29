"""Stable v4 environment contract surface."""

from .types import (
    V4Action,
    V4AuthoritativeState,
    V4ContractVersion,
    V4Observation,
    V4StepResult,
    V4TerminalSignal,
    V4TransitionRecord,
)
from .validators import (
    derive_terminal_signal,
    validate_authoritative_state,
    validate_observation_payload,
    validate_v4_action,
    validate_v4_observation,
    validate_v4_step_result,
    validate_v4_terminal_signal,
    validate_v4_transition_record,
)

VERSION = V4ContractVersion.V4_0_0

__all__ = [
    "V4Observation",
    "V4Action",
    "V4AuthoritativeState",
    "V4TerminalSignal",
    "V4TransitionRecord",
    "V4StepResult",
    "V4ContractVersion",
    "VERSION",
    "validate_observation_payload",
    "validate_v4_observation",
    "validate_v4_action",
    "validate_authoritative_state",
    "derive_terminal_signal",
    "validate_v4_terminal_signal",
    "validate_v4_transition_record",
    "validate_v4_step_result",
]
