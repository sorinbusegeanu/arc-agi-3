from __future__ import annotations

from enum import IntEnum

from v7.memory.ids import MemoryLevel


class CognitiveState(IntEnum):
    """Whether a semantic memory may participate in cognition."""

    ACTIVE = 1
    PROBE_ONLY = 2
    QUARANTINED = 3
    RETIRED = 4


class GateValidationState(IntEnum):
    """Scientific validation state for the developmental gate that produced a memory."""

    STRUCTURAL_CANDIDATE = 1
    PROBE_ELIGIBLE = 2
    TRANSFER_TESTED = 3
    VALIDATED = 4
    REJECTED = 5
    TRUSTED = 6


class GateId(IntEnum):
    NONE = 0
    G01 = 1
    G12 = 2
    G23C = 3
    G23R = 4
    G34 = 5
    G45 = 6
    G56 = 7


_GATE_NAMES = {
    GateId.NONE: "NONE",
    GateId.G01: "G01",
    GateId.G12: "G12",
    GateId.G23C: "G23C",
    GateId.G23R: "G23R",
    GateId.G34: "G34",
    GateId.G45: "G45",
    GateId.G56: "G56",
}


def gate_name(gate_id: int | GateId) -> str:
    try:
        return _GATE_NAMES[GateId(int(gate_id))]
    except (TypeError, ValueError, KeyError):
        return "NONE"


def gate_for_identity(level: MemoryLevel | int, type_id: int) -> GateId:
    """Map the canonical v7 hierarchy to the gate that produces each reusable memory."""
    level = MemoryLevel(int(level))
    type_id = int(type_id)
    if level == MemoryLevel.M1:
        return GateId.G01
    if level == MemoryLevel.M2:
        return GateId.G12
    if level == MemoryLevel.M3 and type_id == 302:  # TYPE_CARRIER
        return GateId.G23C
    if level == MemoryLevel.M3 and type_id == 300:  # TYPE_ROLE
        return GateId.G23R
    if level == MemoryLevel.M4:
        return GateId.G34
    if level == MemoryLevel.M5:
        return GateId.G45
    if level == MemoryLevel.M6:
        return GateId.G56
    return GateId.NONE


def is_gate_validated(state: int | GateValidationState) -> bool:
    try:
        value = GateValidationState(int(state))
    except (TypeError, ValueError):
        return False
    return value in {GateValidationState.VALIDATED, GateValidationState.TRUSTED}


def is_probe_cognitive_state(state: int | CognitiveState) -> bool:
    try:
        value = CognitiveState(int(state))
    except (TypeError, ValueError):
        return False
    return value in {CognitiveState.ACTIVE, CognitiveState.PROBE_ONLY}


__all__ = [
    "CognitiveState",
    "GateId",
    "GateValidationState",
    "gate_for_identity",
    "gate_name",
    "is_gate_validated",
    "is_probe_cognitive_state",
]
