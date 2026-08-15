from __future__ import annotations

from enum import IntFlag

from v7.memory.models import MemoryNode
from v7.memory.state import CognitiveState, GateId, GateValidationState, is_gate_validated


class MemoryStatus(IntFlag):
    """Compatibility/reporting flags. v7.0.6 state enums are authoritative."""

    ACTIVE = 1 << 0
    PROMOTED = 1 << 1
    DEMOTED = 1 << 2
    REPLAY_QUEUED = 1 << 3


class ConceptValidationStatus(IntFlag):
    """Compatibility mirror for existing M4 reports and tests."""

    CANDIDATE = 1 << 8
    TRANSFER_VALIDATED = 1 << 9
    TRANSFER_REJECTED = 1 << 10
    STRUCTURAL_SUPPORTED = 1 << 11
    TRANSFER_CANDIDATE = 1 << 12
    TRUSTED = 1 << 13


def memory_cognitive_state(node: MemoryNode | None) -> CognitiveState | None:
    if node is None:
        return None
    raw = int(getattr(node, "cognitive_state", CognitiveState.ACTIVE))
    try:
        return CognitiveState(raw)
    except ValueError:
        return CognitiveState.ACTIVE


def memory_validation_state(node: MemoryNode | None) -> GateValidationState | None:
    if node is None:
        return None
    raw = int(getattr(node, "validation_state", GateValidationState.VALIDATED))
    try:
        return GateValidationState(raw)
    except ValueError:
        return GateValidationState.VALIDATED


def memory_is_active(node: MemoryNode | None) -> bool:
    """Return whether a node is allowed in normal cognition."""
    if node is None:
        return False
    if int(node.status_flags) & int(MemoryStatus.DEMOTED):
        return False
    return memory_cognitive_state(node) == CognitiveState.ACTIVE


def memory_is_probe_eligible(node: MemoryNode | None) -> bool:
    """Return whether a node may be used by an explicit probe/replay lane."""
    state = memory_cognitive_state(node)
    if state not in {CognitiveState.ACTIVE, CognitiveState.PROBE_ONLY}:
        return False
    if node is None:
        return False
    if int(node.status_flags) & int(ConceptValidationStatus.TRANSFER_REJECTED):
        return False
    return True


def memory_is_derivation_eligible(node: MemoryNode | None) -> bool:
    """Only active validated/trusted parents may bootstrap higher abstractions."""
    if not memory_is_active(node):
        return False
    assert node is not None
    if int(getattr(node, "gate_id", GateId.NONE)) == int(GateId.NONE):
        return True
    return is_gate_validated(
        int(getattr(node, "validation_state", GateValidationState.VALIDATED))
    )


__all__ = [
    "ConceptValidationStatus",
    "MemoryStatus",
    "memory_cognitive_state",
    "memory_is_active",
    "memory_is_derivation_eligible",
    "memory_is_probe_eligible",
    "memory_validation_state",
]
