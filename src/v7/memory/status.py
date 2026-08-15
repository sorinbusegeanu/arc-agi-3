from __future__ import annotations

from enum import IntFlag

from v7.memory.models import MemoryNode


class MemoryStatus(IntFlag):
    ACTIVE = 1 << 0
    PROMOTED = 1 << 1
    DEMOTED = 1 << 2
    REPLAY_QUEUED = 1 << 3


class ConceptValidationStatus(IntFlag):
    CANDIDATE = 1 << 8
    TRANSFER_VALIDATED = 1 << 9
    TRANSFER_REJECTED = 1 << 10
    STRUCTURAL_SUPPORTED = 1 << 11
    TRANSFER_CANDIDATE = 1 << 12
    TRUSTED = 1 << 13


def memory_is_active(node: MemoryNode | None) -> bool:
    """Return whether a node is eligible to influence active cognition.

    Nodes begin with no lifecycle flags, so absence of ACTIVE is not itself
    inactive. DEMOTED is the explicit exclusion state.
    """
    return node is not None and not bool(
        int(node.status_flags) & int(MemoryStatus.DEMOTED)
    )


__all__ = [
    "ConceptValidationStatus",
    "MemoryStatus",
    "memory_is_active",
]
