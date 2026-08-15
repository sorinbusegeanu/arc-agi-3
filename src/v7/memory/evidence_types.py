from __future__ import annotations

from enum import IntEnum


class EvidenceType(IntEnum):
    EPISODE = 1
    TRAJECTORY = 2
    PROMOTION = 1001
    DEMOTION = 1002
    REPLAY = 1003
    CONCEPT_VALIDATION = 1004
    GATE_VALIDATION = 1005
    RETIREMENT = 1006


__all__ = ["EvidenceType"]
