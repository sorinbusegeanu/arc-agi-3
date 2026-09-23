from __future__ import annotations

SemanticFact = tuple[int, int, int, int, float]

from enum import IntEnum

class SemanticKind(IntEnum):
    GLOBAL=1
    ENTITY=2
    ATTRIBUTE=3
    RELATION=4
    SPATIAL=5
    NUMERIC=6
    TEXT=7
    ACTION=8
    DELTA=9
    TASK=10
