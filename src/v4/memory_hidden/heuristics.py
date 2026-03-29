from __future__ import annotations

from .typedState import MemoryHiddenTypedStateV4


def zero_heuristic(state: MemoryHiddenTypedStateV4) -> int:
    del state
    return 0
