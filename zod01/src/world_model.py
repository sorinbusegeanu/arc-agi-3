from __future__ import annotations

from .memory_episodic import EpisodicMemory
from .types import NormalizedAction, TransitionEdge


class EmpiricalWorldModel:
    def __init__(self, memory: EpisodicMemory) -> None:
        self.memory = memory

    def predict(self, state_hash: str, action: NormalizedAction) -> TransitionEdge | None:
        return self.memory.get_edge(state_hash, action)

    def is_unknown_edge(self, state_hash: str, action: NormalizedAction) -> bool:
        return self.predict(state_hash, action) is None
