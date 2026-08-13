from __future__ import annotations

from dataclasses import dataclass

from v7.memory.models import EdgeState, MemoryNode, MemoryScore


@dataclass(frozen=True, slots=True)
class GenerationDelta:
    """Compact dirty-state payload persisted for one committed generation."""

    nodes: tuple[MemoryNode, ...]
    scores: tuple[MemoryScore, ...]
    edges: tuple[EdgeState, ...]

    @property
    def mutation_count(self) -> int:
        return len(self.nodes) + len(self.scores) + len(self.edges)
