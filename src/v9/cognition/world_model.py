from __future__ import annotations

from dataclasses import dataclass

from v9.memory.identity import MemoryUid


@dataclass(frozen=True, slots=True)
class WorldModelView:
    validated_concepts: tuple[MemoryUid, ...]
    consequence_structures: tuple[MemoryUid, ...]
    explanatory_edges: tuple[tuple[MemoryUid, MemoryUid], ...]

