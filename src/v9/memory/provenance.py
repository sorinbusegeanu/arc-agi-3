from __future__ import annotations

from dataclasses import dataclass

from .identity import EpisodeId, EventUid, MemoryUid


@dataclass(frozen=True, slots=True)
class ProvenanceRoot:
    event_uid: EventUid
    environment_instance_id: int
    episode_id: EpisodeId
    causal_watermark: int


@dataclass(frozen=True, slots=True)
class DerivationProvenance:
    parents: tuple[MemoryUid, ...]
    evidence: tuple[MemoryUid, ...]
    formation_scope: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        parents = tuple(dict.fromkeys(self.parents))
        evidence = tuple(dict.fromkeys(self.evidence))
        formation_scope = tuple(dict.fromkeys(int(value) for value in self.formation_scope))
        if not parents:
            raise ValueError("derived memory must retain at least one parent")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "formation_scope", formation_scope)

