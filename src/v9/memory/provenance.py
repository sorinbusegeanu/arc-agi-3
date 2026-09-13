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
        if not self.parents:
            raise ValueError("derived memory must retain at least one parent")

