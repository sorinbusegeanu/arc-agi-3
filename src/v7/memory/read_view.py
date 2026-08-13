from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionScoreInput, CognitionIndexes
from v7.memory.models import MemoryNode, MemoryScore

EdgeKey = tuple[MemoryId, int]


@dataclass(frozen=True, slots=True)
class MemoryReadView:
    """Immutable generation-specific cognition view."""

    generation_id: GenerationId
    nodes: Mapping[MemoryId, MemoryNode]
    scores: Mapping[MemoryId, MemoryScore]
    adjacency: Mapping[EdgeKey, tuple[MemoryId, ...]]
    cognition_indexes: CognitionIndexes

    @classmethod
    def freeze(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[EdgeKey, tuple[MemoryId, ...] | list[MemoryId] | set[MemoryId]],
        cognition_indexes: CognitionIndexes | None = None,
    ) -> "MemoryReadView":
        frozen_adjacency = {
            key: tuple(sorted(values, key=int))
            for key, values in adjacency.items()
        }
        return cls(
            generation_id=generation_id,
            nodes=MappingProxyType(dict(nodes)),
            scores=MappingProxyType(dict(scores)),
            adjacency=MappingProxyType(frozen_adjacency),
            cognition_indexes=cognition_indexes or CognitionIndexes.empty(),
        )

    def get_nodes(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...]) -> tuple[MemoryNode | None, ...]:
        return tuple(self.nodes.get(memory_id) for memory_id in memory_ids)

    def get_scores(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...]) -> tuple[MemoryScore | None, ...]:
        return tuple(self.scores.get(memory_id) for memory_id in memory_ids)

    def neighbors(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...], relation_type: int) -> tuple[tuple[MemoryId, ...], ...]:
        return tuple(self.adjacency.get((memory_id, int(relation_type)), ()) for memory_id in memory_ids)

    def score_inputs(
        self,
        *,
        context_signature: int,
        action_ids: list[int] | tuple[int, ...],
        family_ids_by_action: Mapping[int, MemoryId] | None = None,
        role_limit: int = 64,
        concept_limit: int = 128,
    ) -> tuple[ActionScoreInput, ...]:
        return self.cognition_indexes.score_inputs(
            context_signature=context_signature,
            action_ids=action_ids,
            family_ids_by_action=family_ids_by_action,
            role_limit=role_limit,
            concept_limit=concept_limit,
        )
