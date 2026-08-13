from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from v7.memory.arenas.compact import CompactMemoryArena
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionScoreInput, CognitionIndexes
from v7.memory.indexes.packed import PackedCognitionIndexes
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
    packed_cognition: PackedCognitionIndexes
    compact_arena: CompactMemoryArena

    @classmethod
    def freeze(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[EdgeKey, tuple[MemoryId, ...] | list[MemoryId] | set[MemoryId]],
        cognition_indexes: CognitionIndexes | None = None,
        previous_view: "MemoryReadView | None" = None,
        nodes_dirty: bool = True,
        scores_dirty: bool = True,
        adjacency_dirty: bool = True,
        cognition_dirty: bool = True,
    ) -> "MemoryReadView":
        frozen_nodes = MappingProxyType(dict(nodes))
        frozen_scores = MappingProxyType(dict(scores))
        frozen_adjacency_dict = {key: tuple(sorted(values, key=int)) for key, values in adjacency.items()}
        frozen_adjacency = MappingProxyType(frozen_adjacency_dict)
        cognition = cognition_indexes or CognitionIndexes.empty()
        arena = CompactMemoryArena.build_incremental(
            generation_id=generation_id,
            nodes=frozen_nodes,
            scores=frozen_scores,
            adjacency=frozen_adjacency_dict,
            previous=None if previous_view is None else previous_view.compact_arena,
            nodes_dirty=nodes_dirty,
            scores_dirty=scores_dirty,
            adjacency_dirty=adjacency_dirty,
        )
        packed = previous_view.packed_cognition if previous_view is not None and not cognition_dirty else PackedCognitionIndexes.build(cognition)
        return cls(generation_id, frozen_nodes, frozen_scores, frozen_adjacency, cognition, packed, arena)

    @classmethod
    def from_compact_arena(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[EdgeKey, tuple[MemoryId, ...]],
        cognition_indexes: CognitionIndexes,
        compact_arena: CompactMemoryArena,
        packed_cognition: PackedCognitionIndexes | None = None,
    ) -> "MemoryReadView":
        return cls(
            generation_id=generation_id,
            nodes=nodes,
            scores=scores,
            adjacency=adjacency,
            cognition_indexes=cognition_indexes,
            packed_cognition=packed_cognition or PackedCognitionIndexes.build(cognition_indexes),
            compact_arena=compact_arena,
        )

    def get_nodes(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...]) -> tuple[MemoryNode | None, ...]:
        return tuple(self.compact_arena.nodes.get(memory_id) for memory_id in memory_ids)

    def get_scores(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...]) -> tuple[MemoryScore | None, ...]:
        return tuple(self.compact_arena.scores.get(memory_id) for memory_id in memory_ids)

    def neighbors(self, memory_ids: list[MemoryId] | tuple[MemoryId, ...], relation_type: int) -> tuple[tuple[MemoryId, ...], ...]:
        return tuple(self.compact_arena.adjacency.neighbors(memory_id, relation_type) for memory_id in memory_ids)

    def score_inputs(
        self,
        *,
        context_signature: int,
        action_ids: list[int] | tuple[int, ...],
        family_ids_by_action: Mapping[int, MemoryId] | None = None,
        role_limit: int = 64,
        concept_limit: int = 128,
    ) -> tuple[ActionScoreInput, ...]:
        if role_limit < 0 or concept_limit < 0:
            raise ValueError("limits must be non-negative")
        family_ids_by_action = family_ids_by_action or {}
        rows: list[ActionScoreInput] = []
        packed = self.packed_cognition
        for raw_action in action_ids:
            action_id = int(raw_action)
            contingencies = packed.contingencies.lookup(int(context_signature), action_id)
            family_id = family_ids_by_action.get(action_id)
            roles = () if family_id is None else packed.roles_exact.lookup(int(context_signature), action_id, family_id, role_limit)
            if not roles:
                roles = packed.roles_fallback.lookup(int(context_signature), action_id, role_limit)
            concepts: list[MemoryId] = []
            seen: set[MemoryId] = set()
            for role_id in roles:
                for concept_id in packed.concepts(role_id, concept_limit):
                    if concept_id in seen:
                        continue
                    seen.add(concept_id)
                    concepts.append(concept_id)
                    if len(concepts) >= concept_limit:
                        break
                if len(concepts) >= concept_limit:
                    break
            rows.append(ActionScoreInput(action_id, contingencies, packed.action_aggregates.get(action_id), roles, tuple(concepts)))
        return tuple(rows)
