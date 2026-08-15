from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from v7.memory.arenas.compact import CompactMemoryArena
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ActionScoreInput, CognitionIndexes
from v7.memory.indexes.mapped import MappedPackedCognitionIndexes
from v7.memory.indexes.packed import PackedCognitionIndexes
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.status import MemoryStatus, memory_is_active

EdgeKey = tuple[MemoryId, int]
PackedCognitionView = PackedCognitionIndexes | MappedPackedCognitionIndexes


@dataclass(frozen=True, slots=True, weakref_slot=True)
class MemoryReadView:
    """Immutable generation-specific cognition view."""

    generation_id: GenerationId
    nodes: Mapping[MemoryId, MemoryNode]
    scores: Mapping[MemoryId, MemoryScore]
    adjacency: Mapping[EdgeKey, tuple[MemoryId, ...]]
    cognition_indexes: CognitionIndexes
    packed_cognition: PackedCognitionView
    compact_arena: CompactMemoryArena

    @classmethod
    def freeze(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[
            EdgeKey,
            tuple[MemoryId, ...] | list[MemoryId] | set[MemoryId],
        ],
        cognition_indexes: CognitionIndexes | None = None,
        previous_view: "MemoryReadView | None" = None,
        nodes_dirty: bool = True,
        scores_dirty: bool = True,
        adjacency_dirty: bool = True,
        cognition_dirty: bool = True,
    ) -> "MemoryReadView":
        frozen_nodes = MappingProxyType(dict(nodes))
        frozen_scores = MappingProxyType(dict(scores))
        frozen_adjacency_dict = {
            key: tuple(sorted(values, key=int))
            for key, values in adjacency.items()
        }
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
        packed = (
            previous_view.packed_cognition
            if previous_view is not None and not cognition_dirty
            else PackedCognitionIndexes.build(cognition)
        )
        return cls(
            generation_id,
            frozen_nodes,
            frozen_scores,
            frozen_adjacency,
            cognition,
            packed,
            arena,
        )

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
        packed_cognition: PackedCognitionView | None = None,
    ) -> "MemoryReadView":
        return cls(
            generation_id=generation_id,
            nodes=nodes,
            scores=scores,
            adjacency=adjacency,
            cognition_indexes=cognition_indexes,
            packed_cognition=packed_cognition
            or PackedCognitionIndexes.build(cognition_indexes),
            compact_arena=compact_arena,
        )

    def get_nodes(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
    ) -> tuple[MemoryNode | None, ...]:
        return tuple(self.compact_arena.nodes.get(memory_id) for memory_id in memory_ids)

    def get_scores(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
    ) -> tuple[MemoryScore | None, ...]:
        return tuple(self.compact_arena.scores.get(memory_id) for memory_id in memory_ids)

    def neighbors(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
        relation_type: int,
    ) -> tuple[tuple[MemoryId, ...], ...]:
        return tuple(
            self.compact_arena.adjacency.neighbors(memory_id, relation_type)
            for memory_id in memory_ids
        )

    def _memory_priority(self, memory_id: MemoryId) -> tuple[float, int]:
        node = self.nodes.get(memory_id)
        if not memory_is_active(node):
            return (-1.0, -int(memory_id))
        assert node is not None
        score = self.scores.get(memory_id)
        support = 1.0 - math.exp(-max(0, int(node.support_count)) / 4.0)
        semantic = 0.0
        if score is not None:
            semantic = max(
                0.0,
                float(score.significance),
                float(score.learning_value),
                float(score.transfer_prior),
                float(score.explanatory_potential),
                max(0.0, float(score.future_option_delta)),
            )
        status = 0.0
        flags = int(node.status_flags)
        if flags & int(MemoryStatus.PROMOTED):
            status += 0.10
        if node.level == MemoryLevel.M4:
            if flags & int(ConceptValidationStatus.TRANSFER_REJECTED):
                return (-1.0, -int(memory_id))
            if flags & int(ConceptValidationStatus.TRUSTED):
                status += 0.25
            elif flags & int(ConceptValidationStatus.TRANSFER_VALIDATED):
                status += 0.20
            elif flags & int(ConceptValidationStatus.TRANSFER_CANDIDATE):
                status += 0.10
        level_bonus = {
            MemoryLevel.M6: 0.06,
            MemoryLevel.M5: 0.04,
            MemoryLevel.M4: 0.03,
            MemoryLevel.M3: 0.02,
        }.get(node.level, 0.0)
        return (
            0.55 * semantic + 0.30 * support + status + level_bonus,
            -int(memory_id),
        )

    def _rank_active(
        self,
        memory_ids,
        *,
        limit: int | None,
    ) -> tuple[MemoryId, ...]:
        unique = {
            MemoryId(int(memory_id))
            for memory_id in memory_ids
            if memory_is_active(self.nodes.get(MemoryId(int(memory_id))))
        }
        ordered = sorted(
            unique,
            key=lambda memory_id: (
                -self._memory_priority(memory_id)[0],
                int(memory_id),
            ),
        )
        if limit is not None:
            ordered = ordered[: max(0, int(limit))]
        return tuple(ordered)

    def score_inputs(
        self,
        *,
        context_signature: int,
        action_ids: list[int] | tuple[int, ...],
        family_ids_by_action: Mapping[int, MemoryId] | None = None,
        role_limit: int = 64,
        concept_limit: int = 128,
    ) -> tuple[ActionScoreInput, ...]:
        """Return active, relevance-ranked cognition candidates.

        Packed indexes are identity ordered for compact storage. Candidate
        truncation therefore happens only after active-state and semantic
        ranking, so old low IDs and rejected concepts cannot monopolize the
        acting frontier.
        """
        if role_limit < 0 or concept_limit < 0:
            raise ValueError("limits must be non-negative")
        families = family_ids_by_action or {}
        packed = self.packed_cognition
        rows: list[ActionScoreInput] = []
        for raw_action_id in action_ids:
            action_id = int(raw_action_id)
            contingencies = self._rank_active(
                packed.contingencies.lookup(int(context_signature), action_id),
                limit=None,
            )
            family = families.get(action_id)
            role_candidates: tuple[MemoryId, ...] = ()
            if family is not None:
                # Exact-role lookup currently accepts an integer bound. Use a
                # very high scan bound, then rank and truncate below.
                role_candidates = packed.roles_exact.lookup(
                    int(context_signature),
                    action_id,
                    family,
                    2**31 - 1,
                )
            if not role_candidates:
                role_candidates = packed.roles_fallback.lookup(
                    int(context_signature),
                    action_id,
                    None,
                )
            roles = self._rank_active(role_candidates, limit=role_limit)

            concept_candidates: set[MemoryId] = set()
            for role_id in roles:
                concept_candidates.update(
                    packed.concepts_by_role.lookup(int(role_id), 0, None)
                )
            concepts = self._rank_active(
                concept_candidates,
                limit=concept_limit,
            )
            rows.append(
                ActionScoreInput(
                    action_id=action_id,
                    contingency_ids=contingencies,
                    aggregate=packed.action_aggregates.get(action_id),
                    role_ids=roles,
                    concept_ids=concepts,
                )
            )
        return tuple(rows)
