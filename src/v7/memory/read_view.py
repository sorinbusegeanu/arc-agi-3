from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from v7.memory.arenas.compact import CompactMemoryArena
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ActionScoreInput, CognitionIndexes
from v7.memory.indexes.mapped import MappedPackedCognitionIndexes
from v7.memory.indexes.packed import PackedCognitionIndexes
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.state import GateValidationState
from v7.memory.status import (
    ConceptValidationStatus,
    MemoryStatus,
    memory_is_active,
    memory_is_probe_eligible,
)

EdgeKey = tuple[MemoryId, int]
PackedCognitionView = PackedCognitionIndexes | MappedPackedCognitionIndexes
_UNBOUNDED_LOOKUP = (1 << 31) - 1


def _filter_cognition(
    cognition: CognitionIndexes,
    nodes: Mapping[MemoryId, MemoryNode],
    *,
    include_probe: bool,
) -> CognitionIndexes:
    """Build publication-time ACTIVE or ACTIVE+PROBE_ONLY cognition."""

    def allowed(memory_id: MemoryId) -> bool:
        node = nodes.get(memory_id)
        return memory_is_probe_eligible(node) if include_probe else memory_is_active(node)

    def allowed_values(values):
        return tuple(memory_id for memory_id in values if allowed(memory_id))

    return CognitionIndexes.freeze(
        contingency_by_context_action={
            key: allowed_values(values)
            for key, values in cognition.contingency_by_context_action.items()
        },
        role_by_context_action_family={
            key: allowed_values(values)
            for key, values in cognition.role_by_context_action_family.items()
        },
        role_by_context_action={
            key: allowed_values(values)
            for key, values in cognition.role_by_context_action.items()
        },
        concepts_by_role={
            role_id: allowed_values(values)
            for role_id, values in cognition.concepts_by_role.items()
            if allowed(role_id)
        },
        action_aggregates=cognition.action_aggregates,
    )


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
    probe_packed_cognition: PackedCognitionView | None = None

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
        active_cognition = _filter_cognition(
            cognition, frozen_nodes, include_probe=False
        )
        probe_cognition = _filter_cognition(
            cognition, frozen_nodes, include_probe=True
        )
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
            else PackedCognitionIndexes.build(active_cognition)
        )
        probe_packed = (
            previous_view.probe_packed_cognition
            if previous_view is not None
            and not cognition_dirty
            and previous_view.probe_packed_cognition is not None
            else PackedCognitionIndexes.build(probe_cognition)
        )
        return cls(
            generation_id,
            frozen_nodes,
            frozen_scores,
            frozen_adjacency,
            cognition,
            packed,
            arena,
            probe_packed,
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
        probe_packed_cognition: PackedCognitionView | None = None,
    ) -> "MemoryReadView":
        active = packed_cognition or PackedCognitionIndexes.build(
            _filter_cognition(cognition_indexes, nodes, include_probe=False)
        )
        probe = probe_packed_cognition or PackedCognitionIndexes.build(
            _filter_cognition(cognition_indexes, nodes, include_probe=True)
        )
        return cls(
            generation_id=generation_id,
            nodes=nodes,
            scores=scores,
            adjacency=adjacency,
            cognition_indexes=cognition_indexes,
            packed_cognition=active,
            compact_arena=compact_arena,
            probe_packed_cognition=probe,
        )

    def get_nodes(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
    ) -> tuple[MemoryNode | None, ...]:
        return tuple(
            self.compact_arena.nodes.get(memory_id) for memory_id in memory_ids
        )

    def get_scores(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
    ) -> tuple[MemoryScore | None, ...]:
        return tuple(
            self.compact_arena.scores.get(memory_id) for memory_id in memory_ids
        )

    def neighbors(
        self,
        memory_ids: list[MemoryId] | tuple[MemoryId, ...],
        relation_type: int,
    ) -> tuple[tuple[MemoryId, ...], ...]:
        return tuple(
            self.compact_arena.adjacency.neighbors(memory_id, relation_type)
            for memory_id in memory_ids
        )

    def _available(self, node: MemoryNode | None, *, include_probe: bool) -> bool:
        return (
            memory_is_probe_eligible(node)
            if include_probe
            else memory_is_active(node)
        )

    def _memory_priority(
        self,
        memory_id: MemoryId,
        *,
        include_probe: bool = False,
    ) -> tuple[float, int]:
        node = self.nodes.get(memory_id)
        if not self._available(node, include_probe=include_probe):
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
        validation = int(
            getattr(node, "validation_state", GateValidationState.VALIDATED)
        )
        if validation == int(GateValidationState.TRUSTED):
            status += 0.25
        elif validation == int(GateValidationState.VALIDATED):
            status += 0.20
        elif validation == int(GateValidationState.PROBE_ELIGIBLE):
            status += 0.05
        if node.level == MemoryLevel.M4:
            if flags & int(ConceptValidationStatus.TRANSFER_REJECTED):
                return (-1.0, -int(memory_id))
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

    def _rank_available(
        self,
        memory_ids,
        *,
        limit: int | None,
        include_probe: bool,
    ) -> tuple[MemoryId, ...]:
        ranked: list[tuple[float, MemoryId]] = []
        seen: set[MemoryId] = set()
        for raw_memory_id in memory_ids:
            memory_id = MemoryId(int(raw_memory_id))
            if memory_id in seen:
                continue
            seen.add(memory_id)
            if not self._available(
                self.nodes.get(memory_id), include_probe=include_probe
            ):
                continue
            priority = self._memory_priority(
                memory_id, include_probe=include_probe
            )[0]
            if priority < 0.0:
                continue
            ranked.append((priority, memory_id))
        ranked.sort(key=lambda item: (-item[0], int(item[1])))
        values = [memory_id for _priority, memory_id in ranked]
        if limit is not None:
            values = values[: max(0, int(limit))]
        return tuple(values)

    def _score_from_packed(
        self,
        *,
        packed: PackedCognitionView,
        context_signature: int,
        action_ids: list[int] | tuple[int, ...],
        family_ids_by_action: Mapping[int, MemoryId],
        role_limit: int,
        concept_limit: int,
        include_probe: bool,
    ) -> tuple[ActionScoreInput, ...]:
        rows: list[ActionScoreInput] = []
        for raw_action_id in action_ids:
            action_id = int(raw_action_id)
            contingencies = self._rank_available(
                packed.contingencies.lookup(int(context_signature), action_id),
                limit=None,
                include_probe=include_probe,
            )
            family = family_ids_by_action.get(action_id)
            role_candidates: tuple[MemoryId, ...] = ()
            if family is not None:
                role_candidates = packed.roles_exact.lookup(
                    int(context_signature),
                    action_id,
                    family,
                    _UNBOUNDED_LOOKUP,
                )
            if not role_candidates:
                role_candidates = packed.roles_fallback.lookup(
                    int(context_signature),
                    action_id,
                    _UNBOUNDED_LOOKUP,
                )
            roles = self._rank_available(
                role_candidates,
                limit=role_limit,
                include_probe=include_probe,
            )
            concept_candidates: set[MemoryId] = set()
            for role_id in roles:
                concept_candidates.update(
                    packed.concepts_by_role.lookup(int(role_id), 0, None)
                )
            concepts = self._rank_available(
                concept_candidates,
                limit=concept_limit,
                include_probe=include_probe,
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

    def score_inputs(
        self,
        *,
        context_signature: int,
        action_ids: list[int] | tuple[int, ...],
        family_ids_by_action: Mapping[int, MemoryId] | None = None,
        role_limit: int = 64,
        concept_limit: int = 128,
        include_probe: bool = False,
    ) -> tuple[ActionScoreInput, ...]:
        """Return relevance-ranked normal or controlled-probe candidates."""
        if role_limit < 0 or concept_limit < 0:
            raise ValueError("limits must be non-negative")
        packed = (
            self.probe_packed_cognition
            if include_probe and self.probe_packed_cognition is not None
            else self.packed_cognition
        )
        return self._score_from_packed(
            packed=packed,
            context_signature=int(context_signature),
            action_ids=action_ids,
            family_ids_by_action=family_ids_by_action or {},
            role_limit=role_limit,
            concept_limit=concept_limit,
            include_probe=include_probe,
        )

    def probe_score_inputs(self, **kwargs) -> tuple[ActionScoreInput, ...]:
        kwargs["include_probe"] = True
        return self.score_inputs(**kwargs)
