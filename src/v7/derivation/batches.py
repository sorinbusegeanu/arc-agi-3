from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v7.derivation.dependencies import DependencyMutation
from v7.derivation.workers import DerivationTaskResult
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import (
    ActionAggregateDelta,
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class DerivedMutationBatch:
    """Worker-produced mutations for one immutable-generation derivation task.

    Workers do not mutate canonical memory. IDs in node mutations must already be
    canonical/reserved; future candidate-ID resolution remains writer-owned.
    """

    generation_id: GenerationId
    source_level: MemoryLevel
    source_ids: tuple[MemoryId, ...]
    node_mutations: tuple[NodeMutation, ...] = ()
    edge_mutations: tuple[EdgeMutation, ...] = ()
    score_mutations: tuple[ScoreMutation, ...] = ()
    dependencies: tuple[DependencyMutation, ...] = ()
    contingencies: tuple[ContingencyIndexMutation, ...] = ()
    roles: tuple[RoleIndexMutation, ...] = ()
    role_concepts: tuple[RoleConceptIndexMutation, ...] = ()
    action_aggregates: tuple[ActionAggregateDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivedMergeStats:
    batches: int = 0
    nodes: int = 0
    edges: int = 0
    scores: int = 0
    dependencies: int = 0
    contingencies: int = 0
    roles: int = 0
    role_concepts: int = 0
    action_aggregates: int = 0


class DeterministicDerivedBatchMerger:
    """Canonical single-writer merge independent of worker completion order."""

    @staticmethod
    def _task_key(result: DerivationTaskResult[DerivedMutationBatch]) -> tuple[int, tuple[int, ...]]:
        return int(result.task.level), tuple(int(value) for value in result.task.memory_ids)

    def apply(
        self,
        results: Iterable[DerivationTaskResult[DerivedMutationBatch]],
        *,
        writer: CanonicalMemoryWriter,
    ) -> DerivedMergeStats:
        ordered = tuple(sorted(results, key=self._task_key))
        if not ordered:
            return DerivedMergeStats()

        target_generation = writer.mutable_generation_id
        batches: list[DerivedMutationBatch] = []
        for result in ordered:
            batch = result.output
            if result.task.generation_id != batch.generation_id:
                raise ValueError("derived batch generation does not match task")
            if batch.source_level != result.task.level or batch.source_ids != result.task.memory_ids:
                raise ValueError("derived batch provenance does not match task")
            if int(batch.generation_id) > int(target_generation):
                raise ValueError("derived batch is from a future generation")
            batches.append(batch)

        dependencies = tuple(item for batch in batches for item in batch.dependencies)
        nodes = tuple(item for batch in batches for item in batch.node_mutations)
        edges = tuple(item for batch in batches for item in batch.edge_mutations)
        scores = tuple(item for batch in batches for item in batch.score_mutations)
        contingencies = tuple(item for batch in batches for item in batch.contingencies)
        roles = tuple(item for batch in batches for item in batch.roles)
        role_concepts = tuple(item for batch in batches for item in batch.role_concepts)
        aggregates = tuple(item for batch in batches for item in batch.action_aggregates)

        # Dependencies are registered before nodes so a derived node mutation can
        # immediately propagate dirtiness to its already-known dependents.
        if dependencies:
            writer.apply_dependency_batch(dependencies)
        if nodes:
            writer.apply_mutation_batch(nodes)
        if edges:
            writer.apply_edge_batch(edges)
        if scores:
            writer.apply_score_batch(scores)
        if contingencies:
            writer.apply_contingency_index_batch(contingencies)
        if roles:
            writer.apply_role_index_batch(roles)
        if role_concepts:
            writer.apply_role_concept_index_batch(role_concepts)
        if aggregates:
            writer.apply_action_aggregate_batch(aggregates)

        return DerivedMergeStats(
            batches=len(batches),
            nodes=len(nodes),
            edges=len(edges),
            scores=len(scores),
            dependencies=len(dependencies),
            contingencies=len(contingencies),
            roles=len(roles),
            role_concepts=len(role_concepts),
            action_aggregates=len(aggregates),
        )
