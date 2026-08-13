from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from v7.memory.delta import GenerationDelta
from v7.memory.generation import GenerationId, GenerationState
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import (
    ActionAggregateDelta,
    CognitionIndexBuilder,
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.models import EdgeMutation, EdgeState, MemoryNode, MemoryScore, NodeMutation, ScoreMutation
from v7.memory.read_view import MemoryReadView


class CanonicalMemoryWriter:
    """Single-owner mutable frontier for v7 active semantic memory."""

    def __init__(self, *, initial_generation: int = 0) -> None:
        if initial_generation < 0:
            raise ValueError("initial_generation must be non-negative")
        self._published_generation = GenerationId(initial_generation)
        self._mutable_generation = GenerationId(initial_generation + 1)
        self._nodes: dict[MemoryId, MemoryNode] = {}
        self._scores: dict[MemoryId, MemoryScore] = {}
        self._edge_support: dict[tuple[MemoryId, int, MemoryId], int] = {}
        self._cognition_indexes = CognitionIndexBuilder()
        self._dirty_nodes: set[MemoryId] = set()
        self._dirty_scores: set[MemoryId] = set()
        self._dirty_edges: set[tuple[MemoryId, int, MemoryId]] = set()
        self._first_global_step: int | None = None
        self._last_global_step: int | None = None
        self._published_view = MemoryReadView.freeze(
            generation_id=self._published_generation,
            nodes={},
            scores={},
            adjacency={},
            cognition_indexes=self._cognition_indexes.freeze(),
        )

    @property
    def published_view(self) -> MemoryReadView:
        return self._published_view

    @property
    def mutable_generation_id(self) -> GenerationId:
        return self._mutable_generation

    @property
    def dirty_counts(self) -> dict[str, int]:
        return {
            "nodes": len(self._dirty_nodes),
            "scores": len(self._dirty_scores),
            "edges": len(self._dirty_edges),
        }

    def observe_global_step(self, global_step: int) -> None:
        step = int(global_step)
        if step < 0:
            raise ValueError("global_step must be non-negative")
        if self._first_global_step is None:
            self._first_global_step = step
        self._last_global_step = step if self._last_global_step is None else max(self._last_global_step, step)

    def apply_mutation_batch(self, mutations: Iterable[NodeMutation]) -> int:
        coalesced: dict[MemoryId, NodeMutation] = {}
        for mutation in mutations:
            prior = coalesced.get(mutation.memory_id)
            if prior is None:
                coalesced[mutation.memory_id] = mutation
                continue
            if prior.level != mutation.level or prior.type_id != mutation.type_id:
                raise ValueError(f"conflicting node identity for memory_id={int(mutation.memory_id)}")
            coalesced[mutation.memory_id] = NodeMutation(
                memory_id=mutation.memory_id,
                level=mutation.level,
                type_id=mutation.type_id,
                support_delta=prior.support_delta + mutation.support_delta,
                status_flags=mutation.status_flags if mutation.status_flags is not None else prior.status_flags,
            )

        generation = self._mutable_generation
        staged: dict[MemoryId, MemoryNode] = {}
        for memory_id, mutation in coalesced.items():
            current = self._nodes.get(memory_id)
            if current is None:
                support_count = mutation.support_delta
                if support_count < 0:
                    raise ValueError("new node support cannot be negative")
                staged[memory_id] = MemoryNode(
                    memory_id=memory_id,
                    level=mutation.level,
                    type_id=mutation.type_id,
                    created_generation=generation,
                    updated_generation=generation,
                    status_flags=0 if mutation.status_flags is None else mutation.status_flags,
                    support_count=support_count,
                )
            else:
                if current.level != mutation.level or current.type_id != mutation.type_id:
                    raise ValueError(f"memory identity is immutable for memory_id={int(memory_id)}")
                support_count = current.support_count + mutation.support_delta
                if support_count < 0:
                    raise ValueError("node support cannot be negative")
                staged[memory_id] = replace(
                    current,
                    updated_generation=generation,
                    status_flags=current.status_flags if mutation.status_flags is None else mutation.status_flags,
                    support_count=support_count,
                )

        self._nodes.update(staged)
        self._dirty_nodes.update(staged)
        return len(coalesced)

    def apply_edge_batch(self, mutations: Iterable[EdgeMutation]) -> int:
        coalesced: dict[tuple[MemoryId, int, MemoryId], int] = {}
        for mutation in mutations:
            key = (mutation.source_id, int(mutation.relation_type), mutation.target_id)
            coalesced[key] = coalesced.get(key, 0) + int(mutation.support_delta)

        staged: dict[tuple[MemoryId, int, MemoryId], int] = {}
        for key, delta in coalesced.items():
            support = self._edge_support.get(key, 0) + delta
            if support < 0:
                raise ValueError("edge support cannot be negative")
            staged[key] = support

        for key, support in staged.items():
            if support == 0:
                self._edge_support.pop(key, None)
            else:
                self._edge_support[key] = support
        self._dirty_edges.update(staged)
        return len(coalesced)

    def apply_score_batch(self, mutations: Iterable[ScoreMutation]) -> int:
        coalesced: dict[MemoryId, ScoreMutation] = {}
        for mutation in mutations:
            prior = coalesced.get(mutation.memory_id)
            if prior is None:
                coalesced[mutation.memory_id] = mutation
                continue
            coalesced[mutation.memory_id] = ScoreMutation(
                memory_id=mutation.memory_id,
                significance=mutation.significance if mutation.significance is not None else prior.significance,
                prediction_error=mutation.prediction_error if mutation.prediction_error is not None else prior.prediction_error,
                learning_value=mutation.learning_value if mutation.learning_value is not None else prior.learning_value,
                transfer_prior=mutation.transfer_prior if mutation.transfer_prior is not None else prior.transfer_prior,
                explanatory_potential=mutation.explanatory_potential if mutation.explanatory_potential is not None else prior.explanatory_potential,
                future_option_delta=mutation.future_option_delta if mutation.future_option_delta is not None else prior.future_option_delta,
            )

        staged: dict[MemoryId, MemoryScore] = {}
        for memory_id, mutation in coalesced.items():
            current = self._scores.get(memory_id, MemoryScore(memory_id=memory_id))
            values = {
                field: getattr(current, field) if value is None else float(value)
                for field, value in (
                    ("significance", mutation.significance),
                    ("prediction_error", mutation.prediction_error),
                    ("learning_value", mutation.learning_value),
                    ("transfer_prior", mutation.transfer_prior),
                    ("explanatory_potential", mutation.explanatory_potential),
                    ("future_option_delta", mutation.future_option_delta),
                )
            }
            staged[memory_id] = MemoryScore(memory_id=memory_id, **values)

        self._scores.update(staged)
        self._dirty_scores.update(staged)
        return len(coalesced)

    def apply_contingency_index_batch(self, mutations: Iterable[ContingencyIndexMutation]) -> int:
        return self._cognition_indexes.apply_contingency_batch(mutations)

    def apply_role_index_batch(self, mutations: Iterable[RoleIndexMutation]) -> int:
        return self._cognition_indexes.apply_role_batch(mutations)

    def apply_role_concept_index_batch(self, mutations: Iterable[RoleConceptIndexMutation]) -> int:
        return self._cognition_indexes.apply_role_concept_batch(mutations)

    def apply_action_aggregate_batch(self, deltas: Iterable[ActionAggregateDelta]) -> int:
        return self._cognition_indexes.apply_action_aggregate_batch(deltas)

    def commit_generation(self) -> tuple[GenerationState, MemoryReadView, GenerationDelta]:
        adjacency: dict[tuple[MemoryId, int], list[MemoryId]] = {}
        for (source_id, relation_type, target_id), support in self._edge_support.items():
            if support <= 0:
                continue
            adjacency.setdefault((source_id, relation_type), []).append(target_id)

        state = GenerationState(
            generation_id=self._mutable_generation,
            parent_generation_id=self._published_generation,
            first_global_step=self._first_global_step,
            last_global_step=self._last_global_step,
        )
        view = MemoryReadView.freeze(
            generation_id=self._mutable_generation,
            nodes=self._nodes,
            scores=self._scores,
            adjacency=adjacency,
            cognition_indexes=self._cognition_indexes.freeze(),
        )
        delta = GenerationDelta(
            nodes=tuple(self._nodes[memory_id] for memory_id in sorted(self._dirty_nodes, key=int)),
            scores=tuple(self._scores[memory_id] for memory_id in sorted(self._dirty_scores, key=int)),
            edges=tuple(
                EdgeState(
                    source_id=source_id,
                    relation_type=relation_type,
                    target_id=target_id,
                    support_count=self._edge_support.get((source_id, relation_type, target_id), 0),
                )
                for source_id, relation_type, target_id in sorted(
                    self._dirty_edges,
                    key=lambda key: (int(key[0]), key[1], int(key[2])),
                )
            ),
        )

        self._published_generation = self._mutable_generation
        self._mutable_generation = GenerationId(int(self._mutable_generation) + 1)
        self._published_view = view
        self._dirty_nodes.clear()
        self._dirty_scores.clear()
        self._dirty_edges.clear()
        self._first_global_step = None
        self._last_global_step = None
        return state, view, delta
