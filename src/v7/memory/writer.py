from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from v7.derivation.dependencies import (
    DependencyMutation,
    DirtyDerivationPlan,
    MemoryDependencyGraph,
)
from v7.memory.canonical import (
    CanonicalCandidateMutation,
    CanonicalMemoryKey,
    CanonicalMemoryRegistry,
)
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
from v7.memory.models import (
    EdgeMutation,
    EdgeState,
    MemoryNode,
    MemoryScore,
    NodeMutation,
    ScoreMutation,
)
from v7.memory.read_view import MemoryReadView
from v7.memory.state import (
    CognitiveState,
    GateId,
    GateValidationState,
    gate_for_identity,
)

PreparedGeneration = tuple[GenerationState, MemoryReadView, GenerationDelta]


class CanonicalMemoryWriter:
    """Single-owner mutable frontier for v7 active semantic memory."""

    def __init__(
        self,
        *,
        initial_generation: int = 0,
        gate_candidates: bool = False,
    ) -> None:
        if initial_generation < 0:
            raise ValueError("initial_generation must be non-negative")
        self.gate_candidates = bool(gate_candidates)
        self._published_generation = GenerationId(initial_generation)
        self._mutable_generation = GenerationId(initial_generation + 1)
        self._nodes: dict[MemoryId, MemoryNode] = {}
        self._scores: dict[MemoryId, MemoryScore] = {}
        self._edge_support: dict[tuple[MemoryId, int, MemoryId], int] = {}
        self._cognition_indexes = CognitionIndexBuilder()
        self._canonical_registry = CanonicalMemoryRegistry()
        self._dependencies = MemoryDependencyGraph()
        self._dirty_nodes: set[MemoryId] = set()
        self._dirty_scores: set[MemoryId] = set()
        self._dirty_edges: set[tuple[MemoryId, int, MemoryId]] = set()
        self._cognition_dirty = False
        self._first_global_step: int | None = None
        self._last_global_step: int | None = None
        self._pending_generation: PreparedGeneration | None = None
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
    def has_pending_generation(self) -> bool:
        return self._pending_generation is not None

    @property
    def dirty_counts(self) -> dict[str, int]:
        return {
            "nodes": len(self._dirty_nodes),
            "scores": len(self._dirty_scores),
            "edges": len(self._dirty_edges),
            "derivation": self._dependencies.dirty_count,
            "cognition": int(self._cognition_dirty),
        }

    def _ensure_mutable(self) -> None:
        if self._pending_generation is not None:
            raise RuntimeError(
                "generation is prepared; finalize or abort it before mutating"
            )

    def canonical_memory_id(
        self,
        key: CanonicalMemoryKey,
    ) -> MemoryId | None:
        return self._canonical_registry.get(key)

    def observe_global_step(self, global_step: int) -> None:
        self._ensure_mutable()
        step = int(global_step)
        if step < 0:
            raise ValueError("global_step must be non-negative")
        if self._first_global_step is None:
            self._first_global_step = step
        self._last_global_step = (
            step
            if self._last_global_step is None
            else max(self._last_global_step, step)
        )

    def apply_mutation_batch(self, mutations: Iterable[NodeMutation]) -> int:
        self._ensure_mutable()
        coalesced: dict[MemoryId, NodeMutation] = {}
        for mutation in mutations:
            self._canonical_registry.observe_existing_id(mutation.memory_id)
            prior = coalesced.get(mutation.memory_id)
            if prior is None:
                coalesced[mutation.memory_id] = mutation
                continue
            if prior.level != mutation.level or prior.type_id != mutation.type_id:
                raise ValueError(
                    f"conflicting node identity for memory_id={int(mutation.memory_id)}"
                )
            coalesced[mutation.memory_id] = NodeMutation(
                mutation.memory_id,
                mutation.level,
                mutation.type_id,
                prior.support_delta + mutation.support_delta,
                mutation.status_flags
                if mutation.status_flags is not None
                else prior.status_flags,
                mutation.cognitive_state
                if mutation.cognitive_state is not None
                else prior.cognitive_state,
                mutation.validation_state
                if mutation.validation_state is not None
                else prior.validation_state,
                mutation.gate_id if mutation.gate_id is not None else prior.gate_id,
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
                    memory_id,
                    mutation.level,
                    mutation.type_id,
                    generation,
                    generation,
                    0 if mutation.status_flags is None else mutation.status_flags,
                    support_count,
                    int(CognitiveState.ACTIVE)
                    if mutation.cognitive_state is None
                    else int(mutation.cognitive_state),
                    int(GateValidationState.VALIDATED)
                    if mutation.validation_state is None
                    else int(mutation.validation_state),
                    int(GateId.NONE)
                    if mutation.gate_id is None
                    else int(mutation.gate_id),
                )
            else:
                if (
                    current.level != mutation.level
                    or current.type_id != mutation.type_id
                ):
                    raise ValueError(
                        f"memory identity is immutable for memory_id={int(memory_id)}"
                    )
                support_count = current.support_count + mutation.support_delta
                if support_count < 0:
                    raise ValueError("node support cannot be negative")
                staged[memory_id] = replace(
                    current,
                    updated_generation=generation,
                    status_flags=current.status_flags
                    if mutation.status_flags is None
                    else mutation.status_flags,
                    support_count=support_count,
                    cognitive_state=current.cognitive_state
                    if mutation.cognitive_state is None
                    else int(mutation.cognitive_state),
                    validation_state=current.validation_state
                    if mutation.validation_state is None
                    else int(mutation.validation_state),
                    gate_id=current.gate_id
                    if mutation.gate_id is None
                    else int(mutation.gate_id),
                )
        for memory_id, node in staged.items():
            self._dependencies.register_node(memory_id, node.level)
        self._nodes.update(staged)
        self._dirty_nodes.update(staged)
        if staged:
            self._dependencies.mark_dirty(staged)
        return len(coalesced)

    def _aggregate_candidate_score(
        self,
        memory_id: MemoryId,
        group: list[CanonicalCandidateMutation],
        field: str,
    ) -> float | None:
        observations: list[tuple[float, int]] = []
        for candidate in group:
            value = getattr(candidate, field)
            if value is None:
                continue
            observations.append(
                (float(value), max(1, int(candidate.support_delta)))
            )
        if not observations:
            return None
        new_weight = sum(weight for _value, weight in observations)
        new_total = sum(value * weight for value, weight in observations)
        current_node = self._nodes.get(memory_id)
        current_score = self._scores.get(memory_id)
        current_support = (
            0 if current_node is None else max(0, int(current_node.support_count))
        )
        if current_score is None or current_support <= 0:
            return new_total / max(1, new_weight)
        current_value = float(getattr(current_score, field))
        return (
            current_value * current_support + new_total
        ) / max(1, current_support + new_weight)

    def apply_canonical_candidate_batch(
        self,
        candidates: Iterable[CanonicalCandidateMutation],
    ) -> dict[CanonicalMemoryKey, MemoryId]:
        self._ensure_mutable()
        rows = tuple(candidates)
        if not rows:
            return {}
        grouped: dict[CanonicalMemoryKey, list[CanonicalCandidateMutation]] = {}
        for candidate in rows:
            grouped.setdefault(candidate.key, []).append(candidate)
        resolved = self._canonical_registry.resolve_many(grouped)
        node_mutations: list[NodeMutation] = []
        score_mutations: list[ScoreMutation] = []
        dependencies: list[DependencyMutation] = []
        for key in sorted(grouped):
            group = grouped[key]
            memory_id = resolved[key]
            support_delta = sum(item.support_delta for item in group)
            current = self._nodes.get(memory_id)
            gate = gate_for_identity(key.level, key.type_id)
            cognitive_state = None
            validation_state = None
            gate_id = None
            if self.gate_candidates and current is None and gate != GateId.NONE:
                cognitive_state = int(CognitiveState.PROBE_ONLY)
                validation_state = int(
                    GateValidationState.PROBE_ELIGIBLE
                    if support_delta >= 2
                    else GateValidationState.STRUCTURAL_CANDIDATE
                )
                gate_id = int(gate)
            node_mutations.append(
                NodeMutation(
                    memory_id,
                    key.level,
                    key.type_id,
                    support_delta=support_delta,
                    cognitive_state=cognitive_state,
                    validation_state=validation_state,
                    gate_id=gate_id,
                )
            )
            score_mutations.append(
                ScoreMutation(
                    memory_id=memory_id,
                    significance=self._aggregate_candidate_score(
                        memory_id, group, "significance"
                    ),
                    prediction_error=self._aggregate_candidate_score(
                        memory_id, group, "prediction_error"
                    ),
                    learning_value=self._aggregate_candidate_score(
                        memory_id, group, "learning_value"
                    ),
                    transfer_prior=self._aggregate_candidate_score(
                        memory_id, group, "transfer_prior"
                    ),
                    explanatory_potential=self._aggregate_candidate_score(
                        memory_id, group, "explanatory_potential"
                    ),
                    future_option_delta=self._aggregate_candidate_score(
                        memory_id, group, "future_option_delta"
                    ),
                )
            )
            parents = tuple(
                sorted(
                    {parent for item in group for parent in item.parents},
                    key=int,
                )
            )
            for parent in parents:
                parent_node = self._nodes.get(parent)
                if (
                    parent_node is not None
                    and int(parent_node.level) < int(key.level)
                ):
                    dependencies.append(
                        DependencyMutation(
                            parent,
                            parent_node.level,
                            memory_id,
                            key.level,
                        )
                    )
        if dependencies:
            self.apply_dependency_batch(dependencies)
        self.apply_mutation_batch(node_mutations)
        self.apply_score_batch(score_mutations)
        return resolved

    def apply_dependency_batch(
        self,
        mutations: Iterable[DependencyMutation],
    ) -> int:
        self._ensure_mutable()
        return self._dependencies.apply_dependency_batch(mutations)

    def dirty_derivation_plan(self) -> DirtyDerivationPlan:
        return self._dependencies.snapshot_plan()

    def consume_dirty_derivation_plan(self) -> DirtyDerivationPlan:
        self._ensure_mutable()
        return self._dependencies.consume_plan()

    def apply_edge_batch(self, mutations: Iterable[EdgeMutation]) -> int:
        self._ensure_mutable()
        coalesced: dict[tuple[MemoryId, int, MemoryId], int] = {}
        for mutation in mutations:
            key = (
                mutation.source_id,
                int(mutation.relation_type),
                mutation.target_id,
            )
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
        self._ensure_mutable()
        coalesced: dict[MemoryId, ScoreMutation] = {}
        for mutation in mutations:
            prior = coalesced.get(mutation.memory_id)
            if prior is None:
                coalesced[mutation.memory_id] = mutation
                continue
            coalesced[mutation.memory_id] = ScoreMutation(
                memory_id=mutation.memory_id,
                significance=mutation.significance
                if mutation.significance is not None
                else prior.significance,
                prediction_error=mutation.prediction_error
                if mutation.prediction_error is not None
                else prior.prediction_error,
                learning_value=mutation.learning_value
                if mutation.learning_value is not None
                else prior.learning_value,
                transfer_prior=mutation.transfer_prior
                if mutation.transfer_prior is not None
                else prior.transfer_prior,
                explanatory_potential=mutation.explanatory_potential
                if mutation.explanatory_potential is not None
                else prior.explanatory_potential,
                future_option_delta=mutation.future_option_delta
                if mutation.future_option_delta is not None
                else prior.future_option_delta,
            )
        staged: dict[MemoryId, MemoryScore] = {}
        for memory_id, mutation in coalesced.items():
            current = self._scores.get(
                memory_id,
                MemoryScore(memory_id=memory_id),
            )
            staged[memory_id] = MemoryScore(
                memory_id=memory_id,
                significance=current.significance
                if mutation.significance is None
                else float(mutation.significance),
                prediction_error=current.prediction_error
                if mutation.prediction_error is None
                else float(mutation.prediction_error),
                learning_value=current.learning_value
                if mutation.learning_value is None
                else float(mutation.learning_value),
                transfer_prior=current.transfer_prior
                if mutation.transfer_prior is None
                else float(mutation.transfer_prior),
                explanatory_potential=current.explanatory_potential
                if mutation.explanatory_potential is None
                else float(mutation.explanatory_potential),
                future_option_delta=current.future_option_delta
                if mutation.future_option_delta is None
                else float(mutation.future_option_delta),
            )
        self._scores.update(staged)
        self._dirty_scores.update(staged)
        return len(coalesced)

    def apply_contingency_index_batch(
        self,
        mutations: Iterable[ContingencyIndexMutation],
    ) -> int:
        self._ensure_mutable()
        count = self._cognition_indexes.apply_contingency_batch(mutations)
        self._cognition_dirty |= count > 0
        return count

    def apply_role_index_batch(
        self,
        mutations: Iterable[RoleIndexMutation],
    ) -> int:
        self._ensure_mutable()
        count = self._cognition_indexes.apply_role_batch(mutations)
        self._cognition_dirty |= count > 0
        return count

    def apply_role_concept_index_batch(
        self,
        mutations: Iterable[RoleConceptIndexMutation],
    ) -> int:
        self._ensure_mutable()
        count = self._cognition_indexes.apply_role_concept_batch(mutations)
        self._cognition_dirty |= count > 0
        return count

    def apply_action_aggregate_batch(
        self,
        deltas: Iterable[ActionAggregateDelta],
    ) -> int:
        self._ensure_mutable()
        count = self._cognition_indexes.apply_action_aggregate_batch(deltas)
        self._cognition_dirty |= count > 0
        return count

    def prepare_generation(self) -> PreparedGeneration:
        if self._pending_generation is not None:
            return self._pending_generation
        adjacency: dict[tuple[MemoryId, int], list[MemoryId]] = {}
        for (source_id, relation_type, target_id), support in self._edge_support.items():
            if support > 0:
                adjacency.setdefault((source_id, relation_type), []).append(target_id)
        state = GenerationState(
            self._mutable_generation,
            self._published_generation,
            self._first_global_step,
            self._last_global_step,
        )
        view = MemoryReadView.freeze(
            generation_id=self._mutable_generation,
            nodes=self._nodes,
            scores=self._scores,
            adjacency=adjacency,
            cognition_indexes=self._cognition_indexes.freeze(),
            previous_view=self._published_view,
            nodes_dirty=bool(self._dirty_nodes),
            scores_dirty=bool(self._dirty_scores),
            adjacency_dirty=bool(self._dirty_edges),
            cognition_dirty=self._cognition_dirty or bool(self._dirty_nodes),
        )
        delta = GenerationDelta(
            nodes=tuple(
                self._nodes[memory_id]
                for memory_id in sorted(self._dirty_nodes, key=int)
            ),
            scores=tuple(
                self._scores[memory_id]
                for memory_id in sorted(self._dirty_scores, key=int)
            ),
            edges=tuple(
                EdgeState(
                    source_id,
                    relation_type,
                    target_id,
                    self._edge_support.get(
                        (source_id, relation_type, target_id),
                        0,
                    ),
                )
                for source_id, relation_type, target_id in sorted(
                    self._dirty_edges,
                    key=lambda key: (int(key[0]), key[1], int(key[2])),
                )
            ),
        )
        self._pending_generation = (state, view, delta)
        return self._pending_generation

    def finalize_generation(self) -> PreparedGeneration:
        prepared = self._pending_generation
        if prepared is None:
            raise RuntimeError("no prepared generation to finalize")
        state, view, delta = prepared
        self._published_generation = state.generation_id
        self._mutable_generation = GenerationId(int(state.generation_id) + 1)
        self._published_view = view
        self._dirty_nodes.clear()
        self._dirty_scores.clear()
        self._dirty_edges.clear()
        self._cognition_dirty = False
        self._first_global_step = None
        self._last_global_step = None
        self._pending_generation = None
        return state, view, delta

    def abort_generation(self) -> None:
        if self._pending_generation is None:
            raise RuntimeError("no prepared generation to abort")
        self._pending_generation = None

    def commit_generation(self) -> PreparedGeneration:
        self.prepare_generation()
        return self.finalize_generation()
