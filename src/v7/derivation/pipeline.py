from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v7.derivation.scientific import EpisodeEvidence, ScientificDerivationKernels
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, ProvenanceRecord
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregateDelta, ContingencyIndexMutation, RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class HierarchyDerivationResult:
    memory_id: MemoryId
    level: int


class MemoryLearningPipeline:
    """Clean v7 API that turns evidence and structural support into M1-M6 memories."""

    def __init__(self, writer: CanonicalMemoryWriter, evidence_lifecycle: EvidenceLifecycleStore | None = None) -> None:
        self.writer = writer
        self.evidence_lifecycle = evidence_lifecycle

    def _record_parents(self, memory_id: MemoryId, parents: Iterable[MemoryId]) -> None:
        if self.evidence_lifecycle is None:
            return
        generation_id = int(self.writer.mutable_generation_id)
        rows = tuple(
            ProvenanceRecord(memory_id=memory_id, parent_memory_id=parent_id, generation_id=generation_id)
            for parent_id in sorted(set(parents), key=int)
        )
        if rows:
            self.evidence_lifecycle.append_provenance(rows)

    def observe_episode(self, evidence: EpisodeEvidence) -> MemoryId:
        candidate = ScientificDerivationKernels.m1_from_episode(evidence)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self.writer.apply_contingency_index_batch((ContingencyIndexMutation(evidence.context_signature, evidence.action_id, memory_id),))
        self.writer.apply_action_aggregate_batch((ActionAggregateDelta(
            action_id=evidence.action_id,
            future_option_sum_delta=evidence.future_option_delta,
            future_option_count_delta=1,
            positive_count_delta=1 if evidence.success else 0,
            negative_count_delta=0 if evidence.success else 1,
            failure_count_delta=0 if evidence.success else 1,
        ),))
        if self.evidence_lifecycle is not None:
            self.evidence_lifecycle.append_provenance((ProvenanceRecord(
                memory_id=memory_id,
                generation_id=int(self.writer.mutable_generation_id),
                source_game=evidence.source_game,
                source_context=evidence.source_context,
                source_global_step=evidence.source_global_step,
            ),))
        return memory_id

    def derive_m2(self, *, action_id: int, member_ids: Iterable[MemoryId], outcome_class: int) -> MemoryId:
        members = tuple(member_ids)
        candidate = ScientificDerivationKernels.m2_family(action_id=action_id, member_ids=members, outcome_class=outcome_class)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m3(self, *, family_id: MemoryId, context_class: int, action_id: int, member_ids: Iterable[MemoryId]) -> MemoryId:
        members = tuple(member_ids)
        candidate = ScientificDerivationKernels.m3_role(family_id=family_id, context_class=context_class, action_id=action_id, member_ids=members)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self.writer.apply_role_index_batch((RoleIndexMutation(context_class, action_id, memory_id, family_id),))
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m4(self, *, role_ids: Iterable[MemoryId], relation_signature: int) -> MemoryId:
        roles = tuple(role_ids)
        candidate = ScientificDerivationKernels.m4_concept(role_ids=roles, relation_signature=relation_signature)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self.writer.apply_role_concept_index_batch(tuple(RoleConceptIndexMutation(role_id, memory_id) for role_id in sorted(set(roles), key=int)))
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m5(self, *, concept_ids: Iterable[MemoryId], transition_signature: int) -> MemoryId:
        concepts = tuple(concept_ids)
        candidate = ScientificDerivationKernels.m5_world_model(concept_ids=concepts, transition_signature=transition_signature)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m6(self, *, world_model_ids: Iterable[MemoryId], action_signature: int, efficiency_gain: float) -> MemoryId:
        models = tuple(world_model_ids)
        candidate = ScientificDerivationKernels.m6_strategy(world_model_ids=models, action_signature=action_signature, efficiency_gain=efficiency_gain)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self._record_parents(memory_id, candidate.parents)
        return memory_id
