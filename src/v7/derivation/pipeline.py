from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v7.derivation.scientific import EpisodeEvidence, ScientificDerivationKernels
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregateDelta, ContingencyIndexMutation, RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class HierarchyDerivationResult:
    memory_id: MemoryId
    level: int


class MemoryLearningPipeline:
    """Clean v7 API that turns evidence and structural support into M1-M6 memories."""

    def __init__(self, writer: CanonicalMemoryWriter) -> None:
        self.writer = writer

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
        return memory_id

    def derive_m2(self, *, action_id: int, member_ids: Iterable[MemoryId], outcome_class: int) -> MemoryId:
        candidate = ScientificDerivationKernels.m2_family(action_id=action_id, member_ids=member_ids, outcome_class=outcome_class)
        return self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]

    def derive_m3(self, *, family_id: MemoryId, context_class: int, action_id: int, member_ids: Iterable[MemoryId]) -> MemoryId:
        candidate = ScientificDerivationKernels.m3_role(family_id=family_id, context_class=context_class, action_id=action_id, member_ids=member_ids)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self.writer.apply_role_index_batch((RoleIndexMutation(context_class, action_id, memory_id, family_id),))
        return memory_id

    def derive_m4(self, *, role_ids: Iterable[MemoryId], relation_signature: int) -> MemoryId:
        roles = tuple(role_ids)
        candidate = ScientificDerivationKernels.m4_concept(role_ids=roles, relation_signature=relation_signature)
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
        self.writer.apply_role_concept_index_batch(tuple(RoleConceptIndexMutation(role_id, memory_id) for role_id in sorted(set(roles), key=int)))
        return memory_id

    def derive_m5(self, *, concept_ids: Iterable[MemoryId], transition_signature: int) -> MemoryId:
        candidate = ScientificDerivationKernels.m5_world_model(concept_ids=concept_ids, transition_signature=transition_signature)
        return self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]

    def derive_m6(self, *, world_model_ids: Iterable[MemoryId], action_signature: int, efficiency_gain: float) -> MemoryId:
        candidate = ScientificDerivationKernels.m6_strategy(world_model_ids=world_model_ids, action_signature=action_signature, efficiency_gain=efficiency_gain)
        return self.writer.apply_canonical_candidate_batch((candidate,))[candidate.key]
