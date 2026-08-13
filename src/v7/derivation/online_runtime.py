from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import TYPE_CONCEPT, TYPE_CONTINGENCY, TYPE_ROLE, TYPE_WORLD_MODEL, ScientificDerivationKernels
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleConceptIndexMutation
from v7.memory.writer import CanonicalMemoryWriter

_MASK63 = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class OnlineDerivationStats:
    families: int = 0
    roles: int = 0
    concepts: int = 0
    world_models: int = 0
    strategies: int = 0

    @property
    def total(self) -> int:
        return self.families + self.roles + self.concepts + self.world_models + self.strategies


class OnlineHierarchyBuilder:
    """Derive bounded higher-order memories from committed interaction evidence."""

    def __init__(self, writer: CanonicalMemoryWriter, pipeline: MemoryLearningPipeline) -> None:
        self.writer = writer
        self.pipeline = pipeline

    def derive(self) -> OnlineDerivationStats:
        registry = getattr(self.writer, "_canonical_registry")
        nodes = getattr(self.writer, "_nodes")
        cognition = getattr(self.writer, "_cognition_indexes")
        families = roles = concepts = world_models = strategies = 0

        grouped_m1: dict[tuple[int, int], list[MemoryId]] = defaultdict(list)
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M1 or node.type_id != TYPE_CONTINGENCY:
                continue
            key = registry.key_for(memory_id)
            if key is None or len(key.parts) < 3:
                continue
            _, action_id, outcome_signature = key.parts[:3]
            grouped_m1[(int(action_id), int(outcome_signature))].append(memory_id)

        family_members: dict[MemoryId, tuple[MemoryId, ...]] = {}
        for (action_id, outcome_signature), member_ids in sorted(grouped_m1.items()):
            members = tuple(sorted(set(member_ids), key=int))
            if len(members) < 2:
                continue
            candidate = ScientificDerivationKernels.m2_family(action_id=action_id, member_ids=members, outcome_class=outcome_signature)
            family_id = self.writer.canonical_memory_id(candidate.key)
            if family_id is None:
                family_id = self.pipeline.derive_m2(action_id=action_id, member_ids=members, outcome_class=outcome_signature)
                families += 1
            family_members[family_id] = members

        roles_by_family: dict[MemoryId, list[MemoryId]] = defaultdict(list)
        for family_id, members in sorted(family_members.items(), key=lambda item: int(item[0])):
            family_key = registry.key_for(family_id)
            if family_key is None or len(family_key.parts) < 2:
                continue
            action_id = int(family_key.parts[0])
            by_context: dict[int, list[MemoryId]] = defaultdict(list)
            for member_id in members:
                member_key = registry.key_for(member_id)
                if member_key is not None and len(member_key.parts) >= 3:
                    by_context[int(member_key.parts[0])].append(member_id)
            for context, context_members in sorted(by_context.items()):
                unique_members = tuple(sorted(set(context_members), key=int))
                candidate = ScientificDerivationKernels.m3_role(family_id=family_id, context_class=context, action_id=action_id, member_ids=unique_members)
                role_id = self.writer.canonical_memory_id(candidate.key)
                if role_id is None:
                    role_id = self.pipeline.derive_m3(family_id=family_id, context_class=context, action_id=action_id, member_ids=unique_members)
                    roles += 1
                roles_by_family[family_id].append(role_id)

        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M3 or node.type_id != TYPE_ROLE:
                continue
            key = registry.key_for(memory_id)
            if key is not None and key.parts:
                roles_by_family[MemoryId(int(key.parts[0]))].append(memory_id)

        for family_id, role_ids in sorted(roles_by_family.items(), key=lambda item: int(item[0])):
            unique_roles = tuple(sorted(set(role_ids), key=int))
            if len(unique_roles) < 2:
                continue
            relation_signature = _mix_signature(int(family_id), 0)
            candidate = ScientificDerivationKernels.m4_concept(role_ids=unique_roles, relation_signature=relation_signature)
            concept_id = self.writer.canonical_memory_id(candidate.key)
            if concept_id is None:
                concept_id = self.pipeline.derive_m4(role_ids=unique_roles, relation_signature=relation_signature)
                concepts += 1
            else:
                self.writer.apply_role_concept_index_batch(RoleConceptIndexMutation(role_id, concept_id) for role_id in unique_roles)

        validated_concepts = tuple(sorted((memory_id for memory_id, node in nodes.items() if node.level == MemoryLevel.M4 and node.type_id == TYPE_CONCEPT and (int(node.status_flags) & int(ConceptValidationStatus.TRANSFER_VALIDATED))), key=int))
        if len(validated_concepts) >= 2:
            transition_signature = _fold_signature(validated_concepts)
            candidate = ScientificDerivationKernels.m5_world_model(concept_ids=validated_concepts, transition_signature=transition_signature)
            if self.writer.canonical_memory_id(candidate.key) is None:
                self.pipeline.derive_m5(concept_ids=validated_concepts, transition_signature=transition_signature)
                world_models += 1

        model_ids = tuple(sorted((memory_id for memory_id, node in nodes.items() if node.level == MemoryLevel.M5 and node.type_id == TYPE_WORLD_MODEL), key=int))
        if model_ids:
            for action_id, aggregate in sorted(cognition.freeze().action_aggregates.items()):
                if aggregate.future_option_count <= 0 or aggregate.future_option_mean <= 0:
                    continue
                candidate = ScientificDerivationKernels.m6_strategy(world_model_ids=model_ids, action_signature=int(action_id), efficiency_gain=float(aggregate.future_option_mean))
                if self.writer.canonical_memory_id(candidate.key) is None:
                    self.pipeline.derive_m6(world_model_ids=model_ids, action_signature=int(action_id), efficiency_gain=float(aggregate.future_option_mean))
                    strategies += 1

        return OnlineDerivationStats(families, roles, concepts, world_models, strategies)


def _mix_signature(a: int, b: int) -> int:
    return ((int(a) * 6364136223846793005) ^ (int(b) * 1442695040888963407)) & _MASK63


def _fold_signature(memory_ids: tuple[MemoryId, ...]) -> int:
    value = 1469598103934665603
    for memory_id in memory_ids:
        value ^= int(memory_id)
        value = (value * 1099511628211) & _MASK63
    return value
