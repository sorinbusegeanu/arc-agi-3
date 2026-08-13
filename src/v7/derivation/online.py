from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import TYPE_CONCEPT, TYPE_CONTINGENCY, TYPE_FAMILY, TYPE_ROLE, TYPE_WORLD_MODEL
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryId, MemoryLevel
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
    """Bounded deterministic structural derivation for live v7 experiments.

    This builder does not invent perceptual objects. It promotes recurring canonical
    interaction structure already present in the memory graph. Empirical transfer is
    still required separately before an M4 candidate is treated as validated.
    """

    def __init__(self, writer: CanonicalMemoryWriter, pipeline: MemoryLearningPipeline) -> None:
        self.writer = writer
        self.pipeline = pipeline

    def derive(self) -> OnlineDerivationStats:
        registry = getattr(self.writer, "_canonical_registry")
        nodes = getattr(self.writer, "_nodes")
        cognition = getattr(self.writer, "_cognition_indexes")

        families = roles = concepts = world_models = strategies = 0

        # M1 -> M2: recurrence across distinct contexts for the same action/outcome.
        grouped_m1: dict[tuple[int, int], list[MemoryId]] = defaultdict(list)
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M1 or node.type_id != TYPE_CONTINGENCY:
                continue
            key = registry.key_for(memory_id)
            if key is None or len(key.parts) < 3:
                continue
            _context, action_id, outcome_signature = key.parts[:3]
            grouped_m1[(int(action_id), int(outcome_signature))].append(memory_id)

        family_members: dict[MemoryId, tuple[MemoryId, ...]] = {}
        for (action_id, outcome_signature), member_ids in sorted(grouped_m1.items()):
            members = tuple(sorted(set(member_ids), key=int))
            if len(members) < 2:
                continue
            before = len(nodes)
            family_id = self.pipeline.derive_m2(action_id=action_id, member_ids=members, outcome_class=outcome_signature)
            family_members[family_id] = members
            families += int(len(nodes) > before)

        # Include existing families so later levels keep progressing after restart.
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M2 or node.type_id != TYPE_FAMILY or memory_id in family_members:
                continue
            key = registry.key_for(memory_id)
            if key is None or len(key.parts) < 3:
                continue
            family_members[memory_id] = tuple(MemoryId(int(value)) for value in key.parts[2:])

        # M2 -> M3: one role position per distinct context participating in a family.
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
                before = len(nodes)
                role_id = self.pipeline.derive_m3(
                    family_id=family_id,
                    context_class=context,
                    action_id=action_id,
                    member_ids=tuple(sorted(set(context_members), key=int)),
                )
                roles_by_family[family_id].append(role_id)
                roles += int(len(nodes) > before)

        # Include pre-existing roles.
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M3 or node.type_id != TYPE_ROLE:
                continue
            key = registry.key_for(memory_id)
            if key is not None and key.parts:
                roles_by_family[MemoryId(int(key.parts[0]))].append(memory_id)

        # M3 -> M4: recurring family roles across at least two contexts form a concept candidate.
        for family_id, role_ids in sorted(roles_by_family.items(), key=lambda item: int(item[0])):
            unique_roles = tuple(sorted(set(role_ids), key=int))
            if len(unique_roles) < 2:
                continue
            relation_signature = _mix_signature(int(family_id), len(unique_roles))
            before = len(nodes)
            self.pipeline.derive_m4(role_ids=unique_roles, relation_signature=relation_signature)
            concepts += int(len(nodes) > before)

        # M4 -> M5 only after empirical transfer validation.
        validated_concepts = tuple(
            sorted(
                (
                    memory_id
                    for memory_id, node in nodes.items()
                    if node.level == MemoryLevel.M4
                    and node.type_id == TYPE_CONCEPT
                    and (int(node.status_flags) & int(ConceptValidationStatus.TRANSFER_VALIDATED))
                ),
                key=int,
            )
        )
        if len(validated_concepts) >= 2:
            transition_signature = _fold_signature(validated_concepts)
            before = len(nodes)
            self.pipeline.derive_m5(concept_ids=validated_concepts, transition_signature=transition_signature)
            world_models += int(len(nodes) > before)

        # M5 -> M6 after a positive observed future-option gain for an action.
        model_ids = tuple(
            sorted(
                (memory_id for memory_id, node in nodes.items() if node.level == MemoryLevel.M5 and node.type_id == TYPE_WORLD_MODEL),
                key=int,
            )
        )
        if model_ids:
            aggregates = cognition.freeze().action_aggregates
            for action_id, aggregate in sorted(aggregates.items()):
                if aggregate.future_option_count <= 0 or aggregate.future_option_mean <= 0:
                    continue
                before = len(nodes)
                self.pipeline.derive_m6(
                    world_model_ids=model_ids,
                    action_signature=int(action_id),
                    efficiency_gain=float(aggregate.future_option_mean),
                )
                strategies += int(len(nodes) > before)

        return OnlineDerivationStats(families, roles, concepts, world_models, strategies)


def _mix_signature(a: int, b: int) -> int:
    return ((int(a) * 6364136223846793005) ^ (int(b) * 1442695040888963407)) & _MASK63


def _fold_signature(memory_ids: tuple[MemoryId, ...]) -> int:
    value = 1469598103934665603
    for memory_id in memory_ids:
        value ^= int(memory_id)
        value = (value * 1099511628211) & _MASK63
    return value
