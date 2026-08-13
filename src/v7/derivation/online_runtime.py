from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import (
    TYPE_CONCEPT,
    TYPE_CONTINGENCY,
    TYPE_FAMILY,
    TYPE_ROLE,
    TYPE_STRATEGY,
    TYPE_WORLD_MODEL,
    ScientificDerivationKernels,
)
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, ProvenanceRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.models import NodeMutation, ScoreMutation
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
        return (
            self.families
            + self.roles
            + self.concepts
            + self.world_models
            + self.strategies
        )


class OnlineHierarchyBuilder:
    """Bounded contextual M2-M6 derivation; all canonical writes remain writer-owned."""

    def __init__(
        self,
        writer: CanonicalMemoryWriter,
        pipeline: MemoryLearningPipeline,
        evidence_store: EvidenceStore,
        lifecycle_store: EvidenceLifecycleStore,
    ) -> None:
        self.writer = writer
        self.pipeline = pipeline
        self.evidence_store = evidence_store
        self.lifecycle_store = lifecycle_store

    def derive(self) -> OnlineDerivationStats:
        nodes = getattr(self.writer, "_nodes")
        registry = getattr(self.writer, "_canonical_registry")
        families = roles = concepts = world_models = strategies = 0

        grouped: dict[tuple[int, int], list[MemoryId]] = defaultdict(list)
        contexts: dict[tuple[int, int], dict[int, list[MemoryId]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M1 or node.type_id != TYPE_CONTINGENCY:
                continue
            key = registry.key_for(memory_id)
            if key is None or len(key.parts) < 3:
                continue
            context, action, outcome = map(int, key.parts[:3])
            grouped[(action, outcome)].append(memory_id)
            contexts[(action, outcome)][context].append(memory_id)

        family_by_member: dict[MemoryId, MemoryId] = {}
        role_by_member: dict[MemoryId, MemoryId] = {}
        for (action, outcome), raw_members in sorted(grouped.items()):
            members = tuple(sorted(set(raw_members), key=int))
            if len(members) < 2:
                continue
            family_candidate = ScientificDerivationKernels.m2_family(
                action_id=action,
                member_ids=members,
                outcome_class=outcome,
            )
            family = self.writer.canonical_memory_id(family_candidate.key)
            if family is None:
                family = self.pipeline.derive_m2(
                    action_id=action,
                    member_ids=members,
                    outcome_class=outcome,
                )
                families += 1
            else:
                self._add_parent_support(
                    family, MemoryLevel.M2, TYPE_FAMILY, members
                )
            for member in members:
                family_by_member[member] = family

            for context, context_members_raw in sorted(
                contexts[(action, outcome)].items()
            ):
                context_members = tuple(
                    sorted(set(context_members_raw), key=int)
                )
                if not context_members:
                    continue
                role_key = self._role_key(
                    family, action, context, context_members
                )
                role = self.writer.canonical_memory_id(role_key)
                if role is None:
                    role = self.pipeline.derive_m3(
                        family_id=family,
                        context_class=context,
                        action_id=action,
                        member_ids=context_members,
                    )
                    roles += 1
                else:
                    self._add_parent_support(
                        role, MemoryLevel.M3, TYPE_ROLE, context_members
                    )
                    self.writer.apply_role_index_batch(
                        (RoleIndexMutation(context, action, role, family),)
                    )
                for member in context_members:
                    role_by_member[member] = role

        episodes = self._load(EvidenceType.EPISODE)
        carrier_roles: dict[int, set[MemoryId]] = defaultdict(set)
        for row in episodes:
            carrier = row.get("carrier_signature")
            memory_id = row.get("memory_id")
            if carrier is None or memory_id is None:
                continue
            role = role_by_member.get(MemoryId(int(memory_id)))
            if role is not None:
                carrier_roles[int(carrier)].add(role)
        for carrier, raw_roles in sorted(carrier_roles.items()):
            role_ids = tuple(sorted(raw_roles, key=int))
            if len(role_ids) < 2:
                continue
            concept_candidate = ScientificDerivationKernels.m4_concept(
                role_ids=role_ids, relation_signature=carrier
            )
            concept = self.writer.canonical_memory_id(concept_candidate.key)
            if concept is None:
                concept = self.pipeline.derive_m4(
                    role_ids=role_ids, relation_signature=carrier
                )
                concepts += 1
            else:
                self._add_parent_support(
                    concept, MemoryLevel.M4, TYPE_CONCEPT, role_ids
                )
                self.writer.apply_role_concept_index_batch(
                    RoleConceptIndexMutation(role, concept) for role in role_ids
                )

        self._update_strategy_reuse_scores(episodes)

        eligible_concepts = {
            int(memory_id)
            for memory_id, node in nodes.items()
            if node.level == MemoryLevel.M4
            and node.type_id == TYPE_CONCEPT
            and node.support_count >= 2
            and not (
                int(node.status_flags)
                & int(ConceptValidationStatus.TRANSFER_REJECTED)
            )
        }
        transition_counts: Counter[int] = Counter()
        transition_concepts: dict[int, set[MemoryId]] = defaultdict(set)
        transition_locations: dict[int, set[tuple[int, int]]] = defaultdict(set)
        by_game: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in episodes:
            if row.get("source_game"):
                by_game[str(row["source_game"])].append(row)

        for game in sorted(by_game):
            prior_concepts: tuple[int, ...] = ()
            prior_action: int | None = None
            prior_context: int | None = None
            for row in sorted(
                by_game[game],
                key=lambda item: int(item.get("source_global_step") or -1),
            ):
                current = tuple(
                    sorted(
                        {
                            int(value)
                            for value in row.get("decision_concept_ids", ())
                            if int(value) in eligible_concepts
                        }
                    )
                )
                if (
                    prior_concepts
                    and current
                    and prior_action is not None
                    and prior_context is not None
                ):
                    union = tuple(sorted(set(prior_concepts) | set(current)))
                    if len(union) >= 2:
                        signature = _transition_key(
                            prior_concepts, prior_action, current
                        )
                        transition_counts[signature] += 1
                        transition_concepts[signature].update(
                            MemoryId(value) for value in union
                        )
                        transition_locations[signature].add(
                            (prior_context, prior_action)
                        )
                prior_concepts = current
                prior_action = int(row.get("action_id") or 0)
                context_values = tuple(
                    int(value)
                    for value in row.get("context_signatures", ()) or ()
                )
                prior_context = (
                    context_values[-2]
                    if len(context_values) >= 2
                    else int(row.get("context_signature") or 0)
                )

        models_by_location: dict[tuple[int, int], set[MemoryId]] = defaultdict(set)
        for signature, count in sorted(transition_counts.items()):
            concept_ids = tuple(
                sorted(transition_concepts[signature], key=int)
            )
            if count < 2 or len(concept_ids) < 2:
                continue
            candidate = ScientificDerivationKernels.m5_world_model(
                concept_ids=concept_ids, transition_signature=signature
            )
            model = self.writer.canonical_memory_id(candidate.key)
            if model is None:
                model = self.pipeline.derive_m5(
                    concept_ids=concept_ids, transition_signature=signature
                )
                world_models += 1
            else:
                self._add_parent_support(
                    model, MemoryLevel.M5, TYPE_WORLD_MODEL, concept_ids
                )
            for context, action in sorted(transition_locations[signature]):
                # The packed context/action lookup is intentionally shared by
                # M3, M5 and M6; readers distinguish them by memory level.
                self.writer.apply_role_index_batch(
                    (RoleIndexMutation(context, action, model, None),)
                )
                models_by_location[(context, action)].add(model)

        all_models = tuple(
            sorted(
                (
                    memory_id
                    for memory_id, node in nodes.items()
                    if node.level == MemoryLevel.M5
                    and node.type_id == TYPE_WORLD_MODEL
                ),
                key=int,
            )
        )[:64]
        trajectories = self._load(EvidenceType.TRAJECTORY)
        best_steps: dict[tuple[str, str], int] = {}
        for row in trajectories:
            if not bool(row.get("success")):
                continue
            actions = tuple(
                int(value) for value in row.get("action_sequence", ()) or ()
            )
            contexts_seq = tuple(
                int(value) for value in row.get("context_sequence", ()) or ()
            )
            if not actions:
                representative = row.get("representative_action")
                if representative is None:
                    continue
                actions = (int(representative),)
            if not contexts_seq:
                contexts_seq = tuple(
                    int(row.get("source_context") or 0) for _ in actions
                )
            pair_count = min(len(actions), len(contexts_seq))
            if pair_count <= 0:
                continue
            actions = actions[:pair_count]
            contexts_seq = contexts_seq[:pair_count]
            relevant_models: set[MemoryId] = set()
            for context, action in zip(contexts_seq, actions, strict=True):
                relevant_models.update(models_by_location.get((context, action), ()))
            model_ids = tuple(sorted(relevant_models, key=int))[:64] or all_models
            if not model_ids:
                continue
            steps = max(1, int(row.get("steps_to_success") or len(actions)))
            group_key = (
                str(row.get("source_game") or ""),
                str(row.get("level_key") or "level"),
            )
            previous_best = best_steps.get(group_key)
            base_efficiency = 1.0 / float(steps)
            improvement = 0.0
            if previous_best is not None and steps < previous_best:
                improvement = (previous_best - steps) / max(1.0, float(previous_best))
            best_steps[group_key] = (
                steps if previous_best is None else min(previous_best, steps)
            )
            efficiency = min(1.0, base_efficiency + improvement)
            action_signature = _sequence_key(actions, contexts_seq)
            candidate = ScientificDerivationKernels.m6_strategy(
                world_model_ids=model_ids,
                action_signature=action_signature,
                efficiency_gain=efficiency,
            )
            existed = self.writer.canonical_memory_id(candidate.key) is not None
            strategy = self.pipeline.derive_m6(
                world_model_ids=model_ids,
                action_signature=action_signature,
                efficiency_gain=efficiency,
            )
            if not existed:
                strategies += 1
            for context, action in zip(contexts_seq, actions, strict=True):
                self.writer.apply_role_index_batch(
                    (RoleIndexMutation(context, action, strategy, None),)
                )

        return OnlineDerivationStats(
            families, roles, concepts, world_models, strategies
        )

    @staticmethod
    def _role_key(
        family_id: MemoryId,
        action_id: int,
        context_class: int,
        member_ids: Iterable[MemoryId],
    ):
        del member_ids
        from v7.memory.canonical import CanonicalMemoryKey

        return CanonicalMemoryKey(
            MemoryLevel.M3,
            TYPE_ROLE,
            (int(family_id), int(action_id), int(context_class)),
        )

    def _update_strategy_reuse_scores(
        self, episodes: list[dict[str, object]]
    ) -> None:
        outcomes: dict[MemoryId, list[int]] = defaultdict(list)
        for row in episodes:
            polarity = int(row.get("terminal_polarity") or 0)
            if polarity == 0:
                continue
            for raw_id in row.get("decision_strategy_ids", ()) or ():
                memory_id = MemoryId(int(raw_id))
                node = getattr(self.writer, "_nodes").get(memory_id)
                if node is not None and node.level == MemoryLevel.M6:
                    outcomes[memory_id].append(polarity)
        mutations = []
        for memory_id, values in sorted(outcomes.items(), key=lambda item: int(item[0])):
            successes = sum(1 for value in values if value > 0)
            failures = sum(1 for value in values if value < 0)
            total = successes + failures
            if total <= 0:
                continue
            success_rate = successes / total
            mutations.append(
                ScoreMutation(
                    memory_id=memory_id,
                    significance=success_rate,
                    learning_value=success_rate,
                    future_option_delta=success_rate - (failures / total),
                )
            )
        if mutations:
            self.writer.apply_score_batch(mutations)

    def _add_parent_support(
        self,
        memory_id: MemoryId,
        level: MemoryLevel,
        type_id: int,
        parents: Iterable[MemoryId],
    ) -> None:
        existing = set(self.lifecycle_store.provenance_parents(memory_id))
        new = tuple(sorted(set(parents) - existing, key=int))
        if not new:
            return
        self.writer.apply_mutation_batch(
            (NodeMutation(memory_id, level, type_id, support_delta=len(new)),)
        )
        generation = int(self.writer.mutable_generation_id)
        self.lifecycle_store.append_provenance(
            ProvenanceRecord(
                memory_id=memory_id,
                parent_memory_id=parent,
                generation_id=generation,
            )
            for parent in new
        )

    def _load(self, evidence_type: EvidenceType) -> list[dict[str, object]]:
        rows = self.evidence_store.connection.execute(
            "SELECT memory_id,source_game,source_context,source_global_step,payload_json,generation_id "
            "FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
            (int(evidence_type),),
        ).fetchall()
        result = []
        for memory_id, game, context, step, payload_json, generation in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except json.JSONDecodeError:
                payload = {}
            payload.update(
                {
                    "memory_id": None if memory_id is None else int(memory_id),
                    "source_game": game,
                    "source_context": context,
                    "source_global_step": step,
                    "generation_id": int(generation),
                }
            )
            result.append(payload)
        return result


def _transition_key(
    prior: tuple[int, ...], action_id: int, current: tuple[int, ...]
) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"world-transition-v2")
    digest.update(str(tuple(prior)).encode("ascii"))
    digest.update(str(int(action_id)).encode("ascii"))
    digest.update(str(tuple(current)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & _MASK63


def _sequence_key(actions: tuple[int, ...], contexts: tuple[int, ...]) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"strategy-sequence-v1")
    digest.update(str(tuple(actions)).encode("ascii"))
    digest.update(str(tuple(contexts)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & _MASK63
