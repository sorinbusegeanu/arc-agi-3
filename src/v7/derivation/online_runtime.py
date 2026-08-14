from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import blake2b
from typing import Any, Iterable

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import (
    TYPE_CARRIER,
    TYPE_CONCEPT,
    TYPE_CONTEXTUAL_ROLE,
    TYPE_CONTINGENCY,
    TYPE_FAMILY,
    TYPE_ROLE,
    TYPE_STRATEGY,
    TYPE_WORLD_MODEL,
    ScientificDerivationKernels,
)
from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.developmental_policy import profile_for_view
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, ProvenanceRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.planning import planning_context
from v7.memory.semantic_relations import (
    TYPE_RELATIONAL_WORLD_MODEL,
    classify_transition_relations,
)
from v7.memory.writer import CanonicalMemoryWriter

_MASK63 = (1 << 63) - 1


def _retrieval_contexts(row: dict[str, Any]) -> tuple[int, ...]:
    """Return bounded general/structural/planning contexts for transfer lookup.

    New five-signature evidence is indexed at C0, C2, and C3 so functional
    roles and world models learned in one game are reachable in another game
    before an exact target context has accumulated evidence. Legacy layouts
    retain their reusable general and planning identities.
    """
    values = tuple(
        int(value) for value in row.get("context_signatures", ()) or ()
    )
    fallback = int(row.get("context_signature") or 0)
    if len(values) >= 5:
        candidates = (values[0], values[2], values[3])
    elif len(values) >= 3:
        candidates = (values[0], planning_context(values, fallback=fallback))
    elif values:
        candidates = (values[0], values[-1])
    else:
        candidates = (fallback,)
    return tuple(dict.fromkeys(int(value) for value in candidates))


@dataclass(frozen=True, slots=True)
class OnlineDerivationStats:
    families: int = 0
    carriers: int = 0
    contextual_roles: int = 0
    roles: int = 0
    concepts: int = 0
    world_models: int = 0
    relational_models: int = 0
    strategies: int = 0
    stage: str = "CONTROL"

    @property
    def total(self) -> int:
        return (
            self.families
            + self.carriers
            + self.contextual_roles
            + self.roles
            + self.concepts
            + self.world_models
            + self.relational_models
            + self.strategies
        )


class OnlineHierarchyBuilder:
    """Bounded M2-M6 derivation with explicit carriers and relational M5.

    Phase 2 changes M2 to action-independent transformation identity and splits
    M3 into carrier -> contextual role instance -> functional role. Phase 3
    requires transfer-validated concepts for M5 and derives typed relational
    world-model evidence from repeated abstract transitions.
    """

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
        profile = profile_for_view(self.writer.published_view)
        budget = max(1, int(profile.abstraction_budget))
        episodes = self._load(EvidenceType.EPISODE)

        families = carriers = contextual_roles = roles = concepts = 0
        world_models = relational_models = strategies = 0

        # ------------------------------------------------------------------
        # M2: action-independent normalized transformation families.
        # ------------------------------------------------------------------
        family_groups: dict[int, list[MemoryId]] = defaultdict(list)
        m1_identity: dict[MemoryId, tuple[int, int, int]] = {}
        for memory_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            if node.level != MemoryLevel.M1 or node.type_id != TYPE_CONTINGENCY:
                continue
            key = registry.key_for(memory_id)
            if key is None or len(key.parts) < 3:
                continue
            context, action, outcome = map(int, key.parts[:3])
            m1_identity[memory_id] = (context, action, outcome)
            family_groups[outcome].append(memory_id)

        family_by_member: dict[MemoryId, MemoryId] = {}
        for outcome, raw_members in list(sorted(family_groups.items()))[:budget]:
            members = tuple(sorted(set(raw_members), key=int))
            if len(members) < 2:
                continue
            candidate = ScientificDerivationKernels.m2_family(
                action_id=0,
                member_ids=members,
                outcome_class=outcome,
            )
            family, created = self._sync_candidate(
                candidate,
                desired_support=len(members),
            )
            families += int(created)
            for member in members:
                family_by_member[member] = family

        # Baseline action/outcome distributions support empirical carrier
        # prediction-lift calculation.
        action_outcomes: dict[int, Counter[int]] = defaultdict(Counter)
        for row in episodes:
            action = int(row.get("action_id") or 0)
            outcome = int(row.get("outcome_signature") or 0)
            action_outcomes[action][outcome] += 1

        # ------------------------------------------------------------------
        # Explicit M3 carrier candidates.
        # ------------------------------------------------------------------
        carrier_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            carrier = row.get("carrier_signature")
            memory_id = row.get("memory_id")
            if carrier is None or memory_id is None:
                continue
            if MemoryId(int(memory_id)) not in family_by_member:
                continue
            carrier_rows[int(carrier)].append(row)

        carrier_id_by_signature: dict[int, MemoryId] = {}
        usable_carriers: set[int] = set()
        for carrier_signature, rows_for_carrier in list(sorted(carrier_rows.items()))[:budget]:
            support_count = len(rows_for_carrier)
            if support_count < 2:
                continue
            member_ids = tuple(
                sorted(
                    {
                        MemoryId(int(row["memory_id"]))
                        for row in rows_for_carrier
                        if row.get("memory_id") is not None
                    },
                    key=int,
                )
            )
            family_ids = tuple(
                sorted(
                    {
                        family_by_member[memory_id]
                        for memory_id in member_ids
                        if memory_id in family_by_member
                    },
                    key=int,
                )
            )
            parent_ids = tuple(sorted(set(member_ids) | set(family_ids), key=int))
            if not parent_ids:
                continue
            contexts = {
                planning_context(
                    row.get("context_signatures", ()) or (),
                    fallback=int(row.get("context_signature") or 0),
                )
                for row in rows_for_carrier
            }
            prediction_lift = self._carrier_prediction_lift(
                rows_for_carrier,
                action_outcomes,
            )
            compression_gain = 1.0 - (1.0 / max(1.0, float(support_count)))
            candidate = ScientificDerivationKernels.m3_carrier(
                carrier_signature=carrier_signature,
                parent_ids=parent_ids,
                support_count=support_count,
                prediction_lift=prediction_lift,
                compression_gain=compression_gain,
            )
            carrier_id, created = self._sync_candidate(
                candidate,
                desired_support=support_count,
            )
            carriers += int(created)
            carrier_id_by_signature[carrier_signature] = carrier_id
            if len(contexts) >= 2 and compression_gain >= 0.50:
                usable_carriers.add(carrier_signature)

        # ------------------------------------------------------------------
        # Contextual M3 role instances.
        # ------------------------------------------------------------------
        instance_groups: dict[
            tuple[MemoryId, MemoryId, int, int],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for carrier_signature in sorted(usable_carriers):
            carrier_id = carrier_id_by_signature[carrier_signature]
            for row in carrier_rows[carrier_signature]:
                raw_memory_id = row.get("memory_id")
                if raw_memory_id is None:
                    continue
                memory_id = MemoryId(int(raw_memory_id))
                family_id = family_by_member.get(memory_id)
                if family_id is None:
                    continue
                context = planning_context(
                    row.get("context_signatures", ()) or (),
                    fallback=int(row.get("context_signature") or 0),
                )
                action = int(row.get("action_id") or 0)
                instance_groups[(family_id, carrier_id, action, context)].append(row)

        instance_info: dict[
            MemoryId,
            tuple[MemoryId, MemoryId, int, int, tuple[MemoryId, ...], list[dict[str, Any]]],
        ] = {}
        for (family_id, carrier_id, action, context), rows_for_instance in list(
            sorted(
                instance_groups.items(),
                key=lambda item: (
                    int(item[0][0]),
                    int(item[0][1]),
                    item[0][2],
                    item[0][3],
                ),
            )
        )[:budget]:
            members = tuple(
                sorted(
                    {
                        MemoryId(int(row["memory_id"]))
                        for row in rows_for_instance
                        if row.get("memory_id") is not None
                    },
                    key=int,
                )
            )
            if not members:
                continue
            candidate = ScientificDerivationKernels.m3_contextual_role(
                family_id=family_id,
                carrier_id=carrier_id,
                context_class=context,
                action_id=action,
                member_ids=members,
            )
            instance_id, created = self._sync_candidate(
                candidate,
                desired_support=len(rows_for_instance),
            )
            contextual_roles += int(created)
            instance_info[instance_id] = (
                family_id,
                carrier_id,
                action,
                context,
                members,
                rows_for_instance,
            )

        # ------------------------------------------------------------------
        # Functional M3 roles: abstract across contexts/carriers.
        # ------------------------------------------------------------------
        functional_groups: dict[int, list[MemoryId]] = defaultdict(list)
        for instance_id, info in instance_info.items():
            family_id, _carrier_id, _action, _context, _members, instance_rows = info
            family_key = registry.key_for(family_id)
            outcome = int(family_key.parts[0]) if family_key and family_key.parts else 0
            signature = _functional_role_signature(outcome, instance_rows)
            functional_groups[signature].append(instance_id)

        functional_role_by_member: dict[MemoryId, set[MemoryId]] = defaultdict(set)
        functional_role_by_carrier: dict[MemoryId, set[MemoryId]] = defaultdict(set)
        for signature, raw_instances in list(sorted(functional_groups.items()))[:budget]:
            instances = tuple(sorted(set(raw_instances), key=int))
            if len(instances) < 2:
                continue
            contexts = {
                int(instance_info[instance_id][3])
                for instance_id in instances
                if instance_id in instance_info
            }
            carrier_ids = {
                instance_info[instance_id][1]
                for instance_id in instances
                if instance_id in instance_info
            }
            games = {
                str(row.get("source_game"))
                for instance_id in instances
                for row in instance_info[instance_id][5]
                if row.get("source_game")
            }
            occurrence_count = sum(
                len(instance_info[instance_id][5]) for instance_id in instances
            )
            transfer_prior = min(1.0, max(len(contexts), len(games)) / 4.0)
            explanatory = min(1.0, max(len(instances), len(carrier_ids)) / 4.0)
            candidate = ScientificDerivationKernels.m3_functional_role(
                function_signature=signature,
                instance_ids=instances,
                support_count=occurrence_count,
                transfer_prior=transfer_prior,
                explanatory_potential=explanatory,
            )
            role_id, created = self._sync_candidate(
                candidate,
                desired_support=occurrence_count,
            )
            roles += int(created)
            for instance_id in instances:
                family_id, carrier_id, action, context, members, instance_rows = instance_info[
                    instance_id
                ]
                retrieval_contexts = {int(context)}
                for row in instance_rows:
                    retrieval_contexts.update(_retrieval_contexts(row))
                self.writer.apply_role_index_batch(
                    RoleIndexMutation(index_context, action, role_id, family_id)
                    for index_context in sorted(retrieval_contexts)
                )
                functional_role_by_carrier[carrier_id].add(role_id)
                for member in members:
                    functional_role_by_member[member].add(role_id)

        # ------------------------------------------------------------------
        # M4 concepts bind multiple functional roles through carrier relation.
        # ------------------------------------------------------------------
        for carrier_signature, carrier_id in list(
            sorted(carrier_id_by_signature.items())
        )[:budget]:
            role_ids = tuple(
                sorted(functional_role_by_carrier.get(carrier_id, ()), key=int)
            )
            if len(role_ids) < 2:
                continue
            candidate = ScientificDerivationKernels.m4_concept(
                role_ids=role_ids,
                relation_signature=carrier_signature,
            )
            concept_id, created = self._sync_candidate(
                candidate,
                desired_support=len(role_ids),
            )
            concepts += int(created)
            self.writer.apply_role_concept_index_batch(
                RoleConceptIndexMutation(role_id, concept_id) for role_id in role_ids
            )

        self._update_strategy_reuse_scores(episodes)

        # ------------------------------------------------------------------
        # M5a: empirical transition models require validated concepts.
        # ------------------------------------------------------------------
        validated_mask = int(
            ConceptValidationStatus.TRANSFER_VALIDATED
            | ConceptValidationStatus.TRUSTED
        )
        eligible_concepts = {
            int(memory_id)
            for memory_id, node in nodes.items()
            if node.level == MemoryLevel.M4
            and node.type_id == TYPE_CONCEPT
            and bool(int(node.status_flags) & validated_mask)
            and not bool(
                int(node.status_flags)
                & int(ConceptValidationStatus.TRANSFER_REJECTED)
            )
        }

        transition_counts: Counter[int] = Counter()
        transition_concepts: dict[int, set[MemoryId]] = defaultdict(set)
        transition_locations: dict[int, set[tuple[int, int]]] = defaultdict(set)
        transition_relation_rows: dict[
            int, list[tuple[tuple[int, ...], tuple[int, ...], float, int]]
        ] = defaultdict(list)
        by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            if row.get("source_game"):
                by_game[str(row["source_game"])].append(row)

        for game in sorted(by_game):
            prior_concepts: tuple[int, ...] = ()
            prior_action: int | None = None
            prior_contexts: tuple[int, ...] = ()
            prior_future = 0.0
            prior_terminal = 0
            for row in sorted(
                by_game[game],
                key=lambda item: int(item.get("source_global_step") or -1),
            ):
                current = tuple(
                    sorted(
                        {
                            int(value)
                            for value in row.get("decision_concept_ids", ()) or ()
                            if int(value) in eligible_concepts
                        }
                    )
                )
                if (
                    prior_terminal == 0
                    and prior_concepts
                    and current
                    and prior_action is not None
                    and prior_contexts
                ):
                    union = tuple(sorted(set(prior_concepts) | set(current)))
                    if len(union) >= 2:
                        signature = _transition_key(
                            prior_concepts,
                            prior_action,
                            current,
                        )
                        transition_counts[signature] += 1
                        transition_concepts[signature].update(
                            MemoryId(value) for value in union
                        )
                        transition_locations[signature].update(
                            (context, prior_action) for context in prior_contexts
                        )
                        transition_relation_rows[signature].append(
                            (
                                prior_concepts,
                                current,
                                prior_future,
                                prior_terminal,
                            )
                        )
                prior_concepts = current
                prior_action = int(row.get("action_id") or 0)
                prior_contexts = _retrieval_contexts(row)
                prior_future = float(row.get("future_option_delta") or 0.0)
                prior_terminal = int(row.get("terminal_polarity") or 0)

        model_by_signature: dict[int, MemoryId] = {}
        models_by_location: dict[tuple[int, int], set[MemoryId]] = defaultdict(set)
        for signature, count in list(sorted(transition_counts.items()))[:budget]:
            concept_ids = tuple(sorted(transition_concepts[signature], key=int))
            if count < 2 or len(concept_ids) < 2:
                continue
            candidate = ScientificDerivationKernels.m5_world_model(
                concept_ids=concept_ids,
                transition_signature=signature,
            )
            model_id, created = self._sync_candidate(
                candidate,
                desired_support=count,
            )
            world_models += int(created)
            model_by_signature[signature] = model_id
            for context, action in sorted(transition_locations[signature]):
                self.writer.apply_role_index_batch(
                    (RoleIndexMutation(context, action, model_id, None),)
                )
                models_by_location[(context, action)].add(model_id)

        # ------------------------------------------------------------------
        # M5b: typed relational model inferred from repeated M5a transitions.
        # ------------------------------------------------------------------
        relation_counts: Counter[tuple[int, int, int]] = Counter()
        relation_models: dict[tuple[int, int, int], set[MemoryId]] = defaultdict(set)
        relation_locations: dict[tuple[int, int, int], set[tuple[int, int]]] = defaultdict(set)
        for signature, rows in transition_relation_rows.items():
            model_id = model_by_signature.get(signature)
            if model_id is None:
                continue
            for prior, current, future_delta, terminal in rows:
                for source, relation_type, target in classify_transition_relations(
                    prior_concepts=prior,
                    current_concepts=current,
                    future_option_delta=future_delta,
                    terminal_polarity=terminal,
                ):
                    key = (source, relation_type, target)
                    relation_counts[key] += 1
                    relation_models[key].add(model_id)
                    relation_locations[key].update(transition_locations[signature])

        desired_relation_edges: Counter[tuple[MemoryId, int, MemoryId]] = Counter()
        for (source, relation_type, target), count in list(
            sorted(relation_counts.items())
        )[:budget]:
            if count < 2:
                continue
            source_id = MemoryId(source)
            target_id = MemoryId(target)
            supporting_models = tuple(
                sorted(relation_models[(source, relation_type, target)], key=int)
            )
            relation_signature = _relation_key(source, relation_type, target)
            key = CanonicalMemoryKey(
                MemoryLevel.M5,
                TYPE_RELATIONAL_WORLD_MODEL,
                (relation_signature,),
            )
            candidate = CanonicalCandidateMutation(
                key=key,
                support_delta=count,
                parents=(source_id, target_id, *supporting_models),
                explanatory_potential=min(1.0, count / 4.0),
                transfer_prior=min(1.0, len(supporting_models) / 3.0),
            )
            relation_id, created = self._sync_candidate(
                candidate,
                desired_support=count,
            )
            relational_models += int(created)
            for context, action in sorted(
                relation_locations[(source, relation_type, target)]
            ):
                self.writer.apply_role_index_batch(
                    (RoleIndexMutation(context, action, relation_id, None),)
                )
                models_by_location[(context, action)].add(relation_id)
            desired_relation_edges[(source_id, relation_type, target_id)] = count
        self._sync_edge_support(desired_relation_edges)

        # ------------------------------------------------------------------
        # M6 efficiency strategies remain as compact hierarchy memories. Phase
        # 1 executable procedures are derived separately from trajectory data.
        # ------------------------------------------------------------------
        all_models = tuple(
            sorted(
                (
                    memory_id
                    for memory_id, node in nodes.items()
                    if node.level == MemoryLevel.M5
                    and node.type_id in {TYPE_WORLD_MODEL, TYPE_RELATIONAL_WORLD_MODEL}
                ),
                key=int,
            )
        )[:64]
        trajectories = self._load(EvidenceType.TRAJECTORY)
        best_steps: dict[tuple[str, str], int] = {}
        strategy_candidates = 0
        for row in trajectories:
            if strategy_candidates >= budget:
                break
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
                improvement = (previous_best - steps) / max(
                    1.0, float(previous_best)
                )
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
            _strategy_id, created = self._sync_candidate(
                candidate,
                desired_support=len(model_ids),
            )
            strategies += int(created)
            strategy_candidates += 1

        return OnlineDerivationStats(
            families=families,
            carriers=carriers,
            contextual_roles=contextual_roles,
            roles=roles,
            concepts=concepts,
            world_models=world_models,
            relational_models=relational_models,
            strategies=strategies,
            stage=profile.stage.name,
        )

    @staticmethod
    def _carrier_prediction_lift(
        rows: list[dict[str, Any]],
        action_outcomes: dict[int, Counter[int]],
    ) -> float:
        by_action: dict[int, Counter[int]] = defaultdict(Counter)
        for row in rows:
            by_action[int(row.get("action_id") or 0)][
                int(row.get("outcome_signature") or 0)
            ] += 1
        weighted = 0.0
        total_weight = 0
        for action, counts in by_action.items():
            local_total = sum(counts.values())
            baseline = action_outcomes.get(action, Counter())
            baseline_total = sum(baseline.values())
            if local_total <= 0 or baseline_total <= 0:
                continue
            local_confidence = max(counts.values()) / local_total
            baseline_confidence = max(baseline.values()) / baseline_total
            weighted += max(0.0, local_confidence - baseline_confidence) * local_total
            total_weight += local_total
        return 0.0 if total_weight <= 0 else min(1.0, weighted / total_weight)

    def _sync_candidate(
        self,
        candidate: CanonicalCandidateMutation,
        *,
        desired_support: int,
    ) -> tuple[MemoryId, bool]:
        desired = max(1, int(desired_support))
        existing = self.writer.canonical_memory_id(candidate.key)
        created = existing is None
        if existing is None:
            adjusted = replace(candidate, support_delta=desired)
            memory_id = self.writer.apply_canonical_candidate_batch((adjusted,))[
                adjusted.key
            ]
        else:
            memory_id = existing
            node = getattr(self.writer, "_nodes")[memory_id]
            if desired > int(node.support_count):
                self.writer.apply_mutation_batch(
                    (
                        NodeMutation(
                            memory_id,
                            node.level,
                            node.type_id,
                            support_delta=desired - int(node.support_count),
                        ),
                    )
                )
            self.writer.apply_score_batch(
                (
                    ScoreMutation(
                        memory_id=memory_id,
                        significance=candidate.significance,
                        prediction_error=candidate.prediction_error,
                        learning_value=candidate.learning_value,
                        transfer_prior=candidate.transfer_prior,
                        explanatory_potential=candidate.explanatory_potential,
                        future_option_delta=candidate.future_option_delta,
                    ),
                )
            )
        self._record_parents(memory_id, candidate.parents)
        return memory_id, created

    def _record_parents(
        self,
        memory_id: MemoryId,
        parents: Iterable[MemoryId],
    ) -> None:
        existing = set(self.lifecycle_store.provenance_parents(memory_id))
        new = tuple(sorted(set(parents) - existing, key=int))
        if not new:
            return
        generation = int(self.writer.mutable_generation_id)
        self.lifecycle_store.append_provenance(
            ProvenanceRecord(
                memory_id=memory_id,
                parent_memory_id=parent,
                generation_id=generation,
            )
            for parent in new
        )

    def _sync_edge_support(
        self,
        desired: Counter[tuple[MemoryId, int, MemoryId]],
    ) -> None:
        edge_support = getattr(self.writer, "_edge_support")
        mutations: list[EdgeMutation] = []
        for key, wanted in sorted(
            desired.items(),
            key=lambda item: (int(item[0][0]), item[0][1], int(item[0][2])),
        ):
            current = int(edge_support.get(key, 0))
            delta = int(wanted) - current
            if delta > 0:
                mutations.append(
                    EdgeMutation(
                        source_id=key[0],
                        relation_type=key[1],
                        target_id=key[2],
                        support_delta=delta,
                    )
                )
        if mutations:
            self.writer.apply_edge_batch(mutations)

    def _update_strategy_reuse_scores(
        self,
        episodes: list[dict[str, Any]],
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
        for memory_id, values in sorted(
            outcomes.items(), key=lambda item: int(item[0])
        ):
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

    def _load(self, evidence_type: EvidenceType) -> list[dict[str, Any]]:
        rows = self.evidence_store.connection.execute(
            "SELECT memory_id,source_game,source_context,source_global_step,payload_json,generation_id "
            "FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
            (int(evidence_type),),
        ).fetchall()
        result: list[dict[str, Any]] = []
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


def _functional_role_signature(
    outcome_signature: int,
    rows: list[dict[str, Any]],
) -> int:
    future = sum(float(row.get("future_option_delta") or 0.0) for row in rows)
    future_bucket = 1 if future > 0 else -1 if future < 0 else 0
    positives = sum(int(row.get("terminal_polarity") or 0) > 0 for row in rows)
    negatives = sum(int(row.get("terminal_polarity") or 0) < 0 for row in rows)
    terminal_bucket = 1 if positives > negatives else -1 if negatives > positives else 0
    changed = sum(int(row.get("changed_cells") or 0) > 0 for row in rows)
    changed_bucket = 1 if changed * 2 >= max(1, len(rows)) else 0
    digest = blake2b(digest_size=8)
    digest.update(b"functional-role-v1")
    for value in (
        int(outcome_signature),
        int(future_bucket),
        int(terminal_bucket),
        int(changed_bucket),
    ):
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest(), "little") & _MASK63


def _transition_key(
    prior: tuple[int, ...],
    action_id: int,
    current: tuple[int, ...],
) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"world-transition-v3")
    digest.update(str(tuple(prior)).encode("ascii"))
    digest.update(str(int(action_id)).encode("ascii"))
    digest.update(str(tuple(current)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & _MASK63


def _relation_key(source: int, relation_type: int, target: int) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"relational-world-model-v1")
    for value in (source, relation_type, target):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(), "little") & _MASK63


def _sequence_key(actions: Iterable[int], contexts: Iterable[int]) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"strategy-sequence-v2")
    digest.update(str(tuple(int(value) for value in actions)).encode("ascii"))
    digest.update(str(tuple(int(value) for value in contexts)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & _MASK63
