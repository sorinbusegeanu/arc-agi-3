from dataclasses import replace

from v9.cognition.grounding import GroundingEvidence
from v9.memory.identity import ContextScopeId, LineageUid, MemoryUid
from v9.memory.m5_consequence import M5ConsequenceStructure
from v9.memory.m6_outcome import M6Outcome
from v9.memory.m7_strategy import M7Strategy
from v9.memory.model import MemoryLevel
from v9.memory.provenance import DerivationProvenance
from v9.mutation.proposals import MutationKind, ProposalClass
from v9.mutation.lineage import LineageAwareDependencyEdge, LineageContextOverlay, RegimeState

from .config import RuntimeConfig, ScientificConfig, ScientificConfigId
from .publication import CanonicalGraph
from .read_view import ReadView
from .optimized_runtime import ContinuousMemoryRuntime as _OptimizedContinuousMemoryRuntime


class ContinuousMemoryRuntime(_OptimizedContinuousMemoryRuntime):
    """Public v9 runtime without an artificial canonical graph record ceiling."""

    def __init__(self, config):
        super().__init__(config)
        # v9 uses dynamic Python graph storage. The former arena-style capacities
        # were inherited from an earlier fixed-storage design and became a hard
        # 1,000,000-node rejection ceiling at the default four shards. They are
        # not a scientific constraint and are intentionally disabled, including
        # for restored snapshots that persisted the old limits.
        self.graph.node_capacity_per_partition = None
        self.graph.edge_capacity_per_partition = None
        self._m5: dict[MemoryUid, M5ConsequenceStructure] = {}
        self._m6: dict[MemoryUid, M6Outcome] = {}
        self._m7: dict[MemoryUid, M7Strategy] = {}
        self._restore_higher_memory_objects()

    def _evidence_payload(self, uid: MemoryUid):
        payload = self.graph.payloads.get(uid)
        if payload is not None:
            return payload
        deferred = self._deferred_base_nodes.get(uid)
        return None if deferred is None else deferred[1]

    def _evidence_environment_scope(self, evidence_uids) -> tuple[int, ...]:
        environments: set[int] = set()
        for uid in evidence_uids:
            payload = self._evidence_payload(uid)
            if payload is None:
                continue
            environment = payload.get("environment_instance_id")
            if environment is not None:
                environments.add(int(environment))
        return tuple(sorted(environments))

    def build_derivation_task(self, signature: int, *, task_id: int):
        task = super().build_derivation_task(signature, task_id=task_id)
        if task is None:
            return None
        evidence = tuple(
            uid
            for row in task.rows
            for uid in row.provenance.evidence
        )
        scope = self._evidence_environment_scope(evidence)
        return replace(task, formation_scope=scope) if scope else task

    def _concept_evidence_scope(self, concept) -> tuple[int, ...]:
        scope = self._evidence_environment_scope(concept.provenance.evidence)
        return scope or tuple(int(value) for value in concept.provenance.formation_scope)

    def transfer_validation_candidates(self, *, limit: int = 32) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows: list[dict[str, object]] = []
            concepts = sorted(
                self._m4.values(),
                key=lambda row: (
                    bool(row.validated),
                    -float(row.compression_benefit),
                    -int(row.explanatory_reach),
                    row.uid,
                ),
            )
            for concept in concepts:
                if concept.validated:
                    continue
                scope = self._concept_evidence_scope(concept)
                source_types: set[str] = set()
                for environment_id in scope:
                    try:
                        source_types.add(str(self.environments.resolve(environment_id).environment_type))
                    except KeyError:
                        pass

                action_stats: dict[int, list[int]] = {}
                for uid in concept.provenance.evidence:
                    payload = self._evidence_payload(uid)
                    if payload is None or payload.get("action_id") is None:
                        continue
                    action = int(payload["action_id"])
                    stats = action_stats.setdefault(action, [0, 0, 0])
                    valence = int(payload.get("primary_valence", 0))
                    stats[0] += int(valence > 0)
                    stats[1] += 1
                    stats[2] += int(valence < 0)
                actions = tuple(
                    action
                    for action, _ in sorted(
                        action_stats.items(),
                        key=lambda item: (-item[1][0], -item[1][1], item[1][2], item[0]),
                    )
                )
                contexts = tuple(
                    sorted(
                        {
                            int(payload["context_signature"])
                            for uid in concept.provenance.evidence
                            for payload in [self._evidence_payload(uid)]
                            if payload is not None and payload.get("context_signature") is not None
                        }
                    )
                )
                if not actions:
                    continue
                positive_evidence = sum(stats[0] for stats in action_stats.values())
                negative_evidence = sum(stats[2] for stats in action_stats.values())
                support = sum(stats[1] for stats in action_stats.values())
                rows.append(
                    {
                        "concept_uid": concept.uid,
                        "formation_scope": scope,
                        "source_environment_types": tuple(sorted(source_types)),
                        "actions": actions,
                        "contexts": contexts,
                        "positive_evidence": int(positive_evidence),
                        "negative_evidence": int(negative_evidence),
                        "support": int(support),
                        "validated": bool(concept.validated),
                    }
                )
            rows.sort(
                key=lambda row: (
                    -int(row["positive_evidence"]),
                    -int(row["support"]),
                    int(row["negative_evidence"]),
                    row["concept_uid"],
                )
            )
            return tuple(rows[: max(1, int(limit))])

    def is_concept_validated(self, concept_uid: MemoryUid) -> bool:
        with self._lock:
            concept = self._m4.get(concept_uid)
            return bool(concept is not None and concept.validated)

    def record_transfer_validation(self, concept_uid: MemoryUid, **kwargs) -> None:
        with self._lock:
            concept = self._m4.get(concept_uid)
            if concept is None or concept_uid not in self.graph.nodes or concept_uid not in self.graph.payloads:
                self._m4.pop(concept_uid, None)
                self._transfer_trials.pop(concept_uid, None)
                return

            # Older broad runs recorded the global set of every seen environment
            # as each concept's formation scope. Replace that over-broad metadata
            # with the environments that actually contributed grounded evidence.
            scope = self._concept_evidence_scope(concept)
            if scope and scope != tuple(concept.provenance.formation_scope):
                provenance = replace(concept.provenance, formation_scope=scope)
                concept = replace(concept, provenance=provenance)
                self._m4[concept_uid] = concept
                self._publish(
                    self.graph.nodes[concept_uid],
                    {**self.graph.payloads[concept_uid], "formation_scope": list(scope)},
                    concept.provenance.evidence,
                    proposal_class=ProposalClass.STATEFUL,
                    mutation_kind=MutationKind.UPDATE_VALIDATION,
                )

            was_validated = bool(concept.validated)
            super().record_transfer_validation(concept_uid, **kwargs)
            concept = self._m4.get(concept_uid)
            if concept is None or was_validated or not concept.validated:
                return

            parent_uid = concept.provenance.parents[0] if concept.provenance.parents else None
            role = None if parent_uid is None else self._m3.get(parent_uid)
            if role is None or parent_uid not in self.graph.nodes or parent_uid not in self.graph.payloads:
                return
            consequence = next(
                (
                    row for row in self._m5.values()
                    if concept.uid in row.provenance.parents
                ),
                None,
            )
            if consequence is None:
                return
            formation_scope = set(concept.provenance.formation_scope)
            admissible = tuple(
                trial
                for trial in self._transfer_trials.get(concept_uid, ())
                if trial["matched"]
                and trial["held_out"]
                and int(trial["target_environment_id"]) not in formation_scope
                and float(trial["enabled_metric"]) - float(trial["ablated_metric"])
                > float(self.config.scientific.transfer_effect_threshold)
            )
            if not admissible:
                return
            action = int(admissible[-1]["target_native_action"])
            target_environment_id = int(kwargs["target_environment_id"])
            best_effect = max(
                float(trial["enabled_metric"]) - float(trial["ablated_metric"])
                for trial in admissible
            )
            context = self.contexts.form(
                (target_environment_id, action),
                evidence_refs=concept.provenance.evidence,
                formation_watermark=self._watermark,
                support=len(admissible),
                improvement=best_effect,
            )
            lineage = self.lineages.derive(LineageUid(0), concept.uid, self._watermark)
            self.lineages.put_overlay(
                LineageContextOverlay(
                    concept.uid,
                    lineage,
                    context.scope_id,
                    RegimeState.PREDICTIVE,
                    support=len(admissible),
                    evidence_opportunities=len(admissible),
                )
            )
            dependency = LineageAwareDependencyEdge(
                concept.uid,
                consequence.uid,
                lineage,
                context.scope_id,
                independent_support=len(admissible),
            )
            self.lineages.dependencies[
                (dependency.source, dependency.target, dependency.lineage_uid, dependency.context_scope)
            ] = dependency
            grounding_key = (
                int(concept.uid.lo),
                int(role.uid.lo),
                target_environment_id,
                int(context.scope_id.value),
                int(lineage.value),
            )
            before_grounding = self.grounding.states.get(grounding_key)
            after_grounding = self.grounding.observe(
                GroundingEvidence(
                    int(concept.uid.lo),
                    int(role.uid.lo),
                    target_environment_id,
                    int(context.scope_id.value),
                    int(lineage.value),
                    int(self._watermark),
                    causal_intervention=True,
                    positive=True,
                )
            )
            if before_grounding is None or int(after_grounding.maturity) > int(before_grounding.maturity):
                self.telemetry["grounding_promotions"] += 1

    @staticmethod
    def _payload_provenance(payload: dict[str, object]) -> DerivationProvenance | None:
        parents = tuple(MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("parents", []))
        if not parents:
            return None
        evidence = tuple(
            MemoryUid(int(raw[0]), int(raw[1]))
            for raw in payload.get("evidence_refs", payload.get("parents", []))
        )
        return DerivationProvenance(parents, evidence)

    def _restore_higher_memory_objects(self) -> None:
        level_index = getattr(self.graph, "_uids_by_level", {})
        for uid in tuple(level_index.get(MemoryLevel.M5, ())):
            payload = self.graph.payloads.get(uid, {})
            provenance = self._payload_provenance(payload)
            if provenance is None:
                continue
            self._m5[uid] = M5ConsequenceStructure(
                uid,
                tuple(int(value) for value in payload.get("descriptor", self.graph.nodes[uid].structural_key)),
                provenance,
                bool(payload.get("mature", False)),
            )
        for uid in tuple(level_index.get(MemoryLevel.M6, ())):
            payload = self.graph.payloads.get(uid, {})
            provenance = self._payload_provenance(payload)
            if provenance is None:
                continue
            self._m6[uid] = M6Outcome(
                uid,
                tuple(int(value) for value in payload.get("class_signature", self.graph.nodes[uid].structural_key)),
                provenance.parents,
                provenance,
                int(payload.get("class_version", 1)),
            )
        for uid in tuple(level_index.get(MemoryLevel.M7, ())):
            payload = self.graph.payloads.get(uid, {})
            provenance = self._payload_provenance(payload)
            target = payload.get("target_outcome")
            if provenance is None or not isinstance(target, (list, tuple)) or len(target) != 2:
                continue
            self._m7[uid] = M7Strategy(
                uid,
                MemoryUid(int(target[0]), int(target[1])),
                int(payload["target_environment_id"]),
                tuple(int(value) for value in payload.get("native_actions", [])),
                int(payload.get("reliability_successes", 0)),
                int(payload.get("reliability_trials", 0)),
                int(payload.get("primary_valence_sum", 0)),
                int(payload.get("realized_cost_sum", 0)),
                provenance,
            )


__all__ = ["CanonicalGraph", "ContinuousMemoryRuntime", "ReadView", "RuntimeConfig", "ScientificConfig", "ScientificConfigId"]