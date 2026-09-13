from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
from typing import Any

from v9.cognition.compression import form_families
from v9.cognition.developmental_stage import DevelopmentalStageTracker, StageEvidence
from v9.cognition.grounding import GroundingRegistry
from v9.cognition.isf import ISFComponents, InteractionSignificanceFunction
from v9.cognition.replay import ReplayCandidate, ReplayResult, ReplayScheduler
from v9.cognition.roles import form_roles
from v9.cognition.similarity import ProgressiveSimilarity, ScaleStatistics, StructuralCandidateIndex, StructuralIndexKey
from v9.cognition.transfer import TransferTrustRegistry, ValidationMode
from v9.environments.registry import EnvironmentRegistry
from v9.environments.schemas import EnvironmentIdentity
from v9.memory.identity import ContextScopeId, EpisodeId, EventUid, LineageUid, MemoryUid, ModalityId, stable_u64
from v9.memory.m0_episode import M0Episode
from v9.memory.m1_grounded import GroundedRelation, M1GroundedContingency
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.m2_family import M2TransformationFamily
from v9.memory.m3_role import M3FunctionalRole
from v9.memory.m4_concept import M4Concept
from v9.memory.m5_consequence import M5ConsequenceStructure
from v9.memory.m6_outcome import M6Outcome
from v9.memory.m7_strategy import M7Strategy
from v9.memory.model import CanonicalNode, CognitiveState, ExperienceEvent, MemoryLevel, MemoryType
from v9.memory.provenance import DerivationProvenance
from v9.memory.relations import RelationEdge, RelationType
from v9.memory.residency import PayloadStore
from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent, PassiveWorldEvent, SYMBOL_MODALITY, TimelineEvent, TimelineIdentity, WORLD_MODALITY
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.mutation.lineage import LineageStore
from v9.mutation.context import ContextRegistry, EffectiveCognitiveState, EffectiveStateResolver
from v9.mutation.proposals import MutationKind, MutationProposal, MutationWrite, ProposalClass
from v9.mutation.read_sets import ReadDependency, ReadSet
from v9.research.evidence import EvidenceLedger
from v9.research.hypotheses import untested_assessment
from v9.research.reports import write_report
from v9.telemetry import (
    ConsolidationSample,
    HGTInferenceSample,
    HGTTrainingSample,
    ModelEvolutionSample,
    OptimizationSample,
    TelemetryProvenance,
    UnifiedTelemetry,
    build_primary_dashboard,
)

from .config import RuntimeConfig, write_scientific_config_manifest
from .lifecycle import LifecycleRegistry
from .partitions import PartitionMap
from .publication import CanonicalGraph, edge_ref, node_ref
from .read_view import ReadView
from .rings import MultimodalTimeline
from .snapshot import SnapshotResult, assert_native_root, latest_snapshot, load_snapshot, write_snapshot


class ContinuousMemoryRuntime:
    """Native, RAM-authoritative Hydra v9 runtime with explicit composition."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        if config.reset_persistent_identity and self.root.exists() and any(self.root.iterdir()):
            raise ValueError("v9.5 does not rewrite an existing identity history; choose a fresh --root")
        assert_native_root(self.root)
        write_scientific_config_manifest(self.root, config.scientific)
        self.partitions = PartitionMap(config.shards)
        self.graph = CanonicalGraph(config.shards, node_capacity_per_partition=config.node_capacity_per_shard, edge_capacity_per_partition=config.edge_capacity_per_shard, applied_proposal_capacity=config.action_capacity_per_shard * config.shards)
        scientific = config.scientific
        self.timeline = MultimodalTimeline(capacity=scientific.passive_event_queue_depth, symbol_budget=scientific.symbol_budget_per_window, symbol_payload_bytes=scientific.symbol_payload_bytes)
        self.environments = EnvironmentRegistry()
        self.lifecycle = LifecycleRegistry()
        self.lineages = LineageStore()
        self.contexts = ContextRegistry(limit=scientific.context_scope_limit, descriptor_limit=scientific.descriptor_component_limit)
        self.effective_states = EffectiveStateResolver(self.contexts, self.lineages)
        self.scale_statistics = ScaleStatistics(sample_threshold=scientific.normalization_bootstrap_samples, contingency_threshold=scientific.normalization_bootstrap_contingencies, reservoir_limit=scientific.normalization_reservoir_limit, component_limit=scientific.descriptor_component_limit, minimum_generation_span=scientific.normalization_minimum_generation_span, allowed_radii=scientific.structural_radii)
        self.similarity = ProgressiveSimilarity(beta_by_radius=dict(scientific.beta_by_radius), candidate_limit=scientific.candidates_per_radius, equivalence_limit=scientific.equivalence_set_size, maximum_radius=max(scientific.structural_radii), ambiguity_threshold=scientific.ambiguity_entropy_threshold, margin_threshold=scientific.top2_margin_threshold, information_threshold=scientific.information_gain_threshold, symmetry_patience=scientific.symmetry_patience, statistics=self.scale_statistics)
        self.structural_index = StructuralCandidateIndex(bucket_capacity=scientific.candidates_per_radius, bucket_scan_limit=scientific.structural_index_bucket_scan_limit)
        self.grounding = GroundingRegistry()
        self.transfer_trust = TransferTrustRegistry(scope_limit=scientific.transfer_trust_scope_limit)
        self.stage_tracker = DevelopmentalStageTracker(history_limit=scientific.isf_decision_hot_limit)
        self.isf = InteractionSignificanceFunction(scientific.isf_weights_by_stage, schema_version=scientific.isf_score_schema_version, hot_limit=scientific.isf_decision_hot_limit)
        self.replay = ReplayScheduler(candidate_limit=scientific.replay_candidates_per_interval)
        self.payloads = PayloadStore()
        self.symbol_codecs: dict[int, DeterministicSymbolCodec] = {}
        self.evidence = EvidenceLedger(self.root / "evidence" / "ledger.jsonl", scientific.config_id.value)
        self._watermark = 0
        self._snapshot_id = 0
        self._producer_sequences: dict[int, int] = {}
        self._started = False
        self._closed = False
        self._lock = RLock()
        self._m1n_occurrences: dict[int, list[M1NormalizedRelation]] = {}
        self._m1n_supports: dict[int, int] = {}
        self._replay_pool: dict[MemoryUid, float] = {}
        self._formation_environments: set[int] = set()
        self._latest_interaction_grounding: dict[tuple[int, int], M1GroundedContingency] = {}
        self._m2: dict[MemoryUid, M2TransformationFamily] = {}
        self._m3: dict[MemoryUid, M3FunctionalRole] = {}
        self._m4: dict[MemoryUid, M4Concept] = {}
        self._transfer_trials: dict[MemoryUid, list[dict[str, Any]]] = {}
        self._modality_events: dict[int, int] = {}
        self._similarity_entropy_by_radius: dict[int, list[float]] = {}
        self._replans_demonstrated = 0
        self._efficient_replans = 0
        self._symbol_prediction_delta_sum = 0.0
        self._prediction_error_sum = 0.0
        self._prediction_error_count = 0
        self.unified_telemetry = UnifiedTelemetry(model_version=scientific.hgt_model_version)
        self.telemetry: dict[str, int] = {
            "events": 0, "proposals": 0, "accepted": 0, "stale": 0,
            "rejected": 0, "cross_partition_transactions": 0,
            "canonical_reuse": 0, "canonical_forks": 0,
            "grounded_action_influence": 0,
            "grounding_promotions": 0, "grounding_suspensions": 0,
            "snapshot_writes": 0, "snapshot_restores": 0,
            "read_set_conflicts": 0, "symbol_conditioned_prediction_observations": 0,
        }
        if config.restore:
            path = latest_snapshot(self.root)
            if path is not None:
                self._restore(load_snapshot(path, expected_config_id=scientific.config_id.value))

    @property
    def watermark(self) -> int:
        return self._watermark

    @property
    def generation(self) -> int:
        return self.graph.generation

    @property
    def read_view(self) -> ReadView:
        return self.graph.read_view()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        self._started = True

    def reserve_producer_sequence(self, producer_id: int, proposed_sequence: int) -> int:
        with self._lock:
            sequence = max(
                int(proposed_sequence),
                self._producer_sequences.get(int(producer_id), 0) + 1,
            )
            self._producer_sequences[int(producer_id)] = sequence
            return sequence

    def make_experience(self, *, producer_id: int, producer_sequence: int, environment_instance_id: int | None = None, source_game_hash: int | None = None, global_step: int, context_signature: int, action_id: int, outcome_signature: int, family_signature: int = 0, carrier_signature: int = 0, future_option_delta: float = 0.0, changed_cells: int = 0, primary_valence: int = 0, trajectory_signature: int = 0, next_context_signature: int = 0, prediction_error: float = 0.0) -> ExperienceEvent:
        with self._lock:
            environment = int(environment_instance_id if environment_instance_id is not None else source_game_hash if source_game_hash is not None else 0)
            sequence = max(int(producer_sequence), self._producer_sequences.get(int(producer_id), 0) + 1)
            self._producer_sequences[int(producer_id)] = sequence
            return ExperienceEvent(EventUid.from_producer(producer_id, sequence), self._watermark + 1, int(producer_id), sequence, environment, int(global_step), int(context_signature), int(action_id), int(outcome_signature), int(family_signature), int(carrier_signature), float(future_option_delta), int(changed_cells), int(primary_valence), int(trajectory_signature), int(next_context_signature), float(prediction_error))

    def submit(self, experience: ExperienceEvent, *, episode_id: EpisodeId | None = None) -> bool:
        identity = TimelineIdentity(experience.event_id, experience.watermark, experience.producer_id, experience.producer_sequence, experience.environment_instance_id, episode_id or EpisodeId(0), WORLD_MODALITY)
        return self.submit_event(InteractionEvent(identity, experience))

    def submit_event(self, event: TimelineEvent) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            if not self.timeline.append(event):
                return False
            self._drain_timeline()
            return True

    def record_interaction(self, adapter: Any, *, producer_id: int, producer_sequence: int, global_step: int, native_action: int, before_observation: Any, after_observation: Any, episode_id: EpisodeId, symbol_codec: DeterministicSymbolCodec | None = None) -> ExperienceEvent:
        with self._lock:
            environment = self.environments.register(adapter.identity()).value
            sequence = max(int(producer_sequence), self._producer_sequences.get(int(producer_id), 0) + 1)
            self._producer_sequences[int(producer_id)] = sequence
            before_signature = int(adapter.encode_observation(before_observation))
            after_signature = int(adapter.encode_observation(after_observation))
            boundary = adapter.boundary_event()
            experience = ExperienceEvent(
                EventUid.from_producer(producer_id, sequence), self._watermark + 1,
                int(producer_id), sequence, environment, int(global_step),
                before_signature, int(adapter.encode_action(native_action)), after_signature,
                stable_u64(adapter.observation_schema().schema_id, int(before_signature != after_signature), person=b"v9-family"),
                stable_u64(adapter.observation_schema().schema_id, before_signature, person=b"v9-carrier"),
                float(len(adapter.available_actions())), int(before_signature != after_signature),
                int(boundary.primary_valence), stable_u64(environment, episode_id.value, person=b"v9-trajectory"),
                after_signature, 0.0,
            )
            self.submit(experience, episode_id=episode_id)
            symbols = tuple(adapter.optional_symbol_stream())
            if symbols and symbol_codec is not None:
                self.symbol_codecs[symbol_codec.vocabulary_id.value] = symbol_codec
                observations = symbol_codec.encode_stream(symbols, stream_name=f"{environment}:{episode_id.value}:{sequence}")
                events = tuple(
                    PassiveSymbolEvent(
                        TimelineIdentity(EventUid.from_producer(producer_id + 1_000_000, sequence * 10_000 + index), self._watermark + index + 1, producer_id + 1_000_000, sequence * 10_000 + index, environment, episode_id, SYMBOL_MODALITY),
                        row.vocabulary_id, row.stream_id, row.symbol_id, row.position.value,
                    )
                    for index, row in enumerate(observations)
                )
                raw_sizes = tuple(1 if isinstance(value, int) else len(value) for value in symbols)
                for event in self.timeline.append_symbols(events, raw_sizes):
                    self._drain_timeline()
            return experience

    def record_symbol_stream(self, adapter: Any, *, producer_id: int, producer_sequence: int, episode_id: EpisodeId, symbol_codec: DeterministicSymbolCodec) -> int:
        with self._lock:
            environment = self.environments.register(adapter.identity()).value
            symbols = tuple(adapter.optional_symbol_stream())
            if not symbols:
                return 0
            self.symbol_codecs[symbol_codec.vocabulary_id.value] = symbol_codec
            observations = symbol_codec.encode_stream(symbols, stream_name=f"{environment}:{episode_id.value}:{producer_sequence}")
            events = tuple(
                PassiveSymbolEvent(
                    TimelineIdentity(
                        EventUid.from_producer(producer_id + 1_000_000, producer_sequence * 10_000 + index),
                        self._watermark + index + 1,
                        producer_id + 1_000_000,
                        producer_sequence * 10_000 + index,
                        environment,
                        episode_id,
                        SYMBOL_MODALITY,
                    ),
                    row.vocabulary_id,
                    row.stream_id,
                    row.symbol_id,
                    row.position.value,
                )
                for index, row in enumerate(observations)
            )
            accepted = 0
            for event in self.timeline.append_symbols(events, tuple(1 for _ in symbols)):
                accepted += 1
                self._drain_timeline()
            return accepted

    def _drain_timeline(self) -> None:
        while (event := self.timeline.pop_next()) is not None:
            self._watermark = max(self._watermark, event.identity.causal_watermark)
            self.telemetry["events"] += 1
            modality = int(event.identity.modality_id.value)
            self._modality_events[modality] = self._modality_events.get(modality, 0) + 1
            stage_before = self.stage_tracker.stage
            signatures = self._ingest(event)
            self._develop(signatures)
            stage_snapshot = self.stage_tracker.close_interval(self._stage_evidence(), evidence_watermark=self._watermark)
            self.evidence.append("DEVELOPMENTAL_STAGE", self._watermark, {"interval_id": stage_snapshot.interval_id, "stage": int(stage_snapshot.stage), "next_stage": int(stage_snapshot.next_stage), "evidence": asdict(stage_snapshot.evidence)})
            if isinstance(event, InteractionEvent):
                experience = event.experience
                recurrence = self._m1n_supports.get(stable_u64(f"ACTION:{experience.action_id}:FAMILY:{experience.family_signature}:OUTCOME:{experience.outcome_signature}", NormalizedChannel.WORLD.value, person=b"v9-m1-normalized"), 0)
                decision = self.isf.score(
                    ISFComponents(abs(experience.primary_valence), abs(experience.future_option_delta), experience.prediction_error, 1.0 / max(1, recurrence), 0.5 if experience.family_signature else 0.0, min(1.0, experience.changed_cells / 16.0)),
                    decision_watermark=self._watermark,
                    evidence_availability_watermark=event.identity.causal_watermark,
                    stage=stage_before,
                    next_stage=stage_snapshot.next_stage,
                    graph_generation=self.graph.generation,
                )
                self._prediction_error_sum += abs(float(experience.prediction_error))
                self._prediction_error_count += 1
                self.evidence.append("ISF_DECISION", self._watermark, {"stage": int(decision.developmental_stage), "next_stage": int(decision.next_developmental_stage), "score": decision.score, "raw": asdict(decision.raw_components), "normalized": asdict(decision.normalized_components), "graph_generation": decision.graph_generation})

    def _stage_evidence(self) -> StageEvidence:
        strategies = [self.graph.payloads[uid] for uid in self.graph.uids_at_level(MemoryLevel.M7)]
        return StageEvidence(
            stable_contingencies=sum(len(rows) >= 2 for rows in self._m1n_occurrences.values()),
            structural_abstractions=len(self._m3),
            held_out_transfer_successes=sum(bool(row.validated) for row in self._m4.values()),
            mature_consequences=sum(bool(self.graph.payloads[uid].get("mature")) for uid in self.graph.uids_at_level(MemoryLevel.M5)),
            outcome_equivalences=self.graph.memory_count(MemoryLevel.M6),
            learned_preferences=sum(int(row.get("primary_valence_sum", 0)) != 0 for row in strategies),
            alternative_strategies=max(0, len(strategies) - 1),
            demonstrated_replans=self._replans_demonstrated,
            efficient_replans=self._efficient_replans,
        )

    def _payload_digest(self, event: TimelineEvent) -> int:
        if isinstance(event, InteractionEvent):
            return stable_u64(event.experience.context_signature, event.experience.action_id, event.experience.outcome_signature, person=b"v9-interaction-payload")
        if isinstance(event, PassiveSymbolEvent):
            return stable_u64(event.vocabulary_id.value, event.stream_id.value, event.symbol_id.value, event.position, person=b"v9-symbol-payload")
        return stable_u64(event.observation_schema_id, event.observation_signature, person=b"v9-world-payload")

    def _publish(self, node: CanonicalNode, payload: dict[str, Any], evidence: tuple[MemoryUid, ...], *, proposal_class: ProposalClass = ProposalClass.ADDITIVE, mutation_kind: MutationKind = MutationKind.UPSERT_NODE) -> None:
        payload = dict(payload)
        payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])
        owner = self.partitions.owner(node.uid)
        ref = node_ref(node.uid)
        read_set = ReadSet.build((ReadDependency(ref, self.graph.versions.get(ref)),), maximum_size=self.config.scientific.maximum_read_set_size)
        proposal = MutationProposal.build(mutation_kind, target_partitions=(owner,), read_set=read_set, evidence_refs=evidence, causal_watermark=self._watermark, writes=(MutationWrite(node=node, payload=payload),), proposal_class=proposal_class)
        self.telemetry["proposals"] += 1
        previous_generation = self.graph.generation
        result = self.graph.publish(proposal)
        if result.outcome.value == "ACCEPTED":
            self.telemetry["accepted"] += 1
            if result.graph_generation == previous_generation:
                self.telemetry["canonical_reuse"] += 1
            else:
                self.structural_index.add(node)
            if self.config.enable_lifecycle and result.graph_generation != previous_generation:
                self.lifecycle.observe(node.uid, support_delta=1, relevant_opportunity=True, watermark=self._watermark)
        elif result.outcome.value == "STALE_READ_SET":
            self.telemetry["stale"] += 1
            self.telemetry["read_set_conflicts"] += 1
        else:
            self.telemetry["rejected"] += 1
        for raw_parent in payload.get("parents", []):
            if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                continue
            parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
            edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
            partitions = tuple(sorted({self.partitions.owner(node.uid), self.partitions.owner(parent)}))
            edge_read = ReadSet.build((ReadDependency(edge_ref(edge), self.graph.versions.get(edge_ref(edge))),), maximum_size=self.config.scientific.maximum_read_set_size)
            edge_proposal = MutationProposal.build(MutationKind.UPSERT_EDGE, target_partitions=partitions, read_set=edge_read, evidence_refs=evidence, causal_watermark=self._watermark, writes=(MutationWrite(edge=edge),))
            self.telemetry["proposals"] += 1
            self.telemetry["cross_partition_transactions"] += int(len(partitions) > 1)
            edge_result = self.graph.publish(edge_proposal)
            if edge_result.outcome.value == "ACCEPTED":
                self.telemetry["accepted"] += 1
            elif edge_result.outcome.value == "STALE_READ_SET":
                self.telemetry["stale"] += 1
                self.telemetry["read_set_conflicts"] += 1
            else:
                self.telemetry["rejected"] += 1

    def _record_normalized(self, relation: M1NormalizedRelation) -> int:
        occurrences = self._m1n_occurrences.setdefault(relation.structural_signature, [])
        support = self._m1n_supports.get(relation.structural_signature, 0) + 1
        self._m1n_supports[relation.structural_signature] = support
        if len(occurrences) < max(2, self.config.scientific.m1n_facts_per_channel):
            occurrences.append(relation)
        self._replay_pool[relation.uid] = float(support)
        if len(self._replay_pool) > self.config.scientific.replay_candidates:
            victim = min(self._replay_pool, key=lambda uid: (self._replay_pool[uid], uid))
            del self._replay_pool[victim]
        self._publish(CanonicalNode(relation.uid, MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (relation.structural_signature,), self._watermark), {"observable_relation": relation.observable_relation, "channel": relation.channel.value, "structural_signature": relation.structural_signature, "support": support, "parents": [[uid.hi, uid.lo] for uid in relation.provenance.parents]}, relation.provenance.evidence)
        return relation.structural_signature

    def _ingest(self, event: TimelineEvent) -> tuple[int, ...]:
        self._formation_environments.add(int(event.identity.environment_instance_id))
        context = event.experience.context_signature if isinstance(event, InteractionEvent) else 0
        m0 = M0Episode.from_event(event, context_signature=context, payload_digest=self._payload_digest(event))
        self._publish(CanonicalNode(m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (event.identity.event_id.hi, event.identity.event_id.lo), self._watermark), {"modality_id": m0.modality_id, "environment_instance_id": m0.provenance.environment_instance_id, "episode_id": m0.provenance.episode_id.value, "context_signature": m0.context_signature, "payload_digest": m0.payload_digest, "action_id": m0.action_id, "outcome_signature": m0.outcome_signature, "next_context_signature": m0.next_context_signature, "symbol_identity": m0.symbol_identity, "primary_valence": m0.primary_valence, "future_option_delta": m0.future_option_delta, "realized_cost": m0.realized_cost}, (m0.uid,))
        if isinstance(event, InteractionEvent):
            relation, channel = GroundedRelation.ACTION_CONDITIONED, NormalizedChannel.WORLD
            observable = f"ACTION:{event.experience.action_id}:FAMILY:{event.experience.family_signature}:OUTCOME:{event.experience.outcome_signature}"
        elif isinstance(event, PassiveSymbolEvent):
            relation, channel = GroundedRelation.SYMBOL_OCCURRED, NormalizedChannel.SYMBOL
            observable = "SYMBOL_OCCURRED"
        else:
            relation, channel = GroundedRelation.PASSIVE_PRECEDES_SETTLED, NormalizedChannel.WORLD
            observable = f"PASSIVE_WORLD:{event.observation_schema_id}"
        m1g = M1GroundedContingency.build(relation, (m0,))
        self._publish(CanonicalNode(m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (m1g.uid.hi, m1g.uid.lo), self._watermark), {"relation": m1g.relation.value, "environment_instance_id": m1g.environment_instance_id, "episode_id": m1g.episode_id, "grounded_context_signature": m1g.grounded_context_signature, "executable_action_token": m1g.executable_action_token, "realized_transition_signature": m1g.realized_transition_signature, "grounded_next_context_signature": m1g.grounded_next_context_signature, "parents": [[m0.uid.hi, m0.uid.lo]]}, (m0.uid,))
        key = (m1g.environment_instance_id, m1g.episode_id)
        if isinstance(event, InteractionEvent):
            self._latest_interaction_grounding[key] = m1g
        m1n = M1NormalizedRelation.build(observable, channel, (m1g,))
        signatures = [self._record_normalized(m1n)]
        if isinstance(event, PassiveSymbolEvent) and key in self._latest_interaction_grounding:
            aligned = M1NormalizedRelation.build("SYMBOL_ALIGNED_WITH_INTERACTION", NormalizedChannel.CROSS_MODAL, (self._latest_interaction_grounding[key], m1g))
            signatures.append(self._record_normalized(aligned))
        self.evidence.append("INGESTION", self._watermark, {"m0": m0.uid.hex(), "m1g": m1g.uid.hex(), "m1n": m1n.uid.hex(), "channel": channel.value})
        return tuple(signatures)

    def apply_prepared_ingestion(self, prepared: Any) -> tuple[int, ...]:
        """Publish worker-prepared interaction memories without running M2-M4 derivation."""
        from v9.runtime.memory_pipeline import PreparedIngestion

        if not isinstance(prepared, PreparedIngestion):
            raise TypeError("prepared ingestion has unexpected type")
        if prepared.event is None:
            return ()
        with self._lock:
            event = prepared.event
            identity = self.environments.register(prepared.identity)
            if int(identity.value) != int(event.identity.environment_instance_id):
                raise RuntimeError("prepared environment identity mismatch")
            self._watermark = max(self._watermark, int(event.identity.causal_watermark))
            self.timeline.events_seen += 1
            self.timeline.actions_committed += 1
            self.timeline.last_ordering_key = event.identity.ordering_key
            self.telemetry["events"] += 1
            modality = int(event.identity.modality_id.value)
            self._modality_events[modality] = self._modality_events.get(modality, 0) + 1
            stage_before = self.stage_tracker.stage
            self._formation_environments.add(int(event.identity.environment_instance_id))

            m0 = prepared.m0
            m1g = prepared.m1g
            m1n = prepared.m1n
            if m0 is None or m1g is None or m1n is None:
                raise RuntimeError("prepared interaction is incomplete")

            self._publish(
                CanonicalNode(
                    m0.uid,
                    MemoryLevel.M0,
                    MemoryType.EPISODE,
                    (event.identity.event_id.hi, event.identity.event_id.lo),
                    self._watermark,
                ),
                {
                    "modality_id": m0.modality_id,
                    "environment_instance_id": m0.provenance.environment_instance_id,
                    "episode_id": m0.provenance.episode_id.value,
                    "context_signature": m0.context_signature,
                    "payload_digest": m0.payload_digest,
                    "action_id": m0.action_id,
                    "outcome_signature": m0.outcome_signature,
                    "next_context_signature": m0.next_context_signature,
                    "symbol_identity": m0.symbol_identity,
                    "primary_valence": m0.primary_valence,
                    "future_option_delta": m0.future_option_delta,
                    "realized_cost": m0.realized_cost,
                },
                (m0.uid,),
            )
            self._publish(
                CanonicalNode(
                    m1g.uid,
                    MemoryLevel.M1,
                    MemoryType.GROUNDED_CONTINGENCY,
                    (m1g.uid.hi, m1g.uid.lo),
                    self._watermark,
                ),
                {
                    "relation": m1g.relation.value,
                    "environment_instance_id": m1g.environment_instance_id,
                    "episode_id": m1g.episode_id,
                    "grounded_context_signature": m1g.grounded_context_signature,
                    "executable_action_token": m1g.executable_action_token,
                    "realized_transition_signature": m1g.realized_transition_signature,
                    "grounded_next_context_signature": m1g.grounded_next_context_signature,
                    "parents": [[m0.uid.hi, m0.uid.lo]],
                },
                (m0.uid,),
            )
            key = (m1g.environment_instance_id, m1g.episode_id)
            self._latest_interaction_grounding[key] = m1g
            signature = self._record_normalized(m1n)
            self.evidence.append(
                "INGESTION",
                self._watermark,
                {
                    "m0": m0.uid.hex(),
                    "m1g": m1g.uid.hex(),
                    "m1n": m1n.uid.hex(),
                    "channel": m1n.channel.value,
                    "worker_prepared": True,
                },
            )

            stage_snapshot = self.stage_tracker.close_interval(
                self._stage_evidence(),
                evidence_watermark=self._watermark,
            )
            self.evidence.append(
                "DEVELOPMENTAL_STAGE",
                self._watermark,
                {
                    "interval_id": stage_snapshot.interval_id,
                    "stage": int(stage_snapshot.stage),
                    "next_stage": int(stage_snapshot.next_stage),
                    "evidence": asdict(stage_snapshot.evidence),
                },
            )
            experience = event.experience
            recurrence = self._m1n_supports.get(signature, 0)
            decision = self.isf.score(
                ISFComponents(
                    abs(experience.primary_valence),
                    abs(experience.future_option_delta),
                    experience.prediction_error,
                    1.0 / max(1, recurrence),
                    0.5 if experience.family_signature else 0.0,
                    min(1.0, experience.changed_cells / 16.0),
                ),
                decision_watermark=self._watermark,
                evidence_availability_watermark=event.identity.causal_watermark,
                stage=stage_before,
                next_stage=stage_snapshot.next_stage,
                graph_generation=self.graph.generation,
            )
            self._prediction_error_sum += abs(float(experience.prediction_error))
            self._prediction_error_count += 1
            self.evidence.append(
                "ISF_DECISION",
                self._watermark,
                {
                    "stage": int(decision.developmental_stage),
                    "next_stage": int(decision.next_developmental_stage),
                    "score": decision.score,
                    "raw": asdict(decision.raw_components),
                    "normalized": asdict(decision.normalized_components),
                    "graph_generation": decision.graph_generation,
                },
            )
            return (signature,)

    def build_derivation_task(self, signature: int, *, task_id: int) -> Any | None:
        from v9.runtime.memory_pipeline import DerivationTask

        with self._lock:
            rows = tuple(self._m1n_occurrences.get(int(signature), ()))
            support = int(self._m1n_supports.get(int(signature), len(rows)))
            distinct_evidence = {
                evidence_uid
                for row in rows
                for evidence_uid in row.provenance.evidence
            }
            if len(rows) < 2 or support < 2 or len(distinct_evidence) < 2:
                return None
            return DerivationTask(
                int(task_id),
                int(signature),
                rows,
                support,
                tuple(sorted(self._formation_environments)),
                int(self._watermark),
            )

    def apply_derivation_result(self, result: Any) -> None:
        from v9.runtime.memory_pipeline import DerivationResult

        if not isinstance(result, DerivationResult):
            raise TypeError("derivation result has unexpected type")
        with self._lock:
            self._watermark = max(self._watermark, int(result.causal_watermark))
            family = result.family
            self._m2[family.uid] = family
            self._publish(
                CanonicalNode(
                    family.uid,
                    MemoryLevel.M2,
                    MemoryType.FAMILY,
                    (family.structural_signature,),
                    self._watermark,
                ),
                {
                    "structural_signature": family.structural_signature,
                    "recurrence": family.recurrence,
                    "compression_benefit": family.compression_benefit,
                    "parents": [[uid.hi, uid.lo] for uid in family.provenance.parents],
                },
                family.provenance.evidence,
            )
            for role in result.roles:
                self._m3[role.uid] = role
                self._publish(
                    CanonicalNode(
                        role.uid,
                        MemoryLevel.M3,
                        MemoryType.ROLE,
                        (role.relational_signature, role.consequence_signature),
                        self._watermark,
                    ),
                    {
                        "relational_signature": role.relational_signature,
                        "consequence_signature": role.consequence_signature,
                        "parents": [[uid.hi, uid.lo] for uid in role.provenance.parents],
                    },
                    role.provenance.evidence,
                )
            for candidate in result.concepts:
                if candidate.uid in self._m4:
                    continue
                self._m4[candidate.uid] = candidate
                self._publish(
                    CanonicalNode(
                        candidate.uid,
                        MemoryLevel.M4,
                        MemoryType.CONCEPT,
                        candidate.invariant_descriptor,
                        self._watermark,
                    ),
                    {
                        "invariant_descriptor": list(candidate.invariant_descriptor),
                        "compression_benefit": candidate.compression_benefit,
                        "explanatory_reach": candidate.explanatory_reach,
                        "transfer_prior": candidate.transfer_prior,
                        "formation_scope": list(candidate.provenance.formation_scope),
                        "held_out_targets": [],
                        "validated": False,
                        "concept_state": candidate.state.value,
                        "parents": [[uid.hi, uid.lo] for uid in candidate.provenance.parents],
                    },
                    candidate.provenance.evidence,
                )

    def _develop(self, signatures: tuple[int, ...] = ()) -> None:
        for signature in tuple(sorted(set(signatures)))[: self.config.scientific.replay_candidates_per_interval]:
            rows = tuple(self._m1n_occurrences[signature])
            support = self._m1n_supports.get(signature, len(rows))
            if len(rows) < 2 or support < 2:
                continue
            family = form_families(rows)[0]
            family = replace(family, recurrence=support, compression_benefit=float(support - 1))
            self._m2[family.uid] = family
            self._publish(CanonicalNode(family.uid, MemoryLevel.M2, MemoryType.FAMILY, (family.structural_signature,), self._watermark), {"structural_signature": family.structural_signature, "recurrence": family.recurrence, "compression_benefit": family.compression_benefit, "parents": [[row.uid.hi, row.uid.lo] for row in rows]}, family.provenance.evidence)
            roles = form_roles((family,), consequence_by_family={family.uid.lo: family.structural_signature})
            for role in roles:
                self._m3[role.uid] = role
                self._publish(CanonicalNode(role.uid, MemoryLevel.M3, MemoryType.ROLE, (role.relational_signature, role.consequence_signature), self._watermark), {"relational_signature": role.relational_signature, "consequence_signature": role.consequence_signature, "parents": [[uid.hi, uid.lo] for uid in role.provenance.parents]}, role.provenance.evidence)
                formation_scope = tuple(sorted(self._formation_environments))
                compression = family.compression_benefit
                candidate = M4Concept.candidate((role,), compression_benefit=compression, explanatory_reach=max(1, len(role.provenance.evidence)), transfer_prior=0.5, formation_scope=formation_scope)
                if candidate.uid not in self._m4:
                    self._m4[candidate.uid] = candidate
                    self._publish(CanonicalNode(candidate.uid, MemoryLevel.M4, MemoryType.CONCEPT, candidate.invariant_descriptor, self._watermark), {"invariant_descriptor": list(candidate.invariant_descriptor), "compression_benefit": candidate.compression_benefit, "explanatory_reach": candidate.explanatory_reach, "transfer_prior": candidate.transfer_prior, "formation_scope": list(candidate.provenance.formation_scope), "held_out_targets": [], "validated": False, "concept_state": candidate.state.value, "parents": [[uid.hi, uid.lo] for uid in candidate.provenance.parents]}, candidate.provenance.evidence)

    def effective_state(self, uid: MemoryUid, *, lineage_uid: LineageUid | None = None, context_scope_id: ContextScopeId | None = None, target_environment_id: int | None = None) -> EffectiveCognitiveState:
        return self.effective_states.resolve(self.graph.nodes[uid], lineage_uid=lineage_uid, context_scope_id=context_scope_id, target_environment_id=target_environment_id, target_trust=self.transfer_trust.score_map())

    def retrieve_structural_candidates(self, keys: tuple[StructuralIndexKey, ...], *, limit: int | None = None) -> tuple[MemoryUid, ...]:
        return self.structural_index.retrieve(keys, limit=limit or self.config.scientific.candidates_per_radius)

    def search_structural(self, query: dict[int, Any], candidates: dict[int, dict[int, Any]], *, compute_budget: int):
        outcome = self.similarity.search(query, candidates, compute_budget=compute_budget)
        for scale in outcome.scales:
            rows = self._similarity_entropy_by_radius.setdefault(scale.radius, [])
            rows.append(scale.entropy)
            del rows[:-self.config.scientific.normalization_reservoir_limit]
        return outcome

    def replay_once(self) -> ReplayResult:
        before_m0 = self.graph.memory_count(MemoryLevel.M0)
        candidates = tuple(ReplayCandidate(uid, fitness) for uid, fitness in sorted(self._replay_pool.items()))

        def process(_candidate: ReplayCandidate) -> tuple[int, int, int]:
            before_nodes, before_generation = len(self.graph.nodes), self.graph.generation
            signature = int(self.graph.payloads[_candidate.uid]["structural_signature"])
            self._develop((signature,))
            return len(self.graph.nodes) - before_nodes, int(self.graph.generation > before_generation), 0

        result = self.replay.run(candidates, process)
        if self.graph.memory_count(MemoryLevel.M0) != before_m0:
            raise RuntimeError("replay may not fabricate M0 environment evidence")
        self.evidence.append("REPLAY", self._watermark, asdict(result))
        return result

    def record_replanning_evidence(self, *, improved_efficiency: bool) -> None:
        self._replans_demonstrated += 1
        self._efficient_replans += int(improved_efficiency)
        self.evidence.append("REPLANNING", self._watermark, {"improved_efficiency": bool(improved_efficiency)})

    def record_symbol_conditioned_prediction(self, *, baseline: float, conditioned: float, actual: float) -> float:
        from v9.cognition.prediction import PredictionComparison
        comparison = PredictionComparison(float(baseline), float(conditioned), float(actual))
        self._symbol_prediction_delta_sum += comparison.delta
        self.telemetry["symbol_conditioned_prediction_observations"] += 1
        self.evidence.append("SYMBOL_CONDITIONED_PREDICTION", self._watermark, {"baseline": baseline, "conditioned": conditioned, "actual": actual, "delta": comparison.delta})
        return comparison.delta

    def record_transfer_validation(self, concept_uid: MemoryUid, *, target_environment_id: int, target_native_action: int, enabled_metric: float, ablated_metric: float, matched: bool = True, held_out: bool = True, context_scope_id: int = 0) -> None:
        if concept_uid not in self._m4:
            raise KeyError("transfer validation requires an existing M4 concept candidate")
        row = {"target_environment_id": int(target_environment_id), "target_native_action": int(target_native_action), "enabled_metric": float(enabled_metric), "ablated_metric": float(ablated_metric), "matched": bool(matched), "held_out": bool(held_out), "context_scope_id": int(context_scope_id)}
        trial_rows = self._transfer_trials.setdefault(concept_uid, [])
        trial_rows.append(row)
        hot_trial_limit = max(self.config.scientific.transfer_minimum_trials, self.config.scientific.transfer_validation_trials_per_interval)
        del trial_rows[:-hot_trial_limit]
        self.evidence.append("TRANSFER_TRIAL", self._watermark, {"concept_uid": concept_uid.hex(), **row})
        formation_scope = set(self._m4[concept_uid].provenance.formation_scope)
        effect = float(enabled_metric) - float(ablated_metric)
        positive = bool(matched and held_out and int(target_environment_id) not in formation_scope and effect > self.config.scientific.transfer_effect_threshold)
        self.transfer_trust.observe(concept_uid, target_environment_id=target_environment_id, context_scope_id=context_scope_id, effect=effect, positive=positive)
        admissible = tuple(trial for trial in self._transfer_trials[concept_uid] if trial["matched"] and trial["held_out"] and trial["target_environment_id"] not in formation_scope and trial["enabled_metric"] - trial["ablated_metric"] > self.config.scientific.transfer_effect_threshold)
        if ValidationMode(self.config.scientific.transfer_validation_mode) is ValidationMode.LEARNING_ONLY:
            return
        if len(admissible) < self.config.scientific.transfer_minimum_trials:
            return
        concept = self._m4[concept_uid].with_validation(tuple(int(trial["target_environment_id"]) for trial in admissible))
        self._m4[concept.uid] = concept
        concept_node = self.graph.nodes[concept.uid]
        self._publish(concept_node, {**self.graph.payloads[concept.uid], "held_out_targets": list(concept.held_out_targets), "validated": concept.validated, "concept_state": concept.state.value}, concept.provenance.evidence, proposal_class=ProposalClass.STATEFUL, mutation_kind=MutationKind.UPDATE_VALIDATION)
        if self.config.enable_lifecycle and concept.uid in self.lifecycle.records:
            self.lifecycle.records[concept.uid] = replace(self.lifecycle.records[concept.uid], state=CognitiveState.VALIDATED, last_transition_watermark=self._watermark)
        role = self._m3[concept.provenance.parents[0]]
        consequence = M5ConsequenceStructure.form((concept,), (role.consequence_signature,))
        self._publish(CanonicalNode(consequence.uid, MemoryLevel.M5, MemoryType.CONSEQUENCE, consequence.consequence_descriptor, self._watermark), {"descriptor": list(consequence.consequence_descriptor), "mature": consequence.mature, "parents": [[uid.hi, uid.lo] for uid in consequence.provenance.parents]}, consequence.provenance.evidence)
        outcome = M6Outcome.form((consequence,), diameter_bound=0)
        self._publish(CanonicalNode(outcome.uid, MemoryLevel.M6, MemoryType.OUTCOME, outcome.class_signature, self._watermark), {"class_signature": list(outcome.class_signature), "class_version": outcome.class_version, "parents": [[uid.hi, uid.lo] for uid in outcome.provenance.parents]}, outcome.provenance.evidence)
        action = int(admissible[-1]["target_native_action"])
        grounded_payloads = [self.graph.payloads[uid] for uid in outcome.provenance.evidence if uid in self.graph.payloads and self.graph.nodes[uid].level is MemoryLevel.M0]
        primary_valence_sum = sum(int(payload.get("primary_valence", 0)) for payload in grounded_payloads)
        realized_cost_sum = sum(max(1, int(payload.get("realized_cost", 0))) for payload in grounded_payloads) or len(admissible)
        strategy = M7Strategy.form(outcome, target_environment_id=int(target_environment_id), native_actions=(action,), successes=len(admissible), trials=len(admissible), primary_valence_sum=primary_valence_sum, realized_cost_sum=realized_cost_sum)
        self._publish(CanonicalNode(strategy.uid, MemoryLevel.M7, MemoryType.STRATEGY, (outcome.uid.hi, outcome.uid.lo, target_environment_id, action), self._watermark), {"target_outcome": [outcome.uid.hi, outcome.uid.lo], "target_environment_id": int(target_environment_id), "native_actions": [action], "reliability_successes": strategy.reliability_successes, "reliability_trials": strategy.reliability_trials, "primary_valence_sum": strategy.primary_valence_sum, "realized_cost_sum": strategy.realized_cost_sum, "parents": [[outcome.uid.hi, outcome.uid.lo]]}, strategy.provenance.evidence)

    def wait_quiescent(self, timeout: float = 300.0) -> None:
        del timeout
        self._drain_timeline()

    def snapshot(self) -> SnapshotResult:
        with self._lock:
            self.wait_quiescent()
            self._snapshot_id += 1
            self.telemetry["snapshot_writes"] += 1
            snapshot_id = self._snapshot_id
            watermark = self._watermark
            generation = self.graph.generation
            fixed_cut = self.state_dict()
        try:
            return write_snapshot(self.root, fixed_cut, snapshot_id=snapshot_id, watermark=watermark, graph_generation=generation, scientific_config_id=self.config.scientific.config_id.value)
        except BaseException:
            with self._lock:
                self.telemetry["snapshot_writes"] -= 1
            raise

    def state_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.state_dict(), "timeline": self.timeline.state_dict(),
            "environments": self.environments.state_dict(), "evidence": self.evidence.state_dict(),
            "lifecycle": self.lifecycle.state_dict(), "lineages": self.lineages.state_dict(), "contexts": self.contexts.state_dict(),
            "scale_statistics": self.scale_statistics.state_dict(), "similarity": self.similarity.state_dict(),
            "structural_index": self.structural_index.state_dict(),
            "grounding": self.grounding.state_dict(), "transfer_trust": self.transfer_trust.state_dict(),
            "developmental_stage": self.stage_tracker.state_dict(), "isf": self.isf.state_dict(), "replay": self.replay.state_dict(),
            "payloads": self.payloads.state_dict(),
            "symbol_codecs": [codec.state_dict() for _, codec in sorted(self.symbol_codecs.items())],
            "watermark": self._watermark, "snapshot_id": self._snapshot_id,
            "producer_sequences": {str(key): value for key, value in self._producer_sequences.items()},
            "telemetry": dict(self.telemetry),
            "modality_events": {str(key): value for key, value in self._modality_events.items()},
            "similarity_entropy_by_radius": {str(key): list(value) for key, value in self._similarity_entropy_by_radius.items()},
            "replans_demonstrated": self._replans_demonstrated, "efficient_replans": self._efficient_replans,
            "symbol_prediction_delta_sum": self._symbol_prediction_delta_sum,
            "prediction_error_sum": self._prediction_error_sum,
            "prediction_error_count": self._prediction_error_count,
            "unified_telemetry": self.unified_telemetry.state_dict(),
            "in_flight_proposals": [],
            "m1n_occurrences": {str(key): len(value) for key, value in self._m1n_occurrences.items()},
            "m1n_supports": {str(key): value for key, value in self._m1n_supports.items()},
            "replay_pool": {uid.hex(): value for uid, value in self._replay_pool.items()},
            "formation_environments": sorted(self._formation_environments),
            "latest_interaction_grounding": [
                {
                    "key": list(key), "uid": [row.uid.hi, row.uid.lo], "relation": row.relation.value,
                    "parents": [[uid.hi, uid.lo] for uid in row.provenance.parents],
                    "evidence": [[uid.hi, uid.lo] for uid in row.provenance.evidence],
                    "environment_instance_id": row.environment_instance_id, "episode_id": row.episode_id,
                    "grounded_context_signature": row.grounded_context_signature,
                    "executable_action_token": row.executable_action_token,
                    "realized_transition_signature": row.realized_transition_signature,
                    "grounded_next_context_signature": row.grounded_next_context_signature,
                    "support": row.support,
                }
                for key, row in sorted(self._latest_interaction_grounding.items())
            ],
            "transfer_trials": {uid.hex(): rows for uid, rows in self._transfer_trials.items()},
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        state = dict(snapshot["state"])
        self.graph = CanonicalGraph.from_state_dict(dict(state["graph"]))
        self.partitions = PartitionMap(self.graph.partition_count)
        self.timeline = MultimodalTimeline.from_state_dict(dict(state["timeline"]))
        self.environments = EnvironmentRegistry.from_state_dict(dict(state["environments"]))
        self.evidence.load_state(dict(state["evidence"]))
        self.lifecycle = LifecycleRegistry.from_state_dict(dict(state.get("lifecycle", {})))
        self.lineages = LineageStore.from_state_dict(dict(state.get("lineages", {})))
        self.contexts = ContextRegistry.from_state_dict(dict(state.get("contexts", {"limit": self.config.scientific.context_scope_limit, "descriptor_limit": self.config.scientific.descriptor_component_limit, "records": []})))
        self.effective_states = EffectiveStateResolver(self.contexts, self.lineages)
        self.scale_statistics = ScaleStatistics.from_state_dict(dict(state["scale_statistics"]))
        scientific = self.config.scientific
        self.similarity = ProgressiveSimilarity(beta_by_radius=dict(scientific.beta_by_radius), candidate_limit=scientific.candidates_per_radius, equivalence_limit=scientific.equivalence_set_size, maximum_radius=max(scientific.structural_radii), ambiguity_threshold=scientific.ambiguity_entropy_threshold, margin_threshold=scientific.top2_margin_threshold, information_threshold=scientific.information_gain_threshold, symmetry_patience=scientific.symmetry_patience, statistics=self.scale_statistics)
        self.similarity.load_state(dict(state.get("similarity", {})))
        self.structural_index = StructuralCandidateIndex.from_state_dict(dict(state.get("structural_index", {"bucket_capacity": scientific.candidates_per_radius, "bucket_scan_limit": scientific.structural_index_bucket_scan_limit})))
        self.grounding = GroundingRegistry.from_state_dict(dict(state.get("grounding", {})))
        self.transfer_trust = TransferTrustRegistry.from_state_dict(dict(state.get("transfer_trust", {"scope_limit": scientific.transfer_trust_scope_limit})))
        self.stage_tracker = DevelopmentalStageTracker.from_state_dict(dict(state.get("developmental_stage", {})))
        self.isf = InteractionSignificanceFunction.from_state_dict(dict(state.get("isf", {"weights_by_stage": scientific.isf_weights_by_stage, "schema_version": scientific.isf_score_schema_version, "hot_limit": scientific.isf_decision_hot_limit})))
        self.replay = ReplayScheduler.from_state_dict(dict(state.get("replay", {"candidate_limit": scientific.replay_candidates_per_interval})))
        self.payloads = PayloadStore.from_state_dict(dict(state.get("payloads", {"byte_budget": 1 << 20, "records": []})))
        self.symbol_codecs = {}
        for raw in state.get("symbol_codecs", []):
            codec = DeterministicSymbolCodec.from_state_dict(dict(raw))
            self.symbol_codecs[codec.vocabulary_id.value] = codec
        self._watermark = int(state["watermark"])
        self._snapshot_id = int(state["snapshot_id"])
        self._producer_sequences = {int(key): int(value) for key, value in dict(state.get("producer_sequences", {})).items()}
        self.telemetry.update({key: int(value) for key, value in dict(state.get("telemetry", {})).items()})
        self.telemetry["snapshot_restores"] += 1
        self._modality_events = {int(key): int(value) for key, value in dict(state.get("modality_events", {})).items()}
        self._similarity_entropy_by_radius = {int(key): [float(value) for value in values] for key, values in dict(state.get("similarity_entropy_by_radius", {})).items()}
        self._replans_demonstrated = int(state.get("replans_demonstrated", 0))
        self._efficient_replans = int(state.get("efficient_replans", 0))
        self._symbol_prediction_delta_sum = float(state.get("symbol_prediction_delta_sum", 0.0))
        self._prediction_error_sum = float(state.get("prediction_error_sum", 0.0))
        self._prediction_error_count = int(state.get("prediction_error_count", 0))
        self.unified_telemetry = UnifiedTelemetry.from_state_dict(dict(state.get("unified_telemetry", UnifiedTelemetry(model_version=scientific.hgt_model_version).state_dict())))
        if state.get("in_flight_proposals"):
            raise RuntimeError("native v9 snapshot contains unsupported in-flight proposals")
        for signature, count in dict(state.get("m1n_occurrences", {})).items():
            node = next((uid for uid, payload in self.graph.payloads.items() if payload.get("structural_signature") == int(signature) and self.graph.nodes[uid].memory_type is MemoryType.NORMALIZED_RELATION), None)
            if node is None:
                continue
            payload = self.graph.payloads[node]
            dummy_parent_uid = node
            evidence_refs = tuple(MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("evidence_refs", [[node.hi, node.lo]]))
            dummy = M1NormalizedRelation(node, str(payload["observable_relation"]), NormalizedChannel(str(payload["channel"])), int(signature), DerivationProvenance((dummy_parent_uid,), evidence_refs))
            self._m1n_occurrences[int(signature)] = [dummy] if int(count) > 0 else []
        self._m1n_supports = {int(key): int(value) for key, value in dict(state.get("m1n_supports", state.get("m1n_occurrences", {}))).items()}
        self._replay_pool = {MemoryUid(int(key[:16], 16), int(key[16:], 16)): float(value) for key, value in dict(state.get("replay_pool", {})).items()}
        self._formation_environments = {int(value) for value in state.get("formation_environments", [])}
        self._latest_interaction_grounding = {}
        for raw in state.get("latest_interaction_grounding", []):
            uid = MemoryUid(int(raw["uid"][0]), int(raw["uid"][1]))
            parents = tuple(MemoryUid(int(value[0]), int(value[1])) for value in raw["parents"])
            evidence = tuple(MemoryUid(int(value[0]), int(value[1])) for value in raw["evidence"])
            row = M1GroundedContingency(uid, GroundedRelation(str(raw["relation"])), DerivationProvenance(parents, evidence), int(raw["environment_instance_id"]), int(raw["episode_id"]), int(raw["grounded_context_signature"]), None if raw.get("executable_action_token") is None else int(raw["executable_action_token"]), int(raw["realized_transition_signature"]), int(raw["grounded_next_context_signature"]), int(raw.get("support", 1)))
            self._latest_interaction_grounding[(int(raw["key"][0]), int(raw["key"][1]))] = row
        for uid, node in self.graph.nodes.items():
            payload = self.graph.payloads[uid]
            parents = tuple(MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("parents", []))
            evidence_refs = tuple(MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("evidence_refs", payload.get("parents", [])))
            provenance = DerivationProvenance(parents, evidence_refs) if parents else None
            if node.level is MemoryLevel.M2 and provenance is not None:
                self._m2[uid] = M2TransformationFamily(uid, int(payload["structural_signature"]), provenance, int(payload["recurrence"]), float(payload["compression_benefit"]))
            elif node.level is MemoryLevel.M3 and provenance is not None:
                self._m3[uid] = M3FunctionalRole(uid, int(payload["relational_signature"]), int(payload["consequence_signature"]), provenance)
            elif node.level is MemoryLevel.M4 and provenance is not None:
                from v9.memory.m4_concept import ConceptState
                self._m4[uid] = M4Concept(uid, tuple(int(value) for value in payload.get("invariant_descriptor", node.structural_key)), DerivationProvenance(parents, evidence_refs, tuple(int(value) for value in payload.get("formation_scope", []))), float(payload["compression_benefit"]), int(payload["explanatory_reach"]), float(payload["transfer_prior"]), tuple(int(value) for value in payload.get("held_out_targets", [])), bool(payload.get("validated", False)), ConceptState(str(payload.get("concept_state", "VALIDATED_CONCEPT" if payload.get("validated") else "TRANSFER_TEST_ELIGIBLE"))))
        for raw_uid, rows in dict(state.get("transfer_trials", {})).items():
            self._transfer_trials[MemoryUid(int(raw_uid[:16], 16), int(raw_uid[16:], 16))] = list(rows)

    def telemetry_provenance(
        self,
        *,
        decision_uid: str | None = None,
        model_version: str | None = None,
        curriculum_step: str | None = None,
        environment_family: str | None = None,
        game_scenario: str | None = None,
        context_uid: str | None = None,
        lineage_uid: str | None = None,
        memory_level: str | None = None,
    ) -> TelemetryProvenance:
        return TelemetryProvenance(
            decision_uid=decision_uid,
            graph_generation=self.graph.generation,
            model_version=model_version or self.unified_telemetry.model_version,
            scientific_config_id=self.config.scientific.config_id.value,
            curriculum_step=curriculum_step,
            environment_family=environment_family,
            game_scenario=game_scenario,
            context_uid=context_uid,
            lineage_uid=lineage_uid,
            memory_level=memory_level,
        )

    def record_deliberation_metrics(self, *, reasoning_cycles: int, initial_score: float, final_score: float, best_score: float, changed: bool, behavior_improved: bool | None, reasoning_cost: float, stop_reason: str, candidate_changes: int = 0, prediction_improvement: float = 0.0, strategy_changes: int = 0, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_deliberation(reasoning_cycles=reasoning_cycles, initial_score=initial_score, final_score=final_score, best_score=best_score, changed=changed, behavior_improved=behavior_improved, reasoning_cost=reasoning_cost, stop_reason=stop_reason, candidate_changes=candidate_changes, prediction_improvement=prediction_improvement, strategy_changes=strategy_changes, provenance=provenance)

    def record_hgt_inference(self, sample: HGTInferenceSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_hgt_inference(sample, provenance=provenance)

    def record_hgt_ablation(self, *, enabled_outcome: float, hydra_baseline_outcome: float, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_hgt_ablation(enabled_outcome=enabled_outcome, hydra_baseline_outcome=hydra_baseline_outcome, provenance=provenance)

    def record_hgt_training(self, sample: HGTTrainingSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_hgt_training(sample, provenance=provenance)

    def record_model_evolution(self, sample: ModelEvolutionSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_model_evolution(sample, provenance=provenance)

    def record_hgt_consolidation(self, sample: ConsolidationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_consolidation(sample, provenance=provenance)

    def record_optimization(self, sample: OptimizationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.unified_telemetry.record_optimization(sample, provenance=provenance)

    def metrics(self) -> dict[str, Any]:
        view = self.read_view
        counts = {f"M{level}": self.graph.memory_count(MemoryLevel(level)) for level in range(8)}
        normalization = {str(radius): self.scale_statistics.state(radius).value for radius in self.config.scientific.structural_radii}
        grounding_counts = {f"G{level}": sum(int(row.maturity) == level for row in self.grounding.states.values()) for level in range(6)}
        persistent_bytes = max(1, len(json.dumps(self.graph.state_dict(), sort_keys=True, separators=(",", ":"))))
        validated_transfers = sum(row.successes for row in self.transfer_trust.records.values())
        prediction_observations = self.telemetry["symbol_conditioned_prediction_observations"]
        strategy_payloads = [payload for uid, payload in view.payloads.items() if view.nodes[uid].level is MemoryLevel.M7]
        strategy_trials = sum(int(row.get("reliability_trials", 0)) for row in strategy_payloads)
        strategy_successes = sum(int(row.get("reliability_successes", 0)) for row in strategy_payloads)
        realized_cost_total = sum(float(row.get("realized_cost_sum", 0.0)) for row in strategy_payloads)
        failed_transfer_scopes = sum(getattr(row.state, "name", str(row.state)) == "FAILED" for row in self.transfer_trust.records.values())
        validated_m4 = sum(bool(row.validated) for row in self._m4.values())
        diagnostic = self.unified_telemetry.diagnostic_metrics()
        retired = int(diagnostic.get("hydra_nodes_retired", 0))
        replaced = int(diagnostic.get("hydra_nodes_replaced_by_abstractions", 0))
        compression_ratio = (retired + replaced) / max(1, len(view.nodes) + retired)
        base = {
            "watermark": self._watermark,
            "graph_generation": view.generation,
            "memories": len(view.nodes),
            "edges": len(view.edges),
            "memory_levels": counts,
            "scientific_config_id": self.config.scientific.config_id.value,
            "timeline_events_seen": self.timeline.events_seen,
            "timeline_events_dropped": self.timeline.events_dropped,
            "events_by_modality": {str(key): value for key, value in sorted(self._modality_events.items())},
            "actions_committed": self.timeline.actions_committed,
            "normalization_states": normalization,
            "normalization_estimator_generation": self.scale_statistics.estimator_generation,
            "structural_index_generation": self.structural_index.generation,
            "structural_index_buckets": len(self.structural_index.buckets),
            "candidate_entropy_by_radius": {str(key): list(value) for key, value in sorted(self._similarity_entropy_by_radius.items())},
            "equivalence_sets": len(self.similarity.equivalence_sets),
            "transfer_validation_mode": self.config.scientific.transfer_validation_mode,
            "transfer_trials": sum(len(rows) for rows in self._transfer_trials.values()),
            "transfer_trust_scopes": len(self.transfer_trust.records),
            "failed_transfer_scopes": failed_transfer_scopes,
            "cross_family_transfer": validated_transfers / max(1, len(self.transfer_trust.records)),
            "grounding_relations": len(self.grounding.states),
            "grounding_counts": grounding_counts,
            "lineage_overlays": len(self.lineages.overlays),
            "lineage_dependencies": len(self.lineages.dependencies),
            "context_scopes": len(self.contexts.records),
            "probation_records": sum(row.state.name == "PROBATION" for row in self.lifecycle.records.values()),
            "probation_transitions": self.lifecycle.transitions,
            "developmental_stage": int(self.stage_tracker.stage),
            "developmental_intervals": self.stage_tracker.interval_id,
            "isf_decisions_hot": len(self.isf.decisions),
            "replay": self.replay.state_dict(),
            "symbol_conditioned_prediction_delta": self._symbol_prediction_delta_sum / max(1, prediction_observations),
            "prediction_error": self._prediction_error_sum / max(1, self._prediction_error_count),
            "persistent_consolidated_bytes": persistent_bytes,
            "persistent_memory_growth_ratio": len(view.nodes) / max(1, self.telemetry["events"]),
            "compression_ratio": compression_ratio,
            "m4_validated": validated_m4,
            "success_rate": strategy_successes / max(1, strategy_trials),
            "trajectory_efficiency": strategy_successes / max(1.0, realized_cost_total),
            "explanatory_reach_per_persistent_byte": sum(int(payload.get("explanatory_reach", 0)) for payload in view.payloads.values()) / persistent_bytes,
            "transfer_quality_per_persistent_byte": validated_transfers / persistent_bytes,
            "prediction_quality_per_persistent_byte": max(0.0, self._symbol_prediction_delta_sum) / persistent_bytes,
            "hot_payload_bytes": self.payloads.hot_bytes,
            **self.telemetry,
        }
        base["telemetry_diagnostics"] = diagnostic
        base["primary_dashboard"] = build_primary_dashboard(base, diagnostic)
        return base



    def scientific_statuses(self) -> dict[str, str]:
        return {"H16": "UNTESTED", "H17": "UNTESTED", "H18": "UNTESTED", "H19": "UNTESTED"}

    def write_scientific_report(self) -> Path:
        assessments = {
            name: asdict(
                untested_assessment(
                    name,
                    scientific_config_id=self.config.scientific.config_id.value,
                    blocker="required matched causal evaluation has not run",
                )
            )
            for name in self.scientific_statuses()
        }
        return write_report(
            self.root,
            "reporting_cut.json",
            {
                "metrics": self.metrics(),
                "hypotheses": self.scientific_statuses(),
                "hypothesis_assessments": assessments,
                "scientific_config": self.config.scientific.as_dict(),
            },
        )

    def close(self, *, normal: bool = True, timeout: float = 300.0) -> SnapshotResult | None:
        del timeout
        if self._closed:
            return None
        result = self.snapshot() if normal and self.config.enable_snapshots else None
        if normal:
            self.write_scientific_report()
        self._closed = True
        return result
