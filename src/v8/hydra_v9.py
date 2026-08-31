from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
import threading
from typing import Iterable

from v8.environment_neutral_transfer import EnvironmentNeutralTransferGate
from v8.environment_registry import EnvironmentIdentityRegistry, EpisodeId
from v8.grounding import GroundingKey, GroundingMaturity, GroundingRegistry
from v8.lineage import ContextScopeId, LineageOverlayStore, LineageUid
from v8.modalities.symbols import DeterministicSymbolCodec, ModalityId, SymbolObservation
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationProposal,
    RelationType,
    ValidationState,
    encode_relation_proposal,
    proposal_fingerprint,
    stable_u64,
)
from v8.multimodal_events import (
    BoundedMultimodalTimeline,
    InteractionTimelineEvent,
    PassiveSymbolEvent,
    TimelineIdentity,
)
from v8.progressive_similarity import ProgressiveStructuralSearch, StructuralDescriptor
from v8.residency import M0ProvenanceRecord, PayloadAvailabilityState, PayloadResidencyStore, PayloadUid
from v8.scientific_config import ScientificConfig, write_scientific_config_manifest
from v8.structural_statistics import ScaleStratifiedStatistics
from v8.symbolic_primitives import (
    SymbolicPrimitive,
    is_symbol_normalized_token,
    symbol_family_key,
    symbol_normalized_primitive,
    symbol_normalized_token,
    symbol_role_token,
)
from v8.versioning import ObjectRef, ReadDependency, StateMutationProposal, StateWrite, VersionedMutationStore


_INSTALLED = False
_THREAD = threading.local()


class LogicalRelationType(str, Enum):
    TEMPORALLY_ALIGNED = "TEMPORALLY_ALIGNED"
    CROSS_MODAL_CORRESPONDENCE = "CROSS_MODAL_CORRESPONDENCE"
    GROUNDS = "GROUNDS"


@dataclass(frozen=True, slots=True)
class LogicalRelationRecord:
    source_uid: MemoryUid
    target_uid: MemoryUid
    logical_type: LogicalRelationType
    watermark: int
    support: int = 1


@dataclass(frozen=True, slots=True)
class ShadowPredictionRecord:
    environment_instance_id: int
    context_signature: int
    action_id: int
    actual_outcome: int
    baseline_correct: bool
    symbol_correct: bool
    symbol_available: bool


class HydraV9State:
    STATE_VERSION = 2

    def __init__(self, runtime, config: ScientificConfig | None = None) -> None:
        self.runtime = runtime
        self.config = config or ScientificConfig()
        self.registry = EnvironmentIdentityRegistry()
        self.timeline = BoundedMultimodalTimeline(
            max_symbols_per_window=self.config.max_symbols_per_window,
            max_symbol_payload_bytes=self.config.max_symbol_payload_bytes,
            max_pending_passive_events=self.config.max_pending_passive_events,
        )
        self.versioning = VersionedMutationStore()
        self.lineage = LineageOverlayStore(
            restore_support_threshold=self.config.probation_restore_support_threshold,
            evidence_opportunity_budget=self.config.probation_evidence_opportunity_budget,
            developmental_age_budget=self.config.probation_developmental_age_budget,
        )
        self.statistics = ScaleStratifiedStatistics(
            n_bootstrap=self.config.normalization_n_bootstrap,
            n_stable_bootstrap=self.config.normalization_n_stable_bootstrap,
            coverage_bootstrap=self.config.normalization_coverage_bootstrap,
            span_bootstrap=self.config.normalization_span_bootstrap,
            max_provisional_samples=self.config.normalization_max_provisional_samples,
        )
        self.progressive = ProgressiveStructuralSearch(
            radii=self.config.structural_radii,
            r_max=self.config.structural_r_max,
            beta_by_radius=dict(self.config.beta_by_radius),
            max_candidates_per_radius=self.config.max_candidates_per_radius,
            top_candidates=self.config.top_candidates,
            ambiguity_threshold=self.config.ambiguity_threshold,
            symmetry_information_threshold=self.config.symmetry_information_threshold,
            symmetry_patience=self.config.symmetry_patience,
        )
        self.grounding = GroundingRegistry()
        self.residency = PayloadResidencyStore()
        self.transfer = EnvironmentNeutralTransferGate(held_out_minimum=self.config.transfer_held_out_validation_minimums)
        self.codecs: dict[int, DeterministicSymbolCodec] = {}
        self._episodes: dict[int, EpisodeId] = {}
        self._context_environment: dict[int, int] = {}
        self._recent_symbols: dict[tuple[int, int], list[tuple[MemoryUid, MemoryUid, SymbolObservation]]] = defaultdict(list)
        self._symbol_support: Counter[MemoryUid] = Counter()
        self._symbol_m1n: set[MemoryUid] = set()
        self._cross_modal_m1n: set[MemoryUid] = set()
        self._association_counts: dict[tuple[MemoryUid, int], Counter[int]] = defaultdict(Counter)
        self._logical_relations: dict[tuple[MemoryUid, MemoryUid, str], LogicalRelationRecord] = {}
        self._correspondence_pairs: set[tuple[MemoryUid, MemoryUid]] = set()
        self._behavior_effects: dict[tuple[int, int, int], tuple[float, GroundingKey, MemoryUid, int]] = {}
        self._shadow: list[ShadowPredictionRecord] = []
        self._seen_interactions: set[tuple[int, int]] = set()
        self._peer_scans = 0
        self._lock = threading.RLock()
        write_scientific_config_manifest(runtime.root, self.config)

    @staticmethod
    def _m0_uid(event_id: EventId) -> MemoryUid:
        return MemoryUid.from_key(MemoryLevel.M0, MemoryType.EPISODE, (int(event_id.hi), int(event_id.lo)))

    @staticmethod
    def _world_m1_uid(event: ExperienceEvent) -> MemoryUid:
        key = (int(event.context_signature), int(event.action_id), int(event.outcome_signature), int(event.next_context_signature))
        return MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)

    def register_environment(self, identity) -> int:
        with self._lock:
            instance = self.registry.register(identity)
            self._episodes.setdefault(instance, self.registry.next_episode(instance))
            return instance

    def start_episode(self, identity) -> EpisodeId:
        with self._lock:
            instance = self.registry.register(identity)
            episode = self.registry.next_episode(instance)
            self._episodes[instance] = episode
            stale = [key for key in self._recent_symbols if key[0] == instance]
            for key in stale: self._recent_symbols.pop(key, None)
            return episode

    def _episode(self, environment_instance_id: int) -> EpisodeId:
        return self._episodes.get(int(environment_instance_id), EpisodeId(0))

    def _submit_node(self, *, level: MemoryLevel, memory_type: MemoryType, key: tuple[int, ...], event_id: EventId, watermark: int, source_environment: int, parent_uid: MemoryUid = MemoryUid(0, 0), relation_type: RelationType = RelationType.EXPLAINS, support_delta: int = 1, cognitive_state: int = -1, validation_state: int = -1) -> MemoryUid:
        uid = MemoryUid.from_key(level, memory_type, key)
        proposal = MemoryProposal(
            uid=uid,
            fingerprint=proposal_fingerprint(level, memory_type, key),
            event_id=event_id,
            watermark=max(0, int(watermark)),
            level=level,
            memory_type=memory_type,
            key_parts=key,
            support_delta=max(0, int(support_delta)),
            score_weight=float(max(0, int(support_delta))),
            parent_uid=parent_uid,
            relation_type=relation_type,
            source_game_hash=int(source_environment),
            cognitive_state=int(cognitive_state),
            validation_state=int(validation_state),
        )
        self.runtime.submit_proposal(proposal)
        return uid

    def _submit_extra_parent(self, uid: MemoryUid, *, level: MemoryLevel, memory_type: MemoryType, key: tuple[int, ...], event_id: EventId, watermark: int, parent_uid: MemoryUid, source_environment: int, relation_type: RelationType = RelationType.EXPLAINS) -> None:
        proposal = MemoryProposal(
            uid=uid,
            fingerprint=proposal_fingerprint(level, memory_type, key),
            event_id=event_id,
            watermark=max(0, int(watermark)),
            level=level,
            memory_type=memory_type,
            key_parts=key,
            support_delta=0,
            score_weight=0.0,
            parent_uid=parent_uid,
            relation_type=relation_type,
            source_game_hash=int(source_environment),
        )
        self.runtime.submit_proposal(proposal)

    def _logical_edge(self, source: MemoryUid, target: MemoryUid, logical_type: LogicalRelationType, *, watermark: int, score: float = 0.0) -> None:
        key = (source, target, logical_type.value)
        prior = self._logical_relations.get(key)
        self._logical_relations[key] = LogicalRelationRecord(source, target, logical_type, int(watermark), 1 if prior is None else prior.support + 1)
        # Existing binary schema carries structural edge evidence through the already
        # edge-only TRANSFER_CORRESPONDENCE relation; exact v9 semantics are retained
        # in the logical-relation ledger persisted with the same snapshot.
        if logical_type in {LogicalRelationType.CROSS_MODAL_CORRESPONDENCE, LogicalRelationType.GROUNDS}:
            self.runtime.submit_relation_proposal(RelationProposal(
                source_uid=source,
                target_uid=target,
                relation_type=RelationType.TRANSFER_CORRESPONDENCE,
                event_id=EventId.from_producer(0x7FFF0001, stable_u64(source.hi, source.lo, target.hi, target.lo, logical_type.value, int(watermark), person=b"v9-edge-event") & 0xFFFFFFFFFFFFFFFF),
                watermark=max(0, int(watermark)),
                support_delta=1,
                score_sum=float(score),
                score_weight=1.0 if score != 0.0 else 0.0,
            ))

    @staticmethod
    def _grounded_key(primitive: SymbolicPrimitive, *parts: object) -> tuple[int, ...]:
        value = stable_u64(primitive.name, *parts, person=b"v9-m1g") & ((1 << 63) - 1)
        return (int(value),)

    def _normalized(self, primitive: SymbolicPrimitive, *structure: object) -> int:
        return symbol_normalized_token(primitive, *structure)

    def observe_symbol_observations(self, observations: Iterable[SymbolObservation], *, environment_identity, context_signature: int, producer_id: int, raw_payload: bytes | None = None) -> tuple[MemoryUid, ...]:
        with self._lock:
            environment = self.registry.register(environment_identity)
            self._context_environment[int(context_signature)] = environment
            episode = self._episodes.get(environment)
            if episode is None:
                episode = self.registry.next_episode(environment); self._episodes[environment] = episode
            rows = tuple(observations)[: self.config.max_symbols_per_window]
            payload_budget = self.config.max_symbol_payload_bytes
            if raw_payload is not None and len(raw_payload) > payload_budget:
                raw_payload = raw_payload[:payload_budget]
            admitted: list[MemoryUid] = []
            previous: tuple[MemoryUid, MemoryUid, SymbolObservation] | None = None
            recent_key = (environment, int(context_signature))
            existing_recent = self._recent_symbols.get(recent_key, [])
            if existing_recent: previous = existing_recent[-1]
            current_recent: list[tuple[MemoryUid, MemoryUid, SymbolObservation]] = []
            for observation in rows:
                sequence = self.timeline.next_producer_sequence(int(producer_id))
                event_id = EventId.from_producer(int(producer_id), sequence)
                identity = TimelineIdentity(event_id, max(0, int(self.runtime.watermark)), int(producer_id), sequence, environment, episode, ModalityId.SYMBOL)
                passive = PassiveSymbolEvent(identity, observation.vocabulary_id, observation.stream_id, observation.symbol_id, observation.position)
                if not self.timeline.append(passive): break
                m0_uid = self._submit_node(level=MemoryLevel.M0, memory_type=MemoryType.EPISODE, key=(event_id.hi, event_id.lo), event_id=event_id, watermark=identity.causal_watermark, source_environment=environment, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                digest = self.residency.digest(raw_payload) if raw_payload is not None else stable_u64(observation.vocabulary_id.value, observation.stream_id.value, observation.symbol_id.value, observation.position.value, person=b"v9-symbol-payload")
                payload_uid = PayloadUid(digest)
                self.residency.register(M0ProvenanceRecord(m0_uid, environment, episode.value, identity.causal_watermark, int(context_signature), payload_uid, digest, PayloadAvailabilityState.ABSENT if raw_payload is None else PayloadAvailabilityState.HOT, int(ModalityId.SYMBOL), vocabulary_id=observation.vocabulary_id.value, stream_id=observation.stream_id.value, symbol_id=observation.symbol_id.value, symbol_position=observation.position.value), raw_payload)

                occurred_key = self._grounded_key(SymbolicPrimitive.SYMBOL_OCCURRED, observation.vocabulary_id.value, observation.stream_id.value, observation.symbol_id.value)
                m1g_uid = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=occurred_key, event_id=event_id, watermark=identity.causal_watermark, source_environment=environment, parent_uid=m0_uid, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                position_bucket = min(7, max(0, int(observation.position.value)))
                normalized_token = self._normalized(SymbolicPrimitive.SYMBOL_OCCURRED, position_bucket)
                m1n_uid = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=(normalized_token,), event_id=event_id, watermark=identity.causal_watermark, source_environment=environment, parent_uid=m1g_uid, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                self._symbol_m1n.add(m1n_uid); self._symbol_support[m1n_uid] += 1
                stable = self._symbol_support[m1n_uid] >= 3
                self.statistics.observe("symbol_relation", 1, float(position_bucket), watermark=identity.causal_watermark, stable=stable, contingency_uid=stable_u64(m1n_uid.hi,m1n_uid.lo,person=b"v9-stat-cont"), descriptor_class=int(SymbolicPrimitive.SYMBOL_OCCURRED))
                current = (m0_uid, m1n_uid, observation)
                if previous is not None:
                    prev_m0, _prev_m1n, prev_obs = previous
                    same = prev_obs.symbol_id == observation.symbol_id and prev_obs.vocabulary_id == observation.vocabulary_id
                    primitive = SymbolicPrimitive.SYMBOL_REPEATED if same else SymbolicPrimitive.SYMBOL_PRECEDES_SYMBOL
                    relation_key = self._grounded_key(primitive, prev_obs.vocabulary_id.value, prev_obs.symbol_id.value, observation.vocabulary_id.value, observation.symbol_id.value)
                    rel_uid = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=relation_key, event_id=event_id, watermark=identity.causal_watermark, source_environment=environment, parent_uid=prev_m0, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                    self._submit_extra_parent(rel_uid, level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=relation_key, event_id=EventId.from_producer(int(producer_id), sequence + (1 << 32)), watermark=identity.causal_watermark, parent_uid=m0_uid, source_environment=environment)
                    structural = (1 if same else 0, min(7, max(0, observation.position.value - prev_obs.position.value)))
                    rel_token = self._normalized(primitive, *structural)
                    rel_n_uid = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=(rel_token,), event_id=event_id, watermark=identity.causal_watermark, source_environment=environment, parent_uid=rel_uid, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                    self._symbol_m1n.add(rel_n_uid); self._symbol_support[rel_n_uid] += 1
                previous = current; current_recent.append(current); admitted.append(m0_uid)
            self._recent_symbols[recent_key] = current_recent[-self.config.max_cross_modal_facts_per_window:]
            return tuple(admitted)

    def _symbol_prediction(self, environment: int, context: int, action: int) -> tuple[int | None, int]:
        combined: Counter[int] = Counter()
        for _m0, source_uid, _obs in self._recent_symbols.get((environment, context), []):
            combined.update(self._association_counts.get((source_uid, int(action)), {}))
        total = sum(combined.values())
        if total < 3 or not combined: return None, total
        return min(combined, key=lambda outcome: (-combined[outcome], outcome)), total

    def observe_interaction(self, event: ExperienceEvent, baseline_distribution: dict[int, float] | None = None) -> None:
        with self._lock:
            event_key = (event.event_id.hi, event.event_id.lo)
            if event_key in self._seen_interactions: return
            self._seen_interactions.add(event_key)
            environment = int(event.source_game_hash); context = int(event.context_signature); self._context_environment[context] = environment
            episode = self._episodes.get(environment, EpisodeId(0))
            identity = TimelineIdentity(event.event_id, int(event.watermark), int(event.producer_id), int(event.producer_sequence), environment, episode, ModalityId.WORLD)
            try: self.timeline.append(InteractionTimelineEvent(identity, event, episode.value == 0))
            except ValueError: pass
            m0_uid = self._m0_uid(event.event_id)
            payload_digest = stable_u64(context, event.outcome_signature, person=b"v9-interaction-payload")
            self.residency.register(M0ProvenanceRecord(m0_uid, environment, episode.value, event.watermark, context, PayloadUid(payload_digest), payload_digest, PayloadAvailabilityState.ABSENT, int(ModalityId.WORLD), action_id=event.action_id, outcome_signature=event.outcome_signature))
            world_m1 = self._world_m1_uid(event)
            baseline_top = None
            if baseline_distribution:
                baseline_top = min(baseline_distribution, key=lambda outcome: (-baseline_distribution[outcome], outcome))
            symbol_top, _support = self._symbol_prediction(environment, context, event.action_id)
            self._shadow.append(ShadowPredictionRecord(environment, context, int(event.action_id), int(event.outcome_signature), baseline_top == event.outcome_signature, symbol_top == event.outcome_signature if symbol_top is not None else baseline_top == event.outcome_signature, symbol_top is not None))

            for symbol_m0, symbol_m1n, observation in self._recent_symbols.get((environment, context), [])[:self.config.max_cross_modal_facts_per_window]:
                relation_key = self._grounded_key(SymbolicPrimitive.SYMBOL_PRECEDES_ACTION, observation.vocabulary_id.value, observation.symbol_id.value, int(event.action_id))
                cross_m1g = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=relation_key, event_id=event.event_id, watermark=event.watermark, source_environment=environment, parent_uid=symbol_m0, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                self._submit_extra_parent(cross_m1g, level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=relation_key, event_id=EventId.from_producer(event.producer_id, event.producer_sequence + (1 << 32)), watermark=event.watermark, parent_uid=m0_uid, source_environment=environment)
                structural = (min(7, max(0, observation.position.value)), 1 if event.terminal_polarity != 0 else 0)
                token = self._normalized(SymbolicPrimitive.SYMBOL_PRECEDES_ACTION, *structural)
                cross_m1n = self._submit_node(level=MemoryLevel.M1, memory_type=MemoryType.CONTINGENCY, key=(token,), event_id=event.event_id, watermark=event.watermark, source_environment=environment, parent_uid=cross_m1g, cognitive_state=int(CognitiveState.ACTIVE), validation_state=int(ValidationState.VALIDATED))
                self._cross_modal_m1n.add(cross_m1n); self._symbol_support[cross_m1n] += 1
                counts = self._association_counts[(symbol_m1n, int(event.action_id))]
                prior_total = sum(counts.values()); prior_dominant = max(counts.values(), default=0)
                predictive = prior_total >= 3 and prior_dominant / max(1, prior_total) >= 0.75
                gkey = GroundingKey(symbol_m1n, world_m1, environment, context, 0)
                self.grounding.observe(gkey, watermark=event.watermark, recurrent=self._symbol_support[symbol_m1n] >= 2, temporally_aligned=True, predictive=predictive, positive=True)
                self._logical_edge(symbol_m1n, world_m1, LogicalRelationType.TEMPORALLY_ALIGNED, watermark=event.watermark)
                counts[int(event.outcome_signature)] += 1
            if not event.terminal_polarity:
                return
            primitive = SymbolicPrimitive.SYMBOL_PRECEDES_BOUNDARY
            for symbol_m0, _symbol_m1n, observation in self._recent_symbols.get((environment, context), [])[:self.config.max_cross_modal_facts_per_window]:
                key=self._grounded_key(primitive,observation.vocabulary_id.value,observation.symbol_id.value,int(event.terminal_polarity)); uid=self._submit_node(level=MemoryLevel.M1,memory_type=MemoryType.CONTINGENCY,key=key,event_id=event.event_id,watermark=event.watermark,source_environment=environment,parent_uid=symbol_m0,cognitive_state=int(CognitiveState.ACTIVE),validation_state=int(ValidationState.VALIDATED)); token=self._normalized(primitive,int(event.terminal_polarity)); nuid=self._submit_node(level=MemoryLevel.M1,memory_type=MemoryType.CONTINGENCY,key=(token,),event_id=event.event_id,watermark=event.watermark,source_environment=environment,parent_uid=uid,cognitive_state=int(CognitiveState.ACTIVE),validation_state=int(ValidationState.VALIDATED)); self._cross_modal_m1n.add(nuid)

    def record_grounding_intervention(self, *, source_symbol_uid: MemoryUid, target_interaction_uid: MemoryUid, environment_instance_id: int, context_signature: int, effect: float, held_out: bool, higher_memory_uid: MemoryUid | None = None, source_environment_instance_id: int | None = None) -> bool:
        with self._lock:
            key = GroundingKey(source_symbol_uid, target_interaction_uid, int(environment_instance_id), int(context_signature), 0)
            ref = ObjectRef("grounding", stable_u64(source_symbol_uid.hi,source_symbol_uid.lo,target_interaction_uid.hi,target_interaction_uid.lo,environment_instance_id,context_signature,person=b"v9-grounding-ref"))
            _value, version = self.versioning.read(ref)
            target_maturity = GroundingMaturity.G5 if held_out and effect > 0 else GroundingMaturity.G4 if effect > 0 else GroundingMaturity.G3
            proposal = StateMutationProposal.build("GROUNDING_INTERVENTION", base_graph_generation=self.versioning.graph_generation, target_partition_ids=(source_symbol_uid.shard(max(1,len(self.runtime.shard_descriptors))), target_interaction_uid.shard(max(1,len(self.runtime.shard_descriptors)))), read_set=(ReadDependency(ref,version),), evidence_refs=(int(abs(effect)*1_000_000),), causal_watermark=self.runtime.watermark, writes=(StateWrite(ref,int(target_maturity)),), event_type_priority=20)
            result = self.versioning.apply_stateful((proposal,))[0]
            if not result.accepted: return False
            row = self.grounding.observe(key, watermark=self.runtime.watermark, recurrent=True, temporally_aligned=True, predictive=True, causal=effect>0, held_out=held_out, positive=effect>0)
            if effect <= 0: return True
            self._logical_edge(source_symbol_uid,target_interaction_uid,LogicalRelationType.GROUNDS,watermark=self.runtime.watermark,score=min(1.0,max(0.0,float(effect))))
            if higher_memory_uid is not None:
                self._validate_higher_memory(higher_memory_uid)
                source_env = int(environment_instance_id if source_environment_instance_id is None else source_environment_instance_id)
                action = self._target_action(target_interaction_uid)
                if action is not None:
                    self._behavior_effects[(int(environment_instance_id),int(context_signature),int(action))]=(max(-0.25,min(0.25,float(effect))),key,higher_memory_uid,source_env)
            return row.maturity >= GroundingMaturity.G4

    def _target_action(self, target_uid: MemoryUid) -> int | None:
        for row in self.runtime.read_view.node_records(level=MemoryLevel.M1):
            if row.uid == target_uid and len(row.key_parts) >= 4:
                from v8.model import signed_u64
                return signed_u64(int(row.key_parts[1]))
        return None

    def _validate_higher_memory(self, uid: MemoryUid) -> None:
        for row in self.runtime.read_view.node_records():
            if row.uid != uid or int(row.level) < int(MemoryLevel.M2): continue
            proposal = MemoryProposal(uid=row.uid,fingerprint=row.fingerprint,event_id=EventId.from_producer(0x7FFF0002,stable_u64(uid.hi,uid.lo,self.runtime.watermark,person=b"v9-validate")&0xFFFFFFFFFFFFFFFF),watermark=self.runtime.watermark,level=MemoryLevel(int(row.level)),memory_type=MemoryType(int(row.memory_type)),key_parts=tuple(int(v) for v in row.key_parts),support_delta=0,score_weight=0.0,cognitive_state=int(CognitiveState.VALIDATED),validation_state=int(ValidationState.VALIDATED)); self.runtime.submit_proposal(proposal); return

    def _higher_validated(self, view, uid: MemoryUid) -> bool:
        for row in view.node_records():
            if row.uid == uid:
                return bool(int(row.level)>=int(MemoryLevel.M2) and (int(row.validation_state)>=int(ValidationState.VALIDATED) or int(row.cognitive_state)==int(CognitiveState.VALIDATED)))
        return False

    def action_delta(self, view, context: int, action: int) -> float:
        environment = self._context_environment.get(int(context))
        if environment is None: return 0.0
        raw = self._behavior_effects.get((environment,int(context),int(action)))
        if raw is None: return 0.0
        effect,key,higher_uid,source_env=raw
        cross = int(source_env)!=int(environment)
        required=GroundingMaturity.G5 if cross else GroundingMaturity.G4
        state=self.grounding.state_for(key)
        if state is None or state.suspended or state.maturity<required or not self._higher_validated(view,higher_uid): return 0.0
        return float(effect)

    def adjust_action_scores(self, view, context: int, rows):
        from v8.publication import ActionScore
        return tuple(ActionScore(row.action_id,row.support_count,float(row.score)+self.action_delta(view,context,row.action_id),row.evidence_shards) for row in rows)

    def _descriptor(self, node, edges, by_uid, radius: int, generation: int) -> StructuralDescriptor:
        frontier={node.uid}; visited={node.uid}; components={stable_u64(int(node.level),int(node.memory_type),len(node.key_parts),person=b"v9-desc-node")}
        adjacency:dict[MemoryUid,list[tuple[int,MemoryUid]]]=defaultdict(list)
        for edge in edges:
            adjacency[edge.source_uid].append((int(edge.relation_type),edge.target_uid)); adjacency[edge.target_uid].append((int(edge.relation_type),edge.source_uid))
        for depth in range(max(0,int(radius))):
            nxt=set()
            for uid in frontier:
                for relation,target in adjacency.get(uid,())[:32]:
                    components.add(stable_u64(depth,relation,person=b"v9-desc-edge")); target_row=by_uid.get(target)
                    if target_row is not None: components.add(stable_u64(depth,int(target_row.level),int(target_row.memory_type),len(target_row.key_parts),person=b"v9-desc-neighbor"))
                    if target not in visited: visited.add(target); nxt.add(target)
            if not nxt: break
            frontier=nxt
        return StructuralDescriptor(node.uid,generation,int(node.updated_watermark),int(radius),1,0,tuple(sorted(components))[:128])

    def peer_cycle(self, view) -> None:
        with self._lock:
            self._peer_scans += 1
            if self._peer_scans % 2: return
            nodes=tuple(view.node_records()); edges=tuple(view.edge_records()); by_uid={row.uid:row for row in nodes}; generation=int(self.runtime.generation)
            symbols=[by_uid[uid] for uid in sorted(self._symbol_m1n|self._cross_modal_m1n) if uid in by_uid]
            world=[row for row in nodes if int(row.level)==int(MemoryLevel.M1) and len(row.key_parts)==1 and row.uid not in self._symbol_m1n and row.uid not in self._cross_modal_m1n]
            if not symbols or not world: return
            for source in symbols[:8]:
                source_by_radius={r:self._descriptor(source,edges,by_uid,r,generation) for r in self.config.structural_radii}
                candidates_by_radius={r:tuple(self._descriptor(row,edges,by_uid,r,generation) for row in world[:self.config.max_candidates_per_radius]) for r in self.config.structural_radii}
                result=self.progressive.search(source_by_radius,candidates_by_radius,current_graph_generation=generation)
                target=result.winner
                if target is None or target==source.uid: continue
                pair=(source.uid,target)
                if pair in self._correspondence_pairs: continue
                self._correspondence_pairs.add(pair); score=result.candidates[0].score if result.candidates else 0.0
                self._logical_edge(source.uid,target,LogicalRelationType.CROSS_MODAL_CORRESPONDENCE,watermark=self.runtime.watermark,score=score)

    def shadow_metrics(self) -> dict[str, float | int]:
        rows=self._shadow
        if not rows: return {"samples":0,"baseline_accuracy":0.0,"symbol_accuracy":0.0,"prediction_improvement":0.0,"symbol_prediction_samples":0}
        total=len(rows); baseline=sum(r.baseline_correct for r in rows)/total; symbol=sum(r.symbol_correct for r in rows)/total
        return {"samples":total,"baseline_accuracy":baseline,"symbol_accuracy":symbol,"prediction_improvement":symbol-baseline,"symbol_prediction_samples":sum(r.symbol_available for r in rows)}

    def metrics(self) -> dict[str, object]:
        return {"scientific_config_id":self.config.config_id,"symbol_m1n_nodes":len(self._symbol_m1n),"cross_modal_m1n_nodes":len(self._cross_modal_m1n),"logical_relation_count":len(self._logical_relations),"grounding_maturity":self.grounding.maturity_distribution(),"grounding_edge_count":self.grounding.grounds_edge_count,"shadow_prediction":self.shadow_metrics(),"versioning":self.versioning.telemetry(),"statistics":self.statistics.telemetry(),"progressive_search":self.progressive.telemetry(),"payload_hot_bytes":self.residency.hot_bytes,"payloads_retired":self.residency.payloads_retired,"transfer_false_count":self.transfer.false_transfer_count,"peer_scans":self._peer_scans}

    def state_dict(self) -> dict[str, object]:
        return {"version":self.STATE_VERSION,"scientific_config_id":self.config.config_id,"registry":self.registry.state_dict(),"timeline":self.timeline.state_dict(),"versioning":self.versioning.state_dict(),"lineage":self.lineage.state_dict(),"statistics":self.statistics.state_dict(),"progressive":self.progressive.state_dict(),"grounding":self.grounding.state_dict(),"residency":self.residency.state_dict(),"transfer":self.transfer.state_dict(),"codecs":[self.codecs[k].state_dict() for k in sorted(self.codecs)],"episodes":{str(k):v.value for k,v in self._episodes.items()},"context_environment":{str(k):v for k,v in self._context_environment.items()},"symbol_m1n":[[u.hi,u.lo] for u in sorted(self._symbol_m1n)],"cross_modal_m1n":[[u.hi,u.lo] for u in sorted(self._cross_modal_m1n)],"symbol_support":[[[u.hi,u.lo],n] for u,n in sorted(self._symbol_support.items())],"associations":[{"source":[u.hi,u.lo],"action":a,"outcomes":dict(c)} for (u,a),c in self._association_counts.items()],"logical_relations":[{"source":[r.source_uid.hi,r.source_uid.lo],"target":[r.target_uid.hi,r.target_uid.lo],"logical_type":r.logical_type.value,"watermark":r.watermark,"support":r.support} for r in self._logical_relations.values()],"correspondence_pairs":[[[a.hi,a.lo],[b.hi,b.lo]] for a,b in sorted(self._correspondence_pairs)],"behavior_effects":[{"environment":env,"context":ctx,"action":action,"effect":effect,"key":{"source":[key.source_symbol_uid.hi,key.source_symbol_uid.lo],"target":[key.target_interaction_uid.hi,key.target_interaction_uid.lo],"environment":key.environment_instance_id,"context":key.context_scope_id,"lineage":key.lineage_uid},"higher":[higher.hi,higher.lo],"source_environment":source_env} for (env,ctx,action),(effect,key,higher,source_env) in self._behavior_effects.items()],"shadow":[asdict(row) for row in self._shadow[-10000:]],"peer_scans":self._peer_scans}

    def load_state(self, state: dict[str, object]) -> None:
        if int(state.get("version",0))>self.STATE_VERSION: raise ValueError("unsupported Hydra v9 state")
        if str(state.get("scientific_config_id",self.config.config_id))!=self.config.config_id: raise RuntimeError("Hydra ScientificConfigId mismatch")
        if isinstance(state.get("registry"),dict): self.registry=EnvironmentIdentityRegistry.from_state_dict(state["registry"])
        if isinstance(state.get("timeline"),dict): self.timeline=BoundedMultimodalTimeline.from_state_dict(state["timeline"])
        if isinstance(state.get("versioning"),dict): self.versioning=VersionedMutationStore.from_state_dict(state["versioning"])
        if isinstance(state.get("lineage"),dict): self.lineage=LineageOverlayStore.from_state_dict(state["lineage"])
        if isinstance(state.get("statistics"),dict): self.statistics=ScaleStratifiedStatistics.from_state_dict(state["statistics"])
        if isinstance(state.get("progressive"),dict): self.progressive=ProgressiveStructuralSearch.from_state_dict(state["progressive"])
        if isinstance(state.get("grounding"),dict): self.grounding=GroundingRegistry.from_state_dict(state["grounding"])
        if isinstance(state.get("residency"),dict): self.residency=PayloadResidencyStore.from_state_dict(state["residency"])
        if isinstance(state.get("transfer"),dict): self.transfer=EnvironmentNeutralTransferGate.from_state_dict(state["transfer"])
        self.codecs={};
        for raw in state.get("codecs",[]):
            if isinstance(raw,dict): codec=DeterministicSymbolCodec.from_state_dict(raw); self.codecs[codec.vocabulary_id.value]=codec
        self._episodes={int(k):EpisodeId(int(v)) for k,v in dict(state.get("episodes",{})).items()}; self._context_environment={int(k):int(v) for k,v in dict(state.get("context_environment",{})).items()}
        self._symbol_m1n={MemoryUid(*map(int,p)) for p in state.get("symbol_m1n",[])}; self._cross_modal_m1n={MemoryUid(*map(int,p)) for p in state.get("cross_modal_m1n",[])}; self._symbol_support=Counter({MemoryUid(*map(int,p)):int(n) for p,n in state.get("symbol_support",[])})
        self._association_counts=defaultdict(Counter)
        for raw in state.get("associations",[]):
            if isinstance(raw,dict): self._association_counts[(MemoryUid(*map(int,raw["source"])),int(raw["action"]))]=Counter({int(k):int(v) for k,v in dict(raw.get("outcomes",{})).items()})
        self._logical_relations={}
        for raw in state.get("logical_relations",[]):
            if isinstance(raw,dict): r=LogicalRelationRecord(MemoryUid(*map(int,raw["source"])),MemoryUid(*map(int,raw["target"])),LogicalRelationType(str(raw["logical_type"])),int(raw["watermark"]),int(raw.get("support",1))); self._logical_relations[(r.source_uid,r.target_uid,r.logical_type.value)]=r
        self._correspondence_pairs={(MemoryUid(*map(int,a)),MemoryUid(*map(int,b))) for a,b in state.get("correspondence_pairs",[])}
        self._behavior_effects={}
        for raw in state.get("behavior_effects",[]):
            if not isinstance(raw,dict): continue
            kr=raw["key"]; key=GroundingKey(MemoryUid(*map(int,kr["source"])),MemoryUid(*map(int,kr["target"])),int(kr["environment"]),int(kr["context"]),int(kr["lineage"])); self._behavior_effects[(int(raw["environment"]),int(raw["context"]),int(raw["action"]))]=(float(raw["effect"]),key,MemoryUid(*map(int,raw["higher"])),int(raw["source_environment"]))
        self._shadow=[ShadowPredictionRecord(**{k:(bool(v) if k in {"baseline_correct","symbol_correct","symbol_available"} else int(v)) for k,v in raw.items()}) for raw in state.get("shadow",[]) if isinstance(raw,dict)]; self._peer_scans=int(state.get("peer_scans",0))


class _HydraAdapterProxy:
    def __init__(self, adapter, runtime) -> None:
        self._adapter=adapter; self._runtime=runtime; self._last_publication=None
        try: runtime.hydra_v9.register_environment(adapter.identity)
        except Exception: pass
    def __getattr__(self,name): return getattr(self._adapter,name)
    def reset(self,*args,**kwargs):
        result=self._adapter.reset(*args,**kwargs); self._last_publication=None
        try: self._runtime.hydra_v9.start_episode(self._adapter.identity)
        except Exception: pass
        return result
    def observe(self,*args,**kwargs):
        observation=self._adapter.observe(*args,**kwargs); self._publish(observation); return observation
    def _publish(self,observation) -> None:
        try:
            context=int(self._adapter.observation_signature(observation)) if hasattr(self._adapter,"observation_signature") else int(self._adapter.cognitive_context_signature()) if hasattr(self._adapter,"cognitive_context_signature") else stable_u64(repr(observation),person=b"v9-proxy-context")
            symbols=tuple(self._adapter.instruction_symbols()) if hasattr(self._adapter,"instruction_symbols") else ()
            raw=getattr(observation,"instruction_bytes",None)
            marker=(context,tuple((s.vocabulary_id.value,s.stream_id.value,s.symbol_id.value,s.position.value) for s in symbols))
            if symbols and marker!=self._last_publication:
                self._runtime.hydra_v9.observe_symbol_observations(symbols,environment_identity=self._adapter.identity,context_signature=context,producer_id=0x60000000+(threading.get_ident()&0xFFFF),raw_payload=raw if isinstance(raw,(bytes,bytearray)) else None); self._last_publication=marker
        except Exception:
            return
    def close(self): return self._adapter.close()


def _patch_normalized_promotion() -> None:
    import v8.normalized_memory_v086 as normalized
    base_family_key=normalized.normalized_family_key
    base_m3=normalized._normalized_m3_candidates
    def family_key(value:int): return symbol_family_key(value) if is_symbol_normalized_token(value) else base_family_key(value)
    normalized.normalized_family_key=family_key
    def m3(engine,nodes,edges,*,limit:int):
        result=list(base_m3(engine,nodes,edges,limit=limit)); remaining=max(0,int(limit)-len(result))
        if remaining<=0: return tuple(result[:limit])
        from v8.promotion import FormationCandidate
        by_uid={row.uid:row for row in nodes}; children=engine._children(tuple(edges))
        families=[row for row in nodes if int(row.level)==int(MemoryLevel.M2) and int(row.memory_type)==int(MemoryType.FAMILY) and len(row.key_parts)>=2 and (int(row.key_parts[0])&(1<<63)) and int(row.support_count)>=int(engine.min_carrier_family_support) and engine._admissible(row)]
        existing={row.uid for row in result}
        for family in sorted(families,key=lambda row:row.uid):
            parents=[by_uid[uid] for uid in children.get(family.uid,()) if uid in by_uid and len(by_uid[uid].key_parts)==1 and is_symbol_normalized_token(int(by_uid[uid].key_parts[0]))]
            if not parents: continue
            grouped=defaultdict(list)
            for parent in parents: grouped[symbol_role_token(int(parent.key_parts[0]))].append(parent)
            for role_token,rows in sorted(grouped.items()):
                support=sum(max(0,int(row.support_count)) for row in rows)
                if support<int(engine.min_carrier_persistence): continue
                key=(stable_u64(family.uid.hi,family.uid.lo,person=b"v9-m3-family"),int(role_token),0); uid=MemoryUid.from_key(MemoryLevel.M3,MemoryType.CARRIER,key)
                if uid in existing: continue
                utility=min(1.0,support/max(2.0,float(len(rows)*2))); result.append(FormationCandidate(uid,MemoryLevel.M3,MemoryType.CARRIER,key,(family.uid,*tuple(sorted(row.uid for row in rows))[:7]),support,utility,utility,0.0,utility,0.0,int(CognitiveState.PROBATION),int(ValidationState.STRUCTURAL),"v9_symbol_role_convergence",utility)); existing.add(uid)
                if len(result)>=limit: return tuple(result[:limit])
        return tuple(result[:limit])
    normalized._normalized_m3_candidates=m3


def install_hydra_v9() -> None:
    global _INSTALLED
    if _INSTALLED: return
    import v8.runtime as runtime_module
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.publication import LiveReadView
    from v8.snapshot import load_latest_auxiliary_state

    _patch_normalized_promotion()
    base_init=V82ContinuousMemoryRuntime.__init__
    base_submit=V82ContinuousMemoryRuntime.submit
    base_aux=V82ContinuousMemoryRuntime._auxiliary_state_json
    base_metrics=V82ContinuousMemoryRuntime.metrics
    base_report=V82ContinuousMemoryRuntime.write_scientific_report

    def init(self,config,*args,**kwargs):
        base_init(self,config,*args,**kwargs); self.hydra_v9=HydraV9State(self); self.read_view._hydra_v9_state=self.hydra_v9
        if bool(config.restore):
            restored=load_latest_auxiliary_state(config.root)
            if isinstance(restored,dict) and isinstance(restored.get("hydra_v9"),dict): self.hydra_v9.load_state(restored["hydra_v9"])
    def submit_relation_proposal(self,proposal:RelationProposal,*,timeout:float=.25):
        if self._closed or self._stop.is_set(): return
        payload=encode_relation_proposal(proposal); shard=proposal.source_uid.shard(len(self._shard_rings))
        self._shard_rings[shard].put(payload,timeout=float(timeout))
    def submit(self,event:ExperienceEvent,*,timeout:float=1.0):
        baseline={}
        try: baseline=self.read_view.outcome_distribution(int(event.context_signature),int(event.action_id))
        except Exception: pass
        base_submit(self,event,timeout=timeout); accepted=event if int(event.watermark)>0 else replace(event,watermark=int(self.watermark)); self.hydra_v9.observe_interaction(accepted,baseline)
    def aux(self):
        payload=json.loads(base_aux(self)); payload["hydra_v9"]=self.hydra_v9.state_dict(); payload["scientific_config_id"]=self.hydra_v9.config.config_id; return json.dumps(payload,sort_keys=True,separators=(",",":"))
    def metrics(self):
        payload=dict(base_metrics(self)); payload["hydra_v9"]=self.hydra_v9.metrics(); return payload
    def report(self):
        base_report(self); target=Path(self.root)/"reports"/"hydra_v9.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps({"metrics":self.hydra_v9.metrics(),"scientific_config":self.hydra_v9.config.as_dict()},indent=2,sort_keys=True),encoding="utf-8")
    V82ContinuousMemoryRuntime.__init__=init; V82ContinuousMemoryRuntime.submit=submit; V82ContinuousMemoryRuntime.submit_relation_proposal=submit_relation_proposal; V82ContinuousMemoryRuntime._auxiliary_state_json=aux; V82ContinuousMemoryRuntime.metrics=metrics; V82ContinuousMemoryRuntime.write_scientific_report=report
    runtime_module.ContinuousMemoryRuntime=V82ContinuousMemoryRuntime

    base_score=LiveReadView.score_actions
    def score_actions(self,context_signature,action_ids):
        rows=base_score(self,context_signature,action_ids); state=getattr(self,"_hydra_v9_state",None); return rows if state is None else state.adjust_action_scores(self,int(context_signature),rows)
    LiveReadView.score_actions=score_actions

    try:
        import v8.mixed_environment_v859 as mixed
        base_make=mixed.make_adapter; base_generic=mixed.run_generic_actor_job
        def make_adapter(*args,**kwargs):
            adapter=base_make(*args,**kwargs); runtime=getattr(_THREAD,"runtime",None); return _HydraAdapterProxy(adapter,runtime) if runtime is not None and hasattr(runtime,"hydra_v9") else adapter
        def run_generic(runtime,job,*,reporting_queue=None):
            prior=getattr(_THREAD,"runtime",None); _THREAD.runtime=runtime
            try: return base_generic(runtime,job,reporting_queue=reporting_queue)
            finally: _THREAD.runtime=prior
        mixed.make_adapter=make_adapter; mixed.run_generic_actor_job=run_generic
    except Exception:
        pass

    try:
        from v8.peers_v82 import V82DevelopmentalPeerSupervisor
        base_peer=V82DevelopmentalPeerSupervisor.run_once
        def peer_run(self):
            base_peer(self); state=getattr(self.read_view,"_hydra_v9_state",None)
            if state is not None: state.peer_cycle(self.read_view)
        V82DevelopmentalPeerSupervisor.run_once=peer_run
    except Exception:
        pass
    _INSTALLED=True
