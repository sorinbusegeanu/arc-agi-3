from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
from typing import Any

from v9.cognition.strategies import strategy_frontier
from v9.cognition.action_selection import scoped_action_key
from v9.cognition.developmental_stage import DevelopmentalStageTracker, StageEvidence
from v9.cognition.grounding import GroundingEvidence, GroundingRegistry
from v9.cognition.isf import ISFComponents, InteractionSignificanceFunction
from v9.cognition.replay import ReplayCandidate, ReplayResult, ReplayScheduler
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

from .actor_policy import ActorPolicySnapshot, ActorStrategyPolicy, ActorOutcomePolicy
from .canonical_store import CanonicalStore, canonical_value
from .canonical_transaction import CanonicalCollection
from .canonical_wal import CanonicalCommitWAL, PersistenceFrontiers
from .config import RuntimeConfig, write_scientific_config_manifest
from .lifecycle import LifecycleRegistry
from .partitions import PartitionMap
from .publication import CanonicalGraph, edge_ref, node_ref
from .read_view import ReadView
from .rings import MultimodalTimeline
from .signature_index import PersistentDirtySignatureWindow, SignatureIndexStore
from .snapshot_backend import SnapshotResult, assert_native_root, latest_snapshot, load_snapshot, load_snapshot_direct, load_graph_shard, decode_graph_shard, write_snapshot


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
        self.evidence = EvidenceLedger(self.root / "evidence" / "ledger.jsonl", scientific.config_id.value, flush_records=256, flush_interval_seconds=0.5)
        self._watermark = 0
        self._snapshot_id = 0
        self._producer_sequences: dict[int, int] = {}
        self._started = False
        self._closed = False
        self._lock = RLock()
        self._m1n_occurrences: dict[int, list[M1NormalizedRelation]] = {}
        self._m1n_supports: dict[int, int] = {}
        self.signature_index = SignatureIndexStore(
            self.root / "indexes" / "signatures.sqlite",
            delta_limit=4096,
            page_cache_limit=1024,
            dirty_window_limit=4096,
        )
        self._m1n_dirty = PersistentDirtySignatureWindow(
            self.signature_index,
            lambda signature: int(self._m1n_supports.get(int(signature), 0)),
        )
        self._cross_modal_signatures: dict[int, None] = {}
        self._actor_action_supports: dict[int, float] = {}
        self._actor_policy_generation = 0
        self._deferred_base_nodes: dict[MemoryUid, tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]] = {}
        self._replay_pool: dict[MemoryUid, float] = {}
        self._formation_environments: set[int] = set()
        self._latest_interaction_grounding: dict[tuple[int, int], M1GroundedContingency] = {}
        self._m2: dict[MemoryUid, M2TransformationFamily] = {}
        self._m3: dict[MemoryUid, M3FunctionalRole] = {}
        self._m4: dict[MemoryUid, M4Concept] = {}
        self._m5: dict[MemoryUid, M5ConsequenceStructure] = {}
        self._m6: dict[MemoryUid, M6Outcome] = {}
        self._m7: dict[MemoryUid, M7Strategy] = {}
        self._transfer_trials: dict[MemoryUid, list[dict[str, Any]]] = {}
        self._modality_events: dict[int, int] = {}
        self._similarity_entropy_by_radius: dict[int, list[float]] = {}
        self._replans_demonstrated = 0
        self._recovered_replans = 0
        self._efficient_replans = 0
        self._symbol_prediction_delta_sum = 0.0
        self._prediction_error_sum = 0.0
        self._prediction_error_count = 0
        self._stage_interval_events = 0
        self._stage_interval_size = 256
        self.unified_telemetry = UnifiedTelemetry(model_version=scientific.hgt_model_version)
        self._hgt_action_scores: dict[int, dict[int, float]] = {}
        self._hgt_context_action_scores: dict[int, dict[int, dict[int, float]]] = {}
        self._environment_ids_by_game: dict[str, set[int]] = {}
        self._memory_uids_by_environment: dict[int, set[MemoryUid]] | None = None

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
                direct = load_snapshot_direct(path, expected_config_id=scientific.config_id.value)
                if direct is None:
                    self._restore(load_snapshot(path, expected_config_id=scientific.config_id.value))
                else:
                    runtime_state, graph_header, shard_specs = direct
                    graph = CanonicalGraph.from_sharded_state(graph_header, ())
                    from concurrent.futures import ThreadPoolExecutor
                    workers = max(1, min(len(shard_specs), 4))
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-restore") as pool:
                        for start in range(0, len(shard_specs), workers):
                            batch = shard_specs[start:start + workers]
                            decoded = list(pool.map(
                                lambda row: decode_graph_shard(
                                    load_graph_shard(path, row[0], row[1]),
                                    expected_partition=row[0],
                                ),
                                batch,
                            ))
                            graph.install_sharded_rows(decoded)
                            del decoded
                    runtime_state["graph"] = graph_header
                    self._restore({"state": runtime_state}, graph_override=graph)
            self._restore_hgt_checkpoint()
        if config.enable_canonical_durability:
            self._initialize_canonical_durability()

    def signature_support(self, signature: int, default: int = 0) -> int:
        selected = int(signature)
        cached = int(self._m1n_supports.get(selected, default))
        persistent = int(self.signature_index.get(selected).support)
        return max(cached, persistent)

    def _initialize_canonical_durability(self) -> None:
        """Compose the WAL and immutable canonical root into graph publication."""
        store = CanonicalStore(
            schema_versions={"canonical_graph": int(self.graph.SCHEMA_VERSION)},
            max_chunk_entries=min(8192, int(self.config.canonical_transaction_max_writes)),
            max_chunk_bytes=min(64 * 1024 * 1024, int(self.config.canonical_transaction_max_mutation_bytes)),
            chunk_directory=self.root / "canonical" / "chunks",
            resident_chunk_limit=0,
        )
        self.canonical_wal = CanonicalCommitWAL(
            self.root / "canonical" / "commit.wal",
            max_group_frames=1,
            max_pending_bytes=int(self.config.canonical_transaction_max_mutation_bytes),
        )
        identity = self._canonical_scientific_identity()
        snapshots = self.root / "canonical" / "snapshots"
        complete = []
        if snapshots.exists():
            candidates = tuple(snapshots.glob("snapshot-*")) + tuple(
                snapshots.glob(".snapshot-*.previous")
            )
            complete = sorted(
                (path for path in candidates if (path / "COMPLETE").is_file()),
                key=lambda path: (
                    path.name.removeprefix(".").removesuffix(".previous"),
                    not path.name.startswith("."),
                ),
            )
        if complete:
            store = CanonicalStore.from_snapshot(
                complete[-1], expected_scientific_identity=identity
            )
            store.enable_disk_backing(self.root / "canonical" / "chunks", resident_chunk_limit=0)
        recovery = self.canonical_wal.recover(truncate=True)
        if store.current_handle.canonical_applied_lsn > recovery.wal_durable_lsn:
            raise RuntimeError("canonical snapshot is ahead of its durable WAL")
        for frame in recovery.frames:
            if frame.wal_lsn <= store.current_handle.canonical_applied_lsn:
                continue
            if frame.scientific_identity != identity:
                raise RuntimeError("canonical WAL contains a different scientific identity")
            overlay = store.begin_overlay(
                frame.wal_lsn,
                max_entries=int(self.config.canonical_transaction_max_writes),
                max_bytes=int(self.config.canonical_transaction_max_mutation_bytes),
            )
            for mutation in frame.mutations:
                collection = str(mutation["collection"])
                key = str(mutation["key"])
                if str(mutation.get("operation", "put")) == "delete":
                    overlay.delete(collection, key)
                else:
                    overlay.put(collection, key, mutation.get("value"))
            store.finalize_overlay(overlay)
        self.canonical_store = store
        self._canonical_snapshot_applied_lsn = (
            store.current_handle.canonical_applied_lsn if complete else 0
        )
        self._hgt_checkpoint_lsn = 0
        self.canonical_wal.register_durable_consumer(
            "canonical_snapshot", self._canonical_snapshot_applied_lsn
        )
        if self.canonical_wal.durable_consumer_checkpoint(
            "canonical_snapshot"
        ) != self._canonical_snapshot_applied_lsn:
            raise RuntimeError("WAL snapshot-consumer checkpoint has no matching canonical snapshot")
        self.canonical_wal.register_durable_consumer(
            "hgt_training_evidence", self._hgt_checkpoint_lsn
        )
        self._hgt_checkpoint_lsn = self.canonical_wal.durable_consumer_checkpoint(
            "hgt_training_evidence"
        )
        self.graph.configure_durable_commit(self._commit_canonical_graph_update)

    def _canonical_scientific_identity(self) -> dict[str, str]:
        return {
            "scientific_config_id": self.config.scientific.config_id.value,
            "design_version": self.config.scientific.design_version,
        }

    def write_canonical_snapshot(self, snapshot_id: int | None = None) -> Path | None:
        if not hasattr(self, "canonical_store"):
            return None
        selected = self._snapshot_id if snapshot_id is None else int(snapshot_id)
        path = self.root / "canonical" / "snapshots" / f"snapshot-{selected:08d}"
        result = self.canonical_store.write_snapshot(
            path, scientific_identity=self._canonical_scientific_identity()
        )
        self._canonical_snapshot_applied_lsn = self.canonical_store.current_handle.canonical_applied_lsn
        self.canonical_wal.update_durable_consumer(
            "canonical_snapshot", self._canonical_snapshot_applied_lsn
        )
        return result

    def advance_hgt_checkpoint(self, checkpoint_lsn: int) -> None:
        """Record an atomically manifested contiguous training-evidence frontier."""
        if not hasattr(self, "canonical_wal"):
            raise RuntimeError("canonical durability migration gate is not enabled")
        selected = int(checkpoint_lsn)
        self.canonical_wal.update_durable_consumer("hgt_training_evidence", selected)
        self._hgt_checkpoint_lsn = selected

    def reclaim_canonical_wal(self) -> int:
        if not hasattr(self, "canonical_wal"):
            return 0
        return self.canonical_wal.reclaim_prefix()

    @staticmethod
    def _canonical_node_value(node: CanonicalNode) -> dict[str, object]:
        return {
            "uid": [int(node.uid.hi), int(node.uid.lo)],
            "level": int(node.level),
            "memory_type": int(node.memory_type),
            "structural_key": [int(value) for value in node.structural_key],
            "created_watermark": int(node.created_watermark),
        }

    @staticmethod
    def _canonical_edge_value(edge: RelationEdge) -> dict[str, object]:
        return {
            "source": [int(edge.source.hi), int(edge.source.lo)],
            "relation": edge.relation.value,
            "target": [int(edge.target.hi), int(edge.target.lo)],
            "evidence": [[int(uid.hi), int(uid.lo)] for uid in edge.evidence_uids],
            "authority": edge.authority.value,
            "object_version": int(edge.object_version),
        }

    def _commit_canonical_graph_update(
        self,
        *,
        proposal: MutationProposal,
        node_updates: dict[MemoryUid, tuple[CanonicalNode, dict[str, Any]]],
        node_deletes: dict[MemoryUid, CanonicalNode],
        edge_updates: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge | None],
        next_generation: int,
    ) -> None:
        """Durably publish one validated graph transaction before live visibility."""
        target_lsn = self.canonical_wal.wal_durable_lsn + 1
        overlay = self.canonical_store.begin_overlay(
            target_lsn,
            max_entries=int(self.config.canonical_transaction_max_writes),
            max_bytes=int(self.config.canonical_transaction_max_mutation_bytes),
        )
        mutations: list[dict[str, object]] = []

        def put(collection: CanonicalCollection, key: str, value: object) -> None:
            encoded = canonical_value(value)
            overlay.put(collection, key, encoded)
            mutations.append({"collection": collection.value, "key": key, "operation": "put", "value": encoded})

        def delete(collection: CanonicalCollection, key: str) -> None:
            overlay.delete(collection, key)
            mutations.append({"collection": collection.value, "key": key, "operation": "delete"})

        for uid, (node, payload) in sorted(node_updates.items()):
            key = f"node:{uid.hi:016x}{uid.lo:016x}"
            put(CanonicalCollection.GRAPH, key, self._canonical_node_value(node))
            put(CanonicalCollection.PAYLOAD, key, dict(payload))
            put(CanonicalCollection.LEVEL_INDEX, f"level:{int(node.level)}:{key}", key)
        for uid, node in sorted(node_deletes.items()):
            key = f"node:{uid.hi:016x}{uid.lo:016x}"
            delete(CanonicalCollection.GRAPH, key)
            delete(CanonicalCollection.PAYLOAD, key)
            delete(CanonicalCollection.LEVEL_INDEX, f"level:{int(node.level)}:{key}")
        for key, edge in sorted(edge_updates.items(), key=lambda row: repr(row[0])):
            source, relation, target = key
            store_key = f"edge:{source.hi:016x}{source.lo:016x}:{relation}:{target.hi:016x}{target.lo:016x}"
            if edge is None:
                delete(CanonicalCollection.GRAPH, store_key)
                if relation == RelationType.PROVENANCE.value:
                    delete(CanonicalCollection.PROVENANCE_INDEX, store_key)
            else:
                value = self._canonical_edge_value(edge)
                put(CanonicalCollection.GRAPH, store_key, value)
                if edge.relation is RelationType.PROVENANCE:
                    put(CanonicalCollection.PROVENANCE_INDEX, store_key, value)
        try:
            self.canonical_wal.commit_overlay(
                self.canonical_store,
                overlay,
                transaction_id=f"proposal-{int(proposal.proposal_uid):016x}",
                frame_payload={
                    "mutations": tuple(mutations),
                    "scientific_identity": self._canonical_scientific_identity(),
                    "work_metadata": {
                        "rows": len(node_updates) + len(edge_updates),
                        "writes": len(mutations),
                        "generation": int(next_generation),
                    },
                },
            )
        except BaseException:
            if not overlay.closed:
                self.canonical_store.abort_overlay(overlay)
            raise

    @property
    def canonical_state_handle(self):
        if not hasattr(self, "canonical_store"):
            raise RuntimeError("canonical durability migration gate is not enabled")
        return self.canonical_store.current_handle

    @property
    def persistence_frontiers(self) -> PersistenceFrontiers:
        if not hasattr(self, "canonical_store"):
            return PersistenceFrontiers()
        return PersistenceFrontiers(
            wal_durable_lsn=self.canonical_wal.wal_durable_lsn,
            canonical_applied_lsn=self.canonical_store.current_handle.canonical_applied_lsn,
            snapshot_applied_lsn=self._canonical_snapshot_applied_lsn,
            hgt_checkpoint_lsn=self._hgt_checkpoint_lsn,
        )

    def create_epoch_inference_view(self, *, sampling_epoch_id: int):
        """Pin the complete actor-visible authority for one matched epoch."""
        from v9.research.experiment_manifest import ExperimentManifest
        from .epoch_inference_view import EpochInferenceView
        from .policy_projection import build_policy_projection
        from .scientific_modes import ScientificVisibilityMode

        if self.config.scientific.scientific_visibility_mode is not ScientificVisibilityMode.MATCHED_REASONING:
            raise RuntimeError("EpochInferenceView is required only for MATCHED_REASONING")
        if not hasattr(self, "canonical_store"):
            raise RuntimeError("MATCHED_REASONING requires canonical durability")
        if self.config.experiment_manifest is None:
            raise RuntimeError("MATCHED_REASONING requires an ExperimentManifest")
        manifest = ExperimentManifest.load(self.config.experiment_manifest)
        if manifest.scientific_config_id != self.config.scientific.config_id.value:
            raise RuntimeError("ExperimentManifest ScientificConfigId changed after runtime construction")
        projection = build_policy_projection(self.actor_policy_snapshot())
        return EpochInferenceView(
            store=self.canonical_store,
            experiment_manifest_id=manifest.manifest_id.value,
            sampling_epoch_id=int(sampling_epoch_id),
            policy_projection=projection,
            model_version=str(projection.snapshot.model_version),
            stage_state=self.stage_tracker.state_dict(),
            normalization_state=self.scale_statistics.state_dict(),
            graph_schema_version=int(manifest.graph_schema_version),
            feature_schema_version=int(manifest.feature_schema_version),
            scientific_config_id=self.config.scientific.config_id.value,
        )

    def _restore_hgt_checkpoint(self) -> None:
        """Restore the active accepted/candidate HGT policy from its manifest."""
        manifest_path = self.root / "models" / "hgt_manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        pending = manifest.get("candidate_status") == "TESTING_PENDING_BEHAVIOR" and manifest.get("candidate_model_version")
        version = manifest.get("candidate_model_version") if pending else (manifest.get("last_accepted_model_version") or manifest.get("current_model_version"))
        checkpoint_rel = (
            manifest.get("current_checkpoint")
            if pending
            else (manifest.get("accepted_checkpoint") or manifest.get("current_checkpoint"))
        )
        if not version or not checkpoint_rel:
            return
        checkpoint_path = self.root / str(checkpoint_rel)
        if not checkpoint_path.exists():
            return
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except (ImportError, OSError, RuntimeError, ValueError):
            return
        action_scores = {
            int(environment): {int(action): float(score) for action, score in actions.items()}
            for environment, actions in dict(checkpoint.get("action_scores", {})).items()
        }
        context_scores = {
            int(environment): {
                int(context): {int(action): float(score) for action, score in actions.items()}
                for context, actions in contexts.items()
            }
            for environment, contexts in dict(checkpoint.get("context_action_scores", {})).items()
        }
        self.set_hgt_action_scores(action_scores, context_action_scores=context_scores)
        self.unified_telemetry.model_version = str(version)
        self.set_telemetry_gauge("hgt_pending_behavior_evaluation", bool(pending))
        self.set_telemetry_gauge("hgt_restored_on_startup", 1)
        self.set_telemetry_gauge("hgt_restored_model", str(version))

    @property
    def watermark(self) -> int:
        return self._watermark

    @property
    def generation(self) -> int:
        return self.graph.generation

    @property
    def read_view(self) -> ReadView:
        return self.graph.read_view()

    def apply_environment_evidence_confidence(self, confidence_by_environment: dict[int, float]) -> int:
        """Apply confidence to memories indexed by environment."""
        with self._lock:
            confidence = {int(k): float(v) for k, v in confidence_by_environment.items()}
            self._environment_evidence_confidence = confidence
            if self._memory_uids_by_environment is None:
                self._memory_uids_by_environment = {}
                for uid, payload in self.graph.payloads.items():
                    environment_id = payload.get("environment_instance_id")
                    if environment_id is not None:
                        self._memory_uids_by_environment.setdefault(int(environment_id), set()).add(uid)
            changed = 0
            for environment_id, value in confidence.items():
                for uid in tuple(self._memory_uids_by_environment.get(environment_id, ())):
                    payload = self.graph.payloads.get(uid)
                    if payload is None:
                        self._memory_uids_by_environment[environment_id].discard(uid)
                        continue
                    if float(payload.get("evidence_confidence", 1.0)) != value:
                        payload["evidence_confidence"] = value
                        changed += 1
            return changed

    def set_hgt_action_scores(
        self,
        scores: dict[int, dict[int, float]],
        *,
        context_action_scores: dict[int, dict[int, dict[int, float]]] | None = None,
    ) -> None:
        with self._lock:
            self._hgt_action_scores = {
                int(environment): {int(action): float(score) for action, score in actions.items()}
                for environment, actions in scores.items()
            }
            self._hgt_context_action_scores = {
                int(environment): {
                    int(context): {int(action): float(score) for action, score in actions.items()}
                    for context, actions in contexts.items()
                }
                for environment, contexts in (context_action_scores or {}).items()
            }
            self._actor_policy_generation += 1

    def capture_hgt_policy_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scores": {int(env): dict(actions) for env, actions in self._hgt_action_scores.items()},
                "context_scores": {
                    int(env): {int(ctx): dict(actions) for ctx, actions in contexts.items()}
                    for env, contexts in self._hgt_context_action_scores.items()
                },
                "model_version": self.unified_telemetry.model_version,
            }

    def set_hgt_enabled(self, enabled: bool) -> None:
        with self._lock:
            if enabled:
                saved = getattr(self, "_hgt_disabled_state", None)
                if saved is not None:
                    self._hgt_action_scores = {int(env): dict(actions) for env, actions in saved["scores"].items()}
                    self._hgt_context_action_scores = {
                        int(env): {int(ctx): dict(actions) for ctx, actions in contexts.items()}
                        for env, contexts in saved.get("context_scores", {}).items()
                    }
                    self.unified_telemetry.model_version = saved["model_version"]
                    self._hgt_disabled_state = None
            else:
                if getattr(self, "_hgt_disabled_state", None) is None:
                    self._hgt_disabled_state = {
                        "scores": {int(env): dict(actions) for env, actions in self._hgt_action_scores.items()},
                        "context_scores": {
                            int(env): {int(ctx): dict(actions) for ctx, actions in contexts.items()}
                            for env, contexts in getattr(self, "_hgt_context_action_scores", {}).items()
                        },
                        "model_version": self.unified_telemetry.model_version,
                    }
                self._hgt_action_scores = {}
                self._hgt_context_action_scores = {}
            self._actor_policy_generation += 1

    def record_successful_trajectory(self, *, environment_id: int, episode_id: int, target_outcome_uid: MemoryUid | None = None) -> int:
        with self._lock:
            if target_outcome_uid is None:
                return 0
            outcome = self._m6.get(target_outcome_uid)
            if outcome is None or target_outcome_uid not in self.graph.nodes:
                return 0
            rows: list[tuple[int, dict[str, Any]]] = []
            for uid in tuple(self.graph._uids_by_level[MemoryLevel.M0]):
                node = self.graph.nodes.get(uid)
                payload = self.graph.payloads.get(uid)
                if node is None or payload is None:
                    continue
                if int(payload.get("environment_instance_id", -1)) != int(environment_id) or int(payload.get("episode_id", -1)) != int(episode_id):
                    continue
                if payload.get("action_id") is not None:
                    rows.append((int(node.created_watermark), payload))
            ordered = sorted(rows, key=lambda item: item[0])
            native_actions = tuple(int(payload["action_id"]) for _, payload in ordered)
            if not native_actions:
                return 0
            primary_valence = sum(int(payload.get("primary_valence", 0)) for _, payload in ordered)
            realized_cost = sum(max(1, int(payload.get("realized_cost", 0))) for _, payload in ordered) or len(native_actions)
            candidate = M7Strategy.form(outcome, target_environment_id=int(environment_id), native_actions=native_actions, successes=1, trials=1, primary_valence_sum=primary_valence, realized_cost_sum=realized_cost)
            if candidate.uid in self._m7:
                return 0
            self._m7[candidate.uid] = candidate
            confidence = float(self.graph.payloads.get(target_outcome_uid, {}).get("evidence_confidence", 1.0))
            self._publish(CanonicalNode(candidate.uid, MemoryLevel.M7, MemoryType.STRATEGY, (outcome.uid.hi, outcome.uid.lo, int(environment_id), *native_actions), self._watermark), {"target_outcome": [outcome.uid.hi, outcome.uid.lo], "target_environment_id": int(environment_id), "native_actions": list(native_actions), "reliability_successes": 1, "reliability_trials": 1, "evidence_confidence": confidence, "primary_valence_sum": primary_valence, "realized_cost_sum": realized_cost, "parents": [[outcome.uid.hi, outcome.uid.lo]]}, candidate.provenance.evidence)
            return 1

    def record_outcome_equivalence_evidence(self, outcome_uid: MemoryUid, *, equivalent: bool, context_scope_id: int, environment_id: int) -> None:
        with self._lock:
            outcome = self._m6.get(outcome_uid)
            node = self.graph.nodes.get(outcome_uid)
            payload = self.graph.payloads.get(outcome_uid)
            if outcome is None or node is None or payload is None:
                return
            distinct_members = {uid for uid in outcome.members if uid in self._m5}
            if len(distinct_members) < 2:
                return
            outcome = replace(
                outcome,
                equivalence_trials=outcome.equivalence_trials + 1,
                equivalence_successes=outcome.equivalence_successes + int(bool(equivalent)),
                contexts_observed=tuple(sorted(set((*outcome.contexts_observed, int(context_scope_id))))),
                environments_observed=tuple(sorted(set((*outcome.environments_observed, int(environment_id))))),
            )
            self._m6[outcome_uid] = outcome
            self._publish(node, {**payload, "equivalence_trials": outcome.equivalence_trials, "equivalence_successes": outcome.equivalence_successes, "contexts_observed": list(outcome.contexts_observed), "environments_observed": list(outcome.environments_observed), "primary_valence_sum": outcome.primary_valence_sum, "preference_trials": outcome.preference_trials}, outcome.provenance.evidence, proposal_class=ProposalClass.STATEFUL, mutation_kind=MutationKind.UPDATE_VALIDATION)
            self.evidence.append("M6_OUTCOME_EQUIVALENCE", self._watermark, {"outcome_uid": outcome_uid.hex(), "equivalent": bool(equivalent), "context_scope_id": int(context_scope_id), "environment_id": int(environment_id)})

    def record_strategy_execution(self, strategy_uid: MemoryUid, *, success: bool, realized_cost: int, primary_valence: int = 0) -> None:
        with self._lock:
            strategy = self.__dict__.setdefault("_m7", {}).get(strategy_uid)
            node = self.graph.nodes.get(strategy_uid)
            payload = self.graph.payloads.get(strategy_uid)
            if strategy is None or node is None or payload is None:
                return
            updated = strategy.observe(success=bool(success), realized_cost=max(1, int(realized_cost)), primary_valence=int(primary_valence))
            self._m7[strategy_uid] = updated
            outcome = self._m6.get(strategy.target_outcome)
            if outcome is not None:
                outcome = replace(
                    outcome,
                    primary_valence_sum=outcome.primary_valence_sum + int(primary_valence),
                    preference_trials=outcome.preference_trials + 1,
                )
                self._m6[outcome.uid] = outcome
                outcome_node = self.graph.nodes.get(outcome.uid)
                outcome_payload = self.graph.payloads.get(outcome.uid)
                if outcome_node is not None and outcome_payload is not None:
                    self._publish(
                        outcome_node,
                        {
                            **outcome_payload,
                            "equivalence_trials": outcome.equivalence_trials,
                            "equivalence_successes": outcome.equivalence_successes,
                            "contexts_observed": list(outcome.contexts_observed),
                            "environments_observed": list(outcome.environments_observed),
                            "primary_valence_sum": outcome.primary_valence_sum,
                            "preference_trials": outcome.preference_trials,
                        },
                        outcome.provenance.evidence,
                        proposal_class=ProposalClass.STATEFUL,
                        mutation_kind=MutationKind.UPDATE_VALIDATION,
                    )
            self._publish(
                node,
                {
                    **payload,
                    "reliability_successes": updated.reliability_successes,
                    "reliability_trials": updated.reliability_trials,
                    "primary_valence_sum": updated.primary_valence_sum,
                    "realized_cost_sum": updated.realized_cost_sum,
                },
                updated.provenance.evidence,
                proposal_class=ProposalClass.STATEFUL,
                mutation_kind=MutationKind.UPDATE_VALIDATION,
            )
            self.evidence.append("M7_STRATEGY_EXECUTION", self._watermark, {"strategy_uid": strategy_uid.hex(), "success": bool(success), "realized_cost": int(realized_cost), "primary_valence": int(primary_valence)})

    def hgt_action_scores(self, environment_id: int, actions: tuple[int, ...]) -> dict[int, float]:
        with self._lock:
            source = self._hgt_action_scores.get(int(environment_id), {})
            return {int(action): float(source.get(int(action), 0.0)) for action in actions}

    def _hgt_scores_by_environment_type(self) -> dict[str, dict[int, float]]:
        grouped: dict[str, dict[int, list[float]]] = {}
        for environment_id, actions in self._hgt_action_scores.items():
            try:
                environment_type = self.environments.resolve(int(environment_id)).environment_type
            except KeyError:
                continue
            target = grouped.setdefault(str(environment_type), {})
            for action, score in actions.items():
                target.setdefault(int(action), []).append(float(score))
        return {
            environment_type: {
                action: sum(scores) / len(scores)
                for action, scores in actions.items()
                if scores
            }
            for environment_type, actions in grouped.items()
        }

    def _hgt_context_scores_by_environment_type(self) -> dict[str, dict[int, dict[int, float]]]:
        grouped: dict[str, dict[int, dict[int, list[float]]]] = {}
        for environment_id, contexts in getattr(self, "_hgt_context_action_scores", {}).items():
            try:
                identity = self.environments.resolve(int(environment_id))
            except KeyError:
                continue
            environment_type = str(identity.environment_type)
            for context_signature, scores in contexts.items():
                for action, score in scores.items():
                    grouped.setdefault(environment_type, {}).setdefault(int(context_signature), {}).setdefault(int(action), []).append(float(score))
        return {
            environment_type: {
                context: {action: sum(values) / len(values) for action, values in actions.items()}
                for context, actions in contexts.items()
            }
            for environment_type, contexts in grouped.items()
        }

    def _grounded_policy_scores(self) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, dict[int, float]]]]:
        """Project causally validated G3+ grounding into actor-visible action scores."""
        by_type: dict[str, dict[int, list[float]]] = {}
        by_context: dict[str, dict[int, dict[int, list[float]]]] = {}
        payload_by_low_uid: dict[int, dict[str, Any]] = {
            int(uid.lo): payload for uid, payload in self.graph.payloads.items()
        }
        for key, state in self.grounding.eligible_states():
            _symbol_uid, interaction_uid, environment_id, context_scope_id, _lineage_uid = key
            payload = payload_by_low_uid.get(int(interaction_uid))
            if payload is None or payload.get("action_id") is None:
                continue
            try:
                environment_type = str(self.environments.resolve(int(environment_id)).environment_type)
            except KeyError:
                continue
            action = int(payload["action_id"])
            confidence = state.support / max(1e-9, state.support + state.contradiction)
            maturity = max(0.0, min(1.0, (int(state.maturity) - 2) / 3.0))
            score = confidence * maturity
            by_type.setdefault(environment_type, {}).setdefault(action, []).append(score)
            context = payload.get("context_signature")
            if context is not None:
                by_context.setdefault(environment_type, {}).setdefault(int(context), {}).setdefault(action, []).append(score)
        return (
            {
                environment_type: {action: sum(values) / len(values) for action, values in actions.items()}
                for environment_type, actions in by_type.items()
            },
            {
                environment_type: {
                    context: {action: sum(values) / len(values) for action, values in actions.items()}
                    for context, actions in contexts.items()
                }
                for environment_type, contexts in by_context.items()
            },
        )

    def actor_policy_snapshot(self) -> ActorPolicySnapshot:
        with self._lock:
            grounded_scores, grounded_context_scores = self._grounded_policy_scores()
            live_strategies = tuple(
                strategy for uid, strategy in getattr(self, "_m7", {}).items()
                if (
                    (uid in self.graph.nodes and uid in self.graph.payloads)
                    or getattr(strategy, "reliability_trials", 0) > 0
                )
            )
            efficiencies = strategy_frontier(live_strategies)
            strategies_by_environment: dict[int, list[ActorStrategyPolicy]] = {}
            for strategy in live_strategies:
                strategies_by_environment.setdefault(int(strategy.target_environment_id), []).append(
                    ActorStrategyPolicy(
                        strategy.uid,
                        strategy.target_outcome,
                        int(strategy.target_environment_id),
                        tuple(strategy.native_actions),
                        float(strategy.reliability),
                        None if strategy.expected_cost is None else float(strategy.expected_cost),
                        efficiencies.get(strategy.uid.lo),
                        float(strategy.primary_valence_sum) / max(1, strategy.reliability_trials),
                    )
                )
            published_strategies = {
                environment: tuple(sorted(rows, key=lambda row: (-row.reliability, -row.primary_valence, float("inf") if row.expected_cost is None else row.expected_cost, row.strategy_uid))[:64])
                for environment, rows in strategies_by_environment.items()
            }
            outcomes_by_environment: dict[int, list[ActorOutcomePolicy]] = {}
            for environment, rows in published_strategies.items():
                seen: set[MemoryUid] = set()
                for row in rows:
                    outcome = self._m6.get(row.target_outcome_uid)
                    if outcome is None or outcome.uid in seen:
                        continue
                    seen.add(outcome.uid)
                    outcomes_by_environment.setdefault(environment, []).append(
                        ActorOutcomePolicy(outcome.uid, environment, outcome.equivalence_confidence, outcome.mean_primary_valence)
                    )
            published_outcomes = {environment: tuple(rows) for environment, rows in outcomes_by_environment.items()}
            return ActorPolicySnapshot.build(
                generation=max(self.graph.generation, self._actor_policy_generation),
                normalized_action_supports=self._actor_action_supports,
                hgt_action_scores=self._hgt_action_scores,
                hgt_context_action_scores=getattr(self, "_hgt_context_action_scores", {}),
                hgt_action_scores_by_type=self._hgt_scores_by_environment_type(),
                hgt_context_action_scores_by_type=self._hgt_context_scores_by_environment_type(),
                grounded_action_scores_by_type=grounded_scores,
                grounded_context_action_scores_by_type=grounded_context_scores,
                model_version=self.unified_telemetry.model_version,
                strategies_by_environment=published_strategies,
                outcomes_by_environment=published_outcomes,
            )

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
                recurrence = self.signature_support(stable_u64(f"ACTION:{experience.action_id}:FAMILY:{experience.family_signature}:OUTCOME:{experience.outcome_signature}", NormalizedChannel.WORLD.value, person=b"v9-m1-normalized"))
                recurrence_surprise = 1.0 / max(1.0, float(recurrence))
                prediction_error = abs(float(experience.prediction_error)) if float(experience.prediction_error) != 0.0 else recurrence_surprise
                decision = self.isf.score(
                    ISFComponents(abs(experience.primary_valence), abs(experience.future_option_delta), prediction_error, 1.0 / max(1, recurrence), 0.5 if experience.family_signature else 0.0, min(1.0, experience.changed_cells / 16.0)),
                    decision_watermark=self._watermark,
                    evidence_availability_watermark=event.identity.causal_watermark,
                    stage=stage_before,
                    next_stage=stage_snapshot.next_stage,
                    graph_generation=self.graph.generation,
                )
                self._prediction_error_sum += float(prediction_error)
                self._prediction_error_count += 1
                self.evidence.append("ISF_DECISION", self._watermark, {"stage": int(decision.developmental_stage), "next_stage": int(decision.next_developmental_stage), "score": decision.score, "raw": asdict(decision.raw_components), "normalized": asdict(decision.normalized_components), "graph_generation": decision.graph_generation})

    def _stage_evidence(self) -> StageEvidence:
        strategies = [self.graph.payloads[uid] for uid in self.graph.uids_at_level(MemoryLevel.M7)]
        return StageEvidence(
            stable_contingencies=sum(len(rows) >= 2 for rows in self._m1n_occurrences.values()),
            structural_abstractions=len(self._m3),
            held_out_transfer_successes=sum(bool(row.validated) for row in self._m4.values()),
            mature_consequences=sum(bool(self.graph.payloads[uid].get("mature")) for uid in self.graph.uids_at_level(MemoryLevel.M5)),
            outcome_equivalences=sum(row.equivalence_trials >= 2 and row.equivalence_confidence > 0.5 for row in self._m6.values()),
            learned_preferences=sum(row.preference_trials >= 2 and row.mean_primary_valence != 0.0 for row in self._m6.values()),
            alternative_strategies=sum(
                len({strategy.native_actions for strategy in self._m7.values() if strategy.target_outcome == outcome.uid}) >= 2
                for outcome in self._m6.values()
            ),
            demonstrated_replans=self._replans_demonstrated,
            efficient_replans=self._efficient_replans,
        )

    def _payload_digest(self, event: TimelineEvent) -> int:
        if isinstance(event, InteractionEvent):
            return stable_u64(event.experience.context_signature, event.experience.action_id, event.experience.outcome_signature, person=b"v9-interaction-payload")
        if isinstance(event, PassiveSymbolEvent):
            return stable_u64(event.vocabulary_id.value, event.stream_id.value, event.symbol_id.value, event.position, person=b"v9-symbol-payload")
        return stable_u64(event.observation_schema_id, event.observation_signature, person=b"v9-world-payload")

    def _publish(self, node: CanonicalNode, payload: dict[str, Any], evidence: tuple[MemoryUid, ...], *, proposal_class: ProposalClass = ProposalClass.ADDITIVE, mutation_kind: MutationKind = MutationKind.UPSERT_NODE) -> bool:
        payload = dict(payload)
        payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])

        writes: list[MutationWrite] = [MutationWrite(node=node, payload=payload)]
        target_partitions = {self.partitions.owner(node.uid)}
        dependencies = [ReadDependency(node_ref(node.uid), self.graph.versions.get(node_ref(node.uid)))]

        for raw_parent in payload.get("parents", []):
            if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                continue
            parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
            edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
            writes.append(MutationWrite(edge=edge))
            target_partitions.add(self.partitions.owner(parent))
            edge_reference = edge_ref(edge)
            dependencies.append(ReadDependency(edge_reference, self.graph.versions.get(edge_reference)))

        read_set = ReadSet.build(
            tuple(dependencies),
            maximum_size=self.config.scientific.maximum_read_set_size,
        )
        proposal = MutationProposal.build(
            mutation_kind,
            target_partitions=tuple(sorted(target_partitions)),
            read_set=read_set,
            evidence_refs=evidence,
            causal_watermark=self._watermark,
            writes=tuple(writes),
            proposal_class=proposal_class,
        )

        self.telemetry["proposals"] += 1
        self.telemetry["cross_partition_transactions"] += int(len(target_partitions) > 1)
        previous_generation = self.graph.generation
        result = self.graph.publish(proposal)

        if result.outcome.value == "ACCEPTED":
            self.telemetry["accepted"] += 1
            if result.graph_generation == previous_generation:
                self.telemetry["canonical_reuse"] += 1
            else:
                if node.level >= MemoryLevel.M2:
                    self.structural_index.add(node)
            if self.config.enable_lifecycle and result.graph_generation != previous_generation:
                self.lifecycle.observe(
                    node.uid,
                    support_delta=1,
                    relevant_opportunity=True,
                    watermark=self._watermark,
                )
        elif result.outcome.value == "STALE_READ_SET":
            self.telemetry["stale"] += 1
            self.telemetry["read_set_conflicts"] += 1
        else:
            self.telemetry["rejected"] += 1
        return result.outcome.value == "ACCEPTED"

    def _publish_group(
        self,
        rows: tuple[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]], ...],
    ) -> bool:
        writes: list[MutationWrite] = []
        target_partitions: set[int] = set()
        dependencies: list[ReadDependency] = []
        evidence_refs: set[MemoryUid] = set()

        for node, raw_payload, evidence in rows:
            payload = dict(raw_payload)
            payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])
            writes.append(MutationWrite(node=node, payload=payload))
            target_partitions.add(self.partitions.owner(node.uid))
            dependencies.append(ReadDependency(node_ref(node.uid), self.graph.versions.get(node_ref(node.uid))))
            evidence_refs.update(evidence)
            for raw_parent in payload.get("parents", []):
                if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                    continue
                parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
                edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
                writes.append(MutationWrite(edge=edge))
                target_partitions.add(self.partitions.owner(parent))

        proposal = MutationProposal.build(
            MutationKind.UPSERT_NODE,
            target_partitions=tuple(sorted(target_partitions)),
            read_set=ReadSet.build(
                tuple(dependencies),
                maximum_size=self.config.scientific.maximum_read_set_size,
            ),
            evidence_refs=tuple(sorted(evidence_refs)),
            causal_watermark=self._watermark,
            writes=tuple(writes),
            proposal_class=ProposalClass.ADDITIVE,
        )
        self.telemetry["proposals"] += 1
        self.telemetry["cross_partition_transactions"] += int(len(target_partitions) > 1)
        previous_generation = self.graph.generation
        result = self.graph.publish(proposal)
        if result.outcome.value == "ACCEPTED":
            self.telemetry["accepted"] += 1
            if result.graph_generation == previous_generation:
                self.telemetry["canonical_reuse"] += 1
            else:
                for node, _, _ in rows:
                    if node.level >= MemoryLevel.M2:
                        self.structural_index.add(node)
                    if self.config.enable_lifecycle:
                        self.lifecycle.observe(
                            node.uid,
                            support_delta=1,
                            relevant_opportunity=True,
                            watermark=self._watermark,
                        )
        elif result.outcome.value == "STALE_READ_SET":
            self.telemetry["stale"] += 1
            self.telemetry["read_set_conflicts"] += 1
        else:
            self.telemetry["rejected"] += 1
        return result.outcome.value == "ACCEPTED"

    def _publish_stateful_group(
        self,
        rows: tuple[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]], ...],
    ) -> bool:
        """Publish a bounded set of authoritative replacements in one transaction."""
        writes: list[MutationWrite] = []
        dependencies: list[ReadDependency] = []
        target_partitions: set[int] = set()
        evidence_refs: set[MemoryUid] = set()
        for node, raw_payload, evidence in rows:
            payload = dict(raw_payload)
            payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])
            writes.append(MutationWrite(node=node, payload=payload))
            target_partitions.add(self.partitions.owner(node.uid))
            reference = node_ref(node.uid)
            dependencies.append(ReadDependency(reference, self.graph.versions.get(reference)))
            evidence_refs.update(evidence)
            for raw_parent in payload.get("parents", []):
                if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                    continue
                parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
                edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
                writes.append(MutationWrite(edge=edge))
                target_partitions.add(self.partitions.owner(parent))
        proposal = MutationProposal.build(
            MutationKind.UPSERT_NODE,
            target_partitions=tuple(sorted(target_partitions)),
            read_set=ReadSet.build(
                tuple(dependencies),
                maximum_size=self.config.scientific.maximum_read_set_size,
            ),
            evidence_refs=tuple(sorted(evidence_refs)),
            causal_watermark=self._watermark,
            writes=tuple(writes),
            proposal_class=ProposalClass.STATEFUL,
        )
        self.telemetry["proposals"] += 1
        self.telemetry["cross_partition_transactions"] += int(len(target_partitions) > 1)
        result = self.graph.publish(proposal)
        if result.outcome.value == "ACCEPTED":
            self.telemetry["accepted"] += 1
            return True
        if result.outcome.value == "STALE_READ_SET":
            self.telemetry["stale"] += 1
            self.telemetry["read_set_conflicts"] += 1
        else:
            self.telemetry["rejected"] += 1
        return False

    def replace_canonical_payloads(
        self, updates: dict[MemoryUid, dict[str, Any]]
    ) -> int:
        """Durably replace actor-visible payload fields in bounded transactions."""
        rows: list[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]] = []
        for uid, patch in sorted(updates.items()):
            node = self.graph.nodes.get(uid)
            current = self.graph.payloads.get(uid)
            if node is None or current is None:
                continue
            merged = dict(current)
            merged.update(patch)
            if merged == current:
                continue
            evidence = tuple(
                MemoryUid(int(raw[0]), int(raw[1]))
                for raw in merged.get("evidence_refs", ())
                if isinstance(raw, (list, tuple)) and len(raw) == 2
            )
            rows.append((node, merged, evidence))
        maximum = int(self.config.scientific.maximum_read_set_size)
        applied = 0
        for offset in range(0, len(rows), maximum):
            chunk = tuple(rows[offset : offset + maximum])
            if self._publish_stateful_group(chunk):
                applied += len(chunk)
        return applied

    def _defer_base_group(
        self,
        rows: tuple[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]], ...],
    ) -> None:
        for node, payload, evidence in rows:
            current = self._deferred_base_nodes.get(node.uid)
            if current is None:
                stored = dict(payload)
            else:
                stored = dict(current[1])
                stored.update(payload)
                if "parents" in payload:
                    stored["parents"] = sorted(
                        {tuple(map(int, row)) for row in current[1].get("parents", [])}
                        | {tuple(map(int, row)) for row in payload.get("parents", [])}
                    )
            stored.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])
            self._deferred_base_nodes[node.uid] = (node, stored, tuple(evidence))

    def _record_normalized(self, relation: M1NormalizedRelation, *, defer_publication: bool = False, payload_extra: dict[str, Any] | None = None) -> int:
        occurrences = self._m1n_occurrences.setdefault(relation.structural_signature, [])
        if relation.channel is NormalizedChannel.CROSS_MODAL:
            signature_key = int(relation.structural_signature)
            self._cross_modal_signatures.pop(signature_key, None)
            self._cross_modal_signatures[signature_key] = None
            while len(self._cross_modal_signatures) > 8192:
                self._cross_modal_signatures.pop(next(iter(self._cross_modal_signatures)))
        support = self.signature_support(relation.structural_signature) + 1
        self._m1n_supports[relation.structural_signature] = support
        observable = str(relation.observable_relation)
        parts = observable.split(":")
        if len(parts) >= 5 and parts[0] == "ACTION":
            try:
                scoped = scoped_action_key(
                    int(parts[3]),
                    action_schema_id=int(parts[1]),
                    environment_type=parts[2],
                )
            except ValueError:
                scoped = None
            if scoped is not None:
                self._actor_action_supports[scoped] = self._actor_action_supports.get(scoped, 0.0) + 1.0
                self._actor_policy_generation += 1
        if len(occurrences) < max(2, self.config.scientific.m1n_facts_per_channel):
            occurrences.append(relation)
        self._replay_pool[relation.uid] = float(support)
        if len(self._replay_pool) > self.config.scientific.replay_candidates:
            victim = min(self._replay_pool, key=lambda uid: (self._replay_pool[uid], uid))
            del self._replay_pool[victim]
        retained_parents = tuple(
            uid
            for occurrence in occurrences
            for uid in occurrence.provenance.parents
        )
        retained_evidence = tuple(
            uid
            for occurrence in occurrences
            for uid in occurrence.provenance.evidence
        )
        if support == 1:
            normalized_node = CanonicalNode(
                relation.uid,
                MemoryLevel.M1,
                MemoryType.NORMALIZED_RELATION,
                (relation.structural_signature,),
                self._watermark,
            )
            normalized_payload = {
                "observable_relation": relation.observable_relation,
                "channel": relation.channel.value,
                "structural_signature": relation.structural_signature,
                "support": support,
                "parents": [[uid.hi, uid.lo] for uid in retained_parents],
            }
            if payload_extra:
                normalized_payload.update(dict(payload_extra))
            if defer_publication:
                self._defer_base_group(((normalized_node, normalized_payload, retained_evidence),))
            else:
                self._publish(
                    normalized_node,
                    normalized_payload,
                    retained_evidence,
                    proposal_class=ProposalClass.STATEFUL,
                )
        else:
            self._m1n_dirty.add(int(relation.structural_signature))
        return relation.structural_signature

    def _parallel_worker_count(self, item_count: int) -> int:
        return max(1, min(int(item_count), 16, os.cpu_count() or 1))

    @staticmethod
    def _prepare_dirty_m1n_row(snapshot):
        signature, rows, support, watermark = snapshot
        if not rows:
            return None
        relation = rows[0]
        parents = tuple(uid for occurrence in rows for uid in occurrence.provenance.parents)
        evidence = tuple(uid for occurrence in rows for uid in occurrence.provenance.evidence)
        return signature, support, (
            CanonicalNode(
                relation.uid,
                MemoryLevel.M1,
                MemoryType.NORMALIZED_RELATION,
                (relation.structural_signature,),
                watermark,
            ),
            {
                "observable_relation": relation.observable_relation,
                "channel": relation.channel.value,
                "structural_signature": relation.structural_signature,
                "support": support,
                "parents": [[uid.hi, uid.lo] for uid in parents],
            },
            evidence,
        )

    def _prepare_dirty_rows_parallel(self, snapshots):
        rows = tuple(snapshots)
        workers = self._parallel_worker_count(len(rows))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-consolidate") as pool:
                prepared = tuple(pool.map(self._prepare_dirty_m1n_row, rows))
        else:
            prepared = tuple(self._prepare_dirty_m1n_row(row) for row in rows)
        return tuple(row for row in prepared if row is not None)

    def _capture_consolidation_cut_locked(self):
        pending = tuple(sorted(self._deferred_base_nodes.items(), key=lambda item: item[0]))
        dirty_snapshots = []
        watermark = int(self._watermark)
        for signature in tuple(self._m1n_dirty):
            selected = int(signature)
            rows = tuple(self._m1n_occurrences.get(selected, ()))
            if not rows:
                continue
            dirty_snapshots.append(
                (
                    selected,
                    rows,
                    self.signature_support(selected, len(rows)),
                    watermark,
                )
            )
        return int(self.graph.generation), watermark, pending, tuple(dirty_snapshots)

    def _developmental_cut_is_current_locked(
        self, generation, watermark, signature_supports, pending=()
    ) -> bool:
        if int(self.graph.generation) != int(generation) or int(self._watermark) != int(watermark):
            return False
        if any(self._deferred_base_nodes.get(uid) != snapshotted for uid, snapshotted in pending):
            return False
        return all(
            self.signature_support(int(signature)) == int(support)
            for signature, support in signature_supports
        )

    def _commit_consolidation_cut_locked(self, pending, dirty_rows) -> None:
        batch_size = 16384
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset : offset + batch_size]
            writes: list[MutationWrite] = []
            target_partitions: set[int] = set()
            evidence_refs: set[MemoryUid] = set()
            for _uid, (node, raw_payload, evidence) in chunk:
                payload = dict(raw_payload)
                writes.append(MutationWrite(node=node, payload=payload))
                target_partitions.add(self.partitions.owner(node.uid))
                evidence_refs.update(evidence)
                for raw_parent in payload.get("parents", []):
                    if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                        continue
                    parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
                    writes.append(
                        MutationWrite(
                            edge=RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
                        )
                    )
                    target_partitions.add(self.partitions.owner(parent))
            if not writes:
                continue
            proposal = MutationProposal.build(
                MutationKind.UPSERT_NODE,
                target_partitions=tuple(sorted(target_partitions)),
                read_set=ReadSet.build(
                    (),
                    maximum_size=self.config.scientific.maximum_read_set_size,
                ),
                evidence_refs=tuple(sorted(evidence_refs)),
                causal_watermark=self._watermark,
                writes=tuple(writes),
                proposal_class=ProposalClass.ADDITIVE,
            )
            self.telemetry["proposals"] += 1
            self.telemetry["cross_partition_transactions"] += int(len(target_partitions) > 1)
            result = self.graph.publish(proposal)
            if result.outcome.value == "ACCEPTED":
                self.telemetry["accepted"] += 1
                for uid, snapshotted in chunk:
                    if self._deferred_base_nodes.get(uid) == snapshotted:
                        self._deferred_base_nodes.pop(uid, None)
                if self.config.enable_lifecycle:
                    for _uid, (node, _payload, _evidence) in chunk:
                        self.lifecycle.observe(
                            node.uid,
                            support_delta=1,
                            relevant_opportunity=True,
                            watermark=self._watermark,
                        )
            elif result.outcome.value == "STALE_READ_SET":
                self.telemetry["stale"] += 1
                self.telemetry["read_set_conflicts"] += 1
            else:
                self.telemetry["rejected"] += 1

        maximum_dependencies = int(self.config.scientific.maximum_read_set_size)
        if not hasattr(self, "canonical_store"):
            for signature, support, (node, payload, evidence) in dirty_rows:
                if self._publish(
                    node,
                    payload,
                    evidence,
                    proposal_class=ProposalClass.STATEFUL,
                ):
                    self.signature_index.mark_derived(int(signature), int(support))
            return

        pending_rows = []
        pending_dependencies = 0
        for signature, support, row in dirty_rows:
            cost = 1
            if cost > maximum_dependencies:
                raise ValueError("one dirty M1 row exceeds the configured read-set bound")
            if pending_rows and pending_dependencies + cost > maximum_dependencies:
                if self._publish_stateful_group(tuple(row for _, _, row in pending_rows)):
                    for pending_signature, pending_support, _ in pending_rows:
                        self.signature_index.mark_derived(
                            int(pending_signature), int(pending_support)
                        )
                pending_rows = []
                pending_dependencies = 0
            pending_rows.append((signature, support, row))
            pending_dependencies += cost
        if pending_rows and self._publish_stateful_group(
            tuple(row for _, _, row in pending_rows)
        ):
            for pending_signature, pending_support, _ in pending_rows:
                self.signature_index.mark_derived(
                    int(pending_signature), int(pending_support)
                )

    def flush_deferred_memory_updates(self) -> None:
        # Freeze a complete developmental cut, do pure CPU work outside the runtime
        # lock, and publish only if the authoritative cut is still current.
        for _attempt in range(3):
            with self._lock:
                generation, watermark, pending, snapshots = self._capture_consolidation_cut_locked()
            prepared = self._prepare_dirty_rows_parallel(snapshots)
            signature_supports = tuple(
                (int(signature), int(support))
                for signature, _rows, support, _watermark in snapshots
            )
            with self._lock:
                if not self._developmental_cut_is_current_locked(
                    generation, watermark, signature_supports, pending
                ):
                    continue
                self._commit_consolidation_cut_locked(pending, prepared)
                return

        # Progress fallback for an unexpectedly busy runtime: freeze by retaining
        # the runtime lock while workers derive from the same immutable cut.
        with self._lock:
            _generation, _watermark, pending, snapshots = self._capture_consolidation_cut_locked()
            prepared = self._prepare_dirty_rows_parallel(snapshots)
            self._commit_consolidation_cut_locked(pending, prepared)

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

            m0_node = CanonicalNode(
                m0.uid,
                MemoryLevel.M0,
                MemoryType.EPISODE,
                (event.identity.event_id.hi, event.identity.event_id.lo),
                self._watermark,
            )
            transition = prepared.transition
            m0_payload = {
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
                "task_success": bool(transition.task_success),
                "task_failure": bool(transition.task_failure),
                "task_truncated": bool(transition.task_truncated),
                "level_index": int(transition.level_index),
                "levels_completed": int(transition.levels_completed),
                **({"semantic_before": [list(row) for row in transition.semantic_before]} if transition.semantic_before else {}),
                **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition.semantic_action else {}),
                **({"semantic_options": [list(row) for row in transition.semantic_options]} if transition.semantic_options else {}),
                **({"semantic_after": [list(row) for row in transition.semantic_after]} if transition.semantic_after else {}),
                **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition.semantic_delta else {}),
            }
            m1g_node = CanonicalNode(
                m1g.uid,
                MemoryLevel.M1,
                MemoryType.GROUNDED_CONTINGENCY,
                (m1g.uid.hi, m1g.uid.lo),
                self._watermark,
            )
            m1g_payload = {
                "relation": m1g.relation.value,
                "environment_instance_id": m1g.environment_instance_id,
                "episode_id": m1g.episode_id,
                "grounded_context_signature": m1g.grounded_context_signature,
                "executable_action_token": m1g.executable_action_token,
                "realized_transition_signature": m1g.realized_transition_signature,
                "grounded_next_context_signature": m1g.grounded_next_context_signature,
                **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition.semantic_action else {}),
                **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition.semantic_delta else {}),
                "parents": [[m0.uid.hi, m0.uid.lo]],
            }
            self._defer_base_group(
                (
                    (m0_node, m0_payload, (m0.uid,)),
                    (m1g_node, m1g_payload, (m0.uid,)),
                )
            )
            key = (m1g.environment_instance_id, m1g.episode_id)
            self._latest_interaction_grounding[key] = m1g
            if prepared.symbol_codec_state:
                codec = DeterministicSymbolCodec.from_state_dict(dict(prepared.symbol_codec_state))
                self.symbol_codecs[int(codec.vocabulary_id.value)] = codec
            for symbol_row in prepared.symbols:
                symbol_m0 = symbol_row.m0
                symbol_m1g = symbol_row.m1g
                symbol_event = symbol_row.event
                self._modality_events[int(symbol_event.identity.modality_id.value)] = self._modality_events.get(int(symbol_event.identity.modality_id.value), 0) + 1
                symbol_payload = {
                    "modality_id": symbol_m0.modality_id,
                    "environment_instance_id": symbol_m0.provenance.environment_instance_id,
                    "episode_id": symbol_m0.provenance.episode_id.value,
                    "context_signature": symbol_m0.context_signature,
                    "payload_digest": symbol_m0.payload_digest,
                    "symbol_identity": symbol_m0.symbol_identity,
                    "symbol_causal_watermark": int(symbol_event.identity.causal_watermark),
                    "symbol_source_step": int(transition.global_step),
                }
                symbol_m0_node = CanonicalNode(symbol_m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (symbol_event.identity.event_id.hi, symbol_event.identity.event_id.lo), self._watermark)
                symbol_m1g_node = CanonicalNode(symbol_m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (symbol_m1g.uid.hi, symbol_m1g.uid.lo), self._watermark)
                symbol_m1g_payload = {
                    "relation": symbol_m1g.relation.value,
                    "environment_instance_id": symbol_m1g.environment_instance_id,
                    "episode_id": symbol_m1g.episode_id,
                    "parents": [[symbol_m0.uid.hi, symbol_m0.uid.lo]],
                    "symbol_identity": symbol_m0.symbol_identity,
                }
                self._defer_base_group(((symbol_m0_node, symbol_payload, (symbol_m0.uid,)), (symbol_m1g_node, symbol_m1g_payload, (symbol_m0.uid,))))
                self._record_normalized(symbol_row.m1n, defer_publication=True, payload_extra={"symbol_identity": symbol_m0.symbol_identity, "symbol_relations": ["SYMBOL_PRECEDES_ACTION", "SYMBOL_PRECEDES_NORMALIZED_CHANGE", "SYMBOL_NEAR_BOUNDARY" if bool(transition.done) else "SYMBOL_COINCIDENT_WITH_PROGRESS", "SYMBOL_COINCIDENT_WITH_OUTCOME" if int(transition.primary_valence) != 0 else "SYMBOL_COINCIDENT_WITH_PROGRESS"], "temporal_offsets": [0], "causal_watermark": int(symbol_event.identity.causal_watermark), "support": 1.0, "contradiction": 0.0})
                if symbol_row.aligned_m1n is None:
                    continue
                self._record_normalized(symbol_row.aligned_m1n, defer_publication=True, payload_extra={"symbol_identity": symbol_m0.symbol_identity, "aligned_interaction_uid": [m1g.uid.hi, m1g.uid.lo], "cross_modal_evidence": ["SYMBOL_INTERACTION_ALIGNMENT", "SYMBOL_TO_INTERACTION_PREDICTION"], "interaction_only_support": 1.0, "symbol_only_support": 1.0, "aligned_cross_modal_support": 1.0, "heldout_transfer_support": 0.0, "causal_watermark": int(symbol_event.identity.causal_watermark)})
                grounding_key = (
                    int(symbol_row.m1g.uid.lo),
                    int(m1g.uid.lo),
                    int(m1g.environment_instance_id),
                    0,
                    0,
                )
                before_grounding = self.grounding.states.get(grounding_key)
                after_grounding = self.grounding.observe(
                    GroundingEvidence(
                        int(symbol_row.m1g.uid.lo),
                        int(m1g.uid.lo),
                        int(m1g.environment_instance_id),
                        0,
                        0,
                        int(self._watermark),
                        recurrent_symbol=True,
                        cross_modal_association=True,
                    )
                )
                if before_grounding is None or int(after_grounding.maturity) > int(before_grounding.maturity):
                    self.telemetry["grounding_promotions"] += 1
            prior_support = self.signature_support(int(m1n.structural_signature))
            normalized_extra = {}
            if transition.semantic_before:
                normalized_extra["semantic_before"] = [list(row) for row in transition.semantic_before]
            if transition.semantic_action:
                normalized_extra["semantic_action"] = [list(row) for row in transition.semantic_action]
            if transition.semantic_options:
                normalized_extra["semantic_options"] = [list(row) for row in transition.semantic_options]
            if transition.semantic_after:
                normalized_extra["semantic_after"] = [list(row) for row in transition.semantic_after]
            if transition.semantic_delta:
                normalized_extra["semantic_effects"] = [list(row) for row in transition.semantic_delta]
            signature = self._record_normalized(m1n, defer_publication=True, payload_extra=normalized_extra)
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

            self._stage_interval_events += 1
            next_stage = self.stage_tracker.stage
            if self._stage_interval_events >= self._stage_interval_size:
                stage_snapshot = self.stage_tracker.close_interval(
                    self._stage_evidence(),
                    evidence_watermark=self._watermark,
                )
                self._stage_interval_events = 0
                next_stage = stage_snapshot.next_stage
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
            recurrence = self.signature_support(signature)
            recurrence_surprise = 1.0 / max(1.0, float(prior_support + 1))
            decision = self.isf.score(
                ISFComponents(
                    abs(experience.primary_valence),
                    abs(experience.future_option_delta),
                    recurrence_surprise,
                    1.0 / max(1, recurrence),
                    0.5 if experience.family_signature else 0.0,
                    min(1.0, experience.changed_cells / 16.0),
                ),
                decision_watermark=self._watermark,
                evidence_availability_watermark=event.identity.causal_watermark,
                stage=stage_before,
                next_stage=next_stage,
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
            support = self.signature_support(int(signature), len(rows))
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
            confidence = float(getattr(result, "evidence_confidence", 1.0))
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
                    "evidence_confidence": confidence,
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
                        "evidence_confidence": confidence,
                        "parents": [[uid.hi, uid.lo] for uid in role.provenance.parents],
                    },
                    role.provenance.evidence,
                )
            for candidate in result.concepts:
                if candidate.uid in self._m4 and candidate.uid in self.graph.nodes:
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
                        "evidence_confidence": confidence,
                        "formation_scope": list(candidate.provenance.formation_scope),
                        "held_out_targets": [],
                        "validated": False,
                        "concept_state": candidate.state.value,
                        "parents": [[uid.hi, uid.lo] for uid in candidate.provenance.parents],
                    },
                    candidate.provenance.evidence,
                )

    def apply_derivation_results_batch(self, results) -> None:
        for result in tuple(results):
            self.apply_derivation_result(result)

    def _derive_tasks_parallel(self, tasks):
        from v9.runtime.memory_pipeline import derive_memory

        rows = tuple(tasks)
        workers = self._parallel_worker_count(len(rows))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-replay") as pool:
                return tuple(pool.map(derive_memory, rows))
        return tuple(derive_memory(row) for row in rows)

    def _capture_replay_cut_locked(self):
        from v9.cognition.replay import select_replay

        candidates = tuple(
            ReplayCandidate(
                uid,
                fitness
                * float(
                    self.graph.payloads.get(uid, {}).get("evidence_confidence", 1.0)
                ),
            )
            for uid, fitness in sorted(self._replay_pool.items())
            if uid in self.graph.nodes and uid in self.graph.payloads
        )
        selected = select_replay(candidates, limit=self.replay.candidate_limit)
        tasks = []
        for task_id, candidate in enumerate(selected, start=1):
            payload = self.graph.payloads.get(candidate.uid)
            if payload is None or candidate.uid not in self.graph.nodes:
                continue
            task = self.build_derivation_task(
                int(payload["structural_signature"]),
                task_id=task_id,
            )
            if task is not None:
                tasks.append(task)
        return (
            int(self.graph.generation),
            int(self._watermark),
            selected,
            tuple(tasks),
        )

    def _replay_cut_is_current_locked(self, generation, watermark, tasks) -> bool:
        return self._developmental_cut_is_current_locked(
            generation,
            watermark,
            tuple(
                (int(task.structural_signature), int(task.support))
                for task in tasks
            ),
        )

    def _develop(self, signatures: tuple[int, ...] = ()) -> None:
        from v9.runtime.memory_pipeline import derive_memory

        tasks = []
        for task_id, signature in enumerate(
            tuple(sorted(set(signatures)))[: self.config.scientific.replay_candidates_per_interval],
            start=1,
        ):
            task = self.build_derivation_task(int(signature), task_id=task_id)
            if task is not None:
                tasks.append(task)
        self.apply_derivation_results_batch(tuple(derive_memory(task) for task in tasks))

    def effective_state(self, uid: MemoryUid, *, lineage_uid: LineageUid | None = None, context_scope_id: ContextScopeId | None = None, target_environment_id: int | None = None) -> EffectiveCognitiveState:
        node = self.graph.nodes.get(uid)
        if node is None:
            raise KeyError(f"effective state requires live canonical memory {uid.hex()}")
        return self.effective_states.resolve(node, lineage_uid=lineage_uid, context_scope_id=context_scope_id, target_environment_id=target_environment_id, target_trust=self.transfer_trust.score_map())

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
        # Derive against one immutable graph/support cut. If anything mutates the
        # authoritative state before commit, discard the work and retry the cut.
        for _attempt in range(3):
            with self._lock:
                generation, watermark, selected, tasks = self._capture_replay_cut_locked()
            derived = self._derive_tasks_parallel(tasks)
            with self._lock:
                if not self._replay_cut_is_current_locked(
                    generation, watermark, tasks
                ):
                    continue
                before_m0 = self.graph.memory_count(MemoryLevel.M0)
                before_nodes = len(self.graph.nodes)
                before_generation = int(self.graph.generation)
                self.apply_derivation_results_batch(derived)
                new_memories = len(self.graph.nodes) - before_nodes
                revisions = len(derived) if self.graph.generation > before_generation else 0
                if self.graph.memory_count(MemoryLevel.M0) != before_m0:
                    raise RuntimeError("replay may not fabricate M0 environment evidence")
                processed = len(selected)
                self.replay.selected += len(selected)
                self.replay.processed += processed
                self.replay.new_memories += new_memories
                self.replay.revisions += revisions
                result = ReplayResult(
                    len(selected), processed, new_memories, revisions, 0
                )
                self.evidence.append("REPLAY", self._watermark, asdict(result))
                return result

        # Progress fallback: keep the cut frozen with the runtime lock while pure
        # derivation runs in worker threads, then commit the same deterministic batch.
        with self._lock:
            _generation, _watermark, selected, tasks = self._capture_replay_cut_locked()
            derived = self._derive_tasks_parallel(tasks)
            before_m0 = self.graph.memory_count(MemoryLevel.M0)
            before_nodes = len(self.graph.nodes)
            before_generation = int(self.graph.generation)
            self.apply_derivation_results_batch(derived)
            new_memories = len(self.graph.nodes) - before_nodes
            revisions = len(derived) if self.graph.generation > before_generation else 0
            if self.graph.memory_count(MemoryLevel.M0) != before_m0:
                raise RuntimeError("replay may not fabricate M0 environment evidence")
            processed = len(selected)
            self.replay.selected += len(selected)
            self.replay.processed += processed
            self.replay.new_memories += new_memories
            self.replay.revisions += revisions
            result = ReplayResult(len(selected), processed, new_memories, revisions, 0)
            self.evidence.append("REPLAY", self._watermark, asdict(result))
            return result

    def record_replanning_evidence(self, *, recovered: bool = False, improved_efficiency: bool = False) -> None:
        self._replans_demonstrated += 1
        self._recovered_replans += int(bool(recovered))
        self._efficient_replans += int(bool(recovered) and bool(improved_efficiency))
        self.evidence.append("REPLANNING", self._watermark, {"recovered": bool(recovered), "improved_efficiency": bool(improved_efficiency)})

    def record_symbol_conditioned_prediction(self, *, baseline: float, conditioned: float, actual: float) -> float:
        from v9.cognition.prediction import PredictionComparison
        comparison = PredictionComparison(float(baseline), float(conditioned), float(actual))
        self._symbol_prediction_delta_sum += comparison.delta
        self.telemetry["symbol_conditioned_prediction_observations"] += 1
        self.evidence.append("SYMBOL_CONDITIONED_PREDICTION", self._watermark, {"baseline": baseline, "conditioned": conditioned, "actual": actual, "delta": comparison.delta})
        return comparison.delta

    def record_transfer_validation(self, concept_uid: MemoryUid, *, target_environment_id: int, target_native_action: int, enabled_metric: float, ablated_metric: float, matched: bool = True, held_out: bool = True, context_scope_id: int = 0) -> None:
        with self._lock:
            concept = self._m4.get(concept_uid)
            if concept is None or concept_uid not in self.graph.nodes or concept_uid not in self.graph.payloads:
                # Transfer trials run concurrently with lifecycle/compaction. A concept
                # selected for a trial may be retired before the result is recorded.
                self._m4.pop(concept_uid, None)
                self._transfer_trials.pop(concept_uid, None)
                self.evidence.append(
                    "TRANSFER_TRIAL_STALE_CONCEPT",
                    self._watermark,
                    {"concept_uid": concept_uid.hex(), "target_environment_id": int(target_environment_id)},
                )
                return
            was_validated = bool(concept.validated)
            row = {"target_environment_id": int(target_environment_id), "target_native_action": int(target_native_action), "enabled_metric": float(enabled_metric), "ablated_metric": float(ablated_metric), "matched": bool(matched), "held_out": bool(held_out), "context_scope_id": int(context_scope_id)}
            trial_rows = self._transfer_trials.setdefault(concept_uid, [])
            trial_rows.append(row)
            hot_trial_limit = max(self.config.scientific.transfer_minimum_trials, self.config.scientific.transfer_validation_trials_per_interval)
            del trial_rows[:-hot_trial_limit]
            self.evidence.append("TRANSFER_TRIAL", self._watermark, {"concept_uid": concept_uid.hex(), **row})
            formation_scope = set(concept.provenance.formation_scope)
            effect = float(enabled_metric) - float(ablated_metric)
            positive = bool(matched and held_out and int(target_environment_id) not in formation_scope and effect > self.config.scientific.transfer_effect_threshold)
            self.transfer_trust.observe(concept_uid, target_environment_id=target_environment_id, context_scope_id=context_scope_id, effect=effect, positive=positive)
            admissible = tuple(trial for trial in self._transfer_trials[concept_uid] if trial["matched"] and trial["held_out"] and trial["target_environment_id"] not in formation_scope and trial["enabled_metric"] - trial["ablated_metric"] > self.config.scientific.transfer_effect_threshold)
            if ValidationMode(self.config.scientific.transfer_validation_mode) is ValidationMode.LEARNING_ONLY:
                return
            if len(admissible) < self.config.scientific.transfer_minimum_trials:
                return
            if was_validated:
                return
            concept = concept.with_validation(tuple(int(trial["target_environment_id"]) for trial in admissible))
            self._m4[concept.uid] = concept
            concept_node = self.graph.nodes.get(concept.uid)
            concept_payload = self.graph.payloads.get(concept.uid)
            if concept_node is None or concept_payload is None:
                self.evidence.append("TRANSFER_TRIAL_STALE_CONCEPT", self._watermark, {"concept_uid": concept.uid.hex(), "target_environment_id": int(target_environment_id)})
                return
            self._publish(concept_node, {**concept_payload, "held_out_targets": list(concept.held_out_targets), "validated": concept.validated, "concept_state": concept.state.value}, concept.provenance.evidence, proposal_class=ProposalClass.STATEFUL, mutation_kind=MutationKind.UPDATE_VALIDATION)
            if self.config.enable_lifecycle and concept.uid in self.lifecycle.records:
                self.lifecycle.records[concept.uid] = replace(self.lifecycle.records[concept.uid], state=CognitiveState.VALIDATED, last_transition_watermark=self._watermark)
            parent_uid = concept.provenance.parents[0] if concept.provenance.parents else None
            role = None if parent_uid is None else self._m3.get(parent_uid)
            if role is None or parent_uid not in self.graph.nodes or parent_uid not in self.graph.payloads:
                self.evidence.append("TRANSFER_TRIAL_STALE_PROVENANCE", self._watermark, {"concept_uid": concept_uid.hex()})
                return
            consequence = M5ConsequenceStructure.form((concept,), (role.consequence_signature,))
            concept_confidence = float(self.graph.payloads.get(concept.uid, {}).get("evidence_confidence", 1.0))
            self._publish(CanonicalNode(consequence.uid, MemoryLevel.M5, MemoryType.CONSEQUENCE, consequence.consequence_descriptor, self._watermark), {"descriptor": list(consequence.consequence_descriptor), "mature": consequence.mature, "evidence_confidence": concept_confidence, "parents": [[uid.hi, uid.lo] for uid in consequence.provenance.parents]}, consequence.provenance.evidence)
            comparable = tuple(
                row for row in self._m5.values()
                if row.uid != consequence.uid
                and row.mature
                and row.consequence_descriptor == consequence.consequence_descriptor
            )
            outcome = M6Outcome.form(tuple(sorted((*comparable, consequence), key=lambda row: row.uid)), diameter_bound=0)
            self._publish(CanonicalNode(outcome.uid, MemoryLevel.M6, MemoryType.OUTCOME, outcome.class_signature, self._watermark), {"class_signature": list(outcome.class_signature), "class_version": outcome.class_version, "equivalence_trials": outcome.equivalence_trials, "equivalence_successes": outcome.equivalence_successes, "contexts_observed": list(outcome.contexts_observed), "environments_observed": list(outcome.environments_observed), "primary_valence_sum": outcome.primary_valence_sum, "preference_trials": outcome.preference_trials, "evidence_confidence": concept_confidence, "parents": [[uid.hi, uid.lo] for uid in outcome.provenance.parents]}, outcome.provenance.evidence)
            action = int(admissible[-1]["target_native_action"])
            episode_rows: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for uid in tuple(self.graph._uids_by_level[MemoryLevel.M0]):
                node = self.graph.nodes.get(uid)
                payload = self.graph.payloads.get(uid)
                if node is None or payload is None or int(payload.get("environment_instance_id", -1)) != int(target_environment_id):
                    continue
                if payload.get("action_id") is None or payload.get("episode_id") is None:
                    continue
                episode_rows.setdefault(int(payload["episode_id"]), []).append((int(node.created_watermark), payload))
            formed = 0
            for rows in episode_rows.values():
                ordered = sorted(rows, key=lambda item: item[0])
                if not any(bool(payload.get("task_success", False)) for _, payload in ordered):
                    continue
                native_actions = tuple(int(payload["action_id"]) for _, payload in ordered)
                if not native_actions:
                    continue
                primary_valence_sum = sum(int(payload.get("primary_valence", 0)) for _, payload in ordered)
                realized_cost_sum = sum(max(1, int(payload.get("realized_cost", 0))) for _, payload in ordered)
                strategy = M7Strategy.form(
                    outcome,
                    target_environment_id=int(target_environment_id),
                    native_actions=native_actions,
                    successes=1,
                    trials=1,
                    primary_valence_sum=primary_valence_sum,
                    realized_cost_sum=realized_cost_sum or len(native_actions),
                )
                existing = self.__dict__.setdefault("_m7", {}).get(strategy.uid)
                if existing is not None:
                    strategy = existing.observe(success=True, realized_cost=realized_cost_sum or len(native_actions), primary_valence=primary_valence_sum)
                self.__dict__.setdefault("_m7", {})[strategy.uid] = strategy
                self._publish(
                    CanonicalNode(strategy.uid, MemoryLevel.M7, MemoryType.STRATEGY, (outcome.uid.hi, outcome.uid.lo, target_environment_id, *native_actions), self._watermark),
                    {"target_outcome": [outcome.uid.hi, outcome.uid.lo], "target_environment_id": int(target_environment_id), "native_actions": list(native_actions), "reliability_successes": strategy.reliability_successes, "reliability_trials": strategy.reliability_trials, "evidence_confidence": concept_confidence, "primary_valence_sum": strategy.primary_valence_sum, "realized_cost_sum": strategy.realized_cost_sum, "context_scope_id": int(context_scope_id), "parents": [[outcome.uid.hi, outcome.uid.lo]]},
                    strategy.provenance.evidence,
                )
                formed += 1
            if not formed:
                strategy = M7Strategy.form(outcome, target_environment_id=int(target_environment_id), native_actions=(action,), successes=len(admissible), trials=len(admissible), primary_valence_sum=0, realized_cost_sum=len(admissible))
                self.__dict__.setdefault("_m7", {})[strategy.uid] = strategy
                self._publish(CanonicalNode(strategy.uid, MemoryLevel.M7, MemoryType.STRATEGY, (outcome.uid.hi, outcome.uid.lo, target_environment_id, action), self._watermark), {"target_outcome": [outcome.uid.hi, outcome.uid.lo], "target_environment_id": int(target_environment_id), "native_actions": [action], "reliability_successes": strategy.reliability_successes, "reliability_trials": strategy.reliability_trials, "evidence_confidence": concept_confidence, "primary_valence_sum": 0, "realized_cost_sum": strategy.realized_cost_sum, "context_scope_id": int(context_scope_id), "parents": [[outcome.uid.hi, outcome.uid.lo]]}, strategy.provenance.evidence)
            self.__dict__.setdefault("_m5", {})[consequence.uid] = consequence
            self.__dict__.setdefault("_m6", {})[outcome.uid] = outcome
            if len(outcome.members) >= 2:
                self.record_outcome_equivalence_evidence(
                    outcome.uid,
                    equivalent=True,
                    context_scope_id=int(context_scope_id),
                    environment_id=int(target_environment_id),
                )
    
    def wait_quiescent(self, timeout: float = 300.0) -> None:
        del timeout
        self._drain_timeline()

    def capture_experiment_state(self) -> dict[str, Any]:
        """Capture an in-memory scientific state cut for matched branch evaluation."""
        # Quiesce before taking the runtime lock so drain/flush work can complete.
        self.wait_quiescent()
        self.flush_deferred_memory_updates()
        with self._lock:
            return {"state": self.state_dict()}

    def restore_experiment_state(self, captured: dict[str, Any]) -> None:
        """Restore an in-memory scientific state cut without creating a disk snapshot."""
        self.wait_quiescent()
        with self._lock:
            self._restore(captured)

    def snapshot(self) -> SnapshotResult:
        with self._lock:
            self.wait_quiescent()
            self.flush_deferred_memory_updates()
            self.evidence.flush()
            self._snapshot_id += 1
            self.telemetry["snapshot_writes"] += 1
            snapshot_id = self._snapshot_id
            watermark = self._watermark
            generation = self.graph.generation
            fixed_cut = self.state_dict()
        try:
            result = write_snapshot(self.root, fixed_cut, snapshot_id=snapshot_id, watermark=watermark, graph_generation=generation, scientific_config_id=self.config.scientific.config_id.value)
            self.write_canonical_snapshot(snapshot_id)
            return result
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
            "hgt_action_scores": {str(env): {str(action): score for action, score in actions.items()} for env, actions in self._hgt_action_scores.items()},
            "hgt_context_action_scores": {
                str(env): {
                    str(context): {str(action): score for action, score in actions.items()}
                    for context, actions in contexts.items()
                }
                for env, contexts in getattr(self, "_hgt_context_action_scores", {}).items()
            },
            "in_flight_proposals": [],
            "m1n_occurrences": {str(key): len(value) for key, value in self._m1n_occurrences.items()},
            "m1n_supports": {str(key): value for key, value in self._m1n_supports.items()},
            "cross_modal_signatures": list(self._cross_modal_signatures),
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

    def _restore(self, snapshot: dict[str, Any], *, graph_override: CanonicalGraph | None = None) -> None:
        state = dict(snapshot["state"])
        self.graph = graph_override if graph_override is not None else CanonicalGraph.from_state_dict(dict(state["graph"]))
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
        self._hgt_action_scores = {int(env): {int(action): float(score) for action, score in dict(actions).items()} for env, actions in dict(state.get("hgt_action_scores", {})).items()}
        self._hgt_context_action_scores = {
            int(env): {
                int(context): {int(action): float(score) for action, score in dict(actions).items()}
                for context, actions in dict(contexts).items()
            }
            for env, contexts in dict(state.get("hgt_context_action_scores", {})).items()
        }
        if state.get("in_flight_proposals"):
            raise RuntimeError("native v9 snapshot contains unsupported in-flight proposals")
        normalized_by_signature: dict[int, MemoryUid] = {}
        for uid in self.graph._uids_by_level[MemoryLevel.M1]:
            node = self.graph.nodes.get(uid)
            if node is None or node.memory_type is not MemoryType.NORMALIZED_RELATION:
                continue
            signature = self.graph.payloads.get(uid, {}).get("structural_signature")
            if signature is not None:
                normalized_by_signature.setdefault(int(signature), uid)
        for signature, count in dict(state.get("m1n_occurrences", {})).items():
            node = normalized_by_signature.get(int(signature))
            if node is None:
                continue
            payload = self.graph.payloads[node]
            dummy_parent_uid = node
            evidence_refs = tuple(MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("evidence_refs", [[node.hi, node.lo]]))
            dummy = M1NormalizedRelation(node, str(payload["observable_relation"]), NormalizedChannel(str(payload["channel"])), int(signature), DerivationProvenance((dummy_parent_uid,), evidence_refs))
            self._m1n_occurrences[int(signature)] = [dummy] if int(count) > 0 else []
        self._m1n_supports = {int(key): int(value) for key, value in dict(state.get("m1n_supports", state.get("m1n_occurrences", {}))).items()}
        for record in self.signature_index.dirty_window():
            signature = int(record.signature)
            self._m1n_supports[signature] = max(
                int(self._m1n_supports.get(signature, 0)), int(record.support)
            )
            if signature in self._m1n_occurrences:
                continue
            uid = normalized_by_signature.get(signature)
            if uid is None:
                continue
            payload = self.graph.payloads[uid]
            evidence_refs = tuple(
                MemoryUid(int(raw[0]), int(raw[1]))
                for raw in payload.get("evidence_refs", [[uid.hi, uid.lo]])
            )
            self._m1n_occurrences[signature] = [
                M1NormalizedRelation(
                    uid,
                    str(payload["observable_relation"]),
                    NormalizedChannel(str(payload["channel"])),
                    signature,
                    DerivationProvenance((uid,), evidence_refs),
                )
            ]
        self._cross_modal_signatures = {int(value): None for value in state.get("cross_modal_signatures", [])}
        self._actor_action_supports = {}
        self._actor_policy_generation = self.graph.generation
        for signature, support in self._m1n_supports.items():
            rows = self._m1n_occurrences.get(int(signature), ())
            if not rows:
                continue
            observable = str(rows[0].observable_relation)
            parts = observable.split(":")
            if len(parts) < 5 or parts[0] != "ACTION":
                continue
            try:
                scoped = scoped_action_key(int(parts[3]), action_schema_id=int(parts[1]), environment_type=parts[2])
            except ValueError:
                continue
            self._actor_action_supports[scoped] = self._actor_action_supports.get(scoped, 0.0) + float(support)
        self._replay_pool = {MemoryUid(int(key[:16], 16), int(key[16:], 16)): float(value) for key, value in dict(state.get("replay_pool", {})).items()}
        self._formation_environments = {int(value) for value in state.get("formation_environments", [])}
        self._latest_interaction_grounding = {}
        for raw in state.get("latest_interaction_grounding", []):
            uid = MemoryUid(int(raw["uid"][0]), int(raw["uid"][1]))
            parents = tuple(MemoryUid(int(value[0]), int(value[1])) for value in raw["parents"])
            evidence = tuple(MemoryUid(int(value[0]), int(value[1])) for value in raw["evidence"])
            row = M1GroundedContingency(uid, GroundedRelation(str(raw["relation"])), DerivationProvenance(parents, evidence), int(raw["environment_instance_id"]), int(raw["episode_id"]), int(raw["grounded_context_signature"]), None if raw.get("executable_action_token") is None else int(raw["executable_action_token"]), int(raw["realized_transition_signature"]), int(raw["grounded_next_context_signature"]), int(raw.get("support", 1)))
            self._latest_interaction_grounding[(int(raw["key"][0]), int(raw["key"][1]))] = row
        for level in (MemoryLevel.M2, MemoryLevel.M3, MemoryLevel.M4, MemoryLevel.M5, MemoryLevel.M6, MemoryLevel.M7):
            for uid in tuple(self.graph._uids_by_level[level]):
                node = self.graph.nodes.get(uid)
                payload = self.graph.payloads.get(uid)
                if node is None or payload is None:
                    self.graph._uids_by_level[level].discard(uid)
                    continue
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
                elif node.level is MemoryLevel.M5 and provenance is not None:
                    self._m5[uid] = M5ConsequenceStructure(uid, tuple(int(value) for value in payload.get("descriptor", node.structural_key)), provenance, bool(payload.get("mature", False)))
                elif node.level is MemoryLevel.M6 and provenance is not None:
                    members = tuple(parents)
                    self._m6[uid] = M6Outcome(uid, tuple(int(value) for value in payload.get("class_signature", node.structural_key)), members, provenance, int(payload.get("class_version", 1)), int(payload.get("equivalence_trials", 0)), int(payload.get("equivalence_successes", 0)), tuple(int(value) for value in payload.get("contexts_observed", ())), tuple(int(value) for value in payload.get("environments_observed", ())), int(payload.get("primary_valence_sum", 0)), int(payload.get("preference_trials", 0)))
                elif node.level is MemoryLevel.M7 and provenance is not None:
                    target_raw = payload.get("target_outcome", parents[0] if parents else None)
                    if isinstance(target_raw, MemoryUid):
                        target_uid = target_raw
                    elif isinstance(target_raw, (list, tuple)) and len(target_raw) == 2:
                        target_uid = MemoryUid(int(target_raw[0]), int(target_raw[1]))
                    elif parents:
                        target_uid = parents[0]
                    else:
                        continue
                    self._m7[uid] = M7Strategy(uid, target_uid, int(payload["target_environment_id"]), tuple(int(value) for value in payload.get("native_actions", ())), int(payload.get("reliability_successes", 0)), int(payload.get("reliability_trials", 0)), int(payload.get("primary_valence_sum", 0)), int(payload.get("realized_cost_sum", 0)), provenance)
        for raw_uid, rows in dict(state.get("transfer_trials", {})).items():
            self._transfer_trials[MemoryUid(int(raw_uid[:16], 16), int(raw_uid[16:], 16))] = list(rows)
        if hasattr(self, "canonical_store"):
            self.graph.configure_durable_commit(self._commit_canonical_graph_update)

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
        with self._lock:
            self.unified_telemetry.record_deliberation(reasoning_cycles=reasoning_cycles, initial_score=initial_score, final_score=final_score, best_score=best_score, changed=changed, behavior_improved=behavior_improved, reasoning_cost=reasoning_cost, stop_reason=stop_reason, candidate_changes=candidate_changes, prediction_improvement=prediction_improvement, strategy_changes=strategy_changes, provenance=provenance)

    def record_hgt_inference(self, sample: HGTInferenceSample, *, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_hgt_inference(sample, provenance=provenance)

    def record_hgt_ablation(self, *, enabled_outcome: float, hydra_baseline_outcome: float, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_hgt_ablation(enabled_outcome=enabled_outcome, hydra_baseline_outcome=hydra_baseline_outcome, provenance=provenance)

    def record_hgt_training(self, sample: HGTTrainingSample, *, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_hgt_training(sample, provenance=provenance)

    def record_model_evolution(self, sample: ModelEvolutionSample, *, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_model_evolution(sample, provenance=provenance)

    def record_hgt_consolidation(self, sample: ConsolidationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_consolidation(sample, provenance=provenance)

    def record_optimization(self, sample: OptimizationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        with self._lock:
            self.unified_telemetry.record_optimization(sample, provenance=provenance)

    def record_curriculum_event(self, *, step: str | None, environment_family: str, game_scenario: str) -> None:
        with self._lock:
            self.unified_telemetry.record_curriculum_event(step=step, environment_family=environment_family, game_scenario=game_scenario)

    def set_telemetry_gauge(self, key: str, value: float | int | str) -> None:
        with self._lock:
            self.unified_telemetry.set_gauge(key, value)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return self._metrics_locked()

    def _metrics_locked(self) -> dict[str, Any]:
        counts = {f"M{level}": self.graph.memory_count(MemoryLevel(level)) for level in range(8)}
        deferred_levels = {0: 0, 1: 0}
        for node, _, _ in self._deferred_base_nodes.values():
            if int(node.level) in deferred_levels and node.uid not in self.graph.nodes:
                deferred_levels[int(node.level)] += 1
        counts["M0"] += deferred_levels[0]
        counts["M1"] += deferred_levels[1]
        normalization = {str(radius): self.scale_statistics.state(radius).value for radius in self.config.scientific.structural_radii}
        grounding_counts = {f"G{level}": sum(int(row.maturity) == level for row in self.grounding.states.values()) for level in range(6)}
        persistent_bytes = max(
            1,
            self.graph.memory_count() * 192 + len(self.graph.edges) * 96,
        )
        validated_transfers = sum(row.successes for row in self.transfer_trust.records.values())
        prediction_observations = self.telemetry["symbol_conditioned_prediction_observations"]
        strategy_payloads = [
            payload
            for uid, payload in self.graph.payloads.items()
            if self.graph.nodes[uid].level is MemoryLevel.M7
        ]
        strategy_trials = sum(int(row.get("reliability_trials", 0)) for row in strategy_payloads)
        strategy_successes = sum(int(row.get("reliability_successes", 0)) for row in strategy_payloads)
        realized_cost_total = sum(float(row.get("realized_cost_sum", 0.0)) for row in strategy_payloads)
        failed_transfer_scopes = sum(getattr(row.state, "name", str(row.state)) == "FAILED" for row in self.transfer_trust.records.values())
        multi_strategy_outcomes = sum(len({s.native_actions for s in self._m7.values() if s.target_outcome == o.uid}) >= 2 for o in self._m6.values())
        equivalence_confidences = [o.equivalence_confidence for o in self._m6.values()]
        preference_outcomes = [o for o in self._m6.values() if o.preference_trials > 0]
        validated_m4 = sum(bool(row.validated) for row in self._m4.values())
        diagnostic = self.unified_telemetry.diagnostic_metrics()
        retired = int(diagnostic.get("hydra_nodes_retired", 0))
        replaced = int(diagnostic.get("hydra_nodes_replaced_by_abstractions", 0))
        total_memories = sum(counts.values())
        compression_ratio = (retired + replaced) / max(1, total_memories + retired)
        base = {
            "watermark": self._watermark,
            "graph_generation": self.graph.generation,
            "memories": total_memories,
            "edges": len(self.graph.edges),
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
            **{f"grounding_G{level}_count": grounding_counts[f"G{level}"] for level in range(6)},
            "grounding_active_count": sum(int(row.behavior_eligible) for row in self.grounding.states.values()),
            "grounding_suspended_count": sum(int(row.suspended) for row in self.grounding.states.values()),
            "grounding_mean_confidence": (
                sum(row.support / max(1e-9, row.support + row.contradiction) for row in self.grounding.states.values())
                / max(1, len(self.grounding.states))
            ),
            "lineage_overlays": len(self.lineages.overlays),
            "lineage_dependencies": len(self.lineages.dependencies),
            "context_scopes": len(self.contexts.records),
            "probation_records": sum(row.state.name == "PROBATION" for row in self.lifecycle.records.values()),
            "probation_transitions": self.lifecycle.transitions,
            "developmental_stage": int(self.stage_tracker.stage),
            "m6_equivalence_confidence": sum(equivalence_confidences) / max(1, len(equivalence_confidences)),
            "m6_preference_relations": len(preference_outcomes),
            "m7_multi_strategy_outcomes": multi_strategy_outcomes,
            "m7_replan_attempts": self._replans_demonstrated,
            "m7_replan_successes": self._efficient_replans,
            "m7_replanning_recovery_rate": self._recovered_replans / max(1, self._replans_demonstrated),
            "m7_replanning_efficiency_rate": self._efficient_replans / max(1, self._replans_demonstrated),
            "developmental_intervals": self.stage_tracker.interval_id,
            "isf_decisions_hot": len(self.isf.decisions),
            "replay": self.replay.state_dict(),
            "symbol_conditioned_prediction_delta": self._symbol_prediction_delta_sum / max(1, prediction_observations),
            "symbol_prediction_gain": self._symbol_prediction_delta_sum / max(1, prediction_observations),
            "prediction_error": self._prediction_error_sum / max(1, self._prediction_error_count),
            "persistent_consolidated_bytes": persistent_bytes,
            "persistent_memory_growth_ratio": total_memories / max(1, self.telemetry["events"]),
            "compression_ratio": compression_ratio,
            "m4_validated": validated_m4,
            "m7_strategy_success_rate": strategy_successes / max(1, strategy_trials),
            "success_rate": float(diagnostic.get("behavioral_success_rate", 0.0)),
            "trajectory_efficiency": float(
                diagnostic.get(
                    "environment_trajectory_efficiency",
                    strategy_successes / max(1.0, realized_cost_total),
                )
            ),
            "explanatory_reach_per_persistent_byte": sum(int(payload.get("explanatory_reach", 0)) for payload in self.graph.payloads.values()) / persistent_bytes,
            "transfer_quality_per_persistent_byte": validated_transfers / persistent_bytes,
            "prediction_quality_per_persistent_byte": (
                max(0.0, 1.0 - (self._prediction_error_sum / max(1, self._prediction_error_count)))
                + max(0.0, self._symbol_prediction_delta_sum / max(1, prediction_observations))
            ) / persistent_bytes,
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
        self.evidence.flush()
        result = self.snapshot() if normal and self.config.enable_snapshots else None
        if normal:
            self.write_scientific_report()
        if hasattr(self, "canonical_store"):
            self.canonical_store.close()
        self.signature_index.close()
        self._closed = True
        return result
