from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Iterable

from v9.cognition.isf import ISFComponents
from v9.cognition.grounding import GroundingEvidence
from v9.cognition.action_selection import scoped_action_key
from v9.cognition.similarity import StructuralDescriptor
from v9.memory.identity import MemoryUid
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.provenance import DerivationProvenance
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.memory_pipeline import DerivationResult, PreparedIngestion, PreparedSymbolIngestion
from v9.runtime.runtime import ContinuousMemoryRuntime as BaseContinuousMemoryRuntime
from v9.runtime.snapshot_backend import latest_snapshot
from v9.telemetry import build_primary_dashboard, read_gpu_snapshot


class ContinuousMemoryRuntime(BaseContinuousMemoryRuntime):
    """Performance-oriented v9 runtime preserving the canonical ordered boundary."""

    def __init__(self, config: Any) -> None:
        self._hgt_context_action_scores: dict[int, dict[int, dict[int, float]]] = {}
        self._normalized_action_cache: dict[int, int | None] = {}
        self._metrics_cache: dict[str, Any] | None = None
        self._metrics_cache_at = 0.0
        restore_path = latest_snapshot(config.root) if bool(getattr(config, "restore", False)) else None
        restore_started = time.perf_counter()
        if restore_path is not None:
            print(f"{time.strftime('[%H:%M]')} restoring snapshot: {restore_path}", flush=True)
        super().__init__(config)
        if restore_path is not None:
            print(
                f"{time.strftime('[%H:%M]')} restore complete "
                f"memories={len(self.graph.nodes)} edges={len(self.graph.edges)} "
                f"seconds={time.perf_counter() - restore_started:.2f}",
                flush=True,
            )

    def set_hgt_action_scores(self, scores: dict[int, dict[int, float]], *, context_action_scores: dict[int, dict[int, dict[int, float]]] | None = None) -> None:
        with self._lock:
            self._hgt_action_scores = {int(environment): {int(action): float(score) for action, score in actions.items()} for environment, actions in scores.items()}
            if context_action_scores is not None:
                self._hgt_context_action_scores = {
                    int(environment): {
                        int(context): {int(action): float(score) for action, score in actions.items()}
                        for context, actions in contexts.items()
                    }
                    for environment, contexts in context_action_scores.items()
                }
            self._actor_policy_generation += 1

    def actor_policy_snapshot(self) -> ActorPolicySnapshot:
        # Keep the optimized runtime behaviorally identical to the canonical
        # M6 -> M7 -> planner policy construction.
        return super().actor_policy_snapshot()

    def full_metrics(self) -> dict[str, Any]:
        metrics = super().metrics()
        with self._lock:
            self._metrics_cache = metrics
            self._metrics_cache_at = time.monotonic()
        return metrics

    def metrics(self) -> dict[str, Any]:
        # Programmatic callers rely on exact post-mutation semantics. Never serve
        # a stale cached graph cut here; the cache is dashboard-only.
        return self.full_metrics()

    def dashboard_metrics(self) -> dict[str, Any]:
        # The dashboard polls every two seconds. On million-node restored runs a
        # full metrics traversal here competes directly with process startup. Use
        # the last exact epoch cut and refresh only O(1) live counters/level sizes.
        with self._lock:
            result = dict(self._metrics_cache or {})
            result["watermark"] = self._watermark
            result["graph_generation"] = self.graph.generation
            result["edges"] = len(self.graph.edges)
            result["memories"] = len(self.graph.nodes)
            result["memory_levels"] = {
                level.name: len(self.graph._uids_by_level[level])
                for level in MemoryLevel
            }
            result["timeline_events_seen"] = self.timeline.events_seen
            result["timeline_events_dropped"] = self.timeline.events_dropped
            result["actions_committed"] = self.timeline.actions_committed
            diagnostic = self.unified_telemetry.diagnostic_metrics()
            gpu = read_gpu_snapshot()
            diagnostic = dict(diagnostic)
            diagnostic["gpu_memory_current_bytes"] = int(gpu.memory_used_bytes)
            diagnostic["gpu_utilization_current"] = float(gpu.utilization_percent)
            result["telemetry_diagnostics"] = diagnostic
            total_memories = sum(int(v) for v in result["memory_levels"].values())
            result["success_rate"] = float(diagnostic.get("behavioral_success_rate", 0.0))
            result["trajectory_efficiency"] = float(
                diagnostic.get("environment_trajectory_efficiency", 0.0)
            )
            result["prediction_error"] = self._prediction_error_sum / max(1, self._prediction_error_count)
            result["persistent_memory_growth_ratio"] = total_memories / max(1, self.telemetry["events"])
            retired = int(diagnostic.get("hydra_nodes_retired", 0))
            replaced = int(diagnostic.get("hydra_nodes_replaced_by_abstractions", 0))
            result["compression_ratio"] = (retired + replaced) / max(1, total_memories + retired)
            result["m4_validated"] = sum(bool(row.validated) for row in self._m4.values())
            result["primary_dashboard"] = build_primary_dashboard(result, diagnostic)
            return result

    def _advance_passive_stage_interval(self, count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            self._stage_interval_events += 1
            if self._stage_interval_events < self._stage_interval_size:
                continue
            stage_snapshot = self.stage_tracker.close_interval(self._stage_evidence(), evidence_watermark=self._watermark)
            self._stage_interval_events = 0
            self.evidence.append("DEVELOPMENTAL_STAGE", self._watermark, {"interval_id": stage_snapshot.interval_id, "stage": int(stage_snapshot.stage), "next_stage": int(stage_snapshot.next_stage), "evidence": asdict(stage_snapshot.evidence)})

    def _apply_prepared_symbol(self, row: PreparedSymbolIngestion) -> tuple[int, ...]:
        event = row.event
        self._watermark = max(self._watermark, int(event.identity.causal_watermark))
        self.timeline.events_seen += 1
        modality = int(event.identity.modality_id.value)
        self._modality_events[modality] = self._modality_events.get(modality, 0) + 1
        self.telemetry["events"] += 1
        self._formation_environments.add(int(event.identity.environment_instance_id))
        m0, m1g, m1n = row.m0, row.m1g, row.m1n
        m0_node = CanonicalNode(m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (event.identity.event_id.hi, event.identity.event_id.lo), self._watermark)
        m0_payload = {"modality_id": m0.modality_id, "environment_instance_id": m0.provenance.environment_instance_id, "episode_id": m0.provenance.episode_id.value, "context_signature": m0.context_signature, "payload_digest": m0.payload_digest, "action_id": m0.action_id, "outcome_signature": m0.outcome_signature, "next_context_signature": m0.next_context_signature, "symbol_identity": m0.symbol_identity, "primary_valence": m0.primary_valence, "future_option_delta": m0.future_option_delta, "realized_cost": m0.realized_cost}
        m1g_node = CanonicalNode(m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (m1g.uid.hi, m1g.uid.lo), self._watermark)
        m1g_payload = {"relation": m1g.relation.value, "environment_instance_id": m1g.environment_instance_id, "episode_id": m1g.episode_id, "grounded_context_signature": m1g.grounded_context_signature, "executable_action_token": m1g.executable_action_token, "realized_transition_signature": m1g.realized_transition_signature, "grounded_next_context_signature": m1g.grounded_next_context_signature, "parents": [[m0.uid.hi, m0.uid.lo]]}
        self._defer_base_group(((m0_node, m0_payload, (m0.uid,)), (m1g_node, m1g_payload, (m0.uid,))))
        signatures = [self._record_normalized(m1n, defer_publication=True)]
        if row.aligned_m1n is not None:
            signatures.append(self._record_normalized(row.aligned_m1n, defer_publication=True))
        self.evidence.append("INGESTION", self._watermark, {"m0": m0.uid.hex(), "m1g": m1g.uid.hex(), "m1n": m1n.uid.hex(), "channel": m1n.channel.value, "worker_prepared": True, "passive_symbol": True})
        self._advance_passive_stage_interval()
        return tuple(signatures)

    def apply_prepared_ingestion(self, prepared: Any) -> tuple[int, ...]:
        if not isinstance(prepared, PreparedIngestion):
            raise TypeError("prepared ingestion has unexpected type")
        signatures: list[int] = []
        if prepared.event is not None:
            signatures.extend(super().apply_prepared_ingestion(prepared))
        with self._lock:
            if prepared.symbol_codec_state:
                codec = DeterministicSymbolCodec.from_state_dict(dict(prepared.symbol_codec_state))
                self.symbol_codecs[codec.vocabulary_id.value] = codec
            for row in prepared.symbols:
                signatures.extend(self._apply_prepared_symbol(row))
        return tuple(signatures)

    def _record_normalized_deferred_batch(
        self,
        relation: M1NormalizedRelation,
        deferred_rows: list[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]],
        transition: Any = None,
    ) -> int:
        signature = int(relation.structural_signature)
        occurrences = self._m1n_occurrences.setdefault(signature, [])
        if relation.channel is NormalizedChannel.CROSS_MODAL:
            self._cross_modal_signatures.pop(signature, None)
            self._cross_modal_signatures[signature] = None
            while len(self._cross_modal_signatures) > 8192:
                self._cross_modal_signatures.pop(next(iter(self._cross_modal_signatures)))
        support = int(self._m1n_supports.get(signature, 0)) + 1
        self._m1n_supports[signature] = support
        if signature not in self._normalized_action_cache:
            observable = str(relation.observable_relation)
            parts = observable.split(":")
            scoped: int | None = None
            if len(parts) >= 5 and parts[0] == "ACTION":
                try:
                    scoped = scoped_action_key(
                        int(parts[3]),
                        action_schema_id=int(parts[1]),
                        environment_type=parts[2],
                    )
                except ValueError:
                    scoped = None
            self._normalized_action_cache[signature] = scoped
        scoped = self._normalized_action_cache[signature]
        if scoped is not None:
            self._actor_action_supports[scoped] = self._actor_action_supports.get(scoped, 0.0) + 1.0
            self._actor_policy_generation += 1
        if len(occurrences) < max(2, self.config.scientific.m1n_facts_per_channel):
            occurrences.append(relation)
        self._replay_pool[relation.uid] = float(support)
        if support == 1:
            retained_parents = tuple(uid for occurrence in occurrences for uid in occurrence.provenance.parents)
            retained_evidence = tuple(uid for occurrence in occurrences for uid in occurrence.provenance.evidence)
            deferred_rows.append((
                CanonicalNode(relation.uid, MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (relation.structural_signature,), self._watermark),
                {
                    "observable_relation": relation.observable_relation,
                    "channel": relation.channel.value,
                    "structural_signature": relation.structural_signature,
                    "support": support,
                    "parents": [[uid.hi, uid.lo] for uid in retained_parents],
                    **({"semantic_before": [list(row) for row in transition.semantic_before]} if transition is not None and transition.semantic_before else {}),
                    **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition is not None and transition.semantic_action else {}),
                    **({"semantic_options": [list(row) for row in transition.semantic_options]} if transition is not None and transition.semantic_options else {}),
                    **({"semantic_after": [list(row) for row in transition.semantic_after]} if transition is not None and transition.semantic_after else {}),
                    **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition is not None and transition.semantic_delta else {}),
                },
                retained_evidence,
            ))
        else:
            self._m1n_dirty.add(signature)
        return signature

    def _advance_stage_interval_batch(self) -> Any:
        self._stage_interval_events += 1
        next_stage = self.stage_tracker.stage
        if self._stage_interval_events >= self._stage_interval_size:
            stage_snapshot = self.stage_tracker.close_interval(self._stage_evidence(), evidence_watermark=self._watermark)
            self._stage_interval_events = 0
            next_stage = stage_snapshot.next_stage
        return next_stage

    def _trim_replay_pool_batch(self) -> None:
        limit = int(self.config.scientific.replay_candidates)
        overflow = len(self._replay_pool) - limit
        if overflow <= 0:
            return
        victims = sorted(self._replay_pool, key=lambda uid: (self._replay_pool[uid], uid))[:overflow]
        for uid in victims:
            self._replay_pool.pop(uid, None)

    def record_curriculum_events_batch(self, rows: Iterable[PreparedIngestion]) -> None:
        prepared_rows = tuple(rows)
        if not prepared_rows:
            return
        with self._lock:
            counts: dict[str, int] = {}
            last_step = "none"
            last_family = ""
            last_scenario = ""
            for prepared in prepared_rows:
                transition = prepared.transition
                last_step = transition.curriculum_step or "none"
                last_family = str(prepared.identity.family)
                last_scenario = str(transition.game_scenario)
                key = f"{last_step}|{last_family}|{last_scenario}"
                counts[key] = counts.get(key, 0) + 1
            self.unified_telemetry.curriculum_counts.update(counts)
            self.unified_telemetry.gauges["curriculum_step"] = last_step
            self.unified_telemetry.gauges["environment_family"] = last_family
            self.unified_telemetry.gauges["game_scenario"] = last_scenario

    def apply_prepared_ingestion_batch(self, rows: Iterable[PreparedIngestion]) -> tuple[tuple[int, ...], ...]:
        """Apply one contiguous ordered prefix under one authoritative runtime lock."""
        prepared_rows = tuple(rows)
        if not prepared_rows:
            return ()
        if any(not isinstance(row, PreparedIngestion) for row in prepared_rows):
            raise TypeError("prepared ingestion has unexpected type")
        with self._lock:
            deferred_rows: list[tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]] = []
            results: list[tuple[int, ...]] = []
            registered_identities: dict[int, Any] = {}
            formation_environments: set[int] = set()
            modality_deltas: dict[int, int] = {}
            timeline_events_delta = 0
            timeline_actions_delta = 0
            telemetry_events_delta = 0
            last_ordering_key = self.timeline.last_ordering_key
            for prepared in prepared_rows:
                signatures: list[int] = []
                event = prepared.event
                if event is not None:
                    environment_id = int(event.identity.environment_instance_id)
                    previous_identity = registered_identities.get(environment_id)
                    if previous_identity is None:
                        identity = self.environments.register(prepared.identity)
                        if int(identity.value) != environment_id:
                            raise RuntimeError("prepared environment identity mismatch")
                        registered_identities[environment_id] = prepared.identity
                    elif previous_identity != prepared.identity:
                        raise RuntimeError("prepared environment identity collision inside batch")
                    self._watermark = max(self._watermark, int(event.identity.causal_watermark))
                    timeline_events_delta += 1
                    timeline_actions_delta += 1
                    telemetry_events_delta += 1
                    last_ordering_key = event.identity.ordering_key
                    modality = int(event.identity.modality_id.value)
                    modality_deltas[modality] = modality_deltas.get(modality, 0) + 1
                    formation_environments.add(environment_id)
                    stage_before = self.stage_tracker.stage
                    m0, m1g, m1n = prepared.m0, prepared.m1g, prepared.m1n
                    if m0 is None or m1g is None or m1n is None:
                        raise RuntimeError("prepared interaction is incomplete")
                    m0_node = CanonicalNode(m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (event.identity.event_id.hi, event.identity.event_id.lo), self._watermark)
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
                    m1g_node = CanonicalNode(m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (m1g.uid.hi, m1g.uid.lo), self._watermark)
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
                    deferred_rows.extend(((m0_node, m0_payload, (m0.uid,)), (m1g_node, m1g_payload, (m0.uid,))))
                    self._latest_interaction_grounding[(m1g.environment_instance_id, m1g.episode_id)] = m1g
                    prior_support = int(self._m1n_supports.get(int(m1n.structural_signature), 0))
                    signature = self._record_normalized_deferred_batch(m1n, deferred_rows, transition)
                    signatures.append(signature)
                    next_stage = self._advance_stage_interval_batch()
                    experience = event.experience
                    recurrence = int(self._m1n_supports.get(signature, 0))
                    recurrence_surprise = 1.0 / max(1.0, float(prior_support + 1))
                    prediction_error = abs(float(experience.prediction_error)) if float(experience.prediction_error) != 0.0 else recurrence_surprise
                    self.isf.score(
                        ISFComponents(
                            abs(experience.primary_valence),
                            abs(experience.future_option_delta),
                            prediction_error,
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
                    self._prediction_error_sum += float(prediction_error)
                    self._prediction_error_count += 1
                if prepared.symbol_codec_state:
                    codec = DeterministicSymbolCodec.from_state_dict(dict(prepared.symbol_codec_state))
                    self.symbol_codecs[codec.vocabulary_id.value] = codec
                for symbol_row in prepared.symbols:
                    symbol_event = symbol_row.event
                    self._watermark = max(self._watermark, int(symbol_event.identity.causal_watermark))
                    timeline_events_delta += 1
                    telemetry_events_delta += 1
                    modality = int(symbol_event.identity.modality_id.value)
                    modality_deltas[modality] = modality_deltas.get(modality, 0) + 1
                    formation_environments.add(int(symbol_event.identity.environment_instance_id))
                    m0, m1g, m1n = symbol_row.m0, symbol_row.m1g, symbol_row.m1n
                    m0_node = CanonicalNode(m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (symbol_event.identity.event_id.hi, symbol_event.identity.event_id.lo), self._watermark)
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
                    }
                    m1g_node = CanonicalNode(m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (m1g.uid.hi, m1g.uid.lo), self._watermark)
                    m1g_payload = {
                        "relation": m1g.relation.value,
                        "environment_instance_id": m1g.environment_instance_id,
                        "episode_id": m1g.episode_id,
                        "grounded_context_signature": m1g.grounded_context_signature,
                        "executable_action_token": m1g.executable_action_token,
                        "realized_transition_signature": m1g.realized_transition_signature,
                        "grounded_next_context_signature": m1g.grounded_next_context_signature,
                        "parents": [[m0.uid.hi, m0.uid.lo]],
                    }
                    deferred_rows.extend(((m0_node, m0_payload, (m0.uid,)), (m1g_node, m1g_payload, (m0.uid,))))
                    signatures.append(self._record_normalized_deferred_batch(m1n, deferred_rows))
                    if symbol_row.aligned_m1n is not None:
                        signatures.append(self._record_normalized_deferred_batch(symbol_row.aligned_m1n, deferred_rows))
                        if prepared.m1g is not None:
                            grounding_key = (
                                int(symbol_row.m1g.uid.lo),
                                int(prepared.m1g.uid.lo),
                                int(prepared.m1g.environment_instance_id),
                                0,
                                0,
                            )
                            before_grounding = self.grounding.states.get(grounding_key)
                            after_grounding = self.grounding.observe(
                                GroundingEvidence(
                                    int(symbol_row.m1g.uid.lo),
                                    int(prepared.m1g.uid.lo),
                                    int(prepared.m1g.environment_instance_id),
                                    0,
                                    0,
                                    int(self._watermark),
                                    recurrent_symbol=True,
                                    cross_modal_association=True,
                                )
                            )
                            if before_grounding is None or int(after_grounding.maturity) > int(before_grounding.maturity):
                                self.telemetry["grounding_promotions"] += 1
                    self._advance_stage_interval_batch()
                results.append(tuple(signatures))
            if deferred_rows:
                self._defer_base_group(tuple(deferred_rows))
            self._trim_replay_pool_batch()
            self._formation_environments.update(formation_environments)
            self.timeline.events_seen += timeline_events_delta
            self.timeline.actions_committed += timeline_actions_delta
            self.timeline.last_ordering_key = last_ordering_key
            self.telemetry["events"] += telemetry_events_delta
            for modality, count in modality_deltas.items():
                self._modality_events[modality] = self._modality_events.get(modality, 0) + count
            return tuple(results)

    @staticmethod
    def _publication_row_dependency_cost(row: tuple[CanonicalNode, dict[str, Any], tuple[Any, ...]]) -> int:
        _node, payload, _evidence = row
        parents = {
            (int(raw_parent[0]), int(raw_parent[1]))
            for raw_parent in payload.get("parents", [])
            if isinstance(raw_parent, (list, tuple)) and len(raw_parent) == 2
        }
        return 1 + len(parents)

    def _publish_derivation_rows(self, publication_rows: Iterable[tuple[CanonicalNode, dict[str, Any], tuple[Any, ...]]]) -> None:
        maximum_dependencies = int(self.config.scientific.maximum_read_set_size)
        maximum_nodes = 384
        batch: list[tuple[CanonicalNode, dict[str, Any], tuple[Any, ...]]] = []
        dependency_budget = 0

        for row in publication_rows:
            row_cost = self._publication_row_dependency_cost(row)
            if row_cost > maximum_dependencies:
                raise ValueError("single derivation row exceeds configured read set bound")
            if batch and (len(batch) >= maximum_nodes or dependency_budget + row_cost > maximum_dependencies):
                self._publish_group(tuple(batch))
                batch = []
                dependency_budget = 0
            batch.append(row)
            dependency_budget += row_cost

        if batch:
            self._publish_group(tuple(batch))

    def apply_derivation_results_batch(self, results: Iterable[DerivationResult]) -> None:
        result_rows = tuple(results)
        if not result_rows:
            return
        with self._lock:
            publication_rows: list[tuple[CanonicalNode, dict[str, Any], tuple[Any, ...]]] = []
            for result in result_rows:
                self._watermark = max(self._watermark, int(result.causal_watermark))
                family = result.family
                self._m2[family.uid] = family
                publication_rows.append((CanonicalNode(family.uid, MemoryLevel.M2, MemoryType.FAMILY, (family.structural_signature,), self._watermark), {"structural_signature": family.structural_signature, "recurrence": family.recurrence, "compression_benefit": family.compression_benefit, "parents": [[uid.hi, uid.lo] for uid in family.provenance.parents]}, family.provenance.evidence))
                for role in result.roles:
                    self._m3[role.uid] = role
                    publication_rows.append((CanonicalNode(role.uid, MemoryLevel.M3, MemoryType.ROLE, (role.relational_signature, role.consequence_signature), self._watermark), {"relational_signature": role.relational_signature, "consequence_signature": role.consequence_signature, "parents": [[uid.hi, uid.lo] for uid in role.provenance.parents]}, role.provenance.evidence))
                for candidate in result.concepts:
                    if candidate.uid in self._m4 and candidate.uid in self.graph.nodes:
                        continue
                    self._m4[candidate.uid] = candidate
                    publication_rows.append((CanonicalNode(candidate.uid, MemoryLevel.M4, MemoryType.CONCEPT, candidate.invariant_descriptor, self._watermark), {"invariant_descriptor": list(candidate.invariant_descriptor), "compression_benefit": candidate.compression_benefit, "explanatory_reach": candidate.explanatory_reach, "transfer_prior": candidate.transfer_prior, "formation_scope": list(candidate.provenance.formation_scope), "held_out_targets": [], "validated": False, "concept_state": candidate.state.value, "parents": [[uid.hi, uid.lo] for uid in candidate.provenance.parents]}, candidate.provenance.evidence))
            self._publish_derivation_rows(publication_rows)

            descriptor_rows: dict[int, dict[int, StructuralDescriptor]] = {}
            for node, payload, _evidence in publication_rows[:64]:
                if not (MemoryLevel.M2 <= node.level <= MemoryLevel.M4):
                    continue
                by_radius: dict[int, StructuralDescriptor] = {}
                key0 = int(node.structural_key[0]) if node.structural_key else 0
                components = (
                    float(int(node.level)),
                    float(int(node.memory_type)),
                    float(key0 & 0xFFFF) / 65535.0,
                    float(payload.get("recurrence", payload.get("support", 0)) or 0),
                    float(payload.get("compression_benefit", 0.0) or 0.0),
                    float(payload.get("explanatory_reach", 0) or 0),
                )
                for radius in self.config.scientific.structural_radii:
                    descriptor = StructuralDescriptor(
                        int(node.uid.lo),
                        int(self.graph.generation),
                        1,
                        int(radius),
                        1,
                        int(self.scale_statistics.estimator_generation),
                        components,
                    )
                    self.scale_statistics.observe(
                        descriptor,
                        stable_contingency_uid=int(node.uid.lo),
                        authoritative_evidence=True,
                    )
                    by_radius[int(radius)] = descriptor
                descriptor_rows[int(node.uid.lo)] = by_radius

            if len(descriptor_rows) >= 2:
                ordered = sorted(descriptor_rows)
                query_uid = ordered[0]
                candidates = {uid: descriptor_rows[uid] for uid in ordered[1:]}
                self.search_structural(
                    descriptor_rows[query_uid],
                    candidates,
                    compute_budget=max(64, int(self.config.scientific.candidates_per_radius) * 16),
                )

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["hgt_context_action_scores"] = {str(environment): {str(context): {str(action): score for action, score in actions.items()} for context, actions in contexts.items()} for environment, contexts in self._hgt_context_action_scores.items()}
        return state

    def _restore(self, snapshot: dict[str, Any]) -> None:
        state = dict(snapshot.get("state", {}))
        saved_occurrences = dict(state.get("m1n_occurrences", {}))

        if saved_occurrences:
            restore_state = dict(state)
            restore_state["m1n_occurrences"] = {}
            if "m1n_supports" not in restore_state:
                restore_state["m1n_supports"] = saved_occurrences
            restore_snapshot = dict(snapshot)
            restore_snapshot["state"] = restore_state
            super()._restore(restore_snapshot)

            normalized_by_signature: dict[int, MemoryUid] = {}
            for uid, node in self.graph.nodes.items():
                if node.memory_type is not MemoryType.NORMALIZED_RELATION:
                    continue
                payload = self.graph.payloads.get(uid, {})
                raw_signature = payload.get("structural_signature")
                if raw_signature is None:
                    continue
                signature = int(raw_signature)
                current = normalized_by_signature.get(signature)
                if current is None or uid < current:
                    normalized_by_signature[signature] = uid

            self._m1n_occurrences = {}
            for raw_signature, count in saved_occurrences.items():
                signature = int(raw_signature)
                uid = normalized_by_signature.get(signature)
                if uid is None:
                    continue
                payload = self.graph.payloads[uid]
                evidence_refs = tuple(
                    MemoryUid(int(raw[0]), int(raw[1]))
                    for raw in payload.get("evidence_refs", [[uid.hi, uid.lo]])
                )
                dummy = M1NormalizedRelation(
                    uid,
                    str(payload["observable_relation"]),
                    NormalizedChannel(str(payload["channel"])),
                    signature,
                    DerivationProvenance((uid,), evidence_refs),
                )
                self._m1n_occurrences[signature] = [dummy] if int(count) > 0 else []

            self._actor_action_supports = {}
            for signature, support in self._m1n_supports.items():
                rows = self._m1n_occurrences.get(int(signature), ())
                if not rows:
                    continue
                observable = str(rows[0].observable_relation)
                parts = observable.split(":")
                if len(parts) < 4 or parts[0] != "ACTION":
                    continue
                try:
                    scoped = scoped_action_key(
                        int(parts[3]),
                        action_schema_id=int(parts[1]),
                        environment_type=parts[2],
                    )
                except ValueError:
                    continue
                self._actor_action_supports[scoped] = self._actor_action_supports.get(scoped, 0.0) + float(support)
            self._actor_policy_generation = self.graph.generation
        else:
            super()._restore(snapshot)

        self._hgt_context_action_scores = {int(environment): {int(context): {int(action): float(score) for action, score in dict(actions).items()} for context, actions in dict(contexts).items()} for environment, contexts in dict(state.get("hgt_context_action_scores", {})).items()}
