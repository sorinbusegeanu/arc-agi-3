from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Iterable

from v9.cognition.grounding import GroundingEvidence, GroundingMaturity
from v9.memory.identity import EpisodeId, EventUid, MemoryUid, SymbolId, SymbolStreamId, SymbolVocabularyId
from v9.memory.m0_episode import M0Episode
from v9.memory.m1_grounded import GroundedRelation, M1GroundedContingency
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge, RelationType
from v9.memory.symbolic_relations import derive_symbolic_relations, shuffled_alignment_control
from v9.modalities.contract import PassiveSymbolEvent
from v9.modalities.symbols import SymbolOccurrence
from v9.mutation.proposals import MutationKind, MutationProposal, MutationWrite, ProposalClass
from v9.mutation.read_sets import ReadDependency, ReadSet
from v9.telemetry import build_primary_dashboard

from . import ContinuousMemoryRuntime as _PublicContinuousMemoryRuntime
from .publication import edge_ref


class CompletedContinuousMemoryRuntime(_PublicContinuousMemoryRuntime):
    """v9.7.8 completion layer for symbolic/cross-modal grounding."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._direct_symbol_rows: dict[tuple[int, int], list[Any]] = {}
        self._symbol_occurrences_by_id: dict[tuple[int, int], SymbolOccurrence] = {}
        self._symbol_occurrences_by_symbol: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._m1n_family_occurrences: dict[int, list[Any]] = getattr(self, "_m1n_family_occurrences", {})
        self._shared_family_support: dict[int, int] = {}
        self._rebuild_symbol_occurrence_index()
        for rows in self._m1n_occurrences.values():
            for row in rows:
                self._register_family_relation(row)

    @staticmethod
    def _occurrence_from_payload(payload: dict[str, Any]) -> SymbolOccurrence | None:
        raw_id = payload.get("symbol_occurrence_id")
        raw_provenance = payload.get("symbol_provenance_id")
        if not isinstance(raw_id, (list, tuple)) or len(raw_id) != 2:
            return None
        if not isinstance(raw_provenance, (list, tuple)) or len(raw_provenance) != 2:
            return None
        required = ("symbol_id", "symbol_position", "symbol_stream_id", "symbol_vocabulary_id", "symbol_codec_id", "symbol_causal_watermark", "symbol_macro_step", "symbol_micro_step", "symbol_environment_instance_id", "symbol_episode_id")
        if any(key not in payload for key in required):
            return None
        return SymbolOccurrence(
            EventUid(int(raw_id[0]), int(raw_id[1])),
            SymbolId(int(payload["symbol_id"])),
            int(payload["symbol_position"]),
            SymbolStreamId(int(payload["symbol_stream_id"])),
            SymbolVocabularyId(int(payload["symbol_vocabulary_id"])),
            str(payload["symbol_codec_id"]),
            int(payload["symbol_causal_watermark"]),
            int(payload["symbol_macro_step"]),
            int(payload["symbol_micro_step"]),
            int(payload["symbol_environment_instance_id"]),
            EpisodeId(int(payload["symbol_episode_id"])),
            EventUid(int(raw_provenance[0]), int(raw_provenance[1])),
            int(payload.get("symbol_modality_id", 2)),
            str(payload.get("symbol_temporal_phase", "COINCIDENT")),
            int(payload.get("symbol_source_sequence", 0)),
        )

    def _index_symbol_occurrences(self, rows: Iterable[SymbolOccurrence]) -> None:
        for row in rows:
            key = (int(row.occurrence_id.hi), int(row.occurrence_id.lo))
            if key in self._symbol_occurrences_by_id:
                continue
            self._symbol_occurrences_by_id[key] = row
            symbol_key = (int(row.vocabulary_id.value), int(row.symbol_id.value))
            self._symbol_occurrences_by_symbol.setdefault(symbol_key, []).append(key)

    def _rebuild_symbol_occurrence_index(self) -> None:
        rows = []
        for uid, node in self.graph.nodes.items():
            if node.level is not MemoryLevel.M0:
                continue
            occurrence = self._occurrence_from_payload(self.graph.payloads.get(uid, {}))
            if occurrence is not None:
                rows.append(occurrence)
        self._index_symbol_occurrences(rows)

    def symbol_occurrences(
        self,
        *,
        vocabulary_id: int | None = None,
        symbol_id: int | None = None,
        environment_instance_id: int | None = None,
        episode_id: int | None = None,
        temporal_phase: str | None = None,
        macro_step_start: int | None = None,
        macro_step_end: int | None = None,
    ) -> tuple[SymbolOccurrence, ...]:
        if vocabulary_id is not None and symbol_id is not None:
            keys = tuple(self._symbol_occurrences_by_symbol.get((int(vocabulary_id), int(symbol_id)), ()))
            rows = [self._symbol_occurrences_by_id[key] for key in keys]
        else:
            rows = list(self._symbol_occurrences_by_id.values())
        rows = [
            row for row in rows
            if (vocabulary_id is None or int(row.vocabulary_id.value) == int(vocabulary_id))
            and (symbol_id is None or int(row.symbol_id.value) == int(symbol_id))
            and (environment_instance_id is None or int(row.environment_instance_id) == int(environment_instance_id))
            and (episode_id is None or int(row.episode_id.value) == int(episode_id))
            and (temporal_phase is None or str(row.temporal_phase) == str(temporal_phase))
            and (macro_step_start is None or int(row.macro_step) >= int(macro_step_start))
            and (macro_step_end is None or int(row.macro_step) <= int(macro_step_end))
        ]
        return tuple(sorted(rows, key=lambda row: (row.causal_watermark, row.macro_step, row.micro_step, row.occurrence_id)))

    def occurrence_provenance(self, occurrence_id: EventUid) -> SymbolOccurrence | None:
        return self._symbol_occurrences_by_id.get((int(occurrence_id.hi), int(occurrence_id.lo)))

    def _persist_prepared_occurrences(self, prepared: Any) -> None:
        self._index_symbol_occurrences(getattr(prepared, "symbol_occurrences", ()))
        durable_updates: dict[MemoryUid, dict[str, Any]] = {}
        for symbol, occurrence in zip(getattr(prepared, "symbols", ()), getattr(prepared, "symbol_occurrences", ())):
            payload = {
                "symbol_occurrence_id": [int(occurrence.occurrence_id.hi), int(occurrence.occurrence_id.lo)],
                "symbol_id": int(occurrence.symbol_id.value),
                "symbol_position": int(occurrence.position),
                "symbol_stream_id": int(occurrence.stream_id.value),
                "symbol_vocabulary_id": int(occurrence.vocabulary_id.value),
                "symbol_codec_id": str(occurrence.codec_id),
                "symbol_causal_watermark": int(occurrence.causal_watermark),
                "symbol_macro_step": int(occurrence.macro_step),
                "symbol_micro_step": int(occurrence.micro_step),
                "symbol_environment_instance_id": int(occurrence.environment_instance_id),
                "symbol_episode_id": int(occurrence.episode_id.value),
                "symbol_provenance_id": [int(occurrence.provenance_id.hi), int(occurrence.provenance_id.lo)],
                "symbol_modality_id": int(occurrence.modality_id),
                "symbol_temporal_phase": str(occurrence.temporal_phase),
                "symbol_source_sequence": int(occurrence.source_sequence),
            }
            uid = symbol.m0.uid
            if uid in self.graph.payloads:
                if hasattr(self, "canonical_store"):
                    durable_updates[uid] = payload
                else:
                    self.graph.payloads[uid].update(payload)
            elif uid in self._deferred_base_nodes:
                node, existing, evidence = self._deferred_base_nodes[uid]
                merged = dict(existing)
                merged.update(payload)
                self._deferred_base_nodes[uid] = (node, merged, evidence)
        if durable_updates:
            self.replace_canonical_payloads(durable_updates)

    def _register_family_relation(self, relation: Any) -> None:
        family = int(getattr(relation, "family_signature", 0) or relation.structural_signature)
        bucket = self._m1n_family_occurrences.setdefault(family, [])
        identity = (relation.uid, relation.channel.value, tuple(relation.provenance.evidence))
        if all((row.uid, row.channel.value, tuple(row.provenance.evidence)) != identity for row in bucket):
            bucket.append(relation)
            limit = max(4, int(self.config.scientific.m1n_facts_per_channel) * 4)
            if len(bucket) > limit:
                del bucket[:-limit]

    def _develop_shared_families(self, family_ids: set[int]) -> None:
        from .memory_pipeline import DerivationTask, derive_memory
        for family_id in sorted(family_ids):
            rows = tuple(self._m1n_family_occurrences.get(int(family_id), ()))
            channels = {row.channel.value for row in rows}
            evidence = {uid for row in rows for uid in row.provenance.evidence}
            if len(rows) < 2 or len(evidence) < 2 or "WORLD" not in channels or not ({"SYMBOL", "CROSS_MODAL"} & channels):
                continue
            support = len(rows)
            if support <= int(self._shared_family_support.get(int(family_id), 0)):
                continue
            result = derive_memory(DerivationTask(0, int(family_id), rows, support, tuple(sorted(self._formation_environments)), int(self._watermark)))
            super().apply_derivation_results_batch((result,))
            self._shared_family_support[int(family_id)] = support

    def _previous_interactions(self, prepared_rows: tuple[Any, ...]) -> dict[tuple[int, int], Any]:
        previous: dict[tuple[int, int], Any] = {}
        with self._lock:
            for prepared in prepared_rows:
                m1g = getattr(prepared, "m1g", None)
                if m1g is None:
                    continue
                key = (int(m1g.environment_instance_id), int(m1g.episode_id))
                previous[key] = self._latest_interaction_grounding.get(key)
        return previous

    def _derive_prepared_symbolic_relations(self, prepared: Any, previous_interaction: Any) -> tuple[tuple[int, ...], set[int]]:
        rows = tuple(getattr(prepared, "symbols", ()) or ())
        if not rows:
            return (), set()
        transition = prepared.transition
        current = getattr(prepared, "m1g", None)
        watermark = max(int(row.event.identity.causal_watermark) for row in rows)
        derived = list(derive_symbolic_relations(rows, interaction_grounding=current, previous_interaction_grounding=previous_interaction, transition=transition, causal_watermark=watermark, occurrences=getattr(prepared, "symbol_occurrences", ())))
        if previous_interaction is not None:
            for index, row in enumerate(rows):
                occurrence = prepared.symbol_occurrences[index] if index < len(prepared.symbol_occurrences) else None
                derived.append(shuffled_alignment_control(row, previous_interaction, causal_watermark=watermark, occurrence=occurrence))
        signatures: list[int] = []
        families: set[int] = set()
        for item in derived:
            signatures.append(self._record_normalized(item.relation, defer_publication=True, payload_extra=item.payload()))
            self._register_family_relation(item.relation)
            families.add(int(item.relation.family_signature or item.relation.structural_signature))
        return tuple(signatures), families

    def apply_prepared_ingestion_batch(self, rows: Iterable[Any]) -> tuple[tuple[int, ...], ...]:
        prepared_rows = tuple(rows)
        if not prepared_rows:
            return ()
        previous = self._previous_interactions(prepared_rows)
        base_results = super().apply_prepared_ingestion_batch(prepared_rows)
        completed: list[tuple[int, ...]] = []
        family_ids: set[int] = set()
        with self._lock:
            for prepared, base in zip(prepared_rows, base_results):
                self._persist_prepared_occurrences(prepared)
                for relation in (getattr(prepared, "m1n", None),):
                    if relation is not None:
                        self._register_family_relation(relation)
                        family_ids.add(int(relation.family_signature or relation.structural_signature))
                for symbol in prepared.symbols:
                    self._register_family_relation(symbol.m1n)
                    family_ids.add(int(symbol.m1n.family_signature or symbol.m1n.structural_signature))
                    if symbol.aligned_m1n is not None:
                        self._register_family_relation(symbol.aligned_m1n)
                        family_ids.add(int(symbol.aligned_m1n.family_signature or symbol.aligned_m1n.structural_signature))
                m1g = getattr(prepared, "m1g", None)
                key = None if m1g is None else (int(m1g.environment_instance_id), int(m1g.episode_id))
                extra, extra_families = self._derive_prepared_symbolic_relations(prepared, None if key is None else previous.get(key))
                family_ids.update(extra_families)
                completed.append(tuple(base) + tuple(extra))
            self._develop_shared_families(family_ids)
        return tuple(completed)

    def apply_prepared_ingestion(self, prepared: Any) -> tuple[int, ...]:
        rows = self.apply_prepared_ingestion_batch((prepared,))
        return rows[0] if rows else ()

    def _ingest(self, event: Any) -> tuple[int, ...]:
        if not isinstance(event, PassiveSymbolEvent):
            return super()._ingest(event)
        key = (int(event.identity.environment_instance_id), int(event.identity.episode_id.value))
        latest_interaction = self._latest_interaction_grounding.get(key)
        signatures = list(super()._ingest(event))
        m0 = M0Episode.from_event(event, context_signature=0, payload_digest=self._payload_digest(event))
        m1g = M1GroundedContingency.build(GroundedRelation.SYMBOL_OCCURRED, (m0,))
        current = SimpleNamespace(m0=m0, m1g=m1g, event=event)
        history = self._direct_symbol_rows.setdefault(key, [])
        relation_rows = tuple(history[-1:] + [current])
        transition = SimpleNamespace(before_signature=0, after_signature=0, boundary_scope="NONE", task_success=False, task_failure=False, task_truncated=False, levels_completed=0, primary_valence=0)
        for item in derive_symbolic_relations(relation_rows, interaction_grounding=None, previous_interaction_grounding=latest_interaction, transition=transition, causal_watermark=int(event.identity.causal_watermark)):
            signatures.append(self._record_normalized(item.relation, payload_extra=item.payload()))
            self._register_family_relation(item.relation)
        history.append(current)
        del history[:-8]
        return tuple(signatures)

    def apply_derivation_results_batch(self, results: Iterable[Any]) -> None:
        result_rows = tuple(results)
        super().apply_derivation_results_batch(result_rows)
        with self._lock:
            for result in result_rows:
                family = result.family
                payload = self.graph.payloads.get(family.uid)
                if payload is not None:
                    payload["modality_support"] = [list(row) for row in family.modality_support]
                    payload["support_decomposition"] = [list(row) for row in family.support_decomposition]
                for role in result.roles:
                    role_payload = self.graph.payloads.get(role.uid)
                    if role_payload is not None:
                        role_payload["support_decomposition"] = [list(row) for row in role.support_decomposition]

    def _develop(self, signatures: tuple[int, ...] = ()) -> None:
        super()._develop(signatures)
        with self._lock:
            for family in self._m2.values():
                payload = self.graph.payloads.get(family.uid)
                if payload is not None:
                    payload["modality_support"] = [list(row) for row in family.modality_support]
                    payload["support_decomposition"] = [list(row) for row in family.support_decomposition]
            for role in self._m3.values():
                payload = self.graph.payloads.get(role.uid)
                if payload is not None:
                    payload["support_decomposition"] = [list(row) for row in role.support_decomposition]

    def _uid_by_low(self, low: int) -> MemoryUid | None:
        candidates = [uid for uid in self.graph.nodes if int(uid.lo) == int(low)]
        return min(candidates) if candidates else None

    def _publish_validated_grounding_edge(self, symbol_structure_uid: int, interaction_structure_uid: int, *, heldout: bool) -> bool:
        symbol_uid = self._uid_by_low(symbol_structure_uid)
        interaction_uid = self._uid_by_low(interaction_structure_uid)
        if symbol_uid is None or interaction_uid is None:
            return False
        relation = RelationType.TRANSFER_VALIDATES if heldout else RelationType.GROUNDS
        edge = RelationEdge(symbol_uid, relation, interaction_uid, (symbol_uid, interaction_uid))
        ref = edge_ref(edge)
        proposal = MutationProposal.build(
            MutationKind.UPSERT_EDGE,
            target_partitions=tuple(sorted({self.partitions.owner(symbol_uid), self.partitions.owner(interaction_uid)})),
            read_set=ReadSet.build((ReadDependency(ref, self.graph.versions.get(ref)),), maximum_size=self.config.scientific.maximum_read_set_size),
            evidence_refs=(symbol_uid, interaction_uid), causal_watermark=self._watermark,
            writes=(MutationWrite(edge=edge),), proposal_class=ProposalClass.STATEFUL,
        )
        result = self.graph.publish(proposal)
        return result.outcome.value == "ACCEPTED"

    def _propagate_grounding_authority(self, interaction_structure_uid: int, state: Any, *, validation_kind: str, trial_id: str) -> float:
        authority = 0.0 if not state.behavior_eligible else max(0.0, min(1.0, (state.support / max(1e-9, state.support + state.contradiction)) * ((int(state.maturity) - 2) / 3.0)))
        root = self._uid_by_low(interaction_structure_uid)
        if root is None:
            return authority
        metadata = {
            "grounding_authority": float(authority),
            "grounding_maturity": int(state.maturity),
            "grounding_active": bool(state.behavior_eligible),
            "prospective_prediction": int(state.maturity) >= int(GroundingMaturity.G3),
            "heldout_transfer": validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
            "novel_composition": validation_kind == "composition",
            "symbol_mediated_learning": validation_kind == "symbol_mediated_learning",
            "grounding_confidence": float(state.support / max(1e-9, state.support + state.contradiction)),
            "grounding_validation_trial_id": str(trial_id),
        }
        durable_updates: dict[MemoryUid, dict[str, Any]] = {}
        frontier = {root}
        if root in self.graph.payloads:
            if hasattr(self, "canonical_store"):
                durable_updates[root] = metadata
            else:
                self.graph.payloads[root].update(metadata)
        for level in (MemoryLevel.M2, MemoryLevel.M3, MemoryLevel.M4, MemoryLevel.M5, MemoryLevel.M6, MemoryLevel.M7):
            next_frontier: set[MemoryUid] = set()
            for uid in self.graph.uids_at_level(level):
                payload = self.graph.payloads.get(uid, {})
                parents = {MemoryUid(int(raw[0]), int(raw[1])) for raw in payload.get("parents", ()) if isinstance(raw, (list, tuple)) and len(raw) == 2}
                if parents & frontier:
                    if hasattr(self, "canonical_store"):
                        durable_updates[uid] = metadata
                    else:
                        payload.update(metadata)
                    next_frontier.add(uid)
            frontier.update(next_frontier)
        if durable_updates:
            self.replace_canonical_payloads(durable_updates)
        self._actor_policy_generation += 1
        return authority

    def grounding_prediction_score(self, interaction_structure_uid: int) -> float:
        values = []
        for key, state in self.grounding.eligible_states():
            if int(key[1]) != int(interaction_structure_uid):
                continue
            values.append(state.support / max(1e-9, state.support + state.contradiction))
        return max(values, default=0.0)

    def grounding_future_option_bonus(self, interaction_structure_uid: int) -> float:
        return 0.25 * self.grounding_prediction_score(interaction_structure_uid)

    def grounding_deliberation_scores(self, candidates: Iterable[MemoryUid]) -> dict[MemoryUid, float]:
        return {uid: float(self.graph.payloads.get(uid, {}).get("grounding_authority", 0.0)) for uid in candidates}

    def record_symbolic_validation(self, *, symbol_structure_uid: int, interaction_structure_uid: int, environment_instance_id: int, validation_kind: str, trial_id: str, effect: float, target_environment_id: int | None = None, context_scope_id: int = 0, lineage_uid: int = 0):
        allowed = {"prediction", "generalization", "heldout_transfer", "composition", "symbol_mediated_learning"}
        if validation_kind not in allowed:
            raise ValueError(f"unsupported symbolic validation kind: {validation_kind}")
        positive = float(effect) > 0.0
        evidence = GroundingEvidence(
            int(symbol_structure_uid), int(interaction_structure_uid), int(environment_instance_id), int(context_scope_id), int(lineage_uid), int(self._watermark),
            recurrent_symbol=True, cross_modal_association=True, prospective_prediction=True,
            heldout_transfer=validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
            novel_composition=validation_kind == "composition", symbol_mediated_learning=validation_kind == "symbol_mediated_learning",
            validation_trial_id=str(trial_id), support=max(0.0, float(effect)) if positive else abs(float(effect)), contradiction=0.0 if positive else abs(float(effect)), positive=positive,
        )
        with self._lock:
            key = (int(symbol_structure_uid), int(interaction_structure_uid), int(environment_instance_id), int(context_scope_id), int(lineage_uid))
            before = self.grounding.states.get(key)
            state = self.grounding.observe(evidence)
            if before is None or int(state.maturity) > int(before.maturity):
                self.telemetry["grounding_promotions"] += 1
            if state.suspended and (before is None or not before.suspended):
                self.telemetry["grounding_suspensions"] += 1
            authority = self._propagate_grounding_authority(interaction_structure_uid, state, validation_kind=validation_kind, trial_id=trial_id)
            if state.behavior_eligible and positive:
                if validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"}:
                    self._publish_validated_grounding_edge(symbol_structure_uid, interaction_structure_uid, heldout=True)
                self._publish_validated_grounding_edge(symbol_structure_uid, interaction_structure_uid, heldout=False)
            self.evidence.append("SYMBOLIC_GROUNDING_VALIDATION", self._watermark, {
                "symbol_structure_uid": int(symbol_structure_uid), "interaction_structure_uid": int(interaction_structure_uid),
                "source_environment_id": int(environment_instance_id), "target_environment_id": None if target_environment_id is None else int(target_environment_id),
                "validation_kind": validation_kind, "trial_id": str(trial_id), "effect": float(effect), "positive": positive,
                "maturity": int(state.maturity), "behavior_eligible": bool(state.behavior_eligible), "grounding_authority": float(authority),
                "support": float(state.support), "contradiction": float(state.contradiction),
            })
            return state

    def retrieve_structural_candidates(self, keys, *, limit: int | None = None):
        rows = tuple(super().retrieve_structural_candidates(keys, limit=limit))
        grounded = {value for key, state in self.grounding.eligible_states() for value in (int(key[0]), int(key[1])) if state.behavior_eligible}
        return tuple(sorted(rows, key=lambda uid: (int(uid.lo) not in grounded, -float(self.graph.payloads.get(uid, {}).get("grounding_authority", 0.0)), uid)))

    def actor_policy_snapshot(self):
        snapshot = super().actor_policy_snapshot()
        strategies = {
            environment: tuple(replace(row, grounding_authority=float(self.graph.payloads.get(row.strategy_uid, {}).get("grounding_authority", 0.0))) for row in rows)
            for environment, rows in snapshot.strategies_by_environment.items()
        }
        outcomes = {
            environment: tuple(replace(row, grounding_authority=float(self.graph.payloads.get(row.outcome_uid, {}).get("grounding_authority", 0.0))) for row in rows)
            for environment, rows in snapshot.outcomes_by_environment.items()
        }
        return replace(snapshot, strategies_by_environment=strategies, outcomes_by_environment=outcomes)

    def metrics(self) -> dict[str, Any]:
        result = dict(super().metrics())
        symbol_nodes: list[MemoryUid] = []
        symbol_m1n = 0
        cross_modal = 0
        candidate_correspondences = 0
        validated_correspondences = 0
        windows: set[tuple[int, int, int]] = set()
        unique_symbols: set[tuple[int, int]] = set()
        phase_counts: dict[str, int] = {}
        for uid, node in self.graph.nodes.items():
            payload = self.graph.payloads.get(uid, {})
            identity = payload.get("symbol_identity")
            if node.level is MemoryLevel.M0 and identity is not None:
                symbol_nodes.append(uid)
                if isinstance(identity, (list, tuple)) and len(identity) >= 3:
                    unique_symbols.add((int(identity[0]), int(identity[2])))
                windows.add((int(payload.get("environment_instance_id", 0)), int(payload.get("episode_id", 0)), int(payload.get("symbol_macro_step", payload.get("symbol_source_step", node.created_watermark)))))
                phase = str(payload.get("symbol_temporal_phase", "COINCIDENT"))
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if node.memory_type is MemoryType.NORMALIZED_RELATION:
                channel = str(payload.get("channel", ""))
                symbol_m1n += int(channel == "SYMBOL")
                cross_modal += int(channel == "CROSS_MODAL")
                candidate_correspondences += int(channel == "CROSS_MODAL")
                validated_correspondences += int(bool(payload.get("heldout_transfer", False)) or bool(payload.get("grounding_active", False)))
        transfer_trials = sum(len(state.validation_trial_ids) for state in self.grounding.states.values() if state.maturity >= GroundingMaturity.G3)
        compositions = sum(int(state.maturity >= GroundingMaturity.G4 and state.behavior_eligible) for state in self.grounding.states.values())
        result.update({
            "symbol_occurrences": len(self._symbol_occurrences_by_id) or len(symbol_nodes), "unique_symbols": len(unique_symbols), "symbol_windows": len(windows),
            "symbolic_M0_count": len(symbol_nodes), "symbolic_M1N_count": symbol_m1n, "cross_modal_correspondences": cross_modal,
            "candidate_correspondences": candidate_correspondences, "validated_correspondences": validated_correspondences,
            "symbol_transfer_trials": transfer_trials, "symbol_composition_successes": compositions,
            "interaction_to_symbol_prediction_gain": float(self._symbol_prediction_delta_sum),
            **{f"symbol_phase_{phase}": count for phase, count in phase_counts.items()},
        })
        diagnostic = dict(self.unified_telemetry.diagnostic_metrics())
        result["primary_dashboard"] = build_primary_dashboard(result, diagnostic)
        return result

    def dashboard_metrics(self) -> dict[str, Any]:
        return self.metrics()
