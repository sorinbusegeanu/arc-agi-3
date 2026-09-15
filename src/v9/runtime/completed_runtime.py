from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

from v9.cognition.grounding import GroundingEvidence, GroundingMaturity
from v9.memory.identity import MemoryUid
from v9.memory.m0_episode import M0Episode
from v9.memory.m1_grounded import GroundedRelation, M1GroundedContingency
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge, RelationType
from v9.memory.symbolic_relations import derive_symbolic_relations, shuffled_alignment_control
from v9.modalities.contract import PassiveSymbolEvent
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

    def _derive_prepared_symbolic_relations(self, prepared: Any, previous_interaction: Any) -> tuple[int, ...]:
        rows = tuple(getattr(prepared, "symbols", ()) or ())
        if not rows:
            return ()
        transition = prepared.transition
        current = getattr(prepared, "m1g", None)
        watermark = max(int(row.event.identity.causal_watermark) for row in rows)
        derived = list(
            derive_symbolic_relations(
                rows,
                interaction_grounding=current,
                previous_interaction_grounding=previous_interaction,
                transition=transition,
                causal_watermark=watermark,
            )
        )
        if current is not None and len(rows) >= 2:
            for row in reversed(rows):
                derived.append(shuffled_alignment_control(row, current, causal_watermark=watermark))
        signatures: list[int] = []
        for item in derived:
            signatures.append(self._record_normalized(item.relation, defer_publication=True, payload_extra=item.payload()))
        return tuple(signatures)

    def apply_prepared_ingestion_batch(self, rows: Iterable[Any]) -> tuple[tuple[int, ...], ...]:
        prepared_rows = tuple(rows)
        if not prepared_rows:
            return ()
        previous = self._previous_interactions(prepared_rows)
        base_results = super().apply_prepared_ingestion_batch(prepared_rows)
        completed: list[tuple[int, ...]] = []
        with self._lock:
            for prepared, base in zip(prepared_rows, base_results):
                m1g = getattr(prepared, "m1g", None)
                key = None if m1g is None else (int(m1g.environment_instance_id), int(m1g.episode_id))
                extra = self._derive_prepared_symbolic_relations(prepared, None if key is None else previous.get(key))
                completed.append(tuple(base) + tuple(extra))
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
        transition = SimpleNamespace(
            before_signature=0,
            after_signature=0,
            boundary_scope="NONE",
            task_success=False,
            task_failure=False,
            task_truncated=False,
            levels_completed=0,
            primary_valence=0,
        )
        for item in derive_symbolic_relations(
            relation_rows,
            interaction_grounding=None,
            previous_interaction_grounding=latest_interaction,
            transition=transition,
            causal_watermark=int(event.identity.causal_watermark),
        ):
            signatures.append(self._record_normalized(item.relation, payload_extra=item.payload()))
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
            wanted = {int(value) for value in signatures}
            for family in self._m2.values():
                if wanted and int(family.structural_signature) not in wanted:
                    continue
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
            evidence_refs=(symbol_uid, interaction_uid),
            causal_watermark=self._watermark,
            writes=(MutationWrite(edge=edge),),
            proposal_class=ProposalClass.STATEFUL,
        )
        result = self.graph.publish(proposal)
        return result.outcome.value == "ACCEPTED"

    def record_symbolic_validation(
        self,
        *,
        symbol_structure_uid: int,
        interaction_structure_uid: int,
        environment_instance_id: int,
        validation_kind: str,
        trial_id: str,
        effect: float,
        target_environment_id: int | None = None,
        context_scope_id: int = 0,
        lineage_uid: int = 0,
    ):
        allowed = {"prediction", "generalization", "heldout_transfer", "composition", "symbol_mediated_learning"}
        if validation_kind not in allowed:
            raise ValueError(f"unsupported symbolic validation kind: {validation_kind}")
        positive = float(effect) > 0.0
        evidence = GroundingEvidence(
            int(symbol_structure_uid), int(interaction_structure_uid), int(environment_instance_id), int(context_scope_id), int(lineage_uid), int(self._watermark),
            recurrent_symbol=True,
            cross_modal_association=True,
            prospective_prediction=True,
            heldout_transfer=validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
            novel_composition=validation_kind == "composition",
            symbol_mediated_learning=validation_kind == "symbol_mediated_learning",
            validation_trial_id=str(trial_id),
            support=max(0.0, float(effect)) if positive else abs(float(effect)),
            contradiction=0.0 if positive else abs(float(effect)),
            positive=positive,
        )
        with self._lock:
            key = (int(symbol_structure_uid), int(interaction_structure_uid), int(environment_instance_id), int(context_scope_id), int(lineage_uid))
            before = self.grounding.states.get(key)
            state = self.grounding.observe(evidence)
            if before is None or int(state.maturity) > int(before.maturity):
                self.telemetry["grounding_promotions"] += 1
            if state.suspended and (before is None or not before.suspended):
                self.telemetry["grounding_suspensions"] += 1
            if state.behavior_eligible and positive:
                if validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"}:
                    self._publish_validated_grounding_edge(symbol_structure_uid, interaction_structure_uid, heldout=True)
                self._publish_validated_grounding_edge(symbol_structure_uid, interaction_structure_uid, heldout=False)
            self.evidence.append(
                "SYMBOLIC_GROUNDING_VALIDATION",
                self._watermark,
                {
                    "symbol_structure_uid": int(symbol_structure_uid),
                    "interaction_structure_uid": int(interaction_structure_uid),
                    "source_environment_id": int(environment_instance_id),
                    "target_environment_id": None if target_environment_id is None else int(target_environment_id),
                    "validation_kind": validation_kind,
                    "trial_id": str(trial_id),
                    "effect": float(effect),
                    "positive": positive,
                    "maturity": int(state.maturity),
                    "behavior_eligible": bool(state.behavior_eligible),
                    "support": float(state.support),
                    "contradiction": float(state.contradiction),
                },
            )
            return state

    def retrieve_structural_candidates(self, keys, *, limit: int | None = None):
        rows = tuple(super().retrieve_structural_candidates(keys, limit=limit))
        grounded = {value for key, state in self.grounding.eligible_states() for value in (int(key[0]), int(key[1])) if state.behavior_eligible}
        return tuple(sorted(rows, key=lambda uid: (int(uid.lo) not in grounded, uid)))

    def metrics(self) -> dict[str, Any]:
        result = dict(super().metrics())
        symbol_nodes: list[MemoryUid] = []
        symbol_m1n = 0
        cross_modal = 0
        windows: set[tuple[int, int, int]] = set()
        unique_symbols: set[tuple[int, int, int]] = set()
        for uid, node in self.graph.nodes.items():
            payload = self.graph.payloads.get(uid, {})
            identity = payload.get("symbol_identity")
            if node.level is MemoryLevel.M0 and identity is not None:
                symbol_nodes.append(uid)
                if isinstance(identity, (list, tuple)) and len(identity) >= 3:
                    unique_symbols.add(tuple(int(value) for value in identity[:3]))
                windows.add((int(payload.get("environment_instance_id", 0)), int(payload.get("episode_id", 0)), int(payload.get("symbol_source_step", node.created_watermark))))
            if node.memory_type is MemoryType.NORMALIZED_RELATION:
                channel = str(payload.get("channel", ""))
                symbol_m1n += int(channel == "SYMBOL")
                cross_modal += int(channel == "CROSS_MODAL")
        transfer_trials = sum(len(state.validation_trial_ids) for state in self.grounding.states.values() if state.maturity >= GroundingMaturity.G3)
        compositions = sum(int(state.maturity >= GroundingMaturity.G4 and state.behavior_eligible) for state in self.grounding.states.values())
        result.update({
            "symbol_occurrences": len(symbol_nodes),
            "unique_symbols": len(unique_symbols),
            "symbol_windows": len(windows),
            "symbolic_M0_count": len(symbol_nodes),
            "symbolic_M1N_count": symbol_m1n,
            "cross_modal_correspondences": cross_modal,
            "symbol_transfer_trials": transfer_trials,
            "symbol_composition_successes": compositions,
        })
        diagnostic = dict(self.unified_telemetry.diagnostic_metrics())
        result["primary_dashboard"] = build_primary_dashboard(result, diagnostic)
        return result

    def dashboard_metrics(self) -> dict[str, Any]:
        return self.metrics()
