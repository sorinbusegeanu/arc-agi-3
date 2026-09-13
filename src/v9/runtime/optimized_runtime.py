from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.memory_pipeline import DerivationResult, PreparedIngestion, PreparedSymbolIngestion
from v9.runtime.runtime import ContinuousMemoryRuntime as BaseContinuousMemoryRuntime


class ContinuousMemoryRuntime(BaseContinuousMemoryRuntime):
    """Performance-oriented v9 runtime preserving the canonical ordered boundary."""

    def __init__(self, config: Any) -> None:
        self._hgt_context_action_scores: dict[int, dict[int, dict[int, float]]] = {}
        super().__init__(config)

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
        with self._lock:
            return ActorPolicySnapshot.build(
                generation=max(self.graph.generation, self._actor_policy_generation),
                normalized_action_supports=self._actor_action_supports,
                hgt_action_scores=self._hgt_action_scores,
                hgt_context_action_scores=self._hgt_context_action_scores,
                hgt_action_scores_by_type=self._hgt_scores_by_environment_type(),
                model_version=self.unified_telemetry.model_version,
            )

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

    def apply_prepared_ingestion_batch(self, rows: Iterable[PreparedIngestion]) -> tuple[tuple[int, ...], ...]:
        prepared_rows = tuple(rows)
        if not prepared_rows:
            return ()
        with self._lock:
            return tuple(self.apply_prepared_ingestion(row) for row in prepared_rows)

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
                    if candidate.uid in self._m4:
                        continue
                    self._m4[candidate.uid] = candidate
                    publication_rows.append((CanonicalNode(candidate.uid, MemoryLevel.M4, MemoryType.CONCEPT, candidate.invariant_descriptor, self._watermark), {"invariant_descriptor": list(candidate.invariant_descriptor), "compression_benefit": candidate.compression_benefit, "explanatory_reach": candidate.explanatory_reach, "transfer_prior": candidate.transfer_prior, "formation_scope": list(candidate.provenance.formation_scope), "held_out_targets": [], "validated": False, "concept_state": candidate.state.value, "parents": [[uid.hi, uid.lo] for uid in candidate.provenance.parents]}, candidate.provenance.evidence))
            node_batch = 384
            for offset in range(0, len(publication_rows), node_batch):
                self._publish_group(tuple(publication_rows[offset:offset + node_batch]))

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["hgt_context_action_scores"] = {str(environment): {str(context): {str(action): score for action, score in actions.items()} for context, actions in contexts.items()} for environment, contexts in self._hgt_context_action_scores.items()}
        return state

    def _restore(self, snapshot: dict[str, Any]) -> None:
        super()._restore(snapshot)
        state = dict(snapshot.get("state", {}))
        self._hgt_context_action_scores = {int(environment): {int(context): {int(action): float(score) for action, score in dict(actions).items()} for context, actions in dict(contexts).items()} for environment, contexts in dict(state.get("hgt_context_action_scores", {})).items()}
