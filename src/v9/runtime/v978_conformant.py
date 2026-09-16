from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any

from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.v978_descriptors import modality_neutral_family_signature

from .final_runtime import FinalContinuousMemoryRuntime
from .memory_pipeline_v2 import CanonicalWrite


class V978ContinuousMemoryRuntime(FinalContinuousMemoryRuntime):
    """Final Design §20 conformance layer for v9.7.8 symbolic grounding."""

    def __init__(self, config: Any) -> None:
        self._v978_evidence_lock = RLock()
        self._v978_legacy_family_map: dict[int, int] = {}
        self._v978_m1n_evidence: dict[int, dict[str, Any]] = {}
        self._interaction_to_symbol_prediction_delta_sum = 0.0
        super().__init__(config)
        self._rebuild_v978_evidence_state()

    def _rebuild_v978_evidence_state(self) -> None:
        for uid, node in self.graph.nodes.items():
            payload = self.graph.payloads.get(uid, {})
            if str(payload.get("channel", "")) not in {"WORLD", "SYMBOL", "CROSS_MODAL"}:
                continue
            signature = payload.get("structural_signature")
            if signature is None:
                continue
            offsets = tuple(int(value) for value in payload.get("temporal_offsets", ()) or ())
            with self._v978_evidence_lock:
                self._v978_m1n_evidence[int(signature)] = {
                    "uid": uid,
                    "support": float(payload.get("support", 1.0)),
                    "contradiction": float(payload.get("contradiction", 0.0)),
                    "offsets": set(offsets),
                    "causal_watermark": int(payload.get("causal_watermark", node.created_watermark)),
                    "family_signature": int(payload.get("family_signature", signature)),
                    "context_signature": int(payload.get("context_signature", 0)),
                }

    def _neutralize_relation(self, relation: M1NormalizedRelation, payload: dict[str, Any] | None = None) -> M1NormalizedRelation:
        legacy = int(relation.family_signature or relation.structural_signature)
        if relation.channel is NormalizedChannel.WORLD:
            neutral = int(modality_neutral_family_signature(relation, payload))
            self._v978_legacy_family_map[legacy] = neutral
        else:
            neutral = int(self._v978_legacy_family_map.get(legacy, legacy))
        if neutral == int(relation.family_signature or relation.structural_signature):
            return relation
        return replace(relation, family_signature=neutral)

    def normalize_m1n_family(self, relation: M1NormalizedRelation, initial_write: CanonicalWrite) -> tuple[M1NormalizedRelation, CanonicalWrite]:
        normalized = self._neutralize_relation(relation, initial_write.payload)
        payload = dict(initial_write.payload)
        payload.update(
            {
                "family_signature": int(normalized.family_signature or normalized.structural_signature),
                "context_signature": int(normalized.context_signature),
                "temporal_offset_range": None if normalized.temporal_offset_range is None else list(normalized.temporal_offset_range),
            }
        )
        return normalized, CanonicalWrite(initial_write.node, payload, initial_write.evidence)

    def accumulate_m1n_evidence(self, relation: M1NormalizedRelation) -> None:
        signature = int(relation.structural_signature)
        with self._v978_evidence_lock:
            current = self._v978_m1n_evidence.get(signature)
            if current is None:
                current = {
                    "uid": relation.uid,
                    "support": 0.0,
                    "contradiction": 0.0,
                    "offsets": set(),
                    "causal_watermark": 0,
                    "family_signature": int(relation.family_signature or relation.structural_signature),
                    "context_signature": int(relation.context_signature),
                }
                self._v978_m1n_evidence[signature] = current
            current["uid"] = relation.uid
            current["support"] = float(current["support"]) + float(relation.support)
            current["contradiction"] = float(current["contradiction"]) + float(relation.contradiction)
            offsets = current["offsets"]
            offsets.update(int(value) for value in relation.temporal_offsets)
            if len(offsets) > 64:
                kept = sorted(offsets, key=lambda value: (abs(value), value))[:64]
                current["offsets"] = set(kept)
            current["causal_watermark"] = max(int(current["causal_watermark"]), int(relation.causal_watermark))
            current["family_signature"] = int(relation.family_signature or relation.structural_signature)
            current["context_signature"] = int(relation.context_signature)

    def _evidence_payload(self, signature: int) -> dict[str, Any]:
        with self._v978_evidence_lock:
            row = self._v978_m1n_evidence.get(int(signature))
            if row is None:
                return {}
            offsets = tuple(sorted(int(value) for value in row["offsets"]))
            return {
                "support": float(row["support"]),
                "contradiction": float(row["contradiction"]),
                "temporal_offsets": list(offsets),
                "temporal_offset_range": None if not offsets else [min(offsets), max(offsets)],
                "causal_watermark": int(row["causal_watermark"]),
                "family_signature": int(row["family_signature"]),
                "context_signature": int(row["context_signature"]),
            }

    def _record_normalized(self, relation: M1NormalizedRelation, *, defer_publication: bool = False, payload_extra: dict[str, Any] | None = None) -> int:
        normalized = self._neutralize_relation(relation, payload_extra)
        self.accumulate_m1n_evidence(normalized)
        extra = dict(payload_extra or {})
        extra.update(self._evidence_payload(normalized.structural_signature))
        return super()._record_normalized(normalized, defer_publication=defer_publication, payload_extra=extra)

    def _register_family_relation(self, relation: Any) -> None:
        normalized = self._neutralize_relation(relation)
        super()._register_family_relation(normalized)

    @staticmethod
    def _rows_with_payload(runtime: "V978ContinuousMemoryRuntime") -> list[tuple[MemoryUid, Any, dict[str, Any]]]:
        rows = [(uid, runtime.graph.nodes[uid], runtime.graph.payloads.get(uid, {})) for uid in runtime.graph.nodes]
        rows.extend((uid, node, payload) for uid, (node, payload, _evidence) in runtime._deferred_base_nodes.items() if uid not in runtime.graph.nodes)
        return rows

    def _backfill_symbol_context(self) -> None:
        rows = self._rows_with_payload(self)
        world: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
        symbols: list[tuple[dict[str, Any], int]] = []
        for _uid, node, payload in rows:
            environment = int(payload.get("environment_instance_id", payload.get("symbol_environment_instance_id", 0)))
            episode = int(payload.get("episode_id", payload.get("symbol_episode_id", 0)))
            if payload.get("symbol_identity") is not None or payload.get("symbol_occurrence_id") is not None:
                symbols.append((payload, int(node.created_watermark)))
            elif payload.get("action_id") is not None:
                world.setdefault((environment, episode), []).append((int(node.created_watermark), payload))
        for values in world.values():
            values.sort(key=lambda row: row[0])
        for payload, watermark in symbols:
            environment = int(payload.get("environment_instance_id", payload.get("symbol_environment_instance_id", 0)))
            episode = int(payload.get("episode_id", payload.get("symbol_episode_id", 0)))
            candidates = world.get((environment, episode), ())
            if not candidates:
                continue
            _, nearby = min(candidates, key=lambda row: (abs(row[0] - watermark), row[0] > watermark, row[0]))
            context = int(nearby.get("context_signature", 0))
            next_context = int(nearby.get("next_context_signature", context) or context)
            outcome_signature = int(nearby.get("outcome_signature", 0) or 0)
            payload.update(
                {
                    "nearby_context_signature": context,
                    "nearby_action_id": int(nearby.get("action_id", 0)),
                    "nearby_transformation_signature": int(stable_u64(context, outcome_signature, next_context, person=b"v9-symbol-nearby")),
                    "nearby_progress": bool(nearby.get("task_success", False) or int(nearby.get("levels_completed", 0)) > 0),
                    "nearby_outcome": int(nearby.get("primary_valence", 0)),
                    "nearby_task_success": bool(nearby.get("task_success", False)),
                    "nearby_task_failure": bool(nearby.get("task_failure", False)),
                }
            )

    def _patch_m1n_evidence(self) -> None:
        with self._v978_evidence_lock:
            evidence_rows = self._v978_m1n_evidence.copy()
            snapshots = tuple(
                (evidence.get("uid"), self._evidence_payload(signature))
                for signature, evidence in evidence_rows.items()
            )
        for uid, evidence_payload in snapshots:
            if uid in self.graph.payloads:
                self.graph.payloads[uid].update(evidence_payload)
            elif uid in self._deferred_base_nodes:
                node, payload, refs = self._deferred_base_nodes[uid]
                merged = dict(payload)
                merged.update(evidence_payload)
                self._deferred_base_nodes[uid] = (node, merged, refs)

    def flush_deferred_memory_updates(self) -> None:
        self._patch_m1n_evidence()
        super().flush_deferred_memory_updates()
        self._patch_m1n_evidence()
        self._backfill_symbol_context()

    def _uid_by_low_any(self, low: int) -> MemoryUid | None:
        candidates = [uid for uid in self.graph.nodes if int(uid.lo) == int(low)]
        candidates.extend(uid for uid in self._deferred_base_nodes if int(uid.lo) == int(low))
        return min(candidates) if candidates else None

    def record_symbolic_validation(self, **kwargs: Any):
        symbol_low = int(kwargs["symbol_structure_uid"])
        interaction_low = int(kwargs["interaction_structure_uid"])
        if self._uid_by_low_any(symbol_low) not in self.graph.nodes or self._uid_by_low_any(interaction_low) not in self.graph.nodes:
            self.flush_deferred_memory_updates()
        state = super().record_symbolic_validation(**kwargs)
        validation_kind = str(kwargs["validation_kind"])
        effect = float(kwargs["effect"])
        relation_name = {
            "prediction": "SYMBOL_TO_INTERACTION_PREDICTION",
            "generalization": "INTERACTION_TO_SYMBOL_GENERALIZATION",
            "heldout_transfer": "CROSS_MODAL_HELDOUT_TRANSFER",
            "composition": "CROSS_MODAL_COMPOSITION",
            "symbol_mediated_learning": "CROSS_MODAL_HELDOUT_TRANSFER",
        }[validation_kind]
        symbol_uid = self._uid_by_low_any(symbol_low)
        interaction_uid = self._uid_by_low_any(interaction_low)
        if symbol_uid is not None and interaction_uid is not None:
            interaction_payload = self.graph.payloads.get(interaction_uid, {})
            family_signature = int(interaction_payload.get("family_signature", interaction_payload.get("structural_signature", stable_u64(interaction_low, person=b"v978-validation-family"))))
            context_signature = int(interaction_payload.get("context_signature", interaction_payload.get("grounded_context_signature", 0)))
            relation = M1NormalizedRelation.from_provenance(
                relation_name,
                NormalizedChannel.CROSS_MODAL,
                parents=(symbol_uid, interaction_uid),
                evidence=(symbol_uid, interaction_uid),
                structural_key=("VALIDATED_GROUNDING", relation_name, symbol_low, family_signature, context_signature),
                family_signature=family_signature,
                support=max(0.0, effect),
                contradiction=max(0.0, -effect),
                causal_watermark=int(self.watermark),
                heldout_transfer=validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
                context_signature=context_signature,
            )
            self._record_normalized(
                relation,
                payload_extra={
                    "symbol_relation": relation_name,
                    "grounding_validation_trial_id": str(kwargs["trial_id"]),
                    "validation_causal_watermark": int(self.watermark),
                    "grounding_evaluation_watermark": int(self.watermark),
                    "grounding_source_environment_id": int(kwargs["environment_instance_id"]),
                    "grounding_target_environment_id": None if kwargs.get("target_environment_id") is None else int(kwargs["target_environment_id"]),
                    "cross_modal_control": "aligned" if effect > 0.0 else "contradicted",
                    "heldout_transfer": validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
                    "novel_composition": validation_kind == "composition",
                    "symbol_mediated_learning": validation_kind == "symbol_mediated_learning",
                    "grounding_active": bool(state.behavior_eligible),
                    "grounding_maturity": int(state.maturity),
                    "grounding_confidence": float(state.support / max(1e-9, state.support + state.contradiction)),
                },
            )
        if validation_kind == "prediction":
            self._symbol_prediction_delta_sum += effect
        elif validation_kind == "generalization":
            self._interaction_to_symbol_prediction_delta_sum += effect
        return state

    def metrics(self) -> dict[str, Any]:
        self._patch_m1n_evidence()
        result = dict(super().metrics())
        grounding_states = self.grounding.states.copy()
        states = tuple(grounding_states.values())
        for maturity in range(6):
            result[f"grounding_G{maturity}_count"] = sum(int(int(state.maturity) == maturity) for state in states)
        result["grounding_active_count"] = sum(int(state.behavior_eligible) for state in states)
        result["symbol_to_interaction_prediction_gain"] = float(self._symbol_prediction_delta_sum)
        result["interaction_to_symbol_generalization_gain"] = float(self._interaction_to_symbol_prediction_delta_sum)
        result["heldout_transfer_successes"] = sum(int(int(state.maturity) >= 3 and state.behavior_eligible and bool(state.validation_trial_ids)) for state in states)
        provenance = dict(result.get("symbol_grounding_provenance", {}))
        provenance.update(
            {
                "environment_ids": sorted({int(key[2]) for key in grounding_states}),
                "context_scope_ids": sorted({int(key[3]) for key in grounding_states}),
                "lineage_ids": sorted({int(key[4]) for key in grounding_states}),
            }
        )
        result["symbol_grounding_provenance"] = provenance
        return result
