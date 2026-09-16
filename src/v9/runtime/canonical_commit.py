from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

from v9.cognition.isf import ISFComponents
from v9.cognition.grounding import GroundingEvidence
from v9.memory.symbolic_relations import shuffled_alignment_control
from v9.modalities.symbols import DeterministicSymbolCodec

from .canonical_commit_derivation import derivation_candidates
from .canonical_commit_isf import score_isf_batch
from .canonical_commit_state import advance_stage_fast, record_normalized_fast, trim_replay_pool
from .memory_pipeline import DerivationTask
from .memory_pipeline_v2 import CommitPlan, _m1n_write


@dataclass(frozen=True, slots=True)
class CanonicalCommitResult:
    signature_rows: tuple[tuple[int, ...], ...]
    derivation_candidates: tuple[DerivationTask, ...]


def apply_canonical_commit_batch(runtime: Any, rows: Iterable[CommitPlan]) -> CanonicalCommitResult:
    plans = tuple(rows)
    if not plans:
        return CanonicalCommitResult((), ())

    with runtime._lock:
        deferred_groups: list[tuple[Any, ...]] = []
        signature_rows: list[tuple[int, ...]] = []
        touched_signatures: set[int] = set()
        registered_identities: dict[int, Any] = {}
        formation_environments: set[int] = set()
        modality_deltas: dict[int, int] = {}
        curriculum_counts: dict[str, int] = {}
        timeline_events_delta = 0
        timeline_actions_delta = 0
        telemetry_events_delta = 0
        symbol_occurrences_delta = 0
        unique_symbols: set[int] = set()
        last_ordering_key = runtime.timeline.last_ordering_key
        last_step = "none"
        last_family = ""
        last_scenario = ""
        isf_rows: list[tuple[ISFComponents, int, int, Any, Any, int]] = []
        max_cross_modal = int(runtime.config.scientific.max_cross_modal_facts_per_macro_event)
        logical_graph_generation = int(runtime.graph.generation)
        publication_generation_delta = getattr(runtime, "_deferred_publication_generation_delta", None)

        for plan in plans:
            plan_deferred_rows: list[Any] = []
            signatures: list[int] = []
            cross_modal_used = 0
            event = plan.event
            previous_interaction = None
            if plan.interaction_grounding is not None:
                g = plan.interaction_grounding
                previous_interaction = runtime._latest_interaction_grounding.get((int(g.environment_instance_id), int(g.episode_id)))

            if event is not None:
                environment_id = int(event.identity.environment_instance_id)
                previous_identity = registered_identities.get(environment_id)
                if previous_identity is None:
                    identity = runtime.environments.register(plan.identity)
                    if int(identity.value) != environment_id:
                        raise RuntimeError("prepared environment identity mismatch")
                    registered_identities[environment_id] = plan.identity
                elif previous_identity != plan.identity:
                    raise RuntimeError("prepared environment identity collision inside batch")

                runtime._watermark = max(runtime._watermark, int(event.identity.causal_watermark))
                timeline_events_delta += 1
                timeline_actions_delta += 1
                telemetry_events_delta += 1
                last_ordering_key = event.identity.ordering_key
                modality = int(event.identity.modality_id.value)
                modality_deltas[modality] = modality_deltas.get(modality, 0) + 1
                formation_environments.add(environment_id)
                stage_before = runtime.stage_tracker.stage
                plan_deferred_rows.extend(write.runtime_row() for write in plan.base_writes)

                if plan.interaction_grounding is not None:
                    grounding = plan.interaction_grounding
                    runtime._latest_interaction_grounding[(grounding.environment_instance_id, grounding.episode_id)] = grounding
                if plan.relation is None or plan.normalized_write is None:
                    raise RuntimeError("interaction commit plan is incomplete")

                prior_support = int(runtime._m1n_supports.get(int(plan.relation.structural_signature), 0))
                signature = record_normalized_fast(runtime, plan.relation, plan.normalized_write, plan_deferred_rows)
                signatures.append(signature)
                touched_signatures.add(signature)
                next_stage = advance_stage_fast(runtime)

                if plan.isf_static is not None:
                    pvi, osi, explicit_pe, tp, ep = plan.isf_static
                    evidence_confidence = float(runtime.__dict__.get("_environment_evidence_confidence", {}).get(environment_id, 1.0))
                    if float(pvi) <= 0.0:
                        explicit_pe *= evidence_confidence
                        tp *= evidence_confidence
                        ep *= evidence_confidence
                    recurrence = int(runtime._m1n_supports.get(signature, 0))
                    recurrence_pe = 1.0 / max(1.0, float(prior_support + 1))
                    pe = abs(float(explicit_pe)) if float(explicit_pe) != 0.0 else recurrence_pe
                    isf_rows.append((ISFComponents(pvi, osi, pe, 1.0 / max(1, recurrence), tp, ep), int(runtime._watermark), int(event.identity.causal_watermark), stage_before, next_stage, logical_graph_generation))
                    runtime._prediction_error_sum += abs(float(pe))
                    runtime._prediction_error_count += 1

            symbol_occurrences_delta += len(plan.symbol_occurrences)
            unique_symbols.update(int(row.symbol_id.value) for row in plan.symbol_occurrences)
            index_occurrences = getattr(runtime, "_index_symbol_occurrences", None)
            if callable(index_occurrences):
                index_occurrences(plan.symbol_occurrences)
            if plan.symbol_codec_state:
                codec = DeterministicSymbolCodec.from_state_dict(dict(plan.symbol_codec_state))
                if codec.vocabulary_id.value not in runtime.symbol_codecs:
                    runtime.symbol_codecs[codec.vocabulary_id.value] = codec

            for symbol in plan.symbols:
                symbol_event = symbol.event
                runtime._watermark = max(runtime._watermark, int(symbol_event.identity.causal_watermark))
                timeline_events_delta += 1
                telemetry_events_delta += 1
                modality = int(symbol_event.identity.modality_id.value)
                modality_deltas[modality] = modality_deltas.get(modality, 0) + 1
                formation_environments.add(int(symbol_event.identity.environment_instance_id))
                plan_deferred_rows.extend(write.runtime_row() for write in symbol.base_writes)

                signature = record_normalized_fast(runtime, symbol.relation, symbol.normalized_write, plan_deferred_rows)
                signatures.append(signature)
                touched_signatures.add(signature)
                if symbol.aligned_relation is not None and cross_modal_used < max_cross_modal:
                    if symbol.aligned_normalized_write is None:
                        raise RuntimeError("aligned symbol commit plan is incomplete")
                    aligned_signature = record_normalized_fast(runtime, symbol.aligned_relation, symbol.aligned_normalized_write, plan_deferred_rows)
                    signatures.append(aligned_signature)
                    touched_signatures.add(aligned_signature)
                    cross_modal_used += 1
                    if plan.interaction_grounding is not None:
                        g = plan.interaction_grounding
                        # Stable symbolic M1N identity is the grounding authority;
                        # occurrence-specific M1G remains provenance only.
                        symbol_structure_uid = symbol.relation.uid
                        grounding_key = (int(symbol_structure_uid.lo), int(g.uid.lo), int(g.environment_instance_id), 0, 0)
                        before_grounding = runtime.grounding.states.get(grounding_key)
                        after_grounding = runtime.grounding.observe(
                            GroundingEvidence(
                                int(symbol_structure_uid.lo),
                                int(g.uid.lo),
                                int(g.environment_instance_id),
                                0,
                                0,
                                int(runtime._watermark),
                                recurrent_symbol=True,
                                cross_modal_association=True,
                            )
                        )
                        if before_grounding is None or int(after_grounding.maturity) > int(before_grounding.maturity):
                            runtime.telemetry["grounding_promotions"] += 1
                advance_stage_fast(runtime)

            for derived in plan.derived_relations:
                if derived.relation.channel.value == "CROSS_MODAL":
                    if cross_modal_used >= max_cross_modal:
                        continue
                    cross_modal_used += 1
                signature = record_normalized_fast(runtime, derived.relation, derived.write, plan_deferred_rows)
                signatures.append(signature)
                touched_signatures.add(signature)

            # Shuffled mismatch evidence is bounded and contradiction-only.
            if previous_interaction is not None and cross_modal_used < max_cross_modal:
                for symbol in plan.symbols:
                    if cross_modal_used >= max_cross_modal:
                        break
                    proxy = SimpleNamespace(m1g=symbol.grounding, base_writes=symbol.base_writes, occurrence=symbol.occurrence)
                    control = shuffled_alignment_control(proxy, previous_interaction, causal_watermark=int(runtime._watermark), occurrence=symbol.occurrence)
                    write = _m1n_write(control.relation, watermark=int(runtime._watermark), occurrence=symbol.occurrence, payload_extra=control.payload())
                    signature = record_normalized_fast(runtime, control.relation, write, plan_deferred_rows)
                    signatures.append(signature)
                    touched_signatures.add(signature)
                    cross_modal_used += 1

            last_step = plan.curriculum_step or "none"
            last_family = str(plan.identity.family)
            last_scenario = str(plan.game_scenario)
            key = f"{last_step}|{last_family}|{last_scenario}"
            curriculum_counts[key] = curriculum_counts.get(key, 0) + 1
            signature_rows.append(tuple(signatures))

            deferred_group = tuple(plan_deferred_rows)
            if deferred_group:
                deferred_groups.append(deferred_group)
                if callable(publication_generation_delta):
                    logical_graph_generation += int(publication_generation_delta(deferred_group))

        if deferred_groups:
            defer_groups = getattr(runtime, "_defer_base_groups", None)
            if callable(defer_groups):
                defer_groups(tuple(deferred_groups))
            else:
                runtime._defer_base_group(tuple(row for group in deferred_groups for row in group))
        trim_replay_pool(runtime)
        runtime._formation_environments.update(formation_environments)
        runtime.timeline.events_seen += timeline_events_delta
        runtime.timeline.actions_committed += timeline_actions_delta
        runtime.timeline.last_ordering_key = last_ordering_key
        runtime.telemetry["events"] += telemetry_events_delta
        runtime.telemetry["symbol_occurrences"] = int(runtime.telemetry.get("symbol_occurrences", 0)) + symbol_occurrences_delta
        runtime.set_telemetry_gauge("symbol_occurrences_ingested", runtime.telemetry["symbol_occurrences"])
        runtime.set_telemetry_gauge("unique_symbols_batch", len(unique_symbols))
        runtime.set_telemetry_gauge("symbolic_m1n_batch", sum(len(plan.symbols) + len(plan.derived_relations) for plan in plans))
        runtime.set_telemetry_gauge("cross_modal_m1n_batch", sum(sum(1 for row in plan.symbols if row.aligned_relation is not None) for plan in plans))
        for modality, count in modality_deltas.items():
            runtime._modality_events[modality] = runtime._modality_events.get(modality, 0) + count

        runtime.unified_telemetry.curriculum_counts.update(curriculum_counts)
        runtime.unified_telemetry.gauges["curriculum_step"] = last_step
        runtime.unified_telemetry.gauges["environment_family"] = last_family
        runtime.unified_telemetry.gauges["game_scenario"] = last_scenario
        score_isf_batch(runtime, isf_rows)
        candidates = derivation_candidates(runtime, touched_signatures)
        return CanonicalCommitResult(tuple(signature_rows), candidates)
