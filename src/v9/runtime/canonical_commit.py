from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from v9.cognition.isf import ISFComponents
from v9.cognition.grounding import GroundingEvidence
from v9.modalities.symbols import DeterministicSymbolCodec

from .canonical_commit_derivation import derivation_candidates
from .canonical_commit_isf import score_isf_batch
from .canonical_commit_state import advance_stage_fast, record_normalized_fast, trim_replay_pool
from .memory_pipeline import DerivationTask
from .memory_pipeline_v2 import CommitPlan


@dataclass(frozen=True, slots=True)
class CanonicalCommitResult:
    signature_rows: tuple[tuple[int, ...], ...]
    derivation_candidates: tuple[DerivationTask, ...]


def apply_canonical_commit_batch(runtime: Any, rows: Iterable[CommitPlan]) -> CanonicalCommitResult:
    plans = tuple(rows)
    if not plans:
        return CanonicalCommitResult((), ())

    with runtime._lock:
        deferred_rows: list[Any] = []
        signature_rows: list[tuple[int, ...]] = []
        touched_signatures: set[int] = set()
        registered_identities: dict[int, Any] = {}
        formation_environments: set[int] = set()
        modality_deltas: dict[int, int] = {}
        curriculum_counts: dict[str, int] = {}
        timeline_events_delta = 0
        timeline_actions_delta = 0
        telemetry_events_delta = 0
        last_ordering_key = runtime.timeline.last_ordering_key
        last_step = "none"
        last_family = ""
        last_scenario = ""
        isf_rows: list[tuple[ISFComponents, int, int, Any, Any, int]] = []

        for plan in plans:
            signatures: list[int] = []
            event = plan.event
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
                deferred_rows.extend(write.runtime_row() for write in plan.base_writes)

                if plan.interaction_grounding is not None:
                    grounding = plan.interaction_grounding
                    runtime._latest_interaction_grounding[(grounding.environment_instance_id, grounding.episode_id)] = grounding
                if plan.relation is None or plan.normalized_write is None:
                    raise RuntimeError("interaction commit plan is incomplete")

                prior_support = int(runtime._m1n_supports.get(int(plan.relation.structural_signature), 0))
                signature = record_normalized_fast(
                    runtime,
                    plan.relation,
                    plan.normalized_write,
                    deferred_rows,
                )
                signatures.append(signature)
                touched_signatures.add(signature)
                next_stage = advance_stage_fast(runtime)

                if plan.isf_static is not None:
                    pvi, osi, explicit_pe, tp, ep = plan.isf_static
                    recurrence = int(runtime._m1n_supports.get(signature, 0))
                    recurrence_pe = 1.0 / max(1.0, float(prior_support + 1))
                    pe = abs(float(explicit_pe)) if float(explicit_pe) != 0.0 else recurrence_pe
                    isf_rows.append(
                        (
                            ISFComponents(pvi, osi, pe, 1.0 / max(1, recurrence), tp, ep),
                            int(runtime._watermark),
                            int(event.identity.causal_watermark),
                            stage_before,
                            next_stage,
                            int(runtime.graph.generation),
                        )
                    )
                    runtime._prediction_error_sum += abs(float(pe))
                    runtime._prediction_error_count += 1

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
                deferred_rows.extend(write.runtime_row() for write in symbol.base_writes)

                signature = record_normalized_fast(
                    runtime,
                    symbol.relation,
                    symbol.normalized_write,
                    deferred_rows,
                )
                signatures.append(signature)
                touched_signatures.add(signature)
                if symbol.aligned_relation is not None:
                    if symbol.aligned_normalized_write is None:
                        raise RuntimeError("aligned symbol commit plan is incomplete")
                    aligned_signature = record_normalized_fast(
                        runtime,
                        symbol.aligned_relation,
                        symbol.aligned_normalized_write,
                        deferred_rows,
                    )
                    signatures.append(aligned_signature)
                    touched_signatures.add(aligned_signature)
                    if plan.interaction_grounding is not None:
                        g = plan.interaction_grounding
                        grounding_key = (
                            int(symbol.relation.uid.lo),
                            int(g.uid.lo),
                            int(g.environment_instance_id),
                            0,
                            0,
                        )
                        before_grounding = runtime.grounding.states.get(grounding_key)
                        after_grounding = runtime.grounding.observe(
                            GroundingEvidence(
                                int(symbol.relation.uid.lo),
                                int(g.uid.lo),
                                int(g.environment_instance_id),
                                0,
                                0,
                                int(runtime._watermark),
                                recurrent_symbol=True,
                                cross_modal_association=True,
                            )
                        )
                advance_stage_fast(runtime)

            last_step = plan.curriculum_step or "none"
            last_family = str(plan.identity.family)
            last_scenario = str(plan.game_scenario)
            key = f"{last_step}|{last_family}|{last_scenario}"
            curriculum_counts[key] = curriculum_counts.get(key, 0) + 1
            signature_rows.append(tuple(signatures))

        if deferred_rows:
            runtime._defer_base_group(tuple(deferred_rows))
        trim_replay_pool(runtime)
        runtime._formation_environments.update(formation_environments)
        runtime.timeline.events_seen += timeline_events_delta
        runtime.timeline.actions_committed += timeline_actions_delta
        runtime.timeline.last_ordering_key = last_ordering_key
        runtime.telemetry["events"] += telemetry_events_delta
        for modality, count in modality_deltas.items():
            runtime._modality_events[modality] = runtime._modality_events.get(modality, 0) + count

        runtime.unified_telemetry.curriculum_counts.update(curriculum_counts)
        runtime.unified_telemetry.gauges["curriculum_step"] = last_step
        runtime.unified_telemetry.gauges["environment_family"] = last_family
        runtime.unified_telemetry.gauges["game_scenario"] = last_scenario
        score_isf_batch(runtime, isf_rows)
        candidates = derivation_candidates(runtime, touched_signatures)
        return CanonicalCommitResult(tuple(signature_rows), candidates)
