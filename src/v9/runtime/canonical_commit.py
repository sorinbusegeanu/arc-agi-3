from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

from v9.cognition.isf import ISFComponents
from v9.cognition.grounding import GroundingEvidence
from v9.memory.symbolic_relations import shuffled_alignment_control
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.modalities.symbols import DeterministicSymbolCodec

from .canonical_commit_derivation import derivation_candidates
from .canonical_commit_isf import score_isf_batch
from .canonical_commit_state import advance_stage_fast, record_normalized_fast, trim_replay_pool
from .canonical_transaction import (
    CanonicalTransactionQuarantined,
    CanonicalTransactionStatus,
    CanonicalWorkBudget,
    CanonicalWorkEstimate,
    _estimate_bytes,
)
from .memory_admission import should_retain_concrete
from .memory_pipeline import CommitPlan, DerivationTask, _m1n_write


@dataclass(frozen=True, slots=True)
class CanonicalCommitResult:
    signature_rows: tuple[tuple[int, ...], ...]
    derivation_candidates: tuple[DerivationTask, ...]
    lock_seconds: float = 0.0


def _plan_writes(plan: CommitPlan) -> Iterable[Any]:
    yield from plan.base_writes
    if plan.normalized_write is not None:
        yield plan.normalized_write
    for symbol in plan.symbols:
        yield from symbol.base_writes
        yield symbol.normalized_write
        if symbol.aligned_normalized_write is not None:
            yield symbol.aligned_normalized_write
    for derived in plan.derived_relations:
        yield derived.write


def _materialize_commit_rows(plans: tuple[CommitPlan, ...]) -> dict[int, Any]:
    materialized_rows: dict[int, Any] = {}
    for plan in plans:
        for write in _plan_writes(plan):
            key = id(write)
            if key not in materialized_rows:
                materialized_rows[key] = write.runtime_row()
    return materialized_rows


def _training_evidence_record_count(node: Any, payload: dict[str, Any]) -> int:
    if node.level is MemoryLevel.M0:
        return int("producer_sequence" in payload and payload.get("symbol_identity") is None)
    if node.level is MemoryLevel.M1:
        return 1 + int(bool(payload.get("heldout_transfer")))
    if node.level in {MemoryLevel.M2, MemoryLevel.M3}:
        return 2
    if node.level is MemoryLevel.M4:
        return 1 + int(bool(payload.get("validated")))
    if node.level in {MemoryLevel.M5, MemoryLevel.M6}:
        return 1
    if node.level is MemoryLevel.M7:
        return 2
    return 0


def _mutation_work(row: Any, *, reserve_training_evidence: bool = False) -> tuple[int, int, int]:
    node, payload, evidence = row
    node_bytes = _estimate_bytes(
        (int(node.uid.hi), int(node.uid.lo)),
        {"node": node, "payload": payload, "evidence": evidence},
    )
    mutation_bytes = node_bytes
    write_count = 3
    maximum_primitive_bytes = node_bytes
    if reserve_training_evidence:
        # Records contain compact fixed-schema labels plus checksums. Reserve a
        # conservative encoded ceiling so WAL evidence can never push an
        # otherwise admitted canonical transaction over its byte budget.
        evidence_bytes = 4096 * _training_evidence_record_count(node, payload)
        mutation_bytes += evidence_bytes
        maximum_primitive_bytes = max(maximum_primitive_bytes, evidence_bytes)
    parents = {
        (int(parent[0]), int(parent[1]))
        for parent in payload.get("parents", ())
        if isinstance(parent, (list, tuple)) and len(parent) == 2
    }
    for parent in parents:
        edge_bytes = _estimate_bytes(
            (int(node.uid.hi), int(node.uid.lo), parent), evidence
        )
        maximum_primitive_bytes = max(maximum_primitive_bytes, edge_bytes)
        mutation_bytes += edge_bytes
        write_count += 2
    return mutation_bytes, write_count, maximum_primitive_bytes


def _work_budget(runtime: Any) -> CanonicalWorkBudget:
    config = getattr(runtime, "config", None)
    return CanonicalWorkBudget(
        max_rows=int(getattr(config, "canonical_transaction_max_rows", 1024)),
        max_input_bytes=int(getattr(config, "canonical_transaction_max_input_bytes", 64 * 1024 * 1024)),
        max_materialized_mutation_bytes=int(getattr(config, "canonical_transaction_max_mutation_bytes", 64 * 1024 * 1024)),
        max_continuation_bytes=int(getattr(config, "canonical_continuation_max_bytes", 16 * 1024 * 1024)),
        max_writes=int(getattr(config, "canonical_transaction_max_writes", 65_536)),
        max_work_units=int(getattr(config, "canonical_transaction_max_work_units", 1_000_000)),
    )


def _estimate_materialized_commit(
    plans: tuple[CommitPlan, ...],
    materialized_rows: dict[int, Any],
    *,
    input_bytes: int,
    reserve_training_evidence: bool = False,
) -> tuple[CanonicalWorkEstimate, int]:
    mutation_bytes = 0
    write_count = 0
    maximum_primitive_bytes = 0
    for row in materialized_rows.values():
        row_bytes, row_writes, primitive_bytes = _mutation_work(
            row, reserve_training_evidence=reserve_training_evidence
        )
        mutation_bytes += row_bytes
        write_count += row_writes
        maximum_primitive_bytes = max(maximum_primitive_bytes, primitive_bytes)
    symbol_count = sum(len(plan.symbols) for plan in plans)
    derived_count = sum(len(plan.derived_relations) for plan in plans)
    grounding_operations = sum(int(plan.interaction_grounding is not None) for plan in plans)
    return (
        CanonicalWorkEstimate(
            rows=len(plans),
            input_bytes=int(input_bytes),
            materialized_mutation_bytes=mutation_bytes,
            write_count=write_count,
            symbol_count=symbol_count,
            derived_relation_count=derived_count,
            grounding_operations=grounding_operations,
            work_units=write_count + symbol_count + derived_count + grounding_operations,
        ),
        maximum_primitive_bytes,
    )


def estimate_canonical_commit_batch(
    rows: Iterable[CommitPlan], *, input_bytes: int = 0
) -> CanonicalWorkEstimate:
    plans = tuple(rows)
    estimate, _ = _estimate_materialized_commit(
        plans, _materialize_commit_rows(plans), input_bytes=int(input_bytes)
    )
    return estimate


def canonical_commit_prefix_length(
    runtime: Any,
    rows: Iterable[CommitPlan],
    *,
    row_input_bytes: Iterable[int],
) -> int:
    """Return the largest ordered prefix satisfying every canonical budget."""
    plans = tuple(rows)
    measured_input = tuple(int(value) for value in row_input_bytes)
    if len(plans) != len(measured_input) or any(value < 0 for value in measured_input):
        raise ValueError("canonical row-byte accounting mismatch")
    if not plans:
        return 0

    budget = _work_budget(runtime)
    materialized_rows: dict[int, Any] = {}
    input_bytes = 0
    mutation_bytes = 0
    write_count = 0
    maximum_primitive_bytes = 0
    symbol_count = 0
    derived_count = 0
    grounding_operations = 0
    for index, plan in enumerate(plans):
        for write in _plan_writes(plan):
            key = id(write)
            if key in materialized_rows:
                continue
            row = write.runtime_row()
            materialized_rows[key] = row
            row_bytes, row_writes, primitive_bytes = _mutation_work(
                row, reserve_training_evidence=hasattr(runtime, "canonical_wal")
            )
            mutation_bytes += row_bytes
            write_count += row_writes
            maximum_primitive_bytes = max(maximum_primitive_bytes, primitive_bytes)
        input_bytes += measured_input[index]
        symbol_count += len(plan.symbols)
        derived_count += len(plan.derived_relations)
        grounding_operations += int(plan.interaction_grounding is not None)
        estimate = CanonicalWorkEstimate(
            rows=index + 1,
            input_bytes=input_bytes,
            materialized_mutation_bytes=mutation_bytes,
            write_count=write_count,
            symbol_count=symbol_count,
            derived_relation_count=derived_count,
            grounding_operations=grounding_operations,
            work_units=write_count + symbol_count + derived_count + grounding_operations,
        )
        status = budget.status(estimate)
        if maximum_primitive_bytes > budget.max_continuation_bytes:
            status = CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE
        if status is CanonicalTransactionStatus.READY:
            continue
        if index:
            return index
        raise CanonicalTransactionQuarantined(
            status,
            sequences=(int(plan.sequence),),
            estimate=estimate,
        )
    return len(plans)


def _append_dirty_normalized(runtime: Any, relation: Any, rows: list[Any]) -> int | None:
    if not hasattr(runtime, "canonical_store"):
        return None
    signature = int(relation.structural_signature)
    if signature not in runtime._m1n_dirty:
        return None
    occurrences = runtime._m1n_occurrences.get(signature, ())
    parents = tuple(uid for occurrence in occurrences for uid in occurrence.provenance.parents)
    evidence = tuple(uid for occurrence in occurrences for uid in occurrence.provenance.evidence)
    rows.append(
        (
            CanonicalNode(
                relation.uid,
                MemoryLevel.M1,
                MemoryType.NORMALIZED_RELATION,
                (signature,),
                int(runtime._watermark),
            ),
            {
                "observable_relation": relation.observable_relation,
                "channel": relation.channel.value,
                "structural_signature": signature,
                "support": int(runtime.signature_support(signature, len(occurrences))),
                "parents": [[uid.hi, uid.lo] for uid in parents],
            },
            evidence,
        )
    )
    return signature


def _relation_evidence_available(runtime: Any, relation: Any, deferred_rows: list[Any]) -> bool:
    available_here = {row[0].uid for row in deferred_rows}
    refs = tuple(relation.provenance.parents) + tuple(relation.provenance.evidence)
    return all(
        uid in runtime.graph.nodes
        or uid in runtime._deferred_base_nodes
        or uid in available_here
        for uid in refs
    )


def _relation_context_is_novel(runtime: Any, relation: Any) -> bool:
    signature = int(relation.structural_signature)
    context = int(getattr(relation, "context_signature", 0) or 0)
    rows = runtime._m1n_occurrences.get(signature, ())
    return not any(int(getattr(row, "context_signature", 0) or 0) == context for row in rows)


def apply_canonical_commit_batch(
    runtime: Any,
    rows: Iterable[CommitPlan],
    *,
    input_bytes: int = 0,
) -> CanonicalCommitResult:
    plans = tuple(rows)
    if not plans:
        return CanonicalCommitResult((), ())

    # Materialize immutable payload expansion before acquiring the authoritative
    # runtime lock. Only state-dependent mutation remains in the critical section.
    materialized_rows = _materialize_commit_rows(plans)
    estimate, maximum_primitive_bytes = _estimate_materialized_commit(
        plans,
        materialized_rows,
        input_bytes=int(input_bytes),
        reserve_training_evidence=hasattr(runtime, "canonical_wal"),
    )
    budget = _work_budget(runtime)
    status = budget.status(estimate)
    if maximum_primitive_bytes > budget.max_continuation_bytes:
        status = CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE
    if status is not CanonicalTransactionStatus.READY:
        raise CanonicalTransactionQuarantined(
            status,
            sequences=tuple(int(plan.sequence) for plan in plans),
            estimate=estimate,
        )

    with runtime._lock:
        lock_acquired = time.perf_counter()
        deferred_groups: list[tuple[Any, ...]] = []
        deferred_dirty_signatures: set[int] = set()
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
        concrete_retained_delta = 0
        concrete_skipped_delta = 0
        concrete_nodes_avoided_delta = 0
        skipped_interaction_training_payloads: list[dict[str, Any]] = []
        admission_reason_counts: dict[str, int] = {}
        admitted_concrete_uids: set[Any] = set()
        batch_materialized_normalized_uids: set[Any] = set()
        max_cross_modal = int(runtime.config.scientific.max_cross_modal_facts_per_macro_event)
        logical_graph_generation = int(runtime.graph.generation)
        publication_generation_delta = getattr(runtime, "_deferred_publication_generation_delta", None)

        for plan in plans:
            plan_deferred_rows: list[Any] = []
            signatures: list[int] = []
            cross_modal_used = 0
            event = plan.event
            previous_interaction = None
            interaction_retained = False

            if plan.interaction_grounding is not None:
                g = plan.interaction_grounding
                previous_interaction = runtime._latest_interaction_grounding.get(
                    (int(g.environment_instance_id), int(g.episode_id))
                )

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

                if plan.relation is None or plan.normalized_write is None:
                    raise RuntimeError("interaction commit plan is incomplete")
                prior_support = int(
                    runtime.signature_support(int(plan.relation.structural_signature))
                )
                interaction_decision = should_retain_concrete(
                    runtime.config.scientific,
                    prior_support=prior_support,
                    context=plan.context,
                    isf_static=plan.isf_static,
                    retained_representatives=len(
                        runtime._m1n_occurrences.get(
                            int(plan.relation.structural_signature), ()
                        )
                    ),
                    novel_context=_relation_context_is_novel(runtime, plan.relation),
                )
                interaction_retained = bool(interaction_decision.retain)
                admission_reason_counts[interaction_decision.reason] = (
                    admission_reason_counts.get(interaction_decision.reason, 0) + 1
                )
                if plan.interaction_grounding is not None:
                    # Working-state continuity follows every observed interaction.
                    # Persistent concrete retention is a separate decision.
                    grounding = plan.interaction_grounding
                    runtime._latest_interaction_grounding[
                        (grounding.environment_instance_id, grounding.episode_id)
                    ] = grounding

                if interaction_retained:
                    plan_deferred_rows.extend(
                        materialized_rows[id(write)] for write in plan.base_writes
                    )
                    admitted_concrete_uids.update(write.node.uid for write in plan.base_writes)
                    concrete_retained_delta += 1
                else:
                    concrete_skipped_delta += 1
                    concrete_nodes_avoided_delta += sum(
                        1
                        for write in plan.base_writes
                        if write.node.uid not in runtime.graph.nodes
                        and write.node.uid not in runtime._deferred_base_nodes
                        and write.node.uid not in admitted_concrete_uids
                    )
                    if plan.base_writes:
                        skipped_interaction_training_payloads.append(
                            dict(materialized_rows[id(plan.base_writes[0])][1])
                        )

                signature = record_normalized_fast(
                    runtime,
                    plan.relation,
                    plan.normalized_write,
                    plan_deferred_rows,
                    materialized_row=materialized_rows[id(plan.normalized_write)],
                    retain_occurrence=interaction_retained,
                    batch_materialized_uids=batch_materialized_normalized_uids,
                )
                dirty_signature = _append_dirty_normalized(
                    runtime, plan.relation, plan_deferred_rows
                )
                if dirty_signature is not None:
                    deferred_dirty_signatures.add(dirty_signature)
                signatures.append(signature)
                touched_signatures.add(signature)
                next_stage = advance_stage_fast(runtime)

                if plan.isf_static is not None:
                    pvi, osi, explicit_pe, tp, ep = plan.isf_static
                    evidence_confidence = float(
                        runtime.__dict__.get("_environment_evidence_confidence", {}).get(
                            environment_id, 1.0
                        )
                    )
                    if float(pvi) <= 0.0:
                        explicit_pe *= evidence_confidence
                        tp *= evidence_confidence
                        ep *= evidence_confidence
                    recurrence = int(runtime.signature_support(signature))
                    recurrence_pe = 1.0 / max(1.0, float(prior_support + 1))
                    pe = (
                        abs(float(explicit_pe))
                        if float(explicit_pe) != 0.0
                        else recurrence_pe
                    )
                    isf_rows.append(
                        (
                            ISFComponents(
                                pvi,
                                osi,
                                pe,
                                1.0 / max(1, recurrence),
                                tp,
                                ep,
                            ),
                            int(runtime._watermark),
                            int(event.identity.causal_watermark),
                            stage_before,
                            next_stage,
                            logical_graph_generation,
                        )
                    )
                    runtime._prediction_error_sum += abs(float(pe))
                    runtime._prediction_error_count += 1

            symbol_occurrences_delta += len(plan.symbol_occurrences)
            unique_symbols.update(
                int(row.symbol_id.value) for row in plan.symbol_occurrences
            )
            index_occurrences = getattr(runtime, "_index_symbol_occurrences", None)
            if callable(index_occurrences):
                index_occurrences(plan.symbol_occurrences)
            if plan.symbol_codec_state:
                codec = DeterministicSymbolCodec.from_state_dict(
                    dict(plan.symbol_codec_state)
                )
                if codec.vocabulary_id.value not in runtime.symbol_codecs:
                    runtime.symbol_codecs[codec.vocabulary_id.value] = codec

            for symbol in plan.symbols:
                symbol_event = symbol.event
                runtime._watermark = max(
                    runtime._watermark, int(symbol_event.identity.causal_watermark)
                )
                timeline_events_delta += 1
                telemetry_events_delta += 1
                modality = int(symbol_event.identity.modality_id.value)
                modality_deltas[modality] = modality_deltas.get(modality, 0) + 1
                formation_environments.add(
                    int(symbol_event.identity.environment_instance_id)
                )

                symbol_prior_support = int(
                    runtime.signature_support(int(symbol.relation.structural_signature))
                )
                symbol_decision = should_retain_concrete(
                    runtime.config.scientific,
                    prior_support=symbol_prior_support,
                    context=plan.context,
                    isf_static=plan.isf_static,
                    retained_representatives=len(
                        runtime._m1n_occurrences.get(
                            int(symbol.relation.structural_signature), ()
                        )
                    ),
                    novel_context=_relation_context_is_novel(runtime, symbol.relation),
                )
                symbol_retained = bool(symbol_decision.retain)
                admission_reason_counts[symbol_decision.reason] = (
                    admission_reason_counts.get(symbol_decision.reason, 0) + 1
                )
                if symbol_retained:
                    plan_deferred_rows.extend(
                        materialized_rows[id(write)] for write in symbol.base_writes
                    )
                    admitted_concrete_uids.update(write.node.uid for write in symbol.base_writes)
                    concrete_retained_delta += 1
                else:
                    concrete_skipped_delta += 1
                    concrete_nodes_avoided_delta += sum(
                        1
                        for write in symbol.base_writes
                        if write.node.uid not in runtime.graph.nodes
                        and write.node.uid not in runtime._deferred_base_nodes
                        and write.node.uid not in admitted_concrete_uids
                    )

                signature = record_normalized_fast(
                    runtime,
                    symbol.relation,
                    symbol.normalized_write,
                    plan_deferred_rows,
                    materialized_row=materialized_rows[id(symbol.normalized_write)],
                    retain_occurrence=symbol_retained,
                    batch_materialized_uids=batch_materialized_normalized_uids,
                )
                dirty_signature = _append_dirty_normalized(
                    runtime, symbol.relation, plan_deferred_rows
                )
                if dirty_signature is not None:
                    deferred_dirty_signatures.add(dirty_signature)
                signatures.append(signature)
                touched_signatures.add(signature)

                if (
                    symbol.aligned_relation is not None
                    and cross_modal_used < max_cross_modal
                ):
                    if symbol.aligned_normalized_write is None:
                        raise RuntimeError("aligned symbol commit plan is incomplete")
                    aligned_retain = bool(
                        not runtime.config.scientific.concrete_admission_enabled
                        or (
                            symbol_retained
                            and interaction_retained
                            and _relation_evidence_available(
                                runtime, symbol.aligned_relation, plan_deferred_rows
                            )
                        )
                    )
                    aligned_signature = record_normalized_fast(
                        runtime,
                        symbol.aligned_relation,
                        symbol.aligned_normalized_write,
                        plan_deferred_rows,
                        materialized_row=materialized_rows[
                            id(symbol.aligned_normalized_write)
                        ],
                        retain_occurrence=aligned_retain,
                        batch_materialized_uids=batch_materialized_normalized_uids,
                    )
                    dirty_signature = _append_dirty_normalized(
                        runtime, symbol.aligned_relation, plan_deferred_rows
                    )
                    if dirty_signature is not None:
                        deferred_dirty_signatures.add(dirty_signature)
                    signatures.append(aligned_signature)
                    touched_signatures.add(aligned_signature)
                    cross_modal_used += 1
                    if plan.interaction_grounding is not None:
                        g = plan.interaction_grounding
                        symbol_structure_uid = symbol.relation.uid
                        interaction_structure_uid = int(plan.relation.uid.lo)
                        grounding_key = (
                            int(symbol_structure_uid.lo),
                            interaction_structure_uid,
                            int(g.environment_instance_id),
                            0,
                            0,
                        )
                        before_grounding = runtime.grounding.states.get(grounding_key)
                        after_grounding = runtime.grounding.observe(
                            GroundingEvidence(
                                int(symbol_structure_uid.lo),
                                interaction_structure_uid,
                                int(g.environment_instance_id),
                                0,
                                0,
                                int(runtime._watermark),
                                recurrent_symbol=True,
                                cross_modal_association=True,
                            )
                        )
                        if (
                            before_grounding is None
                            or int(after_grounding.maturity)
                            > int(before_grounding.maturity)
                        ):
                            runtime.telemetry["grounding_promotions"] += 1
                advance_stage_fast(runtime)

            for derived in plan.derived_relations:
                if derived.relation.channel.value == "CROSS_MODAL":
                    if cross_modal_used >= max_cross_modal:
                        continue
                    cross_modal_used += 1
                retain_derived_occurrence = bool(
                    not runtime.config.scientific.concrete_admission_enabled
                    or _relation_evidence_available(
                        runtime, derived.relation, plan_deferred_rows
                    )
                )
                signature = record_normalized_fast(
                    runtime,
                    derived.relation,
                    derived.write,
                    plan_deferred_rows,
                    materialized_row=materialized_rows[id(derived.write)],
                    retain_occurrence=retain_derived_occurrence,
                    batch_materialized_uids=batch_materialized_normalized_uids,
                )
                dirty_signature = _append_dirty_normalized(
                    runtime, derived.relation, plan_deferred_rows
                )
                if dirty_signature is not None:
                    deferred_dirty_signatures.add(dirty_signature)
                signatures.append(signature)
                touched_signatures.add(signature)

            if previous_interaction is not None and cross_modal_used < max_cross_modal:
                for symbol in plan.symbols:
                    if cross_modal_used >= max_cross_modal:
                        break
                    proxy = SimpleNamespace(
                        m1g=symbol.grounding,
                        base_writes=symbol.base_writes,
                        occurrence=symbol.occurrence,
                    )
                    control = shuffled_alignment_control(
                        proxy,
                        previous_interaction,
                        causal_watermark=int(runtime._watermark),
                        occurrence=symbol.occurrence,
                    )
                    write = _m1n_write(
                        control.relation,
                        watermark=int(runtime._watermark),
                        occurrence=symbol.occurrence,
                        payload_extra=control.payload(),
                    )
                    retain_control_occurrence = bool(
                        not runtime.config.scientific.concrete_admission_enabled
                        or _relation_evidence_available(
                            runtime, control.relation, plan_deferred_rows
                        )
                    )
                    signature = record_normalized_fast(
                        runtime,
                        control.relation,
                        write,
                        plan_deferred_rows,
                        retain_occurrence=retain_control_occurrence,
                        batch_materialized_uids=batch_materialized_normalized_uids,
                    )
                    dirty_signature = _append_dirty_normalized(
                        runtime, control.relation, plan_deferred_rows
                    )
                    if dirty_signature is not None:
                        deferred_dirty_signatures.add(dirty_signature)
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

        commit_skipped_evidence = getattr(
            runtime, "_commit_skipped_interaction_training_evidence", None
        )
        if skipped_interaction_training_payloads and callable(commit_skipped_evidence):
            commit_skipped_evidence(tuple(skipped_interaction_training_payloads))

        if deferred_groups:
            defer_groups = getattr(runtime, "_defer_base_groups", None)
            if callable(defer_groups):
                defer_groups(tuple(deferred_groups))
                for signature in deferred_dirty_signatures:
                    runtime._m1n_dirty.discard(signature)
            else:
                runtime._defer_base_group(tuple(row for group in deferred_groups for row in group))
        trim_replay_pool(runtime)
        runtime._formation_environments.update(formation_environments)
        runtime.timeline.events_seen += timeline_events_delta
        runtime.timeline.actions_committed += timeline_actions_delta
        runtime.timeline.last_ordering_key = last_ordering_key
        runtime.telemetry["events"] += telemetry_events_delta
        runtime.telemetry["concrete_admission_retained_events"] = int(
            runtime.telemetry.get("concrete_admission_retained_events", 0)
        ) + int(concrete_retained_delta)
        runtime.telemetry["concrete_admission_skipped_events"] = int(
            runtime.telemetry.get("concrete_admission_skipped_events", 0)
        ) + int(concrete_skipped_delta)
        runtime.telemetry["concrete_nodes_avoided"] = int(
            runtime.telemetry.get("concrete_nodes_avoided", 0)
        ) + int(concrete_nodes_avoided_delta)
        for reason, count in admission_reason_counts.items():
            key = f"concrete_admission_reason_{reason}"
            runtime.telemetry[key] = int(runtime.telemetry.get(key, 0)) + int(count)
        concrete_total = int(runtime.telemetry["concrete_admission_retained_events"]) + int(
            runtime.telemetry["concrete_admission_skipped_events"]
        )
        runtime.set_telemetry_gauge(
            "concrete_admission_retention_rate",
            float(runtime.telemetry["concrete_admission_retained_events"])
            / max(1, concrete_total),
        )
        runtime.set_telemetry_gauge(
            "concrete_nodes_avoided", int(runtime.telemetry["concrete_nodes_avoided"])
        )
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
        return CanonicalCommitResult(tuple(signature_rows), candidates, time.perf_counter() - lock_acquired)
