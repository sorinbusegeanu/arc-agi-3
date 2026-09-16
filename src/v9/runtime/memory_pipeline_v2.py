from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.m1_grounded import M1GroundedContingency
from v9.memory.m1_normalized import M1NormalizedRelation
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.symbolic_relations import derive_symbolic_relations
from v9.memory.v978_descriptors import modality_neutral_family_signature
from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent

from .memory_pipeline import DerivationTask, IngestionTask, PreparedIngestion, derive_memory, prepare_ingestion
from .multiprocess import WorkerStop
from .shared_batch_transport import publish_shared_batch


@dataclass(frozen=True, slots=True)
class IngestionBatchTask:
    start_sequence: int
    end_sequence: int
    tasks: tuple[IngestionTask, ...]


@dataclass(frozen=True, slots=True)
class DerivationBatchTask:
    start_task_id: int
    end_task_id: int
    tasks: tuple[DerivationTask, ...]


@dataclass(frozen=True, slots=True)
class CanonicalWrite:
    node: CanonicalNode
    payload: dict[str, Any]
    evidence: tuple[MemoryUid, ...]

    def runtime_row(self) -> tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]:
        return self.node, self.payload, self.evidence


@dataclass(frozen=True, slots=True)
class SymbolCommitPlan:
    event: PassiveSymbolEvent
    grounding: M1GroundedContingency
    relation: M1NormalizedRelation
    aligned_relation: M1NormalizedRelation | None
    base_writes: tuple[CanonicalWrite, ...]
    normalized_write: CanonicalWrite
    aligned_normalized_write: CanonicalWrite | None
    occurrence: Any | None = None


@dataclass(frozen=True, slots=True)
class DerivedRelationCommitPlan:
    relation: M1NormalizedRelation
    write: CanonicalWrite


@dataclass(frozen=True, slots=True)
class CommitPlan:
    sequence: int
    identity: Any
    event: InteractionEvent | None
    interaction_grounding: M1GroundedContingency | None
    relation: M1NormalizedRelation | None
    base_writes: tuple[CanonicalWrite, ...]
    normalized_write: CanonicalWrite | None
    symbols: tuple[SymbolCommitPlan, ...]
    symbol_codec_state: dict[str, Any] | None
    symbol_occurrences: tuple[Any, ...]
    curriculum_step: str | None
    game_scenario: str
    isf_static: tuple[float, float, float, float, float] | None
    transition: Any | None = None
    derived_relations: tuple[DerivedRelationCommitPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedCommitBatch:
    start_sequence: int
    end_sequence: int
    rows: tuple[CommitPlan, ...]


def _occurrence_payload(occurrence: Any | None) -> dict[str, Any]:
    if occurrence is None:
        return {}
    return {
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


def _nearby_payload(transition: Any, occurrence: Any | None) -> dict[str, Any]:
    if transition is None or occurrence is None:
        return {}
    transformation = stable_u64(
        int(transition.before_signature),
        int(transition.after_signature),
        int(bool(transition.semantic_delta)),
        person=b"v9-symbol-nearby",
    )
    return {
        "nearby_context_signature": int(transition.before_signature),
        "nearby_action_id": int(transition.action_id),
        "nearby_transformation_signature": int(transformation),
        "nearby_progress": bool(transition.task_success or int(transition.levels_completed) > 0),
        "nearby_outcome": int(transition.primary_valence),
        "nearby_boundary_scope": str(transition.boundary_scope),
        "nearby_task_success": bool(transition.task_success),
        "nearby_task_failure": bool(transition.task_failure),
    }


def _m0_write(m0: Any, event: Any, transition: Any = None, occurrence: Any | None = None) -> CanonicalWrite:
    return CanonicalWrite(
        CanonicalNode(m0.uid, MemoryLevel.M0, MemoryType.EPISODE, (event.identity.event_id.hi, event.identity.event_id.lo), int(event.identity.causal_watermark)),
        {
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
            "evidence_confidence": 1.0,
            **_occurrence_payload(occurrence),
            **_nearby_payload(transition, occurrence),
            **({"task_success": bool(transition.task_success), "task_failure": bool(transition.task_failure), "task_truncated": bool(transition.task_truncated), "level_index": int(transition.level_index), "levels_completed": int(transition.levels_completed)} if transition is not None else {}),
            **({"semantic_before": [list(row) for row in transition.semantic_before]} if transition is not None and transition.semantic_before else {}),
            **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition is not None and transition.semantic_action else {}),
            **({"semantic_options": [list(row) for row in transition.semantic_options]} if transition is not None and transition.semantic_options else {}),
            **({"semantic_after": [list(row) for row in transition.semantic_after]} if transition is not None and transition.semantic_after else {}),
            **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition is not None and transition.semantic_delta else {}),
        },
        (m0.uid,),
    )


def _m1g_write(m1g: Any, m0: Any, event: Any, transition: Any = None, occurrence: Any | None = None) -> CanonicalWrite:
    return CanonicalWrite(
        CanonicalNode(m1g.uid, MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (m1g.uid.hi, m1g.uid.lo), int(event.identity.causal_watermark)),
        {
            "relation": m1g.relation.value,
            "environment_instance_id": m1g.environment_instance_id,
            "episode_id": m1g.episode_id,
            "grounded_context_signature": m1g.grounded_context_signature,
            "executable_action_token": m1g.executable_action_token,
            "realized_transition_signature": m1g.realized_transition_signature,
            "grounded_next_context_signature": m1g.grounded_next_context_signature,
            "evidence_confidence": 1.0,
            **_occurrence_payload(occurrence),
            **_nearby_payload(transition, occurrence),
            **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition is not None and transition.semantic_action else {}),
            **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition is not None and transition.semantic_delta else {}),
            "parents": [[m0.uid.hi, m0.uid.lo]],
        },
        (m0.uid,),
    )


def _m1n_write(relation: M1NormalizedRelation, *, watermark: int, transition: Any = None, occurrence: Any | None = None, payload_extra: dict[str, Any] | None = None) -> CanonicalWrite:
    parents = tuple(relation.provenance.parents)
    evidence = tuple(relation.provenance.evidence)
    payload = {
        "observable_relation": relation.observable_relation,
        "channel": relation.channel.value,
        "structural_signature": relation.structural_signature,
        "family_signature": int(relation.family_signature or relation.structural_signature),
        "context_signature": int(relation.context_signature),
        "support": float(relation.support),
        "contradiction": float(relation.contradiction),
        "temporal_offsets": list(relation.temporal_offsets),
        "temporal_offset_range": None if relation.temporal_offset_range is None else list(relation.temporal_offset_range),
        "causal_watermark": int(relation.causal_watermark or watermark),
        "heldout_transfer": bool(relation.heldout_transfer),
        "evidence_confidence": 1.0,
        **_occurrence_payload(occurrence),
        **_nearby_payload(transition, occurrence),
        "parents": [[uid.hi, uid.lo] for uid in parents],
        **({"semantic_before": [list(row) for row in transition.semantic_before]} if transition is not None and transition.semantic_before else {}),
        **({"semantic_action": [list(row) for row in transition.semantic_action]} if transition is not None and transition.semantic_action else {}),
        **({"semantic_options": [list(row) for row in transition.semantic_options]} if transition is not None and transition.semantic_options else {}),
        **({"semantic_after": [list(row) for row in transition.semantic_after]} if transition is not None and transition.semantic_after else {}),
        **({"semantic_effects": [list(row) for row in transition.semantic_delta]} if transition is not None and transition.semantic_delta else {}),
    }
    if payload_extra:
        payload.update(payload_extra)
    return CanonicalWrite(CanonicalNode(relation.uid, MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (relation.structural_signature,), int(watermark)), payload, evidence)


def _neutralize_prepared_relation(relation: M1NormalizedRelation | None, neutral_family: int | None) -> M1NormalizedRelation | None:
    if relation is None or neutral_family is None:
        return relation
    return replace(relation, family_signature=int(neutral_family))


def build_commit_plan(prepared: PreparedIngestion) -> CommitPlan:
    base_writes: tuple[CanonicalWrite, ...] = ()
    normalized_write: CanonicalWrite | None = None
    isf_static: tuple[float, float, float, float, float] | None = None
    original_family = None if prepared.m1n is None else int(prepared.m1n.family_signature or prepared.m1n.structural_signature)
    neutral_family = None
    relation = prepared.m1n
    if prepared.event is not None:
        if prepared.m0 is None or prepared.m1g is None or prepared.m1n is None:
            raise RuntimeError("prepared interaction is incomplete")
        descriptor_payload = {
            "semantic_effects": [list(row) for row in prepared.transition.semantic_delta],
        }
        neutral_family = int(modality_neutral_family_signature(prepared.m1n, descriptor_payload))
        relation = replace(prepared.m1n, family_signature=neutral_family, context_signature=int(prepared.m1g.grounded_context_signature))
        base_writes = (_m0_write(prepared.m0, prepared.event, prepared.transition), _m1g_write(prepared.m1g, prepared.m0, prepared.event, prepared.transition))
        normalized_write = _m1n_write(relation, watermark=int(prepared.event.identity.causal_watermark), transition=prepared.transition)
        experience = prepared.event.experience
        isf_static = (
            abs(float(experience.primary_valence)), abs(float(experience.future_option_delta)), float(experience.prediction_error),
            0.5 if experience.family_signature else 0.0, min(1.0, float(experience.changed_cells) / 16.0),
        )

    symbols: list[SymbolCommitPlan] = []
    for index, symbol in enumerate(prepared.symbols):
        occurrence = prepared.symbol_occurrences[index] if index < len(prepared.symbol_occurrences) else None
        symbol_relation = symbol.m1n
        aligned_relation = symbol.aligned_m1n
        if neutral_family is not None and original_family is not None:
            if int(symbol_relation.family_signature or symbol_relation.structural_signature) == original_family:
                symbol_relation = replace(symbol_relation, family_signature=neutral_family, context_signature=int(prepared.m1g.grounded_context_signature if prepared.m1g is not None else 0))
            if aligned_relation is not None and int(aligned_relation.family_signature or aligned_relation.structural_signature) == original_family:
                aligned_relation = replace(aligned_relation, family_signature=neutral_family, context_signature=int(prepared.m1g.grounded_context_signature if prepared.m1g is not None else 0))
        writes = (
            _m0_write(symbol.m0, symbol.event, prepared.transition, occurrence=occurrence),
            _m1g_write(symbol.m1g, symbol.m0, symbol.event, prepared.transition, occurrence=occurrence),
        )
        symbols.append(SymbolCommitPlan(
            symbol.event, symbol.m1g, symbol_relation, aligned_relation, writes,
            _m1n_write(symbol_relation, watermark=int(symbol.event.identity.causal_watermark), transition=prepared.transition, occurrence=occurrence),
            None if aligned_relation is None else _m1n_write(aligned_relation, watermark=int(symbol.event.identity.causal_watermark), transition=prepared.transition, occurrence=occurrence),
            occurrence,
        ))

    derived_plans: list[DerivedRelationCommitPlan] = []
    if prepared.symbols:
        watermark = max(int(row.event.identity.causal_watermark) for row in prepared.symbols)
        derived = derive_symbolic_relations(
            prepared.symbols,
            interaction_grounding=prepared.m1g,
            previous_interaction_grounding=None,
            transition=prepared.transition,
            causal_watermark=watermark,
            occurrences=prepared.symbol_occurrences,
        )
        for item in derived:
            derived_relation = item.relation
            if neutral_family is not None and original_family is not None and int(derived_relation.family_signature or derived_relation.structural_signature) == original_family:
                derived_relation = replace(derived_relation, family_signature=neutral_family, context_signature=int(prepared.m1g.grounded_context_signature if prepared.m1g is not None else 0))
            payload_extra = item.payload()
            payload_extra["family_signature"] = int(derived_relation.family_signature or derived_relation.structural_signature)
            derived_plans.append(DerivedRelationCommitPlan(
                derived_relation,
                _m1n_write(derived_relation, watermark=watermark, transition=prepared.transition, payload_extra=payload_extra),
            ))

    return CommitPlan(
        int(prepared.sequence), prepared.identity, prepared.event, prepared.m1g, relation,
        base_writes, normalized_write, tuple(symbols), prepared.symbol_codec_state, prepared.symbol_occurrences,
        prepared.transition.curriculum_step, str(prepared.transition.game_scenario), isf_static, prepared.transition,
        tuple(derived_plans),
    )


def prepare_commit_batch(batch: IngestionBatchTask) -> PreparedCommitBatch:
    if not batch.tasks:
        raise ValueError("ingestion batch must contain tasks")
    rows = tuple(build_commit_plan(prepare_ingestion(task)) for task in batch.tasks)
    if rows[0].sequence != int(batch.start_sequence) or rows[-1].sequence != int(batch.end_sequence):
        raise RuntimeError("prepared commit batch sequence mismatch")
    return PreparedCommitBatch(int(batch.start_sequence), int(batch.end_sequence), rows)


def ingest_batch_worker_main(task_queue: Any, result_queue: Any) -> None:
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, IngestionBatchTask):
            continue
        try:
            result = prepare_commit_batch(item)
            descriptor = publish_shared_batch(result, start_sequence=result.start_sequence, end_sequence=result.end_sequence, rows=len(result.rows))
            result_queue.put(("ingest_batch_shm", result.start_sequence, result.end_sequence, descriptor))
        except BaseException as exc:
            result_queue.put(("worker_error", "ingest", int(item.start_sequence), repr(exc)))


def derivation_batch_worker_main(task_queue: Any, result_queue: Any) -> None:
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, DerivationBatchTask):
            continue
        try:
            results = tuple(derive_memory(task) for task in item.tasks)
            descriptor = publish_shared_batch(results, start_sequence=item.start_task_id, end_sequence=item.end_task_id, rows=len(results))
            result_queue.put(("derivation_batch_shm", item.start_task_id, item.end_task_id, descriptor))
        except BaseException as exc:
            result_queue.put(("worker_error", "derivation", int(item.start_task_id), repr(exc)))
