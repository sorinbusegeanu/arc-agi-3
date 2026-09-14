from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.m1_grounded import M1GroundedContingency
from v9.memory.m1_normalized import M1NormalizedRelation
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
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
    relation: M1NormalizedRelation
    aligned_relation: M1NormalizedRelation | None
    base_writes: tuple[CanonicalWrite, ...]
    normalized_write: CanonicalWrite
    aligned_normalized_write: CanonicalWrite | None


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
    curriculum_step: str | None
    game_scenario: str
    isf_static: tuple[float, float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class PreparedCommitBatch:
    start_sequence: int
    end_sequence: int
    rows: tuple[CommitPlan, ...]


def _m0_write(m0: Any, event: Any, transition: Any = None) -> CanonicalWrite:
    return CanonicalWrite(
        CanonicalNode(
            m0.uid,
            MemoryLevel.M0,
            MemoryType.EPISODE,
            (event.identity.event_id.hi, event.identity.event_id.lo),
            int(event.identity.causal_watermark),
        ),
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
            "semantic_before": [] if transition is None else [list(row) for row in transition.semantic_before],
            "semantic_action": [] if transition is None else [list(row) for row in transition.semantic_action],
            "semantic_effects": [] if transition is None else [list(row) for row in transition.semantic_delta],
        },
        (m0.uid,),
    )


def _m1g_write(m1g: Any, m0: Any, event: Any, transition: Any = None) -> CanonicalWrite:
    return CanonicalWrite(
        CanonicalNode(
            m1g.uid,
            MemoryLevel.M1,
            MemoryType.GROUNDED_CONTINGENCY,
            (m1g.uid.hi, m1g.uid.lo),
            int(event.identity.causal_watermark),
        ),
        {
            "relation": m1g.relation.value,
            "environment_instance_id": m1g.environment_instance_id,
            "episode_id": m1g.episode_id,
            "grounded_context_signature": m1g.grounded_context_signature,
            "executable_action_token": m1g.executable_action_token,
            "realized_transition_signature": m1g.realized_transition_signature,
            "grounded_next_context_signature": m1g.grounded_next_context_signature,
            "semantic_action": [] if transition is None else [list(row) for row in transition.semantic_action],
            "semantic_effects": [] if transition is None else [list(row) for row in transition.semantic_delta],
            "parents": [[m0.uid.hi, m0.uid.lo]],
        },
        (m0.uid,),
    )


def _m1n_write(relation: M1NormalizedRelation, *, watermark: int) -> CanonicalWrite:
    parents = tuple(relation.provenance.parents)
    evidence = tuple(relation.provenance.evidence)
    return CanonicalWrite(
        CanonicalNode(
            relation.uid,
            MemoryLevel.M1,
            MemoryType.NORMALIZED_RELATION,
            (relation.structural_signature,),
            int(watermark),
        ),
        {
            "observable_relation": relation.observable_relation,
            "channel": relation.channel.value,
            "structural_signature": relation.structural_signature,
            "support": 1,
            "parents": [[uid.hi, uid.lo] for uid in parents],
        },
        evidence,
    )


def build_commit_plan(prepared: PreparedIngestion) -> CommitPlan:
    base_writes: tuple[CanonicalWrite, ...] = ()
    normalized_write: CanonicalWrite | None = None
    isf_static: tuple[float, float, float, float, float] | None = None
    if prepared.event is not None:
        if prepared.m0 is None or prepared.m1g is None or prepared.m1n is None:
            raise RuntimeError("prepared interaction is incomplete")
        base_writes = (
            _m0_write(prepared.m0, prepared.event),
            _m1g_write(prepared.m1g, prepared.m0, prepared.event),
        )
        normalized_write = _m1n_write(
            prepared.m1n,
            watermark=int(prepared.event.identity.causal_watermark),
        )
        experience = prepared.event.experience
        isf_static = (
            abs(float(experience.primary_valence)),
            abs(float(experience.future_option_delta)),
            float(experience.prediction_error),
            0.5 if experience.family_signature else 0.0,
            min(1.0, float(experience.changed_cells) / 16.0),
        )

    symbols: list[SymbolCommitPlan] = []
    for symbol in prepared.symbols:
        writes = (
            _m0_write(symbol.m0, symbol.event),
            _m1g_write(symbol.m1g, symbol.m0, symbol.event),
        )
        symbols.append(
            SymbolCommitPlan(
                symbol.event,
                symbol.m1n,
                symbol.aligned_m1n,
                writes,
                _m1n_write(symbol.m1n, watermark=int(symbol.event.identity.causal_watermark)),
                None
                if symbol.aligned_m1n is None
                else _m1n_write(
                    symbol.aligned_m1n,
                    watermark=int(symbol.event.identity.causal_watermark),
                ),
            )
        )

    return CommitPlan(
        int(prepared.sequence),
        prepared.identity,
        prepared.event,
        prepared.m1g,
        prepared.m1n,
        base_writes,
        normalized_write,
        tuple(symbols),
        prepared.symbol_codec_state,
        prepared.transition.curriculum_step,
        str(prepared.transition.game_scenario),
        isf_static,
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
            descriptor = publish_shared_batch(
                result,
                start_sequence=result.start_sequence,
                end_sequence=result.end_sequence,
                rows=len(result.rows),
            )
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
            descriptor = publish_shared_batch(
                results,
                start_sequence=item.start_task_id,
                end_sequence=item.end_task_id,
                rows=len(results),
            )
            result_queue.put(
                ("derivation_batch_shm", item.start_task_id, item.end_task_id, descriptor)
            )
        except BaseException as exc:
            result_queue.put(
                ("worker_error", "derivation", int(item.start_task_id), repr(exc))
            )
