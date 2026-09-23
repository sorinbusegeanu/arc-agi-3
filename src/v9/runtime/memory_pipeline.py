from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from v9.cognition.compression import form_families
from v9.cognition.roles import form_roles
from v9.environments.schemas import EnvironmentIdentity
from v9.memory.identity import EpisodeId, EventUid, MemoryUid, stable_u64
from v9.memory.m0_episode import M0Episode
from v9.memory.m1_grounded import GroundedRelation, M1GroundedContingency
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.m2_family import M2TransformationFamily
from v9.memory.m3_role import M3FunctionalRole
from v9.memory.m4_concept import M4Concept
from v9.memory.model import CanonicalNode, ExperienceEvent, MemoryLevel, MemoryType
from v9.memory.symbolic_relations import derive_symbolic_relations
from v9.memory.v978_descriptors import modality_neutral_family_signature
from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent, SYMBOL_MODALITY, TimelineIdentity, WORLD_MODALITY
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolOccurrence, SymbolTemporalPhase

from .multiprocess import EncodedTransition, WorkerStop
from .shared_batch_transport import (
    SlabOwnership,
    TransportSlabPool,
    encode_transport_value,
    publish_shared_batch,
)


@dataclass(frozen=True, slots=True)
class IngestionTask:
    sequence: int
    causal_watermark: int
    transition: EncodedTransition
    symbol_budget_per_window: int = 8
    symbol_payload_bytes: int = 4096
    max_cross_modal_facts_per_macro_event: int = 16
    symbol_deduplication_policy: str = "token_phase_time"
    symbol_window_time_span: int = 64
    symbol_codec_name: str = "deterministic-opaque"
    symbol_codec_version: int = 1
    input_bytes: int = 0


@dataclass(frozen=True, slots=True)
class PreparedSymbolIngestion:
    event: PassiveSymbolEvent
    m0: M0Episode
    m1g: M1GroundedContingency
    m1n: M1NormalizedRelation
    aligned_m1n: M1NormalizedRelation | None


@dataclass(frozen=True, slots=True)
class PreparedIngestion:
    sequence: int
    transition: EncodedTransition
    identity: EnvironmentIdentity
    event: InteractionEvent | None
    m0: M0Episode | None
    m1g: M1GroundedContingency | None
    m1n: M1NormalizedRelation | None
    symbols: tuple[PreparedSymbolIngestion, ...] = ()
    symbol_codec_state: dict[str, Any] | None = None
    symbol_occurrences: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivationTask:
    task_id: int
    structural_signature: int
    rows: tuple[M1NormalizedRelation, ...]
    support: int
    formation_scope: tuple[int, ...]
    causal_watermark: int
    evidence_confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class DerivationResult:
    task_id: int
    structural_signature: int
    support: int
    family: M2TransformationFamily
    roles: tuple[M3FunctionalRole, ...]
    concepts: tuple[M4Concept, ...]
    causal_watermark: int


def _semantic_signature(rows: tuple[tuple[int, int, int, int, float], ...], fallback: tuple[int, ...], *, person: bytes) -> int:
    if not rows:
        return stable_u64(*fallback, person=person)
    parts: list[object] = []
    for kind, subject, relation, obj, value in rows:
        parts.extend((int(kind), int(subject), int(relation), int(obj), repr(float(value))))
    return stable_u64(*parts, person=person)


def _interaction_family_key(row: M1GroundedContingency) -> tuple[object, ...]:
    """Modality-neutral transformation descriptor used by WORLD/SYMBOL/CROSS_MODAL M1N."""
    return (
        "INTERACTION_TRANSFORMATION",
        int(row.grounded_context_signature),
        int(row.realized_transition_signature),
        int(row.grounded_next_context_signature),
    )


def _symbol_input(raw: object, *, task: IngestionTask, transition: EncodedTransition, index: int) -> tuple[str | bytes | int, str, int, int, int, int]:
    phase = SymbolTemporalPhase.AFTER_ACTION.value
    macro_step = int(transition.global_step)
    micro_step = int(index)
    watermark: int | None = None
    source_sequence = int(transition.producer_sequence)
    token: object = raw

    if isinstance(raw, Mapping):
        if "token" in raw:
            token = raw["token"]
        elif "value" in raw:
            token = raw["value"]
        elif "symbol" in raw:
            token = raw["symbol"]
        else:
            raise ValueError("timed symbol mapping requires token, value, or symbol")
        phase = str(raw.get("phase", phase)).upper()
        macro_step = int(raw.get("macro_step", macro_step))
        micro_step = int(raw.get("micro_step", micro_step))
        watermark = None if raw.get("causal_watermark") is None else int(raw["causal_watermark"])
        source_sequence = int(raw.get("source_sequence", source_sequence))
    elif hasattr(raw, "token") or hasattr(raw, "value"):
        token = getattr(raw, "token", getattr(raw, "value", raw))
        phase = str(getattr(raw, "phase", phase)).upper()
        macro_step = int(getattr(raw, "macro_step", macro_step))
        micro_step = int(getattr(raw, "micro_step", micro_step))
        explicit = getattr(raw, "causal_watermark", None)
        watermark = None if explicit is None else int(explicit)
        source_sequence = int(getattr(raw, "source_sequence", source_sequence))

    SymbolTemporalPhase(phase)
    if not isinstance(token, (str, bytes, int)):
        raise TypeError("symbol token must be str, bytes, or int")
    if watermark is None:
        watermark = max(0, int(task.causal_watermark) - 1) if phase == SymbolTemporalPhase.BEFORE_ACTION.value else int(task.causal_watermark)
    if watermark < 0 or macro_step < 0 or micro_step < 0 or source_sequence < 0:
        raise ValueError("symbol timing fields cannot be negative")
    return token, phase, watermark, macro_step, micro_step, source_sequence


def _symbol_size(token: str | bytes | int) -> int:
    if isinstance(token, bytes):
        return len(token)
    if isinstance(token, str):
        return len(token.encode("utf-8"))
    return 16


def _dedup_key(row: tuple[str | bytes | int, str, int, int, int, int], policy: str, index: int) -> tuple[object, ...]:
    token, phase, watermark, macro_step, micro_step, _source_sequence = row
    if policy == "none":
        return (index,)
    if policy == "token":
        return (token,)
    if policy == "token_phase":
        return token, phase
    if policy == "token_phase_time":
        return token, phase, watermark, macro_step, micro_step
    raise ValueError(f"unsupported symbol deduplication policy: {policy}")


def _prepare_symbols(task: IngestionTask, transition: EncodedTransition, identity: EnvironmentIdentity, interaction_grounding: M1GroundedContingency | None) -> tuple[tuple[PreparedSymbolIngestion, ...], dict[str, Any] | None, tuple[SymbolOccurrence, ...]]:
    if not transition.symbols:
        return (), None, ()
    environment = int(identity.instance_id.value)
    episode_id = EpisodeId(int(transition.episode_id))
    codec = DeterministicSymbolCodec(f"{task.symbol_codec_name}-v{int(task.symbol_codec_version)}")

    timed: list[tuple[str | bytes | int, str, int, int, int, int]] = []
    used_bytes = 0
    seen: set[tuple[object, ...]] = set()
    limit = max(1, int(task.symbol_budget_per_window))
    span = max(1, int(task.symbol_window_time_span))
    for index, raw in enumerate(tuple(transition.symbols)[:limit]):
        row = _symbol_input(raw, task=task, transition=transition, index=index)
        if abs(int(row[2]) - int(task.causal_watermark)) > span or abs(int(row[3]) - int(transition.global_step)) > span:
            continue
        size = _symbol_size(row[0])
        if used_bytes + size > max(1, int(task.symbol_payload_bytes)):
            break
        key = _dedup_key(row, str(task.symbol_deduplication_policy), index)
        if key in seen:
            continue
        seen.add(key)
        used_bytes += size
        timed.append(row)
    if not timed:
        return (), codec.state_dict(), ()

    observations = codec.encode_stream(
        tuple(row[0] for row in timed),
        stream_name=f"{environment}:{episode_id.value}:{transition.producer_sequence}",
    )
    prepared: list[PreparedSymbolIngestion] = []
    occurrences: list[SymbolOccurrence] = []
    family_key = None if interaction_grounding is None else _interaction_family_key(interaction_grounding)
    cross_modal_budget = max(0, int(task.max_cross_modal_facts_per_macro_event))
    for index, (row, timing) in enumerate(zip(observations, timed)):
        _token, phase, watermark, macro_step, micro_step, source_sequence = timing
        producer_sequence = transition.producer_sequence * 10_000 + index
        event = PassiveSymbolEvent(
            TimelineIdentity(
                EventUid.from_producer(transition.actor_id + 1_000_000, producer_sequence),
                int(watermark),
                transition.actor_id + 1_000_000,
                producer_sequence,
                environment,
                episode_id,
                SYMBOL_MODALITY,
            ),
            row.vocabulary_id,
            row.stream_id,
            row.symbol_id,
            row.position.value,
        )
        occurrence = SymbolOccurrence(
            event.identity.event_id,
            row.symbol_id,
            row.position.value,
            row.stream_id,
            row.vocabulary_id,
            f"{codec.vocabulary.codec_name}:{codec.vocabulary.version}",
            int(event.identity.causal_watermark),
            int(macro_step),
            int(micro_step),
            environment,
            episode_id,
            event.identity.event_id,
            int(SYMBOL_MODALITY.value),
            phase,
            int(source_sequence),
        )
        occurrences.append(occurrence)
        payload_digest = stable_u64(row.vocabulary_id.value, row.stream_id.value, row.symbol_id.value, row.position.value, person=b"v9-symbol-payload")
        m0 = M0Episode.from_event(event, context_signature=0, payload_digest=payload_digest)
        m1g = M1GroundedContingency.build(GroundedRelation.SYMBOL_OCCURRED, (m0,))
        symbol_key = ("SYMBOL_OCCURRED", int(row.vocabulary_id.value), int(row.symbol_id.value))
        m1n = M1NormalizedRelation.build(
            "SYMBOL_OCCURRED",
            NormalizedChannel.SYMBOL,
            (m1g,),
            causal_watermark=int(watermark),
            structural_key=symbol_key,
            family_key=family_key or symbol_key,
        )
        aligned = None
        if interaction_grounding is not None and index < cross_modal_budget:
            aligned = M1NormalizedRelation.build(
                "SYMBOL_ALIGNED_WITH_INTERACTION",
                NormalizedChannel.CROSS_MODAL,
                (interaction_grounding, m1g),
                causal_watermark=int(watermark),
                structural_key=("SYMBOL_ALIGNED_WITH_INTERACTION", int(row.vocabulary_id.value), int(row.symbol_id.value), *family_key),
                family_key=family_key,
            )
        prepared.append(PreparedSymbolIngestion(event, m0, m1g, m1n, aligned))
    return tuple(prepared), codec.state_dict(), tuple(occurrences)


def prepare_ingestion(task: IngestionTask) -> PreparedIngestion:
    transition = task.transition
    identity = EnvironmentIdentity(*transition.environment_identity)
    event: InteractionEvent | None = None
    m0: M0Episode | None = None
    m1g: M1GroundedContingency | None = None
    m1n: M1NormalizedRelation | None = None

    if not transition.symbols_only:
        environment = int(identity.instance_id.value)
        episode_id = EpisodeId(int(transition.episode_id))
        producer_id = int(transition.actor_id)
        producer_sequence = int(transition.producer_sequence)
        event_uid = EventUid.from_producer(producer_id, producer_sequence)
        experience = ExperienceEvent(
            event_uid,
            int(task.causal_watermark),
            producer_id,
            producer_sequence,
            environment,
            int(transition.global_step),
            int(transition.before_signature),
            int(transition.action_id),
            int(transition.after_signature),
            _semantic_signature(transition.semantic_delta, (int(transition.observation_schema_id), int(transition.before_signature != transition.after_signature)), person=b"v9-family"),
            _semantic_signature(transition.semantic_before, (int(transition.observation_schema_id), int(transition.before_signature)), person=b"v9-carrier"),
            float(getattr(transition, "future_option_delta", 0.0)),
            max(int(transition.before_signature != transition.after_signature), len(transition.semantic_delta)),
            int(transition.primary_valence),
            stable_u64(environment, int(transition.episode_id), person=b"v9-trajectory"),
            int(transition.after_signature),
            0.0,
        )
        event = InteractionEvent(TimelineIdentity(event_uid, int(task.causal_watermark), producer_id, producer_sequence, environment, episode_id, WORLD_MODALITY), experience)
        payload_digest = stable_u64(experience.context_signature, experience.action_id, experience.outcome_signature, person=b"v9-interaction-payload")
        m0 = M0Episode.from_event(event, context_signature=experience.context_signature, payload_digest=payload_digest)
        m1g = M1GroundedContingency.build(GroundedRelation.ACTION_CONDITIONED, (m0,))
        semantic_action_signature = _semantic_signature(transition.semantic_action, (transition.action_id,), person=b"v9-sem-action")
        semantic_delta_signature = _semantic_signature(transition.semantic_delta, (transition.before_signature, transition.after_signature), person=b"v9-sem-delta")
        observable = (f"ACTION:{transition.action_schema_id}:{identity.environment_type}:{experience.action_id}" f":SEM_ACTION:{semantic_action_signature}:SEM_DELTA:{semantic_delta_signature}" f":FAMILY:{experience.family_signature}:OUTCOME:{experience.outcome_signature}" f":OPTIONS:{transition.available_action_set_signature}:BOUNDARY:{transition.boundary_scope}" f":SUCCESS:{int(transition.task_success)}:FAILURE:{int(transition.task_failure)}" f":TRUNCATED:{int(transition.task_truncated)}:LEVEL:{transition.level_index}" f":LEVELS_COMPLETED:{transition.levels_completed}")
        family_key = _interaction_family_key(m1g)
        m1n = M1NormalizedRelation.build(
            observable,
            NormalizedChannel.WORLD,
            (m1g,),
            causal_watermark=int(task.causal_watermark),
            structural_key=("WORLD_TRANSITION", *family_key),
            family_key=family_key,
        )

    symbols, codec_state, occurrences = _prepare_symbols(task, transition, identity, m1g)
    return PreparedIngestion(task.sequence, transition, identity, event, m0, m1g, m1n, symbols, codec_state, occurrences)


def ingest_worker_main(task_queue: Any, result_queue: Any) -> None:
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, IngestionTask):
            continue
        try:
            result_queue.put(("ingest", item.sequence, prepare_ingestion(item)))
        except BaseException as exc:
            result_queue.put(("worker_error", "ingest", item.sequence, repr(exc)))


def derive_memory(task: DerivationTask) -> DerivationResult:
    if len(task.rows) < 2 or task.support < 2:
        raise ValueError("derivation requires recurrent M1 support")
    families = form_families(task.rows)
    if not families:
        raise ValueError("derivation requires at least one formable recurrent family")
    family = families[0]
    family = replace(family, recurrence=int(task.support), compression_benefit=float(task.support - 1))
    roles = form_roles(
        (family,),
        consequence_by_family={family.uid.lo: family.structural_signature},
    )
    confidence = max(0.10, min(1.0, float(task.evidence_confidence)))
    family = replace(family, compression_benefit=family.compression_benefit * confidence)
    concepts = tuple(
        M4Concept.candidate(
            (role,),
            compression_benefit=family.compression_benefit,
            explanatory_reach=max(1, round(len(role.provenance.evidence) * confidence)),
            transfer_prior=0.5 * confidence,
            formation_scope=task.formation_scope,
        )
        for role in roles
    )
    return DerivationResult(task.task_id, task.structural_signature, task.support, family, roles, concepts, task.causal_watermark)


def derivation_worker_main(task_queue: Any, result_queue: Any) -> None:
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, DerivationTask):
            continue
        try:
            result_queue.put(("derivation", item.task_id, derive_memory(item)))
        except BaseException as exc:
            result_queue.put(("worker_error", "derivation", item.task_id, repr(exc)))


@dataclass(frozen=True, slots=True)
class IngestionBatchTask:
    start_sequence: int
    end_sequence: int
    tasks: tuple[IngestionTask, ...]
    task_input_bytes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.task_input_bytes and len(self.task_input_bytes) != len(self.tasks):
            raise ValueError("ingestion task byte measurements must match task rows")
        if any(int(value) < 0 for value in self.task_input_bytes):
            raise ValueError("ingestion task byte measurements cannot be negative")

    @property
    def input_bytes(self) -> int:
        return sum(_ingestion_task_input_bytes(self))


def _ingestion_task_input_bytes(batch: IngestionBatchTask) -> tuple[int, ...]:
    """Read byte metadata from current and already-in-flight batch objects.

    A long-running coordinator can have queued a slotted dataclass instance
    before this additive field was deployed, then spawn a worker importing the
    newer class.  Unpickling preserves the older state and leaves the new slot
    unset, so all worker-side reads must tolerate that rolling boundary.
    """

    tasks = tuple(batch.tasks)
    measured = tuple(int(value) for value in (getattr(batch, "task_input_bytes", ()) or ()))
    if measured:
        if len(measured) != len(tasks):
            raise ValueError("ingestion task byte measurements must match task rows")
        return measured
    return tuple(int(getattr(task, "input_bytes", 0)) for task in tasks)


@dataclass(frozen=True, slots=True)
class DerivationBatchTask:
    start_task_id: int
    end_task_id: int
    tasks: tuple[DerivationTask, ...]


@dataclass(frozen=True, slots=True)
class TransitionCommitContext:
    before_signature: int
    action_id: int
    after_signature: int
    primary_valence: int
    boundary_scope: str
    task_success: bool
    task_failure: bool
    task_truncated: bool
    level_index: int
    levels_completed: int
    semantic_before: tuple[tuple[int, int, int, int, float], ...]
    semantic_action: tuple[tuple[int, int, int, int, float], ...]
    semantic_options: tuple[tuple[int, int, int, int, float], ...]
    semantic_after: tuple[tuple[int, int, int, int, float], ...]
    semantic_delta: tuple[tuple[int, int, int, int, float], ...]
    actor_id: int
    producer_sequence: int
    global_step: int
    environment_identity: tuple[str, str, str, str]
    game_scenario: str
    sampling_branch: str


def _commit_context(transition: Any) -> TransitionCommitContext:
    return TransitionCommitContext(
        int(transition.before_signature),
        int(transition.action_id),
        int(transition.after_signature),
        int(transition.primary_valence),
        str(transition.boundary_scope),
        bool(transition.task_success),
        bool(transition.task_failure),
        bool(transition.task_truncated),
        int(transition.level_index),
        int(transition.levels_completed),
        tuple(transition.semantic_before),
        tuple(transition.semantic_action),
        tuple(transition.semantic_options),
        tuple(transition.semantic_after),
        tuple(transition.semantic_delta),
        int(transition.actor_id),
        int(transition.producer_sequence),
        int(transition.global_step),
        tuple(str(value) for value in transition.environment_identity),
        str(transition.game_scenario),
        str(getattr(transition, "sampling_branch", "")),
    )


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


def _nearby_payload(context: TransitionCommitContext | None, occurrence: Any | None) -> dict[str, Any]:
    if context is None or occurrence is None:
        return {}
    transformation = stable_u64(
        int(context.before_signature),
        int(context.after_signature),
        int(bool(context.semantic_delta)),
        person=b"v9-symbol-nearby",
    )
    return {
        "nearby_context_signature": int(context.before_signature),
        "nearby_action_id": int(context.action_id),
        "nearby_transformation_signature": int(transformation),
        "nearby_progress": bool(context.task_success or int(context.levels_completed) > 0),
        "nearby_outcome": int(context.primary_valence),
        "nearby_boundary_scope": str(context.boundary_scope),
        "nearby_task_success": bool(context.task_success),
        "nearby_task_failure": bool(context.task_failure),
    }


def _semantic_payload(context: TransitionCommitContext | None, scope: str) -> dict[str, Any]:
    if context is None:
        return {}
    payload: dict[str, Any] = {}
    if scope == "m0":
        payload.update(
            {
                "task_success": bool(context.task_success),
                "task_failure": bool(context.task_failure),
                "task_truncated": bool(context.task_truncated),
                "level_index": int(context.level_index),
                "levels_completed": int(context.levels_completed),
                "actor_id": int(context.actor_id),
                "producer_sequence": int(context.producer_sequence),
                "global_step": int(context.global_step),
                "environment_identity": list(context.environment_identity),
                "game_scenario": str(context.game_scenario),
                "sampling_branch": str(context.sampling_branch),
                "before_signature": int(context.before_signature),
                "after_signature": int(context.after_signature),
            }
        )
    if scope in {"m0", "m1n"}:
        if context.semantic_before:
            payload["semantic_before"] = [list(row) for row in context.semantic_before]
        if context.semantic_action:
            payload["semantic_action"] = [list(row) for row in context.semantic_action]
        if context.semantic_options:
            payload["semantic_options"] = [list(row) for row in context.semantic_options]
        if context.semantic_after:
            payload["semantic_after"] = [list(row) for row in context.semantic_after]
        if context.semantic_delta:
            payload["semantic_effects"] = [list(row) for row in context.semantic_delta]
    elif scope == "m1g":
        if context.semantic_action:
            payload["semantic_action"] = [list(row) for row in context.semantic_action]
        if context.semantic_delta:
            payload["semantic_effects"] = [list(row) for row in context.semantic_delta]
    return payload


@dataclass(frozen=True, slots=True)
class CanonicalWrite:
    node: CanonicalNode
    payload: dict[str, Any]
    evidence: tuple[MemoryUid, ...]
    context: TransitionCommitContext | None = None
    occurrence: Any | None = None
    payload_scope: str = "none"

    def runtime_row(self) -> tuple[CanonicalNode, dict[str, Any], tuple[MemoryUid, ...]]:
        payload = dict(self.payload)
        payload.update(_occurrence_payload(self.occurrence))
        payload.update(_nearby_payload(self.context, self.occurrence))
        payload.update(_semantic_payload(self.context, self.payload_scope))
        return self.node, payload, self.evidence


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
class CanonicalMutationIntent:
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
    context: TransitionCommitContext | None = None
    derived_relations: tuple[DerivedRelationCommitPlan, ...] = ()


# Backward-compatible type name for callers/tests; the runtime boundary is the compiled intent.
CommitPlan = CanonicalMutationIntent


@dataclass(frozen=True, slots=True)
class PreparedCommitBatch:
    start_sequence: int
    end_sequence: int
    rows: tuple[Any, ...]
    input_bytes: int = 0
    row_input_bytes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.row_input_bytes and len(self.row_input_bytes) != len(self.rows):
            raise ValueError("prepared commit byte measurements must match rows")
        if any(int(value) < 0 for value in self.row_input_bytes):
            raise ValueError("prepared commit row bytes cannot be negative")
        if self.row_input_bytes and sum(self.row_input_bytes) != int(self.input_bytes):
            raise ValueError("prepared commit row bytes must equal carried input bytes")


def _m0_write(m0: Any, event: Any, context: TransitionCommitContext | None = None, occurrence: Any | None = None) -> CanonicalWrite:
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
        },
        (m0.uid,),
        context,
        occurrence,
        "m0",
    )


def _m1g_write(m1g: Any, m0: Any, event: Any, context: TransitionCommitContext | None = None, occurrence: Any | None = None) -> CanonicalWrite:
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
            "parents": [[m0.uid.hi, m0.uid.lo]],
        },
        (m0.uid,),
        context,
        occurrence,
        "m1g",
    )


def _m1n_write(
    relation: M1NormalizedRelation,
    *,
    watermark: int,
    context: TransitionCommitContext | None = None,
    transition: Any = None,
    occurrence: Any | None = None,
    payload_extra: dict[str, Any] | None = None,
) -> CanonicalWrite:
    if context is None and transition is not None:
        context = _commit_context(transition)
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
        "parents": [[uid.hi, uid.lo] for uid in parents],
    }
    if payload_extra:
        payload.update(payload_extra)
    return CanonicalWrite(
        CanonicalNode(relation.uid, MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (relation.structural_signature,), int(watermark)),
        payload,
        evidence,
        context,
        occurrence,
        "m1n",
    )


def _neutralize_prepared_relation(relation: M1NormalizedRelation | None, neutral_family: int | None) -> M1NormalizedRelation | None:
    if relation is None or neutral_family is None:
        return relation
    return replace(relation, family_signature=int(neutral_family))


def build_commit_plan(prepared: PreparedIngestion) -> CommitPlan:
    base_writes: tuple[CanonicalWrite, ...] = ()
    normalized_write: CanonicalWrite | None = None
    isf_static: tuple[float, float, float, float, float] | None = None
    context = _commit_context(prepared.transition)
    original_family = None if prepared.m1n is None else int(prepared.m1n.family_signature or prepared.m1n.structural_signature)
    neutral_family = None
    relation = prepared.m1n
    if prepared.event is not None:
        if prepared.m0 is None or prepared.m1g is None or prepared.m1n is None:
            raise RuntimeError("prepared interaction is incomplete")
        descriptor_payload = {
            "semantic_effects": [list(row) for row in context.semantic_delta],
        }
        neutral_family = int(modality_neutral_family_signature(prepared.m1n, descriptor_payload))
        relation = replace(prepared.m1n, family_signature=neutral_family, context_signature=int(prepared.m1g.grounded_context_signature))
        base_writes = (
            _m0_write(prepared.m0, prepared.event, context),
            _m1g_write(prepared.m1g, prepared.m0, prepared.event, context),
        )
        normalized_write = _m1n_write(relation, watermark=int(prepared.event.identity.causal_watermark), context=context)
        experience = prepared.event.experience
        isf_static = (
            abs(float(experience.primary_valence)),
            abs(float(experience.future_option_delta)),
            float(experience.prediction_error),
            0.5 if experience.family_signature else 0.0,
            min(1.0, float(experience.changed_cells) / 16.0),
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
            _m0_write(symbol.m0, symbol.event, context, occurrence=occurrence),
            _m1g_write(symbol.m1g, symbol.m0, symbol.event, context, occurrence=occurrence),
        )
        symbols.append(
            SymbolCommitPlan(
                symbol.event,
                symbol.m1g,
                symbol_relation,
                aligned_relation,
                writes,
                _m1n_write(symbol_relation, watermark=int(symbol.event.identity.causal_watermark), context=context, occurrence=occurrence),
                None if aligned_relation is None else _m1n_write(aligned_relation, watermark=int(symbol.event.identity.causal_watermark), context=context, occurrence=occurrence),
                occurrence,
            )
        )

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
            derived_plans.append(
                DerivedRelationCommitPlan(
                    derived_relation,
                    _m1n_write(derived_relation, watermark=watermark, context=context, payload_extra=payload_extra),
                )
            )

    return CommitPlan(
        int(prepared.sequence),
        prepared.identity,
        prepared.event,
        prepared.m1g,
        relation,
        base_writes,
        normalized_write,
        tuple(symbols),
        prepared.symbol_codec_state,
        prepared.symbol_occurrences,
        prepared.transition.curriculum_step,
        str(prepared.transition.game_scenario),
        isf_static,
        context,
        tuple(derived_plans),
    )


def prepare_commit_batch(batch: IngestionBatchTask) -> PreparedCommitBatch:
    if not batch.tasks:
        raise ValueError("ingestion batch must contain tasks")
    rows = tuple(build_commit_plan(prepare_ingestion(task)) for task in batch.tasks)
    if rows[0].sequence != int(batch.start_sequence) or rows[-1].sequence != int(batch.end_sequence):
        raise RuntimeError("prepared commit batch sequence mismatch")
    return PreparedCommitBatch(
        int(batch.start_sequence), int(batch.end_sequence), rows, int(batch.input_bytes)
    )


_INGEST_RESULT_CHUNK_SIZE = 128


def _publish_compiled_result(
    result_queue: Any,
    pool: TransportSlabPool,
    value: Any,
    *,
    producer_id: int,
    start_sequence: int,
    end_sequence: int,
    rows: int,
    message: tuple[Any, ...],
) -> None:
    encode_started = time.perf_counter()
    payload = encode_transport_value(value)
    descriptor = None
    while descriptor is None:
        try:
            descriptor = pool.write(
                payload,
                producer_id=int(producer_id),
                start_sequence=int(start_sequence),
                end_sequence=int(end_sequence),
                rows=int(rows),
            )
        except BufferError:
            time.sleep(0.001)
    try:
        result_queue.put(
            (*message, descriptor.pack(), 1000.0 * (time.perf_counter() - encode_started))
        )
    except BaseException:
        pool.release(descriptor, owner=SlabOwnership.WORKER_OWNED)
        raise


def ingest_batch_worker_main(task_queue: Any, result_queue: Any, result_pool: TransportSlabPool | None = None, worker_id: int = 0) -> None:
    """Compile immutable canonical mutation intents in parallel ingest processes."""
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, IngestionBatchTask):
            continue
        try:
            tasks = tuple(item.tasks)
            measured_input_bytes = _ingestion_task_input_bytes(item)
            for offset in range(0, len(tasks), _INGEST_RESULT_CHUNK_SIZE):
                chunk = tasks[offset : offset + _INGEST_RESULT_CHUNK_SIZE]
                if not chunk:
                    continue
                compile_started = time.perf_counter()
                rows = tuple(build_commit_plan(prepare_ingestion(task)) for task in chunk)
                compile_ms = 1000.0 * (time.perf_counter() - compile_started)
                result = PreparedCommitBatch(
                    int(chunk[0].sequence),
                    int(chunk[-1].sequence),
                    rows,
                    sum(measured_input_bytes[offset : offset + len(chunk)]),
                    measured_input_bytes[offset : offset + len(chunk)],
                )
                if result_pool is None:
                    descriptor = publish_shared_batch(
                        result,
                        start_sequence=result.start_sequence,
                        end_sequence=result.end_sequence,
                        rows=len(result.rows),
                    )
                    result_queue.put(("ingest_batch_shm", result.start_sequence, result.end_sequence, descriptor, compile_ms))
                else:
                    _publish_compiled_result(
                        result_queue,
                        result_pool,
                        result,
                        producer_id=int(worker_id),
                        start_sequence=result.start_sequence,
                        end_sequence=result.end_sequence,
                        rows=len(result.rows),
                        message=("ingest_batch_slab", result.start_sequence, result.end_sequence, compile_ms),
                    )
        except BaseException as exc:
            result_queue.put(("worker_error", "ingest", int(item.start_sequence), repr(exc)))


def derivation_batch_worker_main(task_queue: Any, result_queue: Any, result_pool: TransportSlabPool | None = None, worker_id: int = 0) -> None:
    while True:
        item = task_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, DerivationBatchTask):
            continue
        try:
            completed = []
            for task in item.tasks:
                try:
                    completed.append(derive_memory(task))
                except BaseException as exc:
                    result_queue.put(("derivation_task_error", int(task.task_id), repr(exc)))
            results = tuple(completed)
            if not results:
                continue
            if result_pool is None:
                descriptor = publish_shared_batch(
                    results,
                    start_sequence=item.start_task_id,
                    end_sequence=item.end_task_id,
                    rows=len(results),
                )
                result_queue.put(("derivation_batch_shm", item.start_task_id, item.end_task_id, descriptor))
            else:
                _publish_compiled_result(
                    result_queue,
                    result_pool,
                    results,
                    producer_id=int(worker_id),
                    start_sequence=item.start_task_id,
                    end_sequence=item.end_task_id,
                    rows=len(results),
                    message=("derivation_batch_slab", item.start_task_id, item.end_task_id),
                )
        except BaseException as exc:
            result_queue.put(("worker_error", "derivation", int(item.start_task_id), repr(exc)))
