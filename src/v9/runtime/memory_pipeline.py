from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from v9.cognition.compression import form_families
from v9.cognition.roles import form_roles
from v9.environments.schemas import EnvironmentIdentity
from v9.memory.identity import EpisodeId, EventUid, stable_u64
from v9.memory.m0_episode import M0Episode
from v9.memory.m1_grounded import GroundedRelation, M1GroundedContingency
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.m2_family import M2TransformationFamily
from v9.memory.m3_role import M3FunctionalRole
from v9.memory.m4_concept import M4Concept
from v9.memory.model import ExperienceEvent
from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent, SYMBOL_MODALITY, TimelineIdentity, WORLD_MODALITY
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolOccurrence

from .multiprocess import EncodedTransition, WorkerStop


@dataclass(frozen=True, slots=True)
class IngestionTask:
    sequence: int
    causal_watermark: int
    transition: EncodedTransition


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


def _prepare_symbols(task: IngestionTask, transition: EncodedTransition, identity: EnvironmentIdentity, interaction_grounding: M1GroundedContingency | None) -> tuple[tuple[PreparedSymbolIngestion, ...], dict[str, Any] | None, tuple[SymbolOccurrence, ...]]:
    if not transition.symbols:
        return (), None, ()
    environment = int(identity.instance_id.value)
    episode_id = EpisodeId(int(transition.episode_id))
    codec = DeterministicSymbolCodec(f"{identity.family}-raw-symbols")
    observations = codec.encode_stream(transition.symbols, stream_name=f"{environment}:{episode_id.value}:{transition.producer_sequence}")
    prepared: list[PreparedSymbolIngestion] = []
    occurrences: list[SymbolOccurrence] = []
    for index, row in enumerate(observations):
        event = PassiveSymbolEvent(
            TimelineIdentity(
                EventUid.from_producer(transition.actor_id + 1_000_000, transition.producer_sequence * 10_000 + index),
                int(task.causal_watermark) + index + 1,
                transition.actor_id + 1_000_000,
                transition.producer_sequence * 10_000 + index,
                environment,
                episode_id,
                SYMBOL_MODALITY,
            ),
            row.vocabulary_id,
            row.stream_id,
            row.symbol_id,
            row.position.value,
        )
        occurrences.append(SymbolOccurrence(
            event.identity.event_id, row.symbol_id, row.position.value, row.stream_id, row.vocabulary_id,
            str(codec.codec_id), int(event.identity.causal_watermark), int(transition.global_step), index,
            environment, episode_id, event.identity.event_id,
        ))
        payload_digest = stable_u64(row.vocabulary_id.value, row.stream_id.value, row.symbol_id.value, row.position.value, person=b"v9-symbol-payload")
        m0 = M0Episode.from_event(event, context_signature=0, payload_digest=payload_digest)
        m1g = M1GroundedContingency.build(GroundedRelation.SYMBOL_OCCURRED, (m0,))
        m1n = M1NormalizedRelation.build("SYMBOL_OCCURRED", NormalizedChannel.SYMBOL, (m1g,))
        aligned = None
        if interaction_grounding is not None:
            aligned = M1NormalizedRelation.build("SYMBOL_ALIGNED_WITH_INTERACTION", NormalizedChannel.CROSS_MODAL, (interaction_grounding, m1g))
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
            float(transition.available_actions_after),
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
        m1n = M1NormalizedRelation.build(observable, NormalizedChannel.WORLD, (m1g,))

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
    family = form_families(task.rows)[0]
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
