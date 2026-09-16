from __future__ import annotations

from collections.abc import Mapping
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
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolOccurrence, SymbolTemporalPhase

from .multiprocess import EncodedTransition, WorkerStop


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
