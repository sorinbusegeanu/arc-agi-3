from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import blake2b
from struct import Struct
from typing import Iterable

_MASK64 = (1 << 64) - 1
_SCHEMA = b"arc-agi3-v8-memory-v2"


class MemoryLevel(IntEnum):
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M7 = 7


class MemoryType(IntEnum):
    EPISODE = 1
    CONTINGENCY = 100
    FAMILY = 200
    CARRIER = 250
    CONTEXTUAL_ROLE = 275
    ROLE = 300
    CONCEPT = 400
    TRANSFER_EVIDENCE = 450
    CONSEQUENCE = 500
    OUTCOME = 600
    STRATEGY = 700
    PREFERENCE_EVIDENCE = 750


class RelationType(IntEnum):
    PROVENANCE = 1
    EXPLAINS = 2
    LEADS_TO = 3
    CONTEXT_REFINES = 4
    SIMILAR_TO = 5
    TRANSFER_CORRESPONDENCE = 6
    SUPERSEDES = 7
    PREFERENCE = 8


class CognitiveState(IntEnum):
    CANDIDATE = 0
    PROBATION = 1
    ACTIVE = 2
    VALIDATED = 3
    QUARANTINED = 4
    RETIRE_PENDING = 5
    RETIRED = 6
    REACTIVATED = 7


class ValidationState(IntEnum):
    UNTESTED = 0
    STRUCTURAL = 1
    TESTED = 2
    VALIDATED = 3
    FAILED = 4


def u64(value: int) -> int:
    return int(value) & _MASK64


def signed_u64(value: int) -> int:
    value = int(value) & _MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def stable_u64(*parts: object, person: bytes = b"v8-stable") -> int:
    digest = blake2b(digest_size=8, person=person[:16])
    for part in parts:
        if isinstance(part, bytes):
            raw = part
        elif isinstance(part, str):
            raw = part.encode("utf-8")
        else:
            raw = int(part).to_bytes(16, "little", signed=True)
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True, slots=True, order=True)
class MemoryUid:
    hi: int
    lo: int

    def __post_init__(self) -> None:
        if not (0 <= int(self.hi) <= _MASK64 and 0 <= int(self.lo) <= _MASK64):
            raise ValueError("MemoryUid components must be uint64")

    @classmethod
    def zero(cls) -> "MemoryUid":
        return cls(0, 0)

    @classmethod
    def from_key(
        cls,
        level: MemoryLevel | int,
        memory_type: MemoryType | int,
        key_parts: Iterable[int],
    ) -> "MemoryUid":
        digest = blake2b(digest_size=16, person=b"arc-v8-memory")
        digest.update(_SCHEMA)
        digest.update(int(level).to_bytes(1, "little", signed=False))
        digest.update(int(memory_type).to_bytes(2, "little", signed=False))
        parts = tuple(int(value) for value in key_parts)
        digest.update(len(parts).to_bytes(1, "little"))
        for value in parts:
            digest.update(value.to_bytes(16, "little", signed=True))
        raw = digest.digest()
        return cls(
            int.from_bytes(raw[:8], "little"),
            int.from_bytes(raw[8:], "little"),
        )

    @property
    def is_zero(self) -> bool:
        return int(self.hi) == 0 and int(self.lo) == 0

    def shard(self, shard_count: int) -> int:
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        return int((self.hi ^ self.lo) % int(shard_count))

    def hex(self) -> str:
        return f"{self.hi:016x}{self.lo:016x}"


@dataclass(frozen=True, slots=True, order=True)
class EventId:
    hi: int
    lo: int

    @classmethod
    def from_producer(cls, producer_id: int, sequence: int) -> "EventId":
        digest = blake2b(digest_size=16, person=b"arc-v8-event")
        digest.update(int(producer_id).to_bytes(8, "little", signed=False))
        digest.update(int(sequence).to_bytes(8, "little", signed=False))
        raw = digest.digest()
        return cls(int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:], "little"))


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    event_id: EventId
    watermark: int
    producer_id: int
    producer_sequence: int
    source_game_hash: int
    global_step: int
    context_signature: int
    action_id: int
    outcome_signature: int
    family_signature: int
    carrier_signature: int
    future_option_delta: float
    changed_cells: int
    terminal_polarity: int
    trajectory_signature: int
    next_context_signature: int = 0

    def __post_init__(self) -> None:
        if self.watermark < 0 or self.global_step < 0 or self.producer_sequence < 0:
            raise ValueError("watermarks and sequence values must be non-negative")
        if self.terminal_polarity not in {-1, 0, 1}:
            raise ValueError("terminal_polarity must be -1, 0 or 1")


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    experience: ExperienceEvent
    parent_uid: MemoryUid = MemoryUid(0, 0)
    current_level: int = -1


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    uid: MemoryUid
    fingerprint: int
    event_id: EventId
    watermark: int
    level: MemoryLevel
    memory_type: MemoryType
    key_parts: tuple[int, ...]
    support_delta: int = 1
    significance_sum: float = 0.0
    prediction_error_sum: float = 0.0
    learning_value_sum: float = 0.0
    transfer_prior_sum: float = 0.0
    explanatory_sum: float = 0.0
    future_option_sum: float = 0.0
    score_weight: float = 1.0
    parent_uid: MemoryUid = MemoryUid(0, 0)
    relation_type: RelationType = RelationType.PROVENANCE
    source_game_hash: int = 0
    cognitive_state: int = -1
    validation_state: int = -1

    def __post_init__(self) -> None:
        if not 0 < len(self.key_parts) <= 4:
            raise ValueError("v8 hot-path canonical keys support 1..4 parts")
        if self.support_delta < 0:
            raise ValueError("support_delta cannot be negative")
        if self.score_weight < 0:
            raise ValueError("score_weight cannot be negative")
        if self.cognitive_state < -1 or self.cognitive_state > 127:
            raise ValueError("invalid cognitive_state")
        if self.validation_state < -1 or self.validation_state > 127:
            raise ValueError("invalid validation_state")


_EXPERIENCE = Struct("<QQQIQQQiQQQdIbQ")
_PIPE_SUFFIX = Struct("<QQb")
_PROPOSAL = Struct("<QQQQQQBHBQQQQqdddddddQQHQbb")

EXPERIENCE_PACKET_SIZE = _EXPERIENCE.size + 16
PIPELINE_PACKET_SIZE = EXPERIENCE_PACKET_SIZE + _PIPE_SUFFIX.size
PROPOSAL_PACKET_SIZE = _PROPOSAL.size


def encode_experience(event: ExperienceEvent) -> bytes:
    return _EXPERIENCE.pack(
        u64(event.event_id.hi),
        u64(event.event_id.lo),
        u64(event.watermark),
        int(event.producer_id) & 0xFFFFFFFF,
        u64(event.producer_sequence),
        u64(event.source_game_hash),
        u64(event.global_step),
        int(event.action_id),
        u64(event.context_signature),
        u64(event.outcome_signature),
        u64(event.family_signature),
        float(event.future_option_delta),
        int(event.changed_cells) & 0xFFFFFFFF,
        int(event.terminal_polarity),
        u64(event.trajectory_signature),
    ) + u64(event.carrier_signature).to_bytes(8, "little") + u64(event.next_context_signature).to_bytes(8, "little")


def decode_experience(payload: bytes) -> ExperienceEvent:
    if len(payload) != EXPERIENCE_PACKET_SIZE:
        raise ValueError(f"invalid experience packet size {len(payload)}")
    base = payload[:-16]
    carrier = int.from_bytes(payload[-16:-8], "little")
    next_context = int.from_bytes(payload[-8:], "little")
    (
        event_hi,
        event_lo,
        watermark,
        producer_id,
        producer_sequence,
        source_game_hash,
        global_step,
        action_id,
        context_signature,
        outcome_signature,
        family_signature,
        future_option_delta,
        changed_cells,
        terminal_polarity,
        trajectory_signature,
    ) = _EXPERIENCE.unpack(base)
    return ExperienceEvent(
        EventId(event_hi, event_lo),
        int(watermark),
        int(producer_id),
        int(producer_sequence),
        int(source_game_hash),
        int(global_step),
        int(context_signature),
        int(action_id),
        int(outcome_signature),
        int(family_signature),
        int(carrier),
        float(future_option_delta),
        int(changed_cells),
        int(terminal_polarity),
        int(trajectory_signature),
        int(next_context),
    )


def encode_pipeline(event: PipelineEvent) -> bytes:
    return encode_experience(event.experience) + _PIPE_SUFFIX.pack(
        u64(event.parent_uid.hi),
        u64(event.parent_uid.lo),
        int(event.current_level),
    )


def decode_pipeline(payload: bytes) -> PipelineEvent:
    if len(payload) != PIPELINE_PACKET_SIZE:
        raise ValueError(f"invalid pipeline packet size {len(payload)}")
    experience = decode_experience(payload[:EXPERIENCE_PACKET_SIZE])
    parent_hi, parent_lo, current_level = _PIPE_SUFFIX.unpack(payload[EXPERIENCE_PACKET_SIZE:])
    return PipelineEvent(experience, MemoryUid(parent_hi, parent_lo), int(current_level))


def proposal_fingerprint(level: int, memory_type: int, key_parts: Iterable[int]) -> int:
    parts = tuple(int(value) for value in key_parts)
    return stable_u64(int(level), int(memory_type), *parts, person=b"v8-key-fp")


def encode_proposal(proposal: MemoryProposal) -> bytes:
    parts = tuple(u64(value) for value in proposal.key_parts)
    padded = parts + (0,) * (4 - len(parts))
    return _PROPOSAL.pack(
        u64(proposal.uid.hi),
        u64(proposal.uid.lo),
        u64(proposal.fingerprint),
        u64(proposal.event_id.hi),
        u64(proposal.event_id.lo),
        u64(proposal.watermark),
        int(proposal.level),
        int(proposal.memory_type),
        len(parts),
        *padded,
        int(proposal.support_delta),
        float(proposal.significance_sum),
        float(proposal.prediction_error_sum),
        float(proposal.learning_value_sum),
        float(proposal.transfer_prior_sum),
        float(proposal.explanatory_sum),
        float(proposal.future_option_sum),
        float(proposal.score_weight),
        u64(proposal.parent_uid.hi),
        u64(proposal.parent_uid.lo),
        int(proposal.relation_type),
        u64(proposal.source_game_hash),
        int(proposal.cognitive_state),
        int(proposal.validation_state),
    )


def decode_proposal(payload: bytes) -> MemoryProposal:
    if len(payload) != PROPOSAL_PACKET_SIZE:
        raise ValueError(f"invalid proposal packet size {len(payload)}")
    values = _PROPOSAL.unpack(payload)
    (
        uid_hi,
        uid_lo,
        fingerprint,
        event_hi,
        event_lo,
        watermark,
        level,
        memory_type,
        key_count,
        k0,
        k1,
        k2,
        k3,
        support_delta,
        significance,
        prediction_error,
        learning_value,
        transfer_prior,
        explanatory,
        future_option,
        score_weight,
        parent_hi,
        parent_lo,
        relation_type,
        source_game_hash,
        cognitive_state,
        validation_state,
    ) = values
    keys = (k0, k1, k2, k3)[: int(key_count)]
    return MemoryProposal(
        uid=MemoryUid(uid_hi, uid_lo),
        fingerprint=int(fingerprint),
        event_id=EventId(event_hi, event_lo),
        watermark=int(watermark),
        level=MemoryLevel(level),
        memory_type=MemoryType(memory_type),
        key_parts=tuple(int(value) for value in keys),
        support_delta=int(support_delta),
        significance_sum=float(significance),
        prediction_error_sum=float(prediction_error),
        learning_value_sum=float(learning_value),
        transfer_prior_sum=float(transfer_prior),
        explanatory_sum=float(explanatory),
        future_option_sum=float(future_option),
        score_weight=float(score_weight),
        parent_uid=MemoryUid(parent_hi, parent_lo),
        relation_type=RelationType(relation_type),
        source_game_hash=int(source_game_hash),
        cognitive_state=int(cognitive_state),
        validation_state=int(validation_state),
    )
