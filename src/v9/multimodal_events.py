from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from struct import Struct
from typing import Iterable

from v8.model import (
    EXPERIENCE_PACKET_SIZE,
    EventId,
    ExperienceEvent,
    decode_experience,
    encode_experience,
)
from v9.environment_registry import EpisodeId
from v9.modalities.symbols import (
    DeterministicSymbolCodec,
    ModalityId,
    SymbolId,
    SymbolStreamId,
    SymbolVocabularyId,
)


class TimelineEventKind(IntEnum):
    INTERACTION = 1
    PASSIVE_WORLD = 2
    PASSIVE_SYMBOL = 3


@dataclass(frozen=True, slots=True)
class TimelineIdentity:
    event_id: EventId
    causal_watermark: int
    producer_id: int
    producer_sequence: int
    environment_instance_id: int
    episode_id: EpisodeId
    modality: ModalityId

    def __post_init__(self) -> None:
        if self.causal_watermark < 0 or self.producer_id < 0 or self.producer_sequence < 0:
            raise ValueError("timeline watermarks and producer fields must be non-negative")
        if self.environment_instance_id < 0 or self.episode_id.value < 0:
            raise ValueError("timeline provenance ids must be non-negative")

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (
            int(self.causal_watermark),
            int(self.producer_id),
            int(self.producer_sequence),
        )


@dataclass(frozen=True, slots=True)
class InteractionTimelineEvent:
    identity: TimelineIdentity
    experience: ExperienceEvent
    legacy_episode_unknown: bool = False

    def __post_init__(self) -> None:
        if self.identity.modality is not ModalityId.WORLD:
            raise ValueError("interaction events must use WORLD modality")
        if self.identity.event_id != self.experience.event_id:
            raise ValueError("timeline and experience EventId must match")
        if self.identity.producer_id != self.experience.producer_id:
            raise ValueError("timeline and experience producer_id must match")
        if self.identity.producer_sequence != self.experience.producer_sequence:
            raise ValueError("timeline and experience producer_sequence must match")


@dataclass(frozen=True, slots=True)
class PassiveWorldEvent:
    identity: TimelineIdentity
    observation_schema_id: int
    observation_signature: int

    def __post_init__(self) -> None:
        if self.identity.modality is not ModalityId.WORLD:
            raise ValueError("passive world events must use WORLD modality")


@dataclass(frozen=True, slots=True)
class PassiveSymbolEvent:
    identity: TimelineIdentity
    vocabulary_id: SymbolVocabularyId
    stream_id: SymbolStreamId
    symbol_id: SymbolId
    position: int

    def __post_init__(self) -> None:
        if self.identity.modality is not ModalityId.SYMBOL:
            raise ValueError("passive symbol events must use SYMBOL modality")
        if self.position < 0:
            raise ValueError("symbol position cannot be negative")


TimelineEvent = InteractionTimelineEvent | PassiveWorldEvent | PassiveSymbolEvent

_HEADER = Struct("<BQQQQQQQB")
_WORLD = Struct("<QQ")
_SYMBOL = Struct("<QQQQ")
_INTERACTION_PREFIX = Struct("<?")


def _encode_header(kind: TimelineEventKind, identity: TimelineIdentity) -> bytes:
    return _HEADER.pack(
        int(kind),
        int(identity.event_id.hi),
        int(identity.event_id.lo),
        int(identity.causal_watermark),
        int(identity.producer_id),
        int(identity.producer_sequence),
        int(identity.environment_instance_id),
        int(identity.episode_id.value),
        int(identity.modality),
    )


def _decode_header(payload: bytes) -> tuple[TimelineEventKind, TimelineIdentity, int]:
    if len(payload) < _HEADER.size:
        raise ValueError("timeline packet shorter than header")
    (
        kind,
        event_hi,
        event_lo,
        watermark,
        producer_id,
        producer_sequence,
        environment_instance_id,
        episode_id,
        modality,
    ) = _HEADER.unpack_from(payload, 0)
    return (
        TimelineEventKind(int(kind)),
        TimelineIdentity(
            EventId(int(event_hi), int(event_lo)),
            int(watermark),
            int(producer_id),
            int(producer_sequence),
            int(environment_instance_id),
            EpisodeId(int(episode_id)),
            ModalityId(int(modality)),
        ),
        _HEADER.size,
    )


def encode_timeline_event(event: TimelineEvent) -> bytes:
    if isinstance(event, PassiveSymbolEvent):
        return _encode_header(TimelineEventKind.PASSIVE_SYMBOL, event.identity) + _SYMBOL.pack(
            int(event.vocabulary_id.value),
            int(event.stream_id.value),
            int(event.symbol_id.value),
            int(event.position),
        )
    if isinstance(event, PassiveWorldEvent):
        return _encode_header(TimelineEventKind.PASSIVE_WORLD, event.identity) + _WORLD.pack(
            int(event.observation_schema_id),
            int(event.observation_signature),
        )
    if isinstance(event, InteractionTimelineEvent):
        return (
            _encode_header(TimelineEventKind.INTERACTION, event.identity)
            + _INTERACTION_PREFIX.pack(bool(event.legacy_episode_unknown))
            + encode_experience(event.experience)
        )
    raise TypeError(f"unsupported timeline event {type(event)!r}")


def decode_timeline_event(payload: bytes) -> TimelineEvent:
    if len(payload) == EXPERIENCE_PACKET_SIZE:
        experience = decode_experience(payload)
        return InteractionTimelineEvent(
            TimelineIdentity(
                experience.event_id,
                int(experience.watermark),
                int(experience.producer_id),
                int(experience.producer_sequence),
                int(experience.source_game_hash),
                EpisodeId(0),
                ModalityId.WORLD,
            ),
            experience,
            legacy_episode_unknown=True,
        )

    kind, identity, offset = _decode_header(payload)
    body = payload[offset:]
    if kind is TimelineEventKind.PASSIVE_SYMBOL:
        if len(body) != _SYMBOL.size:
            raise ValueError("invalid passive-symbol timeline packet size")
        vocabulary, stream, symbol, position = _SYMBOL.unpack(body)
        return PassiveSymbolEvent(
            identity,
            SymbolVocabularyId(int(vocabulary)),
            SymbolStreamId(int(stream)),
            SymbolId(int(symbol)),
            int(position),
        )
    if kind is TimelineEventKind.PASSIVE_WORLD:
        if len(body) != _WORLD.size:
            raise ValueError("invalid passive-world timeline packet size")
        schema, signature = _WORLD.unpack(body)
        return PassiveWorldEvent(identity, int(schema), int(signature))
    if kind is TimelineEventKind.INTERACTION:
        expected = _INTERACTION_PREFIX.size + EXPERIENCE_PACKET_SIZE
        if len(body) != expected:
            raise ValueError("invalid interaction timeline packet size")
        legacy = bool(_INTERACTION_PREFIX.unpack_from(body, 0)[0])
        experience = decode_experience(body[_INTERACTION_PREFIX.size:])
        return InteractionTimelineEvent(identity, experience, legacy_episode_unknown=legacy)
    raise ValueError(f"unsupported timeline event kind {kind}")


@dataclass(slots=True)
class TimelineOverflowTelemetry:
    symbol_observations_seen: int = 0
    symbol_observations_admitted: int = 0
    symbol_limit_dropped: int = 0
    payload_limit_dropped: int = 0
    pending_overflow_dropped: int = 0
    committed_events: int = 0
    committed_actions: int = 0


class BoundedMultimodalTimeline:
    """Bounded passive-event ingress with explicit overflow accounting."""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        max_symbols_per_window: int = 8,
        max_symbol_payload_bytes: int = 4096,
        max_pending_passive_events: int = 64,
    ) -> None:
        if min(max_symbols_per_window, max_symbol_payload_bytes, max_pending_passive_events) <= 0:
            raise ValueError("timeline bounds must be positive")
        self.max_symbols_per_window = int(max_symbols_per_window)
        self.max_symbol_payload_bytes = int(max_symbol_payload_bytes)
        self.max_pending_passive_events = int(max_pending_passive_events)
        self.telemetry = TimelineOverflowTelemetry()
        self._pending: deque[TimelineEvent] = deque()
        self._last_ordering_key: tuple[int, int, int] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def committed_action_count(self) -> int:
        return int(self.telemetry.committed_actions)

    def append(self, event: TimelineEvent) -> bool:
        key = event.identity.ordering_key
        if self._last_ordering_key is not None and key < self._last_ordering_key:
            raise ValueError("timeline events must be appended in causal order")
        if not isinstance(event, InteractionTimelineEvent):
            passive_pending = sum(
                1 for row in self._pending if not isinstance(row, InteractionTimelineEvent)
            )
            if passive_pending >= self.max_pending_passive_events:
                self.telemetry.pending_overflow_dropped += 1
                return False
        self._pending.append(event)
        self._last_ordering_key = key
        return True

    def ingest_symbol_window(
        self,
        codec: DeterministicSymbolCodec,
        tokens: Iterable[str | bytes | int],
        *,
        stream_name: str | int,
        environment_instance_id: int,
        episode_id: EpisodeId,
        causal_watermark: int,
        producer_id: int,
        first_producer_sequence: int,
        start_position: int = 0,
    ) -> tuple[PassiveSymbolEvent, ...]:
        admitted: list[PassiveSymbolEvent] = []
        payload_used = 0
        stream_id = codec.stream_id(stream_name)
        for offset, token in enumerate(tokens):
            self.telemetry.symbol_observations_seen += 1
            if offset >= self.max_symbols_per_window:
                self.telemetry.symbol_limit_dropped += 1
                continue
            if isinstance(token, bytes):
                token_bytes = token
            elif isinstance(token, str):
                token_bytes = token.encode("utf-8")
            else:
                token_bytes = int(token).to_bytes(16, "little", signed=True)
            if payload_used + len(token_bytes) > self.max_symbol_payload_bytes:
                self.telemetry.payload_limit_dropped += 1
                continue
            payload_used += len(token_bytes)
            sequence = int(first_producer_sequence) + offset
            event = PassiveSymbolEvent(
                TimelineIdentity(
                    EventId.from_producer(int(producer_id), sequence),
                    int(causal_watermark),
                    int(producer_id),
                    sequence,
                    int(environment_instance_id),
                    episode_id,
                    ModalityId.SYMBOL,
                ),
                codec.vocabulary_id,
                stream_id,
                codec.symbol_id(token),
                int(start_position) + offset,
            )
            if self.append(event):
                admitted.append(event)
                self.telemetry.symbol_observations_admitted += 1
        return tuple(admitted)

    def pop_next(self) -> TimelineEvent | None:
        if not self._pending:
            return None
        event = self._pending.popleft()
        self.telemetry.committed_events += 1
        if isinstance(event, InteractionTimelineEvent):
            self.telemetry.committed_actions += 1
        return event

    def pending_events(self) -> tuple[TimelineEvent, ...]:
        return tuple(self._pending)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "max_symbols_per_window": self.max_symbols_per_window,
            "max_symbol_payload_bytes": self.max_symbol_payload_bytes,
            "max_pending_passive_events": self.max_pending_passive_events,
            "telemetry": asdict(self.telemetry),
            "last_ordering_key": list(self._last_ordering_key) if self._last_ordering_key else None,
            "pending_packets": [encode_timeline_event(row).hex() for row in self._pending],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "BoundedMultimodalTimeline":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported multimodal timeline state version")
        timeline = cls(
            max_symbols_per_window=int(state["max_symbols_per_window"]),
            max_symbol_payload_bytes=int(state["max_symbol_payload_bytes"]),
            max_pending_passive_events=int(state["max_pending_passive_events"]),
        )
        telemetry = state.get("telemetry", {})
        if isinstance(telemetry, dict):
            timeline.telemetry = TimelineOverflowTelemetry(
                **{field: int(telemetry.get(field, 0)) for field in TimelineOverflowTelemetry.__dataclass_fields__}
            )
        packets = state.get("pending_packets", [])
        if not isinstance(packets, list):
            raise ValueError("pending timeline packets must be a list")
        timeline._pending = deque(decode_timeline_event(bytes.fromhex(str(raw))) for raw in packets)
        last = state.get("last_ordering_key")
        if isinstance(last, list) and len(last) == 3:
            timeline._last_ordering_key = tuple(int(value) for value in last)
        elif timeline._pending:
            timeline._last_ordering_key = timeline._pending[-1].identity.ordering_key
        return timeline
