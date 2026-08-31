from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from struct import Struct

from v8.environment_registry import EpisodeId
from v8.model import EXPERIENCE_PACKET_SIZE, EventId, ExperienceEvent, decode_experience, encode_experience
from v8.modalities.symbols import ModalityId, SymbolId, SymbolPosition, SymbolStreamId, SymbolVocabularyId


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

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (int(self.causal_watermark), int(self.producer_id), int(self.producer_sequence))


@dataclass(frozen=True, slots=True)
class InteractionTimelineEvent:
    identity: TimelineIdentity
    experience: ExperienceEvent
    legacy_episode_unknown: bool = False


@dataclass(frozen=True, slots=True)
class PassiveWorldEvent:
    identity: TimelineIdentity
    observation_schema_id: int
    observation_signature: int


@dataclass(frozen=True, slots=True)
class PassiveSymbolEvent:
    identity: TimelineIdentity
    vocabulary_id: SymbolVocabularyId
    stream_id: SymbolStreamId
    symbol_id: SymbolId
    position: SymbolPosition

    def __post_init__(self) -> None:
        if self.identity.modality is not ModalityId.SYMBOL:
            raise ValueError("passive symbol event must use SYMBOL modality")
        if self.position.value < 0:
            raise ValueError("symbol position must be non-negative")


TimelineEvent = InteractionTimelineEvent | PassiveWorldEvent | PassiveSymbolEvent
_HEADER = Struct("<BQQQQQQQB")
_SYMBOL = Struct("<QQQQ")
_WORLD = Struct("<QQ")
_INTERACTION_PREFIX = Struct("<?")


def encode_timeline_event(event: TimelineEvent) -> bytes:
    if isinstance(event, PassiveSymbolEvent):
        kind = TimelineEventKind.PASSIVE_SYMBOL
    elif isinstance(event, PassiveWorldEvent):
        kind = TimelineEventKind.PASSIVE_WORLD
    elif isinstance(event, InteractionTimelineEvent):
        kind = TimelineEventKind.INTERACTION
    else:
        raise TypeError(type(event))
    i = event.identity
    head = _HEADER.pack(
        int(kind), int(i.event_id.hi), int(i.event_id.lo), int(i.causal_watermark),
        int(i.producer_id), int(i.producer_sequence), int(i.environment_instance_id),
        int(i.episode_id.value), int(i.modality),
    )
    if isinstance(event, PassiveSymbolEvent):
        return head + _SYMBOL.pack(event.vocabulary_id.value, event.stream_id.value, event.symbol_id.value, event.position.value)
    if isinstance(event, PassiveWorldEvent):
        return head + _WORLD.pack(int(event.observation_schema_id), int(event.observation_signature))
    return head + _INTERACTION_PREFIX.pack(bool(event.legacy_episode_unknown)) + encode_experience(event.experience)


def decode_timeline_event(payload: bytes) -> TimelineEvent:
    if len(payload) == EXPERIENCE_PACKET_SIZE:
        e = decode_experience(payload)
        return InteractionTimelineEvent(
            TimelineIdentity(e.event_id, e.watermark, e.producer_id, e.producer_sequence, e.source_game_hash, EpisodeId(0), ModalityId.WORLD),
            e,
            True,
        )
    if len(payload) < _HEADER.size:
        raise ValueError("timeline packet shorter than header")
    raw = _HEADER.unpack_from(payload, 0)
    kind = TimelineEventKind(int(raw[0]))
    identity = TimelineIdentity(
        EventId(int(raw[1]), int(raw[2])), int(raw[3]), int(raw[4]), int(raw[5]),
        int(raw[6]), EpisodeId(int(raw[7])), ModalityId(int(raw[8])),
    )
    body = payload[_HEADER.size:]
    if kind is TimelineEventKind.PASSIVE_SYMBOL:
        if len(body) != _SYMBOL.size:
            raise ValueError("invalid passive symbol packet")
        vocabulary, stream, symbol, position = _SYMBOL.unpack(body)
        return PassiveSymbolEvent(identity, SymbolVocabularyId(vocabulary), SymbolStreamId(stream), SymbolId(symbol), SymbolPosition(position))
    if kind is TimelineEventKind.PASSIVE_WORLD:
        if len(body) != _WORLD.size:
            raise ValueError("invalid passive world packet")
        schema, signature = _WORLD.unpack(body)
        return PassiveWorldEvent(identity, int(schema), int(signature))
    expected = _INTERACTION_PREFIX.size + EXPERIENCE_PACKET_SIZE
    if len(body) != expected:
        raise ValueError("invalid interaction timeline packet")
    legacy = bool(_INTERACTION_PREFIX.unpack_from(body, 0)[0])
    return InteractionTimelineEvent(identity, decode_experience(body[_INTERACTION_PREFIX.size:]), legacy)


@dataclass(slots=True)
class TimelineTelemetry:
    admitted: int = 0
    committed: int = 0
    committed_actions: int = 0
    symbol_limit_dropped: int = 0
    payload_limit_dropped: int = 0
    pending_limit_dropped: int = 0


class BoundedMultimodalTimeline:
    STATE_VERSION = 1

    def __init__(self, *, max_symbols_per_window: int = 64, max_symbol_payload_bytes: int = 4096, max_pending_passive_events: int = 4096) -> None:
        if min(max_symbols_per_window, max_symbol_payload_bytes, max_pending_passive_events) <= 0:
            raise ValueError("timeline limits must be positive")
        self.max_symbols_per_window = int(max_symbols_per_window)
        self.max_symbol_payload_bytes = int(max_symbol_payload_bytes)
        self.max_pending_passive_events = int(max_pending_passive_events)
        self.telemetry = TimelineTelemetry()
        self._pending: deque[TimelineEvent] = deque()
        self._passive_count = 0
        self._last_key: tuple[int, int, int] | None = None
        self._last_sequence: dict[int, int] = {}

    @property
    def pending_passive_count(self) -> int:
        return self._passive_count

    def next_producer_sequence(self, producer_id: int) -> int:
        return int(self._last_sequence.get(int(producer_id), 0)) + 1

    def append(self, event: TimelineEvent) -> bool:
        key = event.identity.ordering_key
        producer = int(event.identity.producer_id)
        sequence = int(event.identity.producer_sequence)
        if sequence <= int(self._last_sequence.get(producer, 0)):
            raise ValueError("producer sequence must increase strictly")
        passive = not isinstance(event, InteractionTimelineEvent)
        if passive and self._passive_count >= self.max_pending_passive_events:
            self.telemetry.pending_limit_dropped += 1
            return False
        self._pending.append(event)
        self._last_sequence[producer] = sequence
        self._last_key = key
        self._passive_count += int(passive)
        self.telemetry.admitted += 1
        return True

    def pop_next(self) -> TimelineEvent | None:
        if not self._pending:
            return None
        event = self._pending.popleft()
        passive = not isinstance(event, InteractionTimelineEvent)
        self._passive_count -= int(passive)
        self.telemetry.committed += 1
        self.telemetry.committed_actions += int(not passive)
        return event

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "max_symbols_per_window": self.max_symbols_per_window,
            "max_symbol_payload_bytes": self.max_symbol_payload_bytes,
            "max_pending_passive_events": self.max_pending_passive_events,
            "telemetry": asdict(self.telemetry),
            "last_sequence": {str(k): v for k, v in sorted(self._last_sequence.items())},
            "pending": [encode_timeline_event(row).hex() for row in self._pending],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "BoundedMultimodalTimeline":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported timeline state")
        obj = cls(
            max_symbols_per_window=int(state["max_symbols_per_window"]),
            max_symbol_payload_bytes=int(state["max_symbol_payload_bytes"]),
            max_pending_passive_events=int(state["max_pending_passive_events"]),
        )
        raw_telemetry = state.get("telemetry", {})
        if isinstance(raw_telemetry, dict):
            obj.telemetry = TimelineTelemetry(**{name: int(raw_telemetry.get(name, 0)) for name in TimelineTelemetry.__dataclass_fields__})
        obj._pending = deque(decode_timeline_event(bytes.fromhex(str(raw))) for raw in state.get("pending", []))
        obj._passive_count = sum(not isinstance(row, InteractionTimelineEvent) for row in obj._pending)
        last_sequence = state.get("last_sequence", {})
        if isinstance(last_sequence, dict):
            obj._last_sequence = {int(k): int(v) for k, v in last_sequence.items()}
        if obj._pending:
            obj._last_key = obj._pending[-1].identity.ordering_key
        if obj._passive_count > obj.max_pending_passive_events:
            raise ValueError("restored passive queue exceeds configured bound")
        return obj
