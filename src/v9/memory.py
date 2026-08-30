from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

from v8.model import EventId, stable_u64
from v9.environment_registry import EpisodeId
from v9.modalities.symbols import ModalityId, SymbolId, SymbolStreamId, SymbolVocabularyId
from v9.multimodal_events import InteractionTimelineEvent, PassiveSymbolEvent, PassiveWorldEvent, TimelineEvent


@dataclass(frozen=True, slots=True)
class PayloadUid:
    value: int


@dataclass(frozen=True, slots=True)
class M0Uid:
    value: int


@dataclass(frozen=True, slots=True)
class M1GUid:
    value: int


@dataclass(frozen=True, slots=True)
class M1NUid:
    value: int


class PayloadAvailabilityState(str, Enum):
    ABSENT = "ABSENT"
    INLINE_IDENTITY = "INLINE_IDENTITY"
    EXTERNAL = "EXTERNAL"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class PayloadRef:
    uid: PayloadUid
    digest: int
    availability: PayloadAvailabilityState


@dataclass(frozen=True, slots=True)
class M0Record:
    uid: M0Uid
    event_id: EventId
    causal_watermark: int
    environment_instance_id: int
    episode_id: EpisodeId
    modality: ModalityId
    context_signature: int
    payload: PayloadRef
    vocabulary_id: SymbolVocabularyId | None = None
    stream_id: SymbolStreamId | None = None
    symbol_id: SymbolId | None = None
    symbol_position: int | None = None
    action_id: int | None = None
    outcome_signature: int | None = None


class GroundedRelationKind(str, Enum):
    SYMBOL_OCCURRED = "SYMBOL_OCCURRED"
    SYMBOL_REPEATED = "SYMBOL_REPEATED"
    SYMBOL_PRECEDES_SYMBOL = "SYMBOL_PRECEDES_SYMBOL"
    SYMBOL_FOLLOWS_SYMBOL = "SYMBOL_FOLLOWS_SYMBOL"
    SYMBOL_PRECEDES_ACTION = "SYMBOL_PRECEDES_ACTION"
    SYMBOL_FOLLOWS_ACTION = "SYMBOL_FOLLOWS_ACTION"
    SYMBOL_PRECEDES_NORMALIZED_CHANGE = "SYMBOL_PRECEDES_NORMALIZED_CHANGE"
    SYMBOL_FOLLOWS_NORMALIZED_CHANGE = "SYMBOL_FOLLOWS_NORMALIZED_CHANGE"
    SYMBOL_PRECEDES_BOUNDARY = "SYMBOL_PRECEDES_BOUNDARY"
    SYMBOL_FOLLOWS_BOUNDARY = "SYMBOL_FOLLOWS_BOUNDARY"


@dataclass(frozen=True, slots=True)
class M1GRecord:
    uid: M1GUid
    kind: GroundedRelationKind
    parent_m0: tuple[M0Uid, ...]
    environment_instance_id: int
    episode_id: EpisodeId

    @property
    def action_authority(self) -> bool:
        return False


class NormalizedChannel(str, Enum):
    WORLD = "WORLD"
    SYMBOL = "SYMBOL"
    CROSS_MODAL = "CROSS_MODAL"


@dataclass(frozen=True, slots=True)
class M1NRecord:
    uid: M1NUid
    primitive: GroundedRelationKind
    channel: NormalizedChannel
    parent_m1g: tuple[M1GUid, ...]

    @property
    def lexical_semantics(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class NormalizedFactBudgets:
    world_facts: int = 8
    symbol_facts: int = 8
    cross_modal_facts: int = 8

    def for_channel(self, channel: NormalizedChannel) -> int:
        if channel is NormalizedChannel.WORLD:
            return int(self.world_facts)
        if channel is NormalizedChannel.SYMBOL:
            return int(self.symbol_facts)
        return int(self.cross_modal_facts)


class MultimodalMemoryStore:
    STATE_VERSION = 1

    def __init__(self, budgets: NormalizedFactBudgets | None = None) -> None:
        self.budgets = budgets or NormalizedFactBudgets()
        self.m0: dict[int, M0Record] = {}
        self.m1g: dict[int, M1GRecord] = {}
        self.m1n: dict[int, M1NRecord] = {}

    @staticmethod
    def _m0_uid(event: TimelineEvent) -> M0Uid:
        identity = event.identity
        return M0Uid(stable_u64(identity.event_id.hi, identity.event_id.lo, identity.environment_instance_id, identity.episode_id.value, int(identity.modality), identity.causal_watermark, person=b"v9-m0-event"))

    @staticmethod
    def _payload_ref(event: TimelineEvent) -> PayloadRef:
        if isinstance(event, PassiveSymbolEvent):
            digest = stable_u64(event.vocabulary_id.value, event.stream_id.value, event.symbol_id.value, event.position, person=b"v9-symbol-payload")
            return PayloadRef(PayloadUid(digest), digest, PayloadAvailabilityState.INLINE_IDENTITY)
        if isinstance(event, PassiveWorldEvent):
            digest = stable_u64(event.observation_schema_id, event.observation_signature, person=b"v9-world-payload")
            return PayloadRef(PayloadUid(digest), digest, PayloadAvailabilityState.ABSENT)
        digest = stable_u64(event.experience.context_signature, event.experience.outcome_signature, person=b"v9-interaction-payload")
        return PayloadRef(PayloadUid(digest), digest, PayloadAvailabilityState.ABSENT)

    def ingest_m0(self, event: TimelineEvent, *, context_signature: int = 0) -> M0Record:
        uid = self._m0_uid(event)
        if uid.value in self.m0:
            return self.m0[uid.value]
        common = dict(uid=uid, event_id=event.identity.event_id, causal_watermark=int(event.identity.causal_watermark), environment_instance_id=int(event.identity.environment_instance_id), episode_id=event.identity.episode_id, modality=event.identity.modality, context_signature=int(context_signature), payload=self._payload_ref(event))
        if isinstance(event, PassiveSymbolEvent):
            record = M0Record(**common, vocabulary_id=event.vocabulary_id, stream_id=event.stream_id, symbol_id=event.symbol_id, symbol_position=int(event.position))
        elif isinstance(event, InteractionTimelineEvent):
            record = M0Record(**common, action_id=int(event.experience.action_id), outcome_signature=int(event.experience.outcome_signature))
        else:
            record = M0Record(**common)
        self.m0[uid.value] = record
        return record

    def ingest_timeline(self, events: Iterable[TimelineEvent], *, context_signature: int = 0) -> tuple[M0Record, ...]:
        return tuple(self.ingest_m0(event, context_signature=context_signature) for event in events)

    @staticmethod
    def _relation_uid(kind: GroundedRelationKind, parents: tuple[M0Uid, ...]) -> M1GUid:
        return M1GUid(stable_u64(kind.value, *(p.value for p in parents), person=b"v9-m1g"))

    def derive_m1g(self, ordered_m0: Iterable[M0Record]) -> tuple[M1GRecord, ...]:
        rows = tuple(ordered_m0)
        output: list[M1GRecord] = []
        previous_symbol: M0Record | None = None
        previous_interaction: M0Record | None = None
        for index, row in enumerate(rows):
            if row.modality is ModalityId.SYMBOL:
                output.append(self._add_m1g(GroundedRelationKind.SYMBOL_OCCURRED, (row,)))
                if previous_symbol is not None:
                    kind = GroundedRelationKind.SYMBOL_REPEATED if previous_symbol.symbol_id == row.symbol_id and previous_symbol.vocabulary_id == row.vocabulary_id else GroundedRelationKind.SYMBOL_PRECEDES_SYMBOL
                    output.append(self._add_m1g(kind, (previous_symbol, row)))
                if previous_interaction is not None:
                    output.append(self._add_m1g(GroundedRelationKind.SYMBOL_FOLLOWS_ACTION, (previous_interaction, row)))
                if index + 1 < len(rows) and rows[index + 1].action_id is not None:
                    output.append(self._add_m1g(GroundedRelationKind.SYMBOL_PRECEDES_ACTION, (row, rows[index + 1])))
                previous_symbol = row
            if row.action_id is not None:
                previous_interaction = row
        return tuple(output)

    def _add_m1g(self, kind: GroundedRelationKind, parents: tuple[M0Record, ...]) -> M1GRecord:
        parent_uids = tuple(p.uid for p in parents)
        uid = self._relation_uid(kind, parent_uids)
        record = M1GRecord(uid, kind, parent_uids, int(parents[0].environment_instance_id), parents[0].episode_id)
        self.m1g.setdefault(uid.value, record)
        return self.m1g[uid.value]

    def normalize_m1g(self, records: Iterable[M1GRecord]) -> tuple[M1NRecord, ...]:
        by_channel: dict[NormalizedChannel, list[M1NRecord]] = {channel: [] for channel in NormalizedChannel}
        seen: set[tuple[str, str]] = set()
        for record in records:
            if record.kind in {GroundedRelationKind.SYMBOL_OCCURRED, GroundedRelationKind.SYMBOL_REPEATED, GroundedRelationKind.SYMBOL_PRECEDES_SYMBOL, GroundedRelationKind.SYMBOL_FOLLOWS_SYMBOL}:
                channel = NormalizedChannel.SYMBOL
            else:
                channel = NormalizedChannel.CROSS_MODAL
            key = (channel.value, record.kind.value)
            if key in seen or len(by_channel[channel]) >= self.budgets.for_channel(channel):
                continue
            seen.add(key)
            uid = M1NUid(stable_u64(channel.value, record.kind.value, person=b"v9-m1n"))
            fact = M1NRecord(uid, record.kind, channel, (record.uid,))
            self.m1n.setdefault(uid.value, fact)
            by_channel[channel].append(self.m1n[uid.value])
        return tuple(fact for channel in NormalizedChannel for fact in by_channel[channel])

    def m0_provenance(self, uid: M0Uid) -> tuple[int, EpisodeId]:
        row = self.m0[int(uid.value)]
        return int(row.environment_instance_id), row.episode_id

    def trace_m1n_to_m0(self, uid: M1NUid) -> tuple[M0Record, ...]:
        fact = self.m1n[int(uid.value)]
        parents: list[M0Record] = []
        for m1g_uid in fact.parent_m1g:
            relation = self.m1g[int(m1g_uid.value)]
            parents.extend(self.m0[int(m0_uid.value)] for m0_uid in relation.parent_m0)
        return tuple(parents)

    def state_dict(self) -> dict[str, object]:
        def m0_row(row: M0Record) -> dict[str, object]:
            return {"uid": row.uid.value, "event_hi": row.event_id.hi, "event_lo": row.event_id.lo, "causal_watermark": row.causal_watermark, "environment_instance_id": row.environment_instance_id, "episode_id": row.episode_id.value, "modality": int(row.modality), "context_signature": row.context_signature, "payload_uid": row.payload.uid.value, "payload_digest": row.payload.digest, "payload_availability": row.payload.availability.value, "vocabulary_id": None if row.vocabulary_id is None else row.vocabulary_id.value, "stream_id": None if row.stream_id is None else row.stream_id.value, "symbol_id": None if row.symbol_id is None else row.symbol_id.value, "symbol_position": row.symbol_position, "action_id": row.action_id, "outcome_signature": row.outcome_signature}
        return {"version": self.STATE_VERSION, "budgets": asdict(self.budgets), "m0": [m0_row(self.m0[key]) for key in sorted(self.m0)], "m1g": [{"uid": row.uid.value, "kind": row.kind.value, "parents": [p.value for p in row.parent_m0], "environment_instance_id": row.environment_instance_id, "episode_id": row.episode_id.value} for key in sorted(self.m1g) for row in (self.m1g[key],)], "m1n": [{"uid": row.uid.value, "primitive": row.primitive.value, "channel": row.channel.value, "parents": [p.value for p in row.parent_m1g]} for key in sorted(self.m1n) for row in (self.m1n[key],)]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "MultimodalMemoryStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported multimodal memory state version")
        budget_state = state.get("budgets", {})
        if not isinstance(budget_state, dict):
            raise ValueError("invalid normalized fact budgets")
        store = cls(NormalizedFactBudgets(**{k: int(v) for k, v in budget_state.items()}))
        for raw in state.get("m0", []):
            if not isinstance(raw, dict):
                continue
            row = M0Record(M0Uid(int(raw["uid"])), EventId(int(raw["event_hi"]), int(raw["event_lo"])), int(raw["causal_watermark"]), int(raw["environment_instance_id"]), EpisodeId(int(raw["episode_id"])), ModalityId(int(raw["modality"])), int(raw["context_signature"]), PayloadRef(PayloadUid(int(raw["payload_uid"])), int(raw["payload_digest"]), PayloadAvailabilityState(str(raw["payload_availability"]))), None if raw.get("vocabulary_id") is None else SymbolVocabularyId(int(raw["vocabulary_id"])), None if raw.get("stream_id") is None else SymbolStreamId(int(raw["stream_id"])), None if raw.get("symbol_id") is None else SymbolId(int(raw["symbol_id"])), None if raw.get("symbol_position") is None else int(raw["symbol_position"]), None if raw.get("action_id") is None else int(raw["action_id"]), None if raw.get("outcome_signature") is None else int(raw["outcome_signature"]))
            store.m0[row.uid.value] = row
        for raw in state.get("m1g", []):
            if not isinstance(raw, dict):
                continue
            row = M1GRecord(M1GUid(int(raw["uid"])), GroundedRelationKind(str(raw["kind"])), tuple(M0Uid(int(x)) for x in raw.get("parents", [])), int(raw["environment_instance_id"]), EpisodeId(int(raw["episode_id"])))
            store.m1g[row.uid.value] = row
        for raw in state.get("m1n", []):
            if not isinstance(raw, dict):
                continue
            row = M1NRecord(M1NUid(int(raw["uid"])), GroundedRelationKind(str(raw["primitive"])), NormalizedChannel(str(raw["channel"])), tuple(M1GUid(int(x)) for x in raw.get("parents", [])))
            store.m1n[row.uid.value] = row
        return store
