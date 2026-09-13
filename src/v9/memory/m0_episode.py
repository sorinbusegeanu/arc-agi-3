from __future__ import annotations

from dataclasses import dataclass

from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent, PassiveWorldEvent, TimelineEvent

from .identity import MemoryUid
from .model import MemoryLevel, MemoryType
from .provenance import ProvenanceRoot


@dataclass(frozen=True, slots=True)
class M0Episode:
    uid: MemoryUid
    provenance: ProvenanceRoot
    modality_id: int
    context_signature: int
    payload_digest: int
    action_id: int | None = None
    outcome_signature: int | None = None
    next_context_signature: int | None = None
    symbol_identity: tuple[int, int, int, int] | None = None
    primary_valence: int = 0
    future_option_delta: float = 0.0
    realized_cost: int = 0

    @classmethod
    def from_event(cls, event: TimelineEvent, *, context_signature: int, payload_digest: int) -> "M0Episode":
        identity = event.identity
        # M0 identity is the EventUid alone. Environment, episode and modality
        # are grounded provenance and must not silently split the identity.
        uid = MemoryUid.from_key(MemoryLevel.M0, MemoryType.EPISODE, (identity.event_id.hi, identity.event_id.lo))
        action: int | None = None
        outcome: int | None = None
        next_context: int | None = None
        symbol: tuple[int, int, int, int] | None = None
        if isinstance(event, InteractionEvent):
            action = event.experience.action_id
            outcome = event.experience.outcome_signature
            next_context = event.experience.next_context_signature
        elif isinstance(event, PassiveSymbolEvent):
            symbol = (
                int(event.vocabulary_id.value), int(event.stream_id.value),
                int(event.symbol_id.value), int(event.position),
            )
        elif not isinstance(event, PassiveWorldEvent):
            raise TypeError(f"unsupported event {type(event)!r}")
        return cls(
            uid,
            ProvenanceRoot(identity.event_id, identity.environment_instance_id, identity.episode_id, identity.causal_watermark),
            identity.modality_id.value,
            int(context_signature), int(payload_digest), action, outcome, next_context, symbol,
            event.experience.primary_valence if isinstance(event, InteractionEvent) else 0,
            event.experience.future_option_delta if isinstance(event, InteractionEvent) else 0.0,
            event.experience.changed_cells if isinstance(event, InteractionEvent) else 0,
        )
