from __future__ import annotations

import pytest

from v9.memory.identity import EpisodeId, EventUid
from v9.memory.model import ExperienceEvent
from v9.modalities import (
    InteractionEvent, PassiveSymbolEvent, SYMBOL_MODALITY, TimelineIdentity,
    WORLD_MODALITY,
)
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.runtime.rings import MultimodalTimeline


def _experience(sequence: int) -> ExperienceEvent:
    return ExperienceEvent(EventUid.from_producer(1, sequence), sequence, 1, sequence, 7, sequence, 2, 3, 4)


def test_passive_symbol_observations_never_count_as_actions() -> None:
    codec = DeterministicSymbolCodec("opaque")
    observation = codec.encode_stream(("x",), stream_name="s")[0]
    identity = TimelineIdentity(EventUid.from_producer(2, 1), 1, 2, 1, 7, EpisodeId(1), SYMBOL_MODALITY)
    symbol = PassiveSymbolEvent(identity, observation.vocabulary_id, observation.stream_id, observation.symbol_id, observation.position.value)
    timeline = MultimodalTimeline(capacity=4, symbol_budget=2, symbol_payload_bytes=8)
    assert timeline.append(symbol)
    assert timeline.pop_next() == symbol
    assert timeline.actions_committed == 0
    interaction = InteractionEvent(TimelineIdentity(EventUid.from_producer(1, 2), 2, 1, 2, 7, EpisodeId(1), WORLD_MODALITY), _experience(2))
    timeline.append(interaction)
    timeline.pop_next()
    assert timeline.actions_committed == 1


def test_symbol_and_payload_budgets_drop_with_telemetry() -> None:
    codec = DeterministicSymbolCodec("opaque")
    encoded = codec.encode_stream(("a", "bb", "c"), stream_name="s")
    events = tuple(PassiveSymbolEvent(TimelineIdentity(EventUid.from_producer(2, i + 1), i + 1, 2, i + 1, 7, EpisodeId(1), SYMBOL_MODALITY), row.vocabulary_id, row.stream_id, row.symbol_id, row.position.value) for i, row in enumerate(encoded))
    timeline = MultimodalTimeline(capacity=8, symbol_budget=2, symbol_payload_bytes=2)
    admitted = timeline.append_symbols(events, (1, 2, 1))
    assert len(admitted) == 2
    assert timeline.events_dropped == 1


def test_timeline_rejects_causal_reordering() -> None:
    timeline = MultimodalTimeline(capacity=2, symbol_budget=1, symbol_payload_bytes=1)
    first = InteractionEvent(TimelineIdentity(EventUid.from_producer(1, 2), 2, 1, 2, 7, EpisodeId(1), WORLD_MODALITY), _experience(2))
    earlier = InteractionEvent(TimelineIdentity(EventUid.from_producer(1, 1), 1, 1, 1, 7, EpisodeId(1), WORLD_MODALITY), _experience(1))
    timeline.append(first)
    with pytest.raises(ValueError):
        timeline.append(earlier)


def test_symbol_identity_is_deterministic_and_semantic_free() -> None:
    left = DeterministicSymbolCodec("opaque")
    right = DeterministicSymbolCodec("opaque")
    assert left.symbol_id("door") == right.symbol_id("door")
    symbol = left.encode_stream(("door",), stream_name="instruction")[0]
    assert not hasattr(symbol, "embedding")
    assert not hasattr(symbol, "meaning")

