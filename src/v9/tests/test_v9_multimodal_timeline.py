from __future__ import annotations

import multiprocessing as mp
import unittest

from v8.model import EventId, ExperienceEvent, encode_experience
from v9.environment_registry import EpisodeId
from v9.modalities.symbols import DeterministicSymbolCodec, ModalityId
from v9.multimodal_events import (
    BoundedMultimodalTimeline,
    InteractionTimelineEvent,
    PassiveSymbolEvent,
    PassiveWorldEvent,
    TimelineIdentity,
    decode_timeline_event,
    encode_timeline_event,
)


def _identity(sequence: int, modality: ModalityId) -> TimelineIdentity:
    return TimelineIdentity(
        EventId.from_producer(3, sequence),
        10,
        3,
        sequence,
        77,
        EpisodeId(99),
        modality,
    )


def _experience(sequence: int) -> ExperienceEvent:
    return ExperienceEvent(
        event_id=EventId.from_producer(3, sequence),
        watermark=10,
        producer_id=3,
        producer_sequence=sequence,
        source_game_hash=77,
        global_step=sequence,
        context_signature=100,
        action_id=2,
        outcome_signature=200,
        family_signature=300,
        carrier_signature=0,
        future_option_delta=0.0,
        changed_cells=1,
        terminal_polarity=0,
        trajectory_signature=400,
    )


class V9MultimodalTimelineTests(unittest.TestCase):
    def test_world_symbol_symbol_action_world_order(self) -> None:
        codec = DeterministicSymbolCodec("timeline")
        timeline = BoundedMultimodalTimeline(max_pending_passive_events=8)
        events = (
            PassiveWorldEvent(_identity(1, ModalityId.WORLD), 11, 111),
            PassiveSymbolEvent(
                _identity(2, ModalityId.SYMBOL),
                codec.vocabulary_id,
                codec.stream_id("s"),
                codec.symbol_id("A"),
                0,
            ),
            PassiveSymbolEvent(
                _identity(3, ModalityId.SYMBOL),
                codec.vocabulary_id,
                codec.stream_id("s"),
                codec.symbol_id("B"),
                1,
            ),
            InteractionTimelineEvent(_identity(4, ModalityId.WORLD), _experience(4)),
            PassiveWorldEvent(_identity(5, ModalityId.WORLD), 11, 112),
        )
        self.assertTrue(all(timeline.append(event) for event in events))
        popped = tuple(timeline.pop_next() for _ in events)
        self.assertEqual(
            tuple(row.identity.producer_sequence for row in popped),
            (1, 2, 3, 4, 5),
        )
        self.assertEqual(timeline.committed_action_count, 1)
        self.assertEqual(timeline.pending_passive_count, 0)

    def test_passive_symbol_has_no_action_id_and_does_not_increment_action_count(self) -> None:
        codec = DeterministicSymbolCodec("passive")
        event = PassiveSymbolEvent(
            _identity(1, ModalityId.SYMBOL),
            codec.vocabulary_id,
            codec.stream_id("s"),
            codec.symbol_id("X"),
            0,
        )
        self.assertFalse(hasattr(event, "action_id"))
        timeline = BoundedMultimodalTimeline()
        timeline.append(event)
        timeline.pop_next()
        self.assertEqual(timeline.committed_action_count, 0)

    def test_symbol_order_survives_multiprocessing_transport(self) -> None:
        codec = DeterministicSymbolCodec("transport")
        original = PassiveSymbolEvent(
            _identity(7, ModalityId.SYMBOL),
            codec.vocabulary_id,
            codec.stream_id("stream"),
            codec.symbol_id("C9"),
            12,
        )
        context = mp.get_context("spawn")
        queue = context.Queue()
        try:
            queue.put(encode_timeline_event(original))
            restored = decode_timeline_event(queue.get(timeout=3.0))
        finally:
            queue.close()
            queue.join_thread()
        self.assertEqual(restored, original)
        self.assertEqual(restored.position, 12)

    def test_bounded_symbol_window_and_pending_overflow_are_explicit(self) -> None:
        codec = DeterministicSymbolCodec("bounded")
        timeline = BoundedMultimodalTimeline(
            max_symbols_per_window=2,
            max_symbol_payload_bytes=3,
            max_pending_passive_events=1,
        )
        admitted = timeline.ingest_symbol_window(
            codec,
            ("AA", "B", "C"),
            stream_name="s",
            environment_instance_id=77,
            episode_id=EpisodeId(99),
            causal_watermark=1,
            producer_id=4,
            first_producer_sequence=1,
        )
        self.assertEqual(len(admitted), 1)
        self.assertEqual(timeline.pending_count, 1)
        self.assertEqual(timeline.pending_passive_count, 1)
        self.assertEqual(timeline.telemetry.symbol_limit_dropped, 1)
        self.assertEqual(timeline.telemetry.pending_overflow_dropped, 1)
        self.assertEqual(timeline.telemetry.symbol_observations_seen, 3)
        self.assertEqual(timeline.telemetry.symbol_observations_admitted, 1)

    def test_payload_bound_is_accounted_without_unbounded_retention(self) -> None:
        codec = DeterministicSymbolCodec("payload")
        timeline = BoundedMultimodalTimeline(
            max_symbols_per_window=4,
            max_symbol_payload_bytes=2,
            max_pending_passive_events=4,
        )
        admitted = timeline.ingest_symbol_window(
            codec,
            ("AAA", "B"),
            stream_name="s",
            environment_instance_id=77,
            episode_id=EpisodeId(99),
            causal_watermark=1,
            producer_id=4,
            first_producer_sequence=1,
        )
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0].position, 1)
        self.assertEqual(timeline.telemetry.payload_limit_dropped, 1)

    def test_legacy_experience_packet_remains_decodable(self) -> None:
        experience = _experience(5)
        restored = decode_timeline_event(encode_experience(experience))
        self.assertIsInstance(restored, InteractionTimelineEvent)
        self.assertEqual(restored.experience, experience)
        self.assertTrue(restored.legacy_episode_unknown)
        self.assertEqual(
            restored.identity.environment_instance_id,
            experience.source_game_hash,
        )
        self.assertEqual(restored.identity.episode_id.value, 0)

    def test_pending_timeline_order_survives_state_restore(self) -> None:
        codec = DeterministicSymbolCodec("restart")
        timeline = BoundedMultimodalTimeline(max_pending_passive_events=4)
        first = PassiveSymbolEvent(
            _identity(1, ModalityId.SYMBOL),
            codec.vocabulary_id,
            codec.stream_id("s"),
            codec.symbol_id("A"),
            0,
        )
        second = PassiveSymbolEvent(
            _identity(2, ModalityId.SYMBOL),
            codec.vocabulary_id,
            codec.stream_id("s"),
            codec.symbol_id("B"),
            1,
        )
        timeline.append(first)
        timeline.append(second)
        restored = BoundedMultimodalTimeline.from_state_dict(timeline.state_dict())
        self.assertEqual(restored.pending_events(), (first, second))
        self.assertEqual(restored.pending_passive_count, 2)
        self.assertEqual(restored.pop_next(), first)
        self.assertEqual(restored.pop_next(), second)


if __name__ == "__main__":
    unittest.main()
