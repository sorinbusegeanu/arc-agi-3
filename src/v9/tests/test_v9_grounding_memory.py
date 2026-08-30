from __future__ import annotations

import unittest

from v8.environments.synthetic_symbolic import SyntheticSymbolicConfig, SyntheticSymbolicEnvironment
from v8.model import EventId, ExperienceEvent
from v9.environment_registry import EpisodeId
from v9.memory import MultimodalMemoryStore, NormalizedChannel, PayloadAvailabilityState
from v9.modalities.symbols import DeterministicSymbolCodec, ModalityId
from v9.multimodal_events import InteractionTimelineEvent, PassiveSymbolEvent, TimelineIdentity


def _symbol(codec: DeterministicSymbolCodec, sequence: int, token: str, position: int, *, environment: int = 10):
    return PassiveSymbolEvent(TimelineIdentity(EventId.from_producer(1, sequence), 4, 1, sequence, environment, EpisodeId(2), ModalityId.SYMBOL), codec.vocabulary_id, codec.stream_id("s"), codec.symbol_id(token), position)


def _interaction(sequence: int, *, environment: int = 10):
    experience = ExperienceEvent(event_id=EventId.from_producer(1, sequence), watermark=4, producer_id=1, producer_sequence=sequence, source_game_hash=environment, global_step=sequence, context_signature=100, action_id=0, outcome_signature=200, family_signature=300, carrier_signature=0, future_option_delta=0.0, changed_cells=1, terminal_polarity=0, trajectory_signature=400)
    return InteractionTimelineEvent(TimelineIdentity(experience.event_id, 4, 1, sequence, environment, EpisodeId(2), ModalityId.WORLD), experience)


class SyntheticGroundingEnvironmentTests(unittest.TestCase):
    def test_fixed_seed_timeline_is_reproducible(self) -> None:
        config = SyntheticSymbolicConfig(seed=7, symbol_condition="shuffled")
        first = SyntheticSymbolicEnvironment(config)
        second = SyntheticSymbolicEnvironment(config)
        actions = (0, 0, 1, 0, 0)
        self.assertEqual(first.timeline_signature(actions), second.timeline_signature(actions))

    def test_control_manipulations_are_available(self) -> None:
        conditions = {"aligned", "permuted", "different", "shuffled", "none", "symbol_only"}
        rows = {condition: SyntheticSymbolicEnvironment(SyntheticSymbolicConfig(symbol_condition=condition)) for condition in conditions}
        self.assertEqual(rows["none"].passive_symbol_tokens(), ())
        self.assertEqual(rows["symbol_only"].observe(), 0)
        self.assertNotEqual(rows["aligned"].passive_symbol_tokens(), rows["different"].passive_symbol_tokens())

    def test_environment_uses_declared_identity_and_schemas(self) -> None:
        env = SyntheticSymbolicEnvironment()
        self.assertEqual(env.identity.family, "synthetic")
        self.assertGreater(env.observation_schema.schema_id, 0)
        self.assertGreater(env.action_schema.schema_id, 0)
        self.assertEqual(env.available_actions(), (0, 1))


class MultimodalMemoryTests(unittest.TestCase):
    def test_same_token_different_vocab_keeps_distinct_grounded_provenance(self) -> None:
        a = DeterministicSymbolCodec("a")
        b = DeterministicSymbolCodec("b")
        store = MultimodalMemoryStore()
        left = store.ingest_m0(_symbol(a, 1, "X", 0))
        right = store.ingest_m0(_symbol(b, 2, "X", 0))
        self.assertNotEqual(left.vocabulary_id, right.vocabulary_id)
        self.assertNotEqual(left.uid, right.uid)

    def test_repeated_token_observations_are_distinct_m0_events(self) -> None:
        codec = DeterministicSymbolCodec("repeat")
        store = MultimodalMemoryStore()
        first = store.ingest_m0(_symbol(codec, 1, "X", 0))
        second = store.ingest_m0(_symbol(codec, 2, "X", 1))
        self.assertNotEqual(first.uid, second.uid)
        self.assertEqual(first.symbol_id, second.symbol_id)
        self.assertEqual(first.payload.availability, PayloadAvailabilityState.INLINE_IDENTITY)

    def test_m1g_is_structural_and_has_no_action_authority(self) -> None:
        codec = DeterministicSymbolCodec("relations")
        store = MultimodalMemoryStore()
        rows = store.ingest_timeline((_symbol(codec, 1, "A", 0), _symbol(codec, 2, "B", 1), _interaction(3)))
        relations = store.derive_m1g(rows)
        self.assertTrue(relations)
        self.assertTrue(all(not row.action_authority for row in relations))
        self.assertTrue(any(row.kind.value == "SYMBOL_PRECEDES_ACTION" for row in relations))

    def test_m1n_is_bounded_deduplicated_and_traceable(self) -> None:
        codec = DeterministicSymbolCodec("normalized")
        store = MultimodalMemoryStore()
        rows = store.ingest_timeline((_symbol(codec, 1, "A", 0), _symbol(codec, 2, "B", 1), _interaction(3)))
        relations = store.derive_m1g(rows)
        facts = store.normalize_m1g(relations + relations)
        self.assertLessEqual(sum(row.channel is NormalizedChannel.SYMBOL for row in facts), store.budgets.symbol_facts)
        self.assertLessEqual(sum(row.channel is NormalizedChannel.CROSS_MODAL for row in facts), store.budgets.cross_modal_facts)
        self.assertEqual(len({row.uid for row in facts}), len(facts))
        for fact in facts:
            self.assertTrue(store.trace_m1n_to_m0(fact.uid))
            self.assertIsNone(fact.lexical_semantics)

    def test_restart_preserves_full_m0_m1_provenance(self) -> None:
        codec = DeterministicSymbolCodec("restart-memory")
        store = MultimodalMemoryStore()
        rows = store.ingest_timeline((_symbol(codec, 1, "A", 0), _interaction(2)))
        facts = store.normalize_m1g(store.derive_m1g(rows))
        restored = MultimodalMemoryStore.from_state_dict(store.state_dict())
        self.assertEqual(restored.state_dict(), store.state_dict())
        self.assertTrue(restored.trace_m1n_to_m0(facts[0].uid))


if __name__ == "__main__":
    unittest.main()
