from __future__ import annotations

import unittest
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import v8
from v8 import actor
from v8 import development
from v8 import model
from v8 import primary_valence
from v8.arena import NodeRecord, SharedNodeArena
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    ValidationState,
    proposal_fingerprint,
)
from v8.observation_contract import ARC_GRID_CONTRACT
from v8.runtime_v82 import V82ContinuousMemoryRuntime


class PrimaryValenceTests(unittest.TestCase):
    def test_installed_actor_worker_is_forkserver_picklable(self) -> None:
        payload = ForkingPickler.dumps(actor.actor_worker)
        self.assertTrue(payload)
        self.assertNotIn("<locals>", actor.actor_worker.__qualname__)

    def test_proposal_codec_preserves_primary_valence_statistics(self) -> None:
        key = (11, 2, 33, 44)
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
        proposal = model.MemoryProposal(
            uid=uid,
            fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
            event_id=EventId.from_producer(1, 2),
            watermark=7,
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=key,
            support_delta=0,
            score_weight=0.0,
            primary_valence_sum=-1.4,
            primary_valence_sq_sum=1.02,
            primary_valence_weight=2.0,
            positive_valence_count=0.25,
            negative_valence_count=1.75,
        )
        restored = model.decode_proposal(model.encode_proposal(proposal))
        self.assertAlmostEqual(restored.primary_valence_sum, -1.4)
        self.assertAlmostEqual(restored.primary_valence_sq_sum, 1.02)
        self.assertAlmostEqual(restored.primary_valence_weight, 2.0)
        self.assertAlmostEqual(restored.positive_valence_count, 0.25)
        self.assertAlmostEqual(restored.negative_valence_count, 1.75)

    def test_node_arena_preserves_primary_valence_without_changing_identity(self) -> None:
        arena = SharedNodeArena(capacity=4)
        try:
            key = (101, 3, 202, 303)
            uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
            record = NodeRecord(
                uid=uid,
                fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
                level=int(MemoryLevel.M1),
                memory_type=int(MemoryType.CONTINGENCY),
                key_parts=key,
                support_count=4,
                significance_sum=2.0,
                prediction_error_sum=0.0,
                learning_value_sum=1.0,
                transfer_prior_sum=0.0,
                explanatory_sum=0.0,
                future_option_sum=0.0,
                score_weight=4.0,
                updated_watermark=5,
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(ValidationState.VALIDATED),
                primary_valence_sum=1.5,
                primary_valence_sq_sum=1.25,
                primary_valence_weight=2.0,
                positive_valence_count=1.75,
                negative_valence_count=0.25,
            )
            arena.begin_write()
            arena.write(0, record)
            arena.end_write(count=1)
            restored = arena.read(0)
            self.assertEqual(restored.uid, uid)
            self.assertEqual(restored.key_parts, key)
            self.assertAlmostEqual(restored.expected_primary_valence, 0.75)
            self.assertGreater(restored.primary_valence_confidence, 0.0)
            self.assertAlmostEqual(restored.positive_valence_count, 1.75)
            self.assertAlmostEqual(restored.negative_valence_count, 0.25)
        finally:
            arena.dispose()

    def _event(self, terminal_polarity: int) -> ExperienceEvent:
        return ExperienceEvent(
            event_id=EventId.from_producer(9, 1),
            watermark=10,
            producer_id=9,
            producer_sequence=1,
            source_game_hash=123,
            global_step=10,
            context_signature=111,
            action_id=2,
            outcome_signature=222,
            family_signature=333,
            carrier_signature=0,
            future_option_delta=0.0,
            changed_cells=1,
            terminal_polarity=terminal_polarity,
            trajectory_signature=444,
            next_context_signature=555,
            prediction_error=0.0,
        )

    def test_terminal_polarity_is_valence_not_memory_identity(self) -> None:
        positive = development.derive_proposal(MemoryLevel.M1, PipelineEvent(self._event(1)))
        negative = development.derive_proposal(MemoryLevel.M1, PipelineEvent(self._event(-1)))
        neutral = development.derive_proposal(MemoryLevel.M1, PipelineEvent(self._event(0)))
        self.assertEqual(positive.uid, negative.uid)
        self.assertEqual(positive.uid, neutral.uid)
        self.assertEqual(positive.key_parts, negative.key_parts)
        self.assertAlmostEqual(positive.primary_valence_sum, 1.0)
        self.assertAlmostEqual(negative.primary_valence_sum, -1.0)
        self.assertAlmostEqual(neutral.primary_valence_weight, 0.0)
        self.assertAlmostEqual(positive.significance_sum, 1.0)
        self.assertAlmostEqual(negative.significance_sum, 1.0)

    def test_delayed_credit_keeps_sign_and_discount(self) -> None:
        primary_valence._PENDING_CREDITS.clear()
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4))
        primary_valence._accumulate_credit(
            uid,
            level=int(MemoryLevel.M1),
            memory_type=int(MemoryType.CONTINGENCY),
            key_parts=(1, 2, 3, 4),
            fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4)),
            value=-0.97,
        )
        credit = primary_valence._credit_tuple()[0]
        self.assertAlmostEqual(credit.valence_sum, -0.97)
        self.assertAlmostEqual(credit.valence_sq_sum, 0.97 * 0.97)
        self.assertEqual(credit.positive_count, 0.0)
        self.assertEqual(credit.negative_count, 1.0)
        primary_valence._PENDING_CREDITS.clear()

    def test_strategy_achievement_reliability_is_not_terminal_valence(self) -> None:
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        primary_valence._WINDOW_ACHIEVEMENT.clear()
        primary_valence._WINDOW_ACHIEVEMENT[strategy] = [3.0, 1.0, 5.0]
        batch = actor._learning_batch(
            job=SimpleNamespace(actor_id=7, game_id="g"),
            strategy_stats={strategy: [3.0, 3.0, 7.0]},
            preference_probes=[],
            replanning_trials=[],
        )
        self.assertIsNotNone(batch)
        stat = batch.strategy_stats[0]
        self.assertEqual(stat.attempts, 3)
        self.assertEqual(stat.successes, 1)
        self.assertAlmostEqual(stat.cost, 5.0)
        primary_valence._WINDOW_ACHIEVEMENT.clear()

    def test_primary_valence_is_admitted_without_task_semantic_reward_fields(self) -> None:
        forbidden = set(ARC_GRID_CONTRACT.forbidden_semantic_fields)
        self.assertIn("reward", forbidden)
        self.assertIn("win_value", forbidden)
        self.assertIn("terminal_value", forbidden)
        self.assertNotIn("terminal_polarity", forbidden)
        self.assertNotIn("primary_valence", forbidden)
        self.assertEqual(ARC_GRID_CONTRACT.contract_id, "arc-grid-v1-primary-valence")

    def test_runtime_reports_v053_semantics(self) -> None:
        self.assertEqual(V82ContinuousMemoryRuntime.research_paper_version, "0.5.3")
        self.assertEqual(V82ContinuousMemoryRuntime.scientific_semantics_version, "v8.3-primary-valence")


if __name__ == "__main__":
    unittest.main()
