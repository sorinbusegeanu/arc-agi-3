from __future__ import annotations

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8.actor import (
    ActorJob,
    ActorLearningBatch,
    PreferenceProbeResult,
    ReplanningTrialResult,
    StrategyRunStat,
    _is_new_terminal_game,
    _merge_learning_batches,
    _publish_learning,
    _refresh_actor_graph_if_due,
    _reset_after_terminal_game,
)
from v8.development import derive_proposal
from v8.model import (
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    RelationType,
    ValidationState,
    proposal_fingerprint,
)
from v8.publication import _LINEAGE_RELATIONS
from v8.shard import _behavior_delta


def experience(*, polarity: int) -> ExperienceEvent:
    return ExperienceEvent(
        EventId.from_producer(1, 1),
        10,
        1,
        1,
        99,
        10,
        7,
        2,
        123,
        456,
        789,
        1.0,
        16,
        polarity,
        555,
        8,
        0.25,
    )


class LiveLearningRegressionTests(unittest.TestCase):
    def test_m1_terminal_label_does_not_change_behavioral_value(self) -> None:
        positive = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=1)))
        neutral = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=0)))
        negative = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=-1)))
        self.assertAlmostEqual(positive.significance_sum, neutral.significance_sum)
        self.assertAlmostEqual(neutral.significance_sum, negative.significance_sum)
        self.assertGreater(positive.significance_sum, 0.0)

    def test_m6_terminal_label_does_not_change_outcome_identity(self) -> None:
        positive = derive_proposal(MemoryLevel.M6, PipelineEvent(experience(polarity=1)))
        neutral = derive_proposal(MemoryLevel.M6, PipelineEvent(experience(polarity=0)))
        negative = derive_proposal(MemoryLevel.M6, PipelineEvent(experience(polarity=-1)))
        self.assertEqual(positive.uid, neutral.uid)
        self.assertEqual(neutral.uid, negative.uid)
        self.assertEqual(positive.key_parts, negative.key_parts)
        self.assertEqual(len(positive.key_parts), 3)

    def test_peer_metrics_change_hot_action_value(self) -> None:
        key = (7, 2, 123, 8)
        base = dict(
            uid=MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
            fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
            event_id=EventId.from_producer(9, 1),
            watermark=20,
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=key,
            support_delta=0,
            score_weight=0.0,
            parent_uid=MemoryUid.zero(),
            relation_type=RelationType.EXPLAINS,
            cognitive_state=-1,
            validation_state=-1,
        )
        prediction = MemoryProposal(**base, prediction_error_sum=1.0)
        transfer = MemoryProposal(**base, transfer_prior_sum=1.0)
        prediction_value, prediction_weight = _behavior_delta(prediction)
        transfer_value, transfer_weight = _behavior_delta(transfer)
        self.assertLess(prediction_value, 0.0)
        self.assertGreater(transfer_value, 0.0)
        self.assertEqual(prediction_weight, 1.0)
        self.assertEqual(transfer_weight, 1.0)

    def test_non_lineage_graph_relations_are_not_ancestry(self) -> None:
        self.assertIn(int(RelationType.EXPLAINS), _LINEAGE_RELATIONS)
        self.assertIn(int(RelationType.LEADS_TO), _LINEAGE_RELATIONS)
        self.assertNotIn(int(RelationType.SIMILAR_TO), _LINEAGE_RELATIONS)
        self.assertNotIn(int(RelationType.PREFERENCE), _LINEAGE_RELATIONS)
        self.assertNotIn(int(RelationType.SUPERSEDES), _LINEAGE_RELATIONS)
        self.assertNotIn(int(RelationType.TRANSFER_CORRESPONDENCE), _LINEAGE_RELATIONS)

    def test_learning_batch_is_published_at_episode_boundary(self) -> None:
        target: queue.Queue[object] = queue.Queue()
        job = ActorJob(1, "game", 10, 0)
        uid = MemoryUid(1, 2)
        published = _publish_learning(
            target,
            job=job,
            strategy_stats={uid: [2.0, 1.0, 3.5]},
            preference_probes=[],
            replanning_trials=[],
        )
        self.assertTrue(published)
        row = target.get_nowait()
        self.assertIsInstance(row, ActorLearningBatch)
        self.assertEqual(row.strategy_stats, (StrategyRunStat(uid, 2, 1, 3.5),))

    def test_full_learning_queue_retains_batch_for_later_retry(self) -> None:
        target: queue.Queue[object] = queue.Queue(maxsize=1)
        target.put_nowait(object())
        job = ActorJob(1, "game", 10, 0)
        uid = MemoryUid(1, 2)

        self.assertFalse(
            _publish_learning(
                target,
                job=job,
                strategy_stats={uid: [2.0, 1.0, 3.5]},
                preference_probes=[],
                replanning_trials=[],
            )
        )
        self.assertEqual(target.qsize(), 1)

    def test_learning_batches_are_merged_losslessly_per_actor(self) -> None:
        uid = MemoryUid(1, 2)
        outcome_a = MemoryUid(3, 4)
        outcome_b = MemoryUid(5, 6)
        probe = PreferenceProbeResult(outcome_a, outcome_b, 7, outcome_a, True)
        trial = ReplanningTrialResult(uid, outcome_a, outcome_b, True)
        merged = _merge_learning_batches(
            (
                ActorLearningBatch(
                    1,
                    "game",
                    (StrategyRunStat(uid, 2, 1, 3.5),),
                    (probe,),
                    (),
                    0,
                ),
                ActorLearningBatch(
                    1,
                    "game",
                    (StrategyRunStat(uid, 3, 2, 4.5),),
                    (),
                    (trial,),
                    1,
                ),
            )
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].strategy_stats, (StrategyRunStat(uid, 5, 3, 8.0),))
        self.assertEqual(merged[0].preference_probes, (probe,))
        self.assertEqual(merged[0].replanning_trials, (trial,))
        self.assertEqual(merged[0].replans, 1)

    def test_terminal_episode_is_counted_waited_and_reset_only_once(self) -> None:
        env = SimpleNamespace(
            last_outcome_state="WIN",
            last_step_was_reset_boundary=False,
            reset=Mock(),
        )
        self.assertTrue(_is_new_terminal_game(env))
        with patch("v8.actor.time.sleep") as sleep:
            _reset_after_terminal_game(env, 1.0)
        sleep.assert_called_once_with(1.0)
        env.reset.assert_called_once_with()

        env.last_step_was_reset_boundary = True
        self.assertFalse(_is_new_terminal_game(env))

    def test_actor_graph_check_occurs_each_thousand_completed_steps(self) -> None:
        read_view = Mock()

        next_check = _refresh_actor_graph_if_due(
            read_view,
            completed_steps=999,
            next_check_step=1_000,
        )
        self.assertEqual(next_check, 1_000)
        read_view.invalidate_strategy_cache.assert_not_called()

        next_check = _refresh_actor_graph_if_due(
            read_view,
            completed_steps=1_000,
            next_check_step=next_check,
        )
        self.assertEqual(next_check, 2_000)
        read_view.invalidate_strategy_cache.assert_called_once_with()

        next_check = _refresh_actor_graph_if_due(
            read_view,
            completed_steps=2_500,
            next_check_step=next_check,
            check_interval_steps=1_000,
        )
        self.assertEqual(next_check, 3_000)
        self.assertEqual(read_view.invalidate_strategy_cache.call_count, 2)

        next_check = _refresh_actor_graph_if_due(
            read_view,
            completed_steps=3_000,
            next_check_step=3_000,
            check_interval_steps=250,
        )
        self.assertEqual(next_check, 3_250)


if __name__ == "__main__":
    unittest.main()
