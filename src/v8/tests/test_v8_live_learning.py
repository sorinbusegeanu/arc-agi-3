from __future__ import annotations

import queue
import unittest

from v8.actor import ActorJob, ActorLearningBatch, StrategyRunStat, _publish_learning
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
    def test_m1_terminal_reward_changes_behavioral_value(self) -> None:
        positive = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=1)))
        neutral = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=0)))
        negative = derive_proposal(MemoryLevel.M1, PipelineEvent(experience(polarity=-1)))
        self.assertGreater(positive.significance_sum, neutral.significance_sum)
        self.assertGreater(neutral.significance_sum, negative.significance_sum)
        self.assertGreater(positive.significance_sum, 0.0)
        self.assertLess(negative.significance_sum, 0.0)

    def test_m6_success_and_failure_are_distinct_outcomes(self) -> None:
        positive = derive_proposal(MemoryLevel.M6, PipelineEvent(experience(polarity=1)))
        negative = derive_proposal(MemoryLevel.M6, PipelineEvent(experience(polarity=-1)))
        self.assertNotEqual(positive.uid, negative.uid)
        self.assertNotEqual(positive.key_parts[:2], negative.key_parts[:2])
        self.assertEqual(len(positive.key_parts), 3)
        self.assertEqual(len(negative.key_parts), 3)

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
        _publish_learning(
            target,
            job=job,
            strategy_stats={uid: [2.0, 1.0, 3.5]},
            preference_probes=[],
            replanning_trials=[],
        )
        row = target.get_nowait()
        self.assertIsInstance(row, ActorLearningBatch)
        self.assertEqual(row.strategy_stats, (StrategyRunStat(uid, 2, 1, 3.5),))


if __name__ == "__main__":
    unittest.main()
