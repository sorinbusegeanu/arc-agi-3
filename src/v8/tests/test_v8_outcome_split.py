from __future__ import annotations

import unittest

from v8.arena import EdgeRecord, SharedActionArena, SharedEdgeArena, SharedNodeArena, NodeRecord
from v8.lifecycle import LifecycleController
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.publication import LiveReadView, ShardReadDescriptor


def outcome_node(key: tuple[int, ...], *, state: CognitiveState, validation: ValidationState) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, key),
        fingerprint=1,
        level=int(MemoryLevel.M6),
        memory_type=int(MemoryType.OUTCOME),
        key_parts=key,
        support_count=8,
        significance_sum=4.0,
        prediction_error_sum=0.0,
        learning_value_sum=2.0,
        transfer_prior_sum=0.0,
        explanatory_sum=2.0,
        future_option_sum=1.0,
        score_weight=8.0,
        updated_watermark=10,
        cognitive_state=int(state),
        validation_state=int(validation),
    )


def strategy_node(action: int, outcome: MemoryUid, context: int) -> NodeRecord:
    key = (action, outcome.hi, outcome.lo, stable_u64(context, person=b"v8-context"))
    return NodeRecord(
        uid=MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key),
        fingerprint=2,
        level=int(MemoryLevel.M7),
        memory_type=int(MemoryType.STRATEGY),
        key_parts=key,
        support_count=5,
        significance_sum=2.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=1.0,
        score_weight=5.0,
        updated_watermark=10,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
        success_sum=4.0,
        cost_sum=5.0,
        attempt_weight=5.0,
    )


def contingency_node(context: int, action: int) -> NodeRecord:
    key = (context, action, 77, context + 1)
    return NodeRecord(
        uid=MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
        fingerprint=3,
        level=int(MemoryLevel.M1),
        memory_type=int(MemoryType.CONTINGENCY),
        key_parts=key,
        support_count=5,
        significance_sum=2.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=5.0,
        score_weight=5.0,
        updated_watermark=10,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.VALIDATED),
    )


class OutcomeSplitTests(unittest.TestCase):
    def test_failed_coarse_outcome_is_quarantined_as_split(self) -> None:
        coarse = outcome_node(
            (1, 2),
            state=CognitiveState.ACTIVE,
            validation=ValidationState.FAILED,
        )
        decision = LifecycleController().decide(coarse)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.QUARANTINED))
        self.assertIn("split", decision.reason)

    def test_strategy_for_inactive_coarse_outcome_is_not_selectable(self) -> None:
        nodes = SharedNodeArena(capacity=8)
        edges = SharedEdgeArena(capacity=8)
        actions = SharedActionArena(capacity=8)
        view = None
        try:
            coarse = outcome_node(
                (1, 2),
                state=CognitiveState.QUARANTINED,
                validation=ValidationState.FAILED,
            )
            fine = outcome_node(
                (1, 2, 3),
                state=CognitiveState.ACTIVE,
                validation=ValidationState.STRUCTURAL,
            )
            coarse_strategy = strategy_node(1, coarse.uid, 99)
            fine_strategy = strategy_node(2, fine.uid, 99)
            fine_contingency = contingency_node(99, 2)
            nodes.begin_write()
            try:
                for index, row in enumerate(
                    (coarse, fine, coarse_strategy, fine_strategy, fine_contingency)
                ):
                    nodes.write(index, row)
            finally:
                nodes.end_write(count=5)

            # The active fine strategy is a scientifically admissible strategy:
            # the observed M1 contingency belongs to the M6 outcome's lineage and
            # the M7 strategy explicitly depends on that same contingency.
            edges.begin_write()
            try:
                edges.write(
                    0,
                    EdgeRecord(
                        fine.uid,
                        int(RelationType.EXPLAINS),
                        fine_contingency.uid,
                        5,
                        10,
                    ),
                )
                edges.write(
                    1,
                    EdgeRecord(
                        fine_strategy.uid,
                        int(RelationType.DEPENDS_ON),
                        fine_contingency.uid,
                        5,
                        10,
                    ),
                )
            finally:
                edges.end_write(count=2)

            descriptor = ShardReadDescriptor(nodes.descriptor, edges.descriptor, actions.descriptor)
            view = LiveReadView((descriptor,))
            candidates = view.plan_candidates(99, (1, 2))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].outcome_uid, fine.uid)
            self.assertEqual(candidates[0].strategy_uid, fine_strategy.uid)
        finally:
            if view is not None:
                view.close()
            nodes.dispose()
            edges.dispose()
            actions.dispose()


if __name__ == "__main__":
    unittest.main()
