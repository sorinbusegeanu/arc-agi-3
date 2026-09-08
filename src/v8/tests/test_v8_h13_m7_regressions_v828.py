from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid
from v8.outcome_holdout_member_aggregation_fix_v828 import (
    _select_occurrence_holdout_class,
)
from v8.outcome_holdout_v828 import _derive_class
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.promotion import EvidenceGatedPromotionEngine
from v8.promotion_strategy_fairness_v828 import install_promotion_strategy_fairness_v828


def _m6_row(lo: int, *, support: int, variant: int = 1):
    return SimpleNamespace(
        uid=MemoryUid(1, lo),
        level=int(MemoryLevel.M6),
        memory_type=int(MemoryType.OUTCOME),
        key_parts=(1, 2, variant),
        support_count=int(support),
        cognitive_state=int(CognitiveState.PROBATION),
        future_option_delta=1.0,
        significance=1.0,
        learning_value=1.0,
    )


def _m1_row(lo: int):
    return SimpleNamespace(
        uid=MemoryUid(2, lo),
        level=int(MemoryLevel.M1),
        memory_type=int(MemoryType.CONTINGENCY),
        key_parts=(100 + lo, 7, 256, 200 + lo),
        support_count=8,
        cognitive_state=int(CognitiveState.PROBATION),
        future_option_delta=1.0,
        significance=1.0,
        learning_value=1.0,
    )


def _m5_row(lo: int):
    return SimpleNamespace(
        uid=MemoryUid(3, lo),
        level=int(MemoryLevel.M5),
        memory_type=int(MemoryType.CONSEQUENCE),
        key_parts=(11, lo, 1000 + lo, 1),
        support_count=8,
        cognitive_state=int(CognitiveState.PROBATION),
        future_option_delta=1.0,
        significance=1.0,
        learning_value=1.0,
        transfer_prior=0.0,
        explanatory_reach=1.0,
    )


class H13M7RegressionTests(unittest.TestCase):
    def test_h13_world_split_preserves_member_level_context_consistency(self) -> None:
        estimator = OutcomeEquivalenceEstimator()
        rows = (_m6_row(1, support=10), _m6_row(2, support=10))
        class_uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2))
        full_class = _derive_class(
            (1, 2), rows, uid=class_uid, version=1, estimator=estimator
        )
        by_uid = {row.uid: row for row in rows}
        occurrences = {
            rows[0].uid: {10: 5, 30: 5},
            rows[1].uid: {20: 5, 30: 5},
        }
        validation = _select_occurrence_holdout_class(
            estimator, full_class, by_uid, occurrences
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertGreaterEqual(validation.training_class.context_consistency, 0.50)
        self.assertTrue(validation.training_persistent)
        self.assertEqual(len(validation.training_class.members), 2)

    def test_m7_is_not_starved_when_m6_candidates_fill_budget(self) -> None:
        install_promotion_strategy_fairness_v828()
        engine = EvidenceGatedPromotionEngine()
        nodes = tuple([_m1_row(1), _m6_row(50, support=8)] + [_m5_row(i) for i in range(1, 20)])
        rows = engine.propose(nodes, (), budget=8)
        self.assertLessEqual(len(rows), 8)
        self.assertTrue(
            any(int(row.level) == int(MemoryLevel.M7) for row in rows),
            "reserved strategy capacity must keep M7 reachable",
        )


if __name__ == "__main__":
    unittest.main()
