from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v8 import actor as actor_module
from v8.actor_read_view_v851 import ActorReadView
from v8.model import MemoryUid
from v8.publication import LiveReadView
from v8 import strategy_empirical_bootstrap_v881 as bootstrap
from v8.strategy_empirical_bootstrap_v881 import (
    _accumulate_generic_strategy_stat,
    _missing_bootstrap_attempts,
)


class StrategyEmpiricalBootstrapTests(unittest.TestCase):
    def test_preference_comparison_requires_exact_causally_reachable_plans(self) -> None:
        context = 41
        bucket = actor_module.stable_u64(context, person=b"v8-context")
        strategy_a = MemoryUid(7, 1)
        strategy_b = MemoryUid(7, 2)
        outcome_a = MemoryUid(6, 1)
        outcome_b = MemoryUid(6, 2)
        primary = SimpleNamespace(
            strategy_uid=strategy_a, outcome_uid=outcome_a, action_id=1
        )
        alternative = SimpleNamespace(
            strategy_uid=strategy_b, outcome_uid=outcome_b, action_id=2
        )
        view = SimpleNamespace(
            _node_by_uid={
                strategy_a: SimpleNamespace(key_parts=(1, 0, 0, bucket)),
                strategy_b: SimpleNamespace(key_parts=(2, 0, 0, bucket)),
            }
        )
        with patch(
            "v8.behavior_recovery._strategy_can_probe", return_value=True
        ):
            self.assertTrue(actor_module._comparison_outcomes_reachable(
                view, primary, alternative, context, (1, 2)
            ))
            self.assertFalse(actor_module._comparison_outcomes_reachable(
                view, primary, alternative, context, (1,)
            ))

    def test_bootstrap_wraps_both_publication_and_actual_actor_views(self) -> None:
        self.assertIs(
            LiveReadView.plan_candidates,
            bootstrap._BASE_PLAN_CANDIDATES,
        )
        self.assertIs(actor_module.LiveReadView, ActorReadView)
        self.assertIs(
            ActorReadView.plan_candidates,
            bootstrap._actor_plan_candidates_v881,
        )
        self.assertIsNotNone(bootstrap._BASE_PLAN_CANDIDATES)
        self.assertIsNotNone(bootstrap._BASE_ACTOR_PLAN_CANDIDATES)

    def test_bootstrap_only_fills_missing_attempts_to_three(self) -> None:
        row = SimpleNamespace(attempt_weight=0.0)
        self.assertEqual(_missing_bootstrap_attempts(row, 0), 3)
        self.assertEqual(_missing_bootstrap_attempts(row, 1), 2)
        self.assertEqual(_missing_bootstrap_attempts(row, 3), 0)

        row = SimpleNamespace(attempt_weight=2.0)
        self.assertEqual(_missing_bootstrap_attempts(row, 0), 1)
        self.assertEqual(_missing_bootstrap_attempts(row, 1), 0)

        row = SimpleNamespace(attempt_weight=3.0)
        self.assertEqual(_missing_bootstrap_attempts(row, 0), 0)

    def test_generic_strategy_execution_accumulates_empirical_stats(self) -> None:
        uid = MemoryUid(10, 20)
        plan = SimpleNamespace(strategy_uid=uid)
        stats = {}

        _accumulate_generic_strategy_stat(stats, plan, success=True, cost=2.5)
        _accumulate_generic_strategy_stat(stats, plan, success=False, cost=1.5)

        self.assertEqual(stats[uid][0], 2.0)
        self.assertEqual(stats[uid][1], 1.0)
        self.assertEqual(stats[uid][2], 4.0)

    def test_no_plan_does_not_create_generic_stats(self) -> None:
        stats = {}
        _accumulate_generic_strategy_stat(stats, None, success=True, cost=1.0)
        self.assertEqual(stats, {})


if __name__ == "__main__":
    unittest.main()
