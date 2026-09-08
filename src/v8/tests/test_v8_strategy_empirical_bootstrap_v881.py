from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.model import MemoryUid
from v8.strategy_empirical_bootstrap_v881 import (
    _accumulate_generic_strategy_stat,
    _missing_bootstrap_attempts,
)


class StrategyEmpiricalBootstrapTests(unittest.TestCase):
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
