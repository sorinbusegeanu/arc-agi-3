from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8 import actor, primary_valence
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.runtime_v82 import V82ContinuousMemoryRuntime
from v8.scientific_traceability import TRACEABILITY
from v8.trajectory_efficiency_v054 import (
    TrajectoryEfficiencyTracker,
    _TRACKER,
    _actions_from_discounted_valence,
    _relative_efficiency,
)


class TrajectoryEfficiencyV054Tests(unittest.TestCase):
    def test_tracker_counts_actions_until_represented_outcome(self) -> None:
        tracker = TrajectoryEfficiencyTracker()
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 9, 2))
        plan = SimpleNamespace(strategy_uid=strategy, outcome_uid=outcome)

        tracker.observe(plan, (), terminal=False)
        tracker.observe(plan, (), terminal=False)
        tracker.observe(plan, (outcome,), terminal=False)

        self.assertEqual(tracker.stats[strategy], [1.0, 1.0, 3.0])

    def test_terminal_boundary_closes_unreached_strategy_with_realized_cost(self) -> None:
        tracker = TrajectoryEfficiencyTracker()
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (5, 6, 7, 8))
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (0, 7, 1))
        plan = SimpleNamespace(strategy_uid=strategy, outcome_uid=outcome)

        tracker.observe(plan, (), terminal=False)
        tracker.observe(plan, (), terminal=True)

        self.assertEqual(tracker.stats[strategy], [1.0, 0.0, 2.0])
        self.assertEqual(tracker.episode_step, 0)

    def test_relative_efficiency_is_only_within_same_outcome_and_context(self) -> None:
        outcome_a = MemoryUid(1, 1)
        outcome_b = MemoryUid(2, 2)
        fast, slow, other, other_context = (MemoryUid(i, i) for i in range(10, 14))
        rows = (
            SimpleNamespace(strategy_uid=fast, outcome_uid=outcome_a, context_bucket=7, mean_cost=4.0),
            SimpleNamespace(strategy_uid=slow, outcome_uid=outcome_a, context_bucket=7, mean_cost=8.0),
            SimpleNamespace(strategy_uid=other, outcome_uid=outcome_b, context_bucket=7, mean_cost=1.0),
            SimpleNamespace(strategy_uid=other_context, outcome_uid=outcome_a, context_bucket=8, mean_cost=2.0),
        )

        relative = _relative_efficiency(rows)

        self.assertAlmostEqual(relative[fast], 1.0)
        self.assertAlmostEqual(relative[slow], 0.5)
        self.assertNotIn(other, relative)
        self.assertNotIn(other_context, relative)

    def test_live_learning_batch_uses_realized_trajectory_stats_not_old_step_cost(self) -> None:
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (3, 4, 5, 6))
        _TRACKER.reset()
        _TRACKER.stats[strategy] = [2.0, 1.0, 7.0]
        primary_valence._PENDING_CREDITS.clear()
        primary_valence._PENDING_VALENCE_PREFERENCES.clear()
        primary_valence._WINDOW_ACHIEVEMENT[strategy] = [9.0, 9.0, 1.0]
        try:
            batch = actor._learning_batch(
                job=SimpleNamespace(actor_id=7, game_id="g"),
                strategy_stats={strategy: [99.0, 99.0, 99.0]},
                preference_probes=[],
                replanning_trials=[],
            )
            self.assertIsNotNone(batch)
            self.assertEqual(len(batch.strategy_stats), 1)
            stat = batch.strategy_stats[0]
            self.assertEqual(stat.attempts, 2)
            self.assertEqual(stat.successes, 1)
            self.assertAlmostEqual(stat.cost, 7.0)
        finally:
            _TRACKER.reset()
            primary_valence._WINDOW_ACHIEVEMENT.clear()

    def test_discounted_primary_valence_exposes_actions_to_valence_boundary(self) -> None:
        gamma = primary_valence._VALENCE_GAMMA
        credit = SimpleNamespace(weight=1.0, valence_sum=gamma ** 4)
        self.assertAlmostEqual(_actions_from_discounted_valence(credit), 5.0)

    def test_h12_requires_outcome_conditioned_efficiency(self) -> None:
        h12 = next(row for row in TRACEABILITY if row.hypothesis_id == "H12")
        self.assertIn("same or explicitly comparable M6 outcome", h12.paper_claim)
        self.assertIn("strategy_efficiency", h12.required_evidence)
        self.assertIn("primary_valence_efficiency", h12.candidate_evidence)

    def test_runtime_reports_v054_semantics(self) -> None:
        self.assertEqual(V82ContinuousMemoryRuntime.research_paper_version, "0.5.4")
        self.assertEqual(
            V82ContinuousMemoryRuntime.scientific_semantics_version,
            "v8.4-outcome-conditioned-efficiency",
        )


if __name__ == "__main__":
    unittest.main()
