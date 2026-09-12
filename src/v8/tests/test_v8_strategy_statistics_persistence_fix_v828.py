from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from v8 import actor
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid
from v8.strategy_statistics_persistence_fix_v828 import (
    _emit_committed_strategy_efficiency,
    _final_residual_batches,
    _merge_strategy_stats,
    _run_actor_jobs_v828,
)
from v8 import strategy_statistics_persistence_fix_v828 as persistence
from v8.trajectory_efficiency_v054 import _TRACKER


class StrategyStatisticsPersistenceFixTests(unittest.TestCase):
    @staticmethod
    def strategy_node(uid, outcome, context, *, attempts, cost, state):
        return SimpleNamespace(
            uid=uid,
            level=int(MemoryLevel.M7),
            memory_type=int(MemoryType.STRATEGY),
            key_parts=(1, outcome.hi, outcome.lo, context),
            cognitive_state=state,
            attempt_weight=attempts,
            cost_sum=cost,
            updated_watermark=20,
        )

    @staticmethod
    def outcome_node(uid):
        return SimpleNamespace(uid=uid, level=int(MemoryLevel.M6))

    def test_merge_preserves_normal_stats_and_adds_tracker_only_uid(self) -> None:
        normal_uid = MemoryUid(1, 2)
        tracker_uid = MemoryUid(3, 4)
        _TRACKER.reset()
        _TRACKER.stats[normal_uid] = [9.0, 9.0, 99.0]
        _TRACKER.stats[tracker_uid] = [2.0, 1.0, 7.0]
        try:
            merged = _merge_strategy_stats(
                actor,
                {normal_uid: [3.0, 2.0, 5.0]},
            )
        finally:
            _TRACKER.reset()

        by_uid = {row.strategy_uid: row for row in merged}
        self.assertEqual(by_uid[normal_uid].attempts, 3)
        self.assertEqual(by_uid[normal_uid].successes, 2)
        self.assertEqual(by_uid[normal_uid].cost, 5.0)
        self.assertEqual(by_uid[tracker_uid].attempts, 2)
        self.assertEqual(by_uid[tracker_uid].successes, 1)
        self.assertEqual(by_uid[tracker_uid].cost, 7.0)

    def test_final_residual_persists_only_unpublished_actor_totals(self) -> None:
        uid = MemoryUid(5, 6)
        result = SimpleNamespace(
            actor_id=7,
            game_id="g",
            strategy_stats=(actor.StrategyRunStat(uid, 5, 3, 10.0),),
        )
        published = defaultdict(lambda: [0.0, 0.0, 0.0])
        published[(7, "g", uid)] = [2.0, 1.0, 4.0]

        batches = _final_residual_batches(actor, (result,), published)

        self.assertEqual(len(batches), 1)
        stat = batches[0].strategy_stats[0]
        self.assertEqual(stat.strategy_uid, uid)
        self.assertEqual(stat.attempts, 3)
        self.assertEqual(stat.successes, 2)
        self.assertEqual(stat.cost, 6.0)

    def test_actor_result_totals_are_not_reingested_as_tracker_residuals(self) -> None:
        result = SimpleNamespace(strategy_stats=(actor.StrategyRunStat(
            MemoryUid(9, 10), 5, 3, 10.0
        ),))
        runtime = SimpleNamespace(record_actor_results=lambda _rows: self.fail(
            "cumulative ActorResult totals must not be ingested"
        ))
        with patch.object(persistence, "_BASE_RUN_ACTOR_JOBS", return_value=(result,)):
            self.assertEqual(_run_actor_jobs_v828(runtime, ()), (result,))

    def test_committed_efficiency_requires_two_empirical_same_cohort(self) -> None:
        outcome = MemoryUid(10, 11)
        fast_uid = MemoryUid(12, 13)
        slow_uid = MemoryUid(14, 15)
        active = int(CognitiveState.ACTIVE)
        node_rows = (
            self.outcome_node(outcome),
            self.strategy_node(fast_uid, outcome, 99, attempts=2.0, cost=4.0, state=active),
            self.strategy_node(slow_uid, outcome, 99, attempts=3.0, cost=12.0, state=active),
        )

        class View:
            def node_records(self):
                return node_rows

        emitted = []

        class Supervisor:
            read_view = View()

            def _fresh(self, *_args):
                return True

            def _append_evidence(self, kind, row, value, **_kwargs):
                emitted.append((kind, row.uid, value))

        count = _emit_committed_strategy_efficiency(Supervisor())

        self.assertEqual(count, 2)
        by_uid = {uid: value for _kind, uid, value in emitted}
        self.assertAlmostEqual(by_uid[fast_uid], 1.0)
        self.assertAlmostEqual(by_uid[slow_uid], 0.5)

    def test_probationary_empirical_strategies_are_comparable(self) -> None:
        outcome = MemoryUid(20, 21)
        first_uid = MemoryUid(22, 23)
        second_uid = MemoryUid(24, 25)
        probation = int(CognitiveState.PROBATION)
        node_rows = (
            self.outcome_node(outcome),
            self.strategy_node(first_uid, outcome, 101, attempts=1.0, cost=2.0, state=probation),
            self.strategy_node(second_uid, outcome, 101, attempts=1.0, cost=3.0, state=probation),
        )

        class View:
            def node_records(self):
                return node_rows

        emitted = []

        class Supervisor:
            read_view = View()

            def _fresh(self, *_args):
                return True

            def _append_evidence(self, kind, row, value, **_kwargs):
                emitted.append((kind, row.uid, value))

        self.assertEqual(_emit_committed_strategy_efficiency(Supervisor()), 2)
        self.assertEqual({kind for kind, _uid, _value in emitted}, {"strategy_efficiency"})

    def test_zero_attempt_strategies_are_reported_as_rejected_inputs(self) -> None:
        outcome = MemoryUid(30, 31)
        strategy_uid = MemoryUid(32, 33)
        source = self.strategy_node(
            strategy_uid, outcome, 202, attempts=0.0, cost=0.0,
            state=int(CognitiveState.ACTIVE),
        )

        class View:
            def node_records(self):
                return (StrategyStatisticsPersistenceFixTests.outcome_node(outcome), source)

        class Supervisor:
            read_view = View()

        with patch("v8.information_flow_diagnostics.emit_bounded") as emit:
            self.assertEqual(_emit_committed_strategy_efficiency(Supervisor()), 0)

        self.assertEqual(emit.call_args.kwargs["input_count"], 1)
        self.assertEqual(
            emit.call_args.kwargs["rejection_counts"],
            {"strategy_without_empirical_attempt": 1},
        )


if __name__ == "__main__":
    unittest.main()
