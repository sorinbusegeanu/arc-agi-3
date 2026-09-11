from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from v8 import actor
from v8.model import CognitiveState, MemoryUid
from v8.strategy_statistics_persistence_fix_v828 import (
    _emit_committed_strategy_efficiency,
    _final_residual_batches,
    _merge_strategy_stats,
)
from v8.trajectory_efficiency_v054 import _TRACKER


class StrategyStatisticsPersistenceFixTests(unittest.TestCase):
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

    def test_committed_efficiency_requires_two_empirical_same_cohort(self) -> None:
        outcome = MemoryUid(10, 11)
        fast_uid = MemoryUid(12, 13)
        slow_uid = MemoryUid(14, 15)
        active = int(CognitiveState.ACTIVE)
        nodes = {
            fast_uid: SimpleNamespace(
                uid=fast_uid,
                cognitive_state=active,
                attempt_weight=2.0,
                updated_watermark=20,
            ),
            slow_uid: SimpleNamespace(
                uid=slow_uid,
                cognitive_state=active,
                attempt_weight=3.0,
                updated_watermark=21,
            ),
        }
        rows = [
            SimpleNamespace(strategy_uid=fast_uid, outcome_uid=outcome, mean_cost=2.0),
            SimpleNamespace(strategy_uid=slow_uid, outcome_uid=outcome, mean_cost=4.0),
        ]

        class View:
            _node_by_uid = nodes
            _strategy_by_context = {99: rows}

            def invalidate_strategy_cache(self):
                return None

            def _refresh_strategy_cache(self):
                return None

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
        nodes = {
            first_uid: SimpleNamespace(
                uid=first_uid,
                cognitive_state=probation,
                attempt_weight=1.0,
                updated_watermark=30,
            ),
            second_uid: SimpleNamespace(
                uid=second_uid,
                cognitive_state=probation,
                attempt_weight=1.0,
                updated_watermark=31,
            ),
        }
        rows = [
            SimpleNamespace(strategy_uid=first_uid, outcome_uid=outcome, mean_cost=2.0),
            SimpleNamespace(strategy_uid=second_uid, outcome_uid=outcome, mean_cost=3.0),
        ]

        class View:
            _node_by_uid = nodes
            _strategy_by_context = {101: rows}

            def invalidate_strategy_cache(self):
                return None

            def _refresh_strategy_cache(self):
                return None

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
        source = SimpleNamespace(
            uid=strategy_uid,
            cognitive_state=int(CognitiveState.ACTIVE),
            attempt_weight=0.0,
            updated_watermark=40,
        )

        class View:
            _node_by_uid = {strategy_uid: source}
            _strategy_by_context = {
                202: [
                    SimpleNamespace(
                        strategy_uid=strategy_uid,
                        outcome_uid=outcome,
                        mean_cost=1.0,
                    )
                ]
            }

            def invalidate_strategy_cache(self):
                return None

            def _refresh_strategy_cache(self):
                return None

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
