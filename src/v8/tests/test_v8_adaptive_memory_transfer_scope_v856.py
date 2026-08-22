from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install production runtime stack
from v8 import adaptive_memory_transfer_integrity_v856 as v856
from v8 import adaptive_memory_transfer_scope_v856 as scope
from v8 import behavior_recovery as behavior
from v8 import sampling_progress_control_v829 as v829
from v8.actor import StrategyRunStat
from v8.model import MemoryLevel, MemoryUid, stable_u64
from v8.primary_valence import PrimaryValenceCredit
from v8.publication import PlannedAction


class _ReadView:
    def __init__(self, provenance) -> None:
        self.provenance = provenance

    def source_games(self, uid):
        return frozenset(self.provenance.get(uid, ()))


class AdaptiveMemoryTransferScopeV856Tests(unittest.TestCase):
    def setUp(self) -> None:
        v856._RECENT_FAILURES.clear()
        v856._TARGET_TRANSFER_FAILURES.clear()
        self.prior_view = behavior._CURRENT_ACTOR_VIEW
        v829._CONTROL_STATE.game_id = "target"
        v829._CONTROL_STATE.context = 123

    def tearDown(self) -> None:
        behavior._CURRENT_ACTOR_VIEW = self.prior_view
        v856._RECENT_FAILURES.clear()
        v856._TARGET_TRANSFER_FAILURES.clear()
        for name in ("game_id", "context"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass

    def test_foreign_strategy_stats_are_removed_but_same_world_stats_are_kept(self) -> None:
        target_hash = int(stable_u64("target", person=b"v8-game"))
        source_hash = int(stable_u64("source", person=b"v8-game"))
        same_uid = MemoryUid(7, 1)
        foreign_uid = MemoryUid(7, 2)
        runtime = SimpleNamespace(
            read_view=_ReadView(
                {
                    same_uid: (target_hash,),
                    foreign_uid: (source_hash,),
                }
            )
        )
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(
                StrategyRunStat(same_uid, 3, 2, 5.0),
                StrategyRunStat(foreign_uid, 4, 0, 8.0),
            ),
            replanning_trials=(),
            replans=0,
            pending_learning=None,
        )

        filtered = scope._filter_target_scoped_learning(runtime, row)

        self.assertEqual(len(filtered.strategy_stats), 1)
        self.assertEqual(filtered.strategy_stats[0].strategy_uid, same_uid)

    def test_foreign_m7_valence_credit_is_removed_without_dropping_lower_level_credit(self) -> None:
        target_hash = int(stable_u64("target", person=b"v8-game"))
        source_hash = int(stable_u64("source", person=b"v8-game"))
        same_uid = MemoryUid(7, 3)
        foreign_uid = MemoryUid(7, 4)
        lower_uid = MemoryUid(1, 5)
        runtime = SimpleNamespace(
            read_view=_ReadView(
                {
                    same_uid: (target_hash,),
                    foreign_uid: (source_hash,),
                    lower_uid: (source_hash,),
                }
            )
        )
        same_credit = PrimaryValenceCredit(same_uid, int(MemoryLevel.M7), 0, (), 1, 1.0, 1.0, 1.0, 1.0, 0.0)
        foreign_credit = PrimaryValenceCredit(foreign_uid, int(MemoryLevel.M7), 0, (), 2, -1.0, 1.0, 1.0, 0.0, 1.0)
        lower_credit = PrimaryValenceCredit(lower_uid, int(MemoryLevel.M1), 0, (), 3, 1.0, 1.0, 1.0, 1.0, 0.0)
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(),
            primary_valence_credits=(same_credit, foreign_credit, lower_credit),
            replanning_trials=(),
            replans=0,
            pending_learning=None,
        )

        filtered = scope._filter_target_scoped_learning(runtime, row)

        self.assertEqual(
            tuple(credit.uid for credit in filtered.primary_valence_credits),
            (same_uid, lower_uid),
        )

    def test_replanning_trial_with_foreign_strategy_is_target_scoped(self) -> None:
        target_hash = int(stable_u64("target", person=b"v8-game"))
        source_hash = int(stable_u64("source", person=b"v8-game"))
        same_a = MemoryUid(7, 6)
        same_b = MemoryUid(7, 7)
        foreign = MemoryUid(7, 8)
        runtime = SimpleNamespace(
            read_view=_ReadView(
                {
                    same_a: (target_hash,),
                    same_b: (target_hash,),
                    foreign: (source_hash,),
                }
            )
        )
        kept = SimpleNamespace(
            primary_strategy_uid=same_a,
            alternative_strategy_uid=same_b,
        )
        dropped = SimpleNamespace(
            primary_strategy_uid=foreign,
            alternative_strategy_uid=same_b,
        )
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(),
            replanning_trials=(kept, dropped),
            replans=2,
            pending_learning=None,
        )

        filtered = scope._filter_target_scoped_learning(runtime, row)

        self.assertEqual(filtered.replanning_trials, (kept,))
        self.assertEqual(filtered.replans, 1)

    def test_pending_learning_is_filtered_recursively(self) -> None:
        target_hash = int(stable_u64("target", person=b"v8-game"))
        source_hash = int(stable_u64("source", person=b"v8-game"))
        foreign = MemoryUid(7, 9)
        runtime = SimpleNamespace(read_view=_ReadView({foreign: (source_hash,)}))
        pending = SimpleNamespace(
            game_id="target",
            strategy_stats=(StrategyRunStat(foreign, 1, 0, 1.0),),
            replanning_trials=(),
            replans=0,
            pending_learning=None,
        )
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(),
            replanning_trials=(),
            replans=0,
            pending_learning=pending,
        )

        filtered = scope._filter_target_scoped_learning(runtime, row)

        self.assertEqual(filtered.pending_learning.strategy_stats, ())
        self.assertEqual(target_hash, int(stable_u64("target", person=b"v8-game")))

    def test_neutral_foreign_outcome_uid_mismatch_is_inconclusive(self) -> None:
        source_hash = int(stable_u64("source", person=b"v8-game"))
        strategy_uid = MemoryUid(7, 10)
        source_outcome = MemoryUid(6, 10)
        target_outcome = MemoryUid(6, 11)
        plan = PlannedAction(2, source_outcome, strategy_uid, 1.0, False)
        view = SimpleNamespace(
            _behavior_last_action=(123, 2),
            _behavior_last_plans=(plan,),
            source_games=lambda uid: frozenset({source_hash}),
        )
        behavior._CURRENT_ACTOR_VIEW = view
        local_key = v856._failure_key("target", 123, strategy_uid)
        target_key = v856._transfer_failure_key("target", strategy_uid)
        v856._RECENT_FAILURES[local_key] = 1
        v856._TARGET_TRANSFER_FAILURES[target_key] = 1

        def inner(**kwargs):
            del kwargs
            v856._record_strategy_result("target", 123, strategy_uid, success=False, foreign=True)
            return (target_outcome, MemoryUid.zero())

        with patch.object(scope, "_BASE_OBSERVED", side_effect=inner):
            result = scope._observed_target_scope_v856(terminal_polarity=0)

        self.assertEqual(result[0], target_outcome)
        self.assertEqual(v856._RECENT_FAILURES[local_key], 1)
        self.assertEqual(v856._TARGET_TRANSFER_FAILURES[target_key], 1)

    def test_negative_foreign_outcome_retains_target_local_backoff(self) -> None:
        source_hash = int(stable_u64("source", person=b"v8-game"))
        strategy_uid = MemoryUid(7, 12)
        source_outcome = MemoryUid(6, 12)
        target_outcome = MemoryUid(6, 13)
        plan = PlannedAction(2, source_outcome, strategy_uid, 1.0, False)
        view = SimpleNamespace(
            _behavior_last_action=(123, 2),
            _behavior_last_plans=(plan,),
            source_games=lambda uid: frozenset({source_hash}),
        )
        behavior._CURRENT_ACTOR_VIEW = view
        target_key = v856._transfer_failure_key("target", strategy_uid)

        def inner(**kwargs):
            del kwargs
            v856._record_strategy_result("target", 123, strategy_uid, success=False, foreign=True)
            return (target_outcome, MemoryUid.zero())

        with patch.object(scope, "_BASE_OBSERVED", side_effect=inner):
            scope._observed_target_scope_v856(terminal_polarity=-1)

        self.assertEqual(v856._TARGET_TRANSFER_FAILURES[target_key], 1)

    def test_same_world_failure_is_not_rewritten_by_transfer_scope(self) -> None:
        target_hash = int(stable_u64("target", person=b"v8-game"))
        strategy_uid = MemoryUid(7, 14)
        outcome_uid = MemoryUid(6, 14)
        plan = PlannedAction(2, outcome_uid, strategy_uid, 1.0, False)
        view = SimpleNamespace(
            _behavior_last_action=(123, 2),
            _behavior_last_plans=(plan,),
            source_games=lambda uid: frozenset({target_hash}),
        )
        behavior._CURRENT_ACTOR_VIEW = view
        local_key = v856._failure_key("target", 123, strategy_uid)

        def inner(**kwargs):
            del kwargs
            v856._record_strategy_result("target", 123, strategy_uid, success=False, foreign=False)
            return (MemoryUid(6, 999), MemoryUid.zero())

        with patch.object(scope, "_BASE_OBSERVED", side_effect=inner):
            scope._observed_target_scope_v856(terminal_polarity=0)

        self.assertEqual(v856._RECENT_FAILURES[local_key], 1)


if __name__ == "__main__":
    unittest.main()
