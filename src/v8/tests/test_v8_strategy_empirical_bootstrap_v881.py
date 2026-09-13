from __future__ import annotations

import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

from v8 import actor as actor_module
from v8.actor_read_view_v851 import ActorReadView
from v8.model import MemoryUid
from v8.publication import LiveReadView
from v8 import strategy_empirical_bootstrap_v881 as bootstrap
from v8 import adaptive_learning_allocation_v819 as v819
from v8.strategy_empirical_bootstrap_v881 import (
    _accumulate_generic_strategy_stat,
    _missing_bootstrap_attempts,
)


class StrategyEmpiricalBootstrapTests(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            del bootstrap._BOOTSTRAP_SCOPE.source_game_hash
        except AttributeError:
            pass

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

    def test_actor_wrapper_preserves_existing_plan_before_bootstrap(self) -> None:
        planned = object()
        with (
            patch.object(
                bootstrap,
                "_BASE_ACTOR_PLAN_CANDIDATES",
                return_value=(planned,),
            ),
            patch.object(bootstrap, "_bootstrap_plan") as fallback,
        ):
            self.assertEqual(
                bootstrap._actor_plan_candidates_v881(object(), 10, (1, 2)),
                (planned,),
            )
        fallback.assert_not_called()

    def test_bootstrap_is_disabled_during_non_discovery_interventions(self) -> None:
        prior = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            for mode in (
                v819.SamplingMode.VERIFY,
                v819.SamplingMode.ALTERNATIVE,
                v819.SamplingMode.TRANSFER,
            ):
                os.environ[v819._SAMPLING_MODE_ENV] = mode.value
                view = SimpleNamespace(_behavior_actor_mode=True)
                self.assertIsNone(bootstrap._bootstrap_plan(view, 10, (1, 2)))
        finally:
            if prior is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior

    def test_bootstrap_provenance_must_include_current_world(self) -> None:
        strategy = MemoryUid(7, 1)
        bootstrap._BOOTSTRAP_SCOPE.source_game_hash = 101
        matching = SimpleNamespace(source_games=lambda uid: frozenset({101, 202}))
        foreign = SimpleNamespace(source_games=lambda uid: frozenset({202}))
        unavailable = SimpleNamespace()

        self.assertTrue(bootstrap._strategy_matches_current_world(matching, strategy))
        self.assertFalse(bootstrap._strategy_matches_current_world(foreign, strategy))
        self.assertFalse(bootstrap._strategy_matches_current_world(unavailable, strategy))

    def test_foreign_strategy_is_rejected_before_bootstrap_scoring(self) -> None:
        context = 41
        strategy = MemoryUid(7, 1)
        outcome = MemoryUid(6, 1)
        item = SimpleNamespace(
            strategy_uid=strategy,
            outcome_uid=outcome,
            action_id=2,
        )
        view = SimpleNamespace(
            _behavior_actor_mode=True,
            _refresh_strategy_cache=lambda: None,
            _strategy_by_context={
                actor_module.stable_u64(context, person=b"v8-context"): (item,)
            },
            _node_by_uid={strategy: SimpleNamespace(attempt_weight=0.0)},
            source_games=lambda uid: frozenset({202}),
        )
        bootstrap._BOOTSTRAP_SCOPE.source_game_hash = 101
        with (
            patch.object(bootstrap, "_strategy_can_probe", return_value=True),
            patch.object(bootstrap, "_score_strategy_rows") as score,
        ):
            self.assertIsNone(bootstrap._bootstrap_plan(view, context, (2,)))
        score.assert_not_called()

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
