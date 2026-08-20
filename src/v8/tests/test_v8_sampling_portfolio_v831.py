from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import v8
from v8 import decision_point_sampling_v821 as sampling
from v8 import sampling_evidence_frontier_v847 as v847
from v8 import sampling_portfolio_v831 as repair
from v8 import sampling_progress_control_v829 as v829
from v8 import sampling_transfer_v833 as v833


class SequencePortfolioTests(unittest.TestCase):
    def setUp(self):
        sampling._SAMPLERS.clear()
        v829._reset_sampling_state_v829()
        v829._CONTROL_STATE.game_id = "ez02"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = None
        v829._CONTROL_STATE.selection_source = "UNKNOWN"
        v829._CONTROL_STATE.planned_actions = frozenset()
        repair._set_mode(None)

    def tearDown(self):
        sampling._SAMPLERS.clear()
        repair._set_mode(None)
        for name in ("game_id", "level", "context", "selection_source", "planned_actions"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass

    def test_historical_bounded_sequence_builder_contains_repeated_two_action_paths(self):
        rows = repair._build_sequences((1, 2, 3, 4))
        self.assertEqual(rows[:4], ((1,), (2,), (3,), (4,)))
        self.assertIn((1, 1), rows)
        self.assertIn((1, 2), rows)
        self.assertIn((4, 4), rows)
        self.assertLessEqual(len(rows), repair._MAX_SEQUENCE_CANDIDATES)

    def test_same_context_noop_remains_expandable_without_fixed_depth(self):
        sampler = repair.PortfolioSampler("ez02", seed=1)
        root = v847._register_current_v847(
            sampler,
            level=0,
            context=10,
            actions=(1,),
            history=(),
        )
        child = v847._record_expansion_v847(
            sampler,
            source_node_id=root.node_id,
            action=1,
            before_level=0,
            before_context=10,
            after_level=0,
            after_context=10,
            after_actions=(1,),
            history_after=(1,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertTrue(child.latent)
        selected = v847._best_expansion_v847(sampler)
        self.assertIs(selected[0], child)
        self.assertEqual(selected[1], 1)

    def test_random_exploration_floor_is_ten_percent_in_both_phases(self):
        sampler = repair.PortfolioSampler("ez02", seed=2)
        cold = [sampler._choose_mode() for _ in range(20)]
        self.assertEqual(cold.count("RANDOM"), 2)
        sampler.decision_count = 0
        sampler.saw_progress = True
        warm = [sampler._choose_mode() for _ in range(20)]
        self.assertEqual(warm.count("RANDOM"), 2)

    def test_random_mode_bypasses_memory_planner(self):
        repair._set_mode("RANDOM")
        sentinel = (object(),)
        with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=sentinel)) as base:
            self.assertEqual(repair._plan_chain_v831(object(), 10, (1, 2)), ())
        base.assert_not_called()

        repair._set_mode("MEMORY")
        with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=sentinel)) as base:
            self.assertIs(repair._plan_chain_v831(object(), 10, (1, 2)), sentinel)
        base.assert_called_once()

    def test_random_mode_allows_action_repetition_after_prior_attempt(self):
        sampler = repair.PortfolioSampler("ez02", seed=3)
        sampler.begin_lease(3)
        point = sampler.base.register_point(
            level=0,
            context=10,
            anchor=(),
            actions=(1,),
        )
        point.tested_actions.add(1)
        repair._set_mode("RANDOM")
        action = sampler.discovery_action(
            level=0,
            context=10,
            actions=(1,),
            history=(),
        )
        self.assertEqual(action, 1)
        self.assertEqual(sampler.base.current.kind, "RANDOM")

    def test_install_reuses_v821_actor_and_v847_wraps_v831_sampler(self):
        self.assertIs(v847._BASE_DISCOVERY_ACTION, v833._discovery_action_v833)
        self.assertEqual(v833._BASE_DISCOVERY_ACTION.__module__, repair.__name__)
        self.assertIs(sampling._sampler_for, v847._sampler_for_v847)
        self.assertIs(v829._BASE_DISCOVERY_ACTOR, sampling._decision_actor_worker)
        self.assertIs(v829._BASE_PLAN_CHAIN, repair._plan_chain_v831)

    def test_sampler_cache_is_portfolio_scoped_by_actor_and_game(self):
        first = repair._sampler_for_v831(SimpleNamespace(actor_id=1, game_id="ez02", seed=7))
        second = repair._sampler_for_v831(SimpleNamespace(actor_id=1, game_id="ez02", seed=8))
        other = repair._sampler_for_v831(SimpleNamespace(actor_id=1, game_id="ez03", seed=7))
        self.assertIs(first, second)
        self.assertIsNot(first, other)


if __name__ == "__main__":
    unittest.main()
