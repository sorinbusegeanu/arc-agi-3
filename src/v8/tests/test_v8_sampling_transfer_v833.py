from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import behavior_recovery as behavior
from v8 import decision_point_sampling_v821 as sampling
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_transfer_v833 as repair
from v8.model import MemoryUid, stable_u64


class SamplingTransferV833Tests(unittest.TestCase):
    def setUp(self):
        sampling._SAMPLERS.clear()
        portfolio._set_mode(None)

    def tearDown(self):
        sampling._SAMPLERS.clear()
        portfolio._set_mode(None)

    def test_cold_portfolio_starts_with_transfer_and_keeps_ten_percent_random(self):
        sampler = portfolio.PortfolioSampler("new-game", seed=1)
        modes = [sampler._choose_mode() for _ in range(20)]
        self.assertEqual(modes[0], "TRANSFER")
        self.assertEqual(modes.count("TRANSFER"), 4)
        self.assertEqual(modes.count("RANDOM"), 2)

    def test_random_is_a_rollout_not_one_step_probe(self):
        sampler = portfolio.PortfolioSampler("ez03", seed=2)
        sampler.begin_lease(2)
        portfolio._set_mode("RANDOM")
        first = sampler.discovery_action(
            level=0, context=10, actions=(1,), history=()
        )
        self.assertEqual(first, 1)
        self.assertTrue(sampler._v833_random_rollout)

        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=0,
            after_context=11,
            after_actions=(1,),
            history_after=(1,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertTrue(sampler._v833_random_rollout)
        second = sampler.forced_action(
            level=0, context=11, actions=(1,), history=(1,)
        )
        self.assertEqual(second, 1)
        self.assertEqual(sampler.base.current.kind, "RANDOM_WALK")

    def test_random_rollout_ends_at_level_boundary(self):
        sampler = portfolio.PortfolioSampler("ez03", seed=3)
        sampler.begin_lease(3)
        sampler._v833_random_rollout = True
        sampler.base.current = sampling.Intervention("RANDOM_WALK", (0, 10), 1, ())
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=1,
            after_context=20,
            after_actions=(1, 2),
            history_after=(1,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertFalse(sampler._v833_random_rollout)
        self.assertTrue(sampler.saw_progress)

    def test_foreign_m7_is_preferred_for_transfer(self):
        sampler = portfolio.PortfolioSampler("new-game", seed=4)
        sampler.begin_lease(4)
        uid = MemoryUid(11, 22)
        outcome = MemoryUid(33, 44)
        row = SimpleNamespace(
            action_id=2,
            outcome_uid=outcome,
            strategy_uid=uid,
            support=8,
            reliability=0.9,
            mean_cost=2.0,
            probationary=False,
        )
        node = SimpleNamespace(transfer_prior=0.8)
        foreign_game = int(stable_u64("old-game", person=b"v8-game"))
        fake = SimpleNamespace(
            _strategy_version=(2, 4),
            _strategy_fallback=(row,),
            _node_by_uid={uid: node},
            _parents={uid: set()},
            _v815_normalized_action_priors={},
        )
        fake._refresh_strategy_cache = lambda: None
        from v8.model import RelationType
        provenance_edge = SimpleNamespace(
            relation_type=int(RelationType.GAME_PROVENANCE),
            source_uid=uid,
            target_uid=MemoryUid(0, foreign_game),
            score=1.0,
        )
        fake.edge_records = lambda: (provenance_edge,)

        with patch.object(behavior, "_CURRENT_ACTOR_VIEW", fake), patch.object(
            behavior, "strategy_can_control", return_value=True
        ):
            selected = repair._cross_game_transfer_action(sampler, (1, 2, 3))
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 2)
        self.assertEqual(selected[1], "M7")

    def test_normalized_m1_is_transfer_fallback(self):
        sampler = portfolio.PortfolioSampler("new-game", seed=5)
        sampler.begin_lease(5)
        fake = SimpleNamespace(
            _strategy_version=(1,),
            _strategy_fallback=(),
            _node_by_uid={},
            _parents={},
            _v815_normalized_action_priors={3: (5, 0.9), 2: (3, 0.4)},
        )
        fake._refresh_strategy_cache = lambda: None
        fake.edge_records = lambda: ()
        with patch.object(behavior, "_CURRENT_ACTOR_VIEW", fake), patch(
            "v8.restart_memory_v815._build_restart_indexes", return_value=None
        ):
            selected = repair._cross_game_transfer_action(sampler, (1, 2, 3))
        self.assertEqual(selected[:2], (3, "NORMALIZED_M1"))


if __name__ == "__main__":
    unittest.main()
