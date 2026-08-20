from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import decision_point_sampling_v821 as sampling
from v8 import sampling_evidence_frontier_v847 as v847
from v8 import sampling_portfolio_v831 as portfolio


class EvidencePrefixFrontierV847Tests(unittest.TestCase):
    def setUp(self) -> None:
        sampling._SAMPLERS.clear()
        self._prior_root = os.environ.pop(v847._TRAJECTORY_ROOT_ENV, None)
        portfolio._set_mode(None)

    def tearDown(self) -> None:
        sampling._SAMPLERS.clear()
        portfolio._set_mode(None)
        if self._prior_root is None:
            os.environ.pop(v847._TRAJECTORY_ROOT_ENV, None)
        else:
            os.environ[v847._TRAJECTORY_ROOT_ENV] = self._prior_root

    def test_final_runtime_uses_v847_sampler_and_lazy_sequence_authority(self) -> None:
        self.assertIs(sampling._sampler_for, v847._sampler_for_v847)
        self.assertIs(portfolio.PortfolioSampler.prepare_step, v847._prepare_step_v847)
        self.assertIs(portfolio.PortfolioSampler.discovery_action, v847._discovery_action_v847)
        self.assertIs(portfolio.PortfolioSampler.observe_transition, v847._observe_transition_v847)
        self.assertIs(
            portfolio.PortfolioSampler._schedule_next_sequence,
            v847._schedule_next_sequence_v847,
        )

    def test_sequence_discovery_never_calls_bounded_product_generator(self) -> None:
        sampler = portfolio.PortfolioSampler("g", seed=1)
        sampler.begin_lease(1)
        portfolio._set_mode("SEQUENCE")
        with patch.object(
            portfolio,
            "_build_sequences",
            side_effect=AssertionError("bounded product tree must not be used"),
        ):
            action = sampler.discovery_action(
                level=0,
                context=10,
                actions=(1, 2),
                history=(),
            )
        self.assertEqual(action, 1)
        self.assertEqual(sampler.base.current.kind, "SEQUENCE")
        self.assertEqual(sampler._v847_active_expansion[1], 1)

    def test_noop_prefix_can_grow_beyond_old_depth_without_internal_horizon(self) -> None:
        sampler = portfolio.PortfolioSampler("g", seed=2)
        node = v847._register_current_v847(
            sampler,
            level=0,
            context=10,
            actions=(1,),
            history=(),
        )
        for depth in range(1, 9):
            node = v847._record_expansion_v847(
                sampler,
                source_node_id=node.node_id,
                action=1,
                before_level=0,
                before_context=10,
                after_level=0,
                after_context=10,
                after_actions=(1,),
                history_after=(1,) * depth,
                changed_cells=0,
                terminal_state="NOT_FINISHED",
                level_advanced=False,
                prediction_error=0.0,
                future_delta=0.0,
            )
            self.assertIsNotNone(node)
            self.assertTrue(node.latent)

        self.assertEqual(len(node.anchor), 8)
        selected = v847._best_expansion_v847(sampler)
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected[0].anchor), 8)
        telemetry = v847.frontier_telemetry_v847("g")
        self.assertIsNone(telemetry["fixed_depth_limit"])
        self.assertIsNone(telemetry["fixed_candidate_limit"])

    def test_changed_semantic_state_uses_transposition_and_keeps_cheapest_anchor(self) -> None:
        sampler = portfolio.PortfolioSampler("g", seed=3)
        first = v847._register_current_v847(
            sampler,
            level=0,
            context=10,
            actions=(1,),
            history=(9, 9),
        )
        destination = v847._record_expansion_v847(
            sampler,
            source_node_id=first.node_id,
            action=1,
            before_level=0,
            before_context=10,
            after_level=0,
            after_context=20,
            after_actions=(2,),
            history_after=(9, 9, 1),
            changed_cells=3,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.2,
            future_delta=0.0,
        )
        self.assertEqual(destination.node_id, "C:0:20")

        second = v847._register_current_v847(
            sampler,
            level=0,
            context=11,
            actions=(2,),
            history=(),
        )
        same = v847._record_expansion_v847(
            sampler,
            source_node_id=second.node_id,
            action=2,
            before_level=0,
            before_context=11,
            after_level=0,
            after_context=20,
            after_actions=(2, 3),
            history_after=(2,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.7,
            future_delta=1.0,
        )

        self.assertIs(same, destination)
        self.assertEqual(same.anchor, (2,))
        self.assertEqual(same.available_actions, {2, 3})
        self.assertEqual(same.prediction_error, 0.7)
        self.assertEqual(
            sum(1 for key in sampler._v847_nodes if key == "C:0:20"),
            1,
        )

    def test_unchanged_observation_retains_path_specific_hidden_state(self) -> None:
        sampler = portfolio.PortfolioSampler("g", seed=4)
        root = v847._register_current_v847(
            sampler,
            level=0,
            context=10,
            actions=(1, 2),
            history=(),
        )
        one = v847._record_expansion_v847(
            sampler,
            source_node_id=root.node_id,
            action=1,
            before_level=0,
            before_context=10,
            after_level=0,
            after_context=10,
            after_actions=(1, 2),
            history_after=(1,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        two = v847._record_expansion_v847(
            sampler,
            source_node_id=root.node_id,
            action=2,
            before_level=0,
            before_context=10,
            after_level=0,
            after_context=10,
            after_actions=(1, 2),
            history_after=(2,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertNotEqual(one.node_id, two.node_id)
        self.assertTrue(one.latent)
        self.assertTrue(two.latent)

    def test_prediction_violation_prioritizes_frontier_without_scalar_reward(self) -> None:
        sampler = portfolio.PortfolioSampler("g", seed=5)
        low = v847.EvidencePrefixNode(
            "C:0:10",
            0,
            10,
            (),
            available_actions={1},
            prediction_error=0.1,
        )
        high = v847.EvidencePrefixNode(
            "C:0:20",
            0,
            20,
            (2, 2, 2),
            available_actions={1},
            prediction_error=0.9,
        )
        v847._upsert_node_v847(sampler, low)
        v847._upsert_node_v847(sampler, high)
        selected = v847._best_expansion_v847(sampler)
        self.assertIs(selected[0], high)

    def test_frontier_state_restores_across_actor_ids_from_existing_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[v847._TRAJECTORY_ROOT_ENV] = tmp
            first = v847._sampler_for_v847(
                SimpleNamespace(actor_id=1, game_id="g", seed=7)
            )
            node = v847._register_current_v847(
                first,
                level=2,
                context=99,
                actions=(1, 3),
                history=(4, 5, 6, 7, 8),
            )
            node.prediction_error = 0.8
            first._v847_dirty = True
            v847._save_sampler_state_v847(first)

            sampling._SAMPLERS.clear()
            second = v847._sampler_for_v847(
                SimpleNamespace(actor_id=2, game_id="g", seed=8)
            )
            restored = second._v847_nodes.get("C:2:99")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.anchor, (4, 5, 6, 7, 8))
            self.assertEqual(restored.available_actions, {1, 3})
            self.assertEqual(restored.prediction_error, 0.8)


if __name__ == "__main__":
    unittest.main()
