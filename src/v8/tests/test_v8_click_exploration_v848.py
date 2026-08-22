from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

import v8
from v8 import click_exploration_v848 as v848
from v8 import sampling_evidence_frontier_v847 as frontier
from v8 import sampling_portfolio_v831 as portfolio
from v8.adaptive_learning_allocation_v819 import AdaptiveLearningCoordinator
from v8.learning_blockers_v055 import pack_action_choice, unpack_action_choice
from v8.model import MemoryUid
from v8.publication import ActionScore, PlannedAction
from v8.sampling_portfolio_v831 import PortfolioSampler


class CompleteClickCoverageTests(unittest.TestCase):
    def test_bounded_pages_cover_every_grid_coordinate(self):
        grid = np.zeros((10, 10), dtype=np.int64)
        pages = v848._coverage_page_count(grid)
        self.assertEqual(pages, 2)

        tokens = {
            token
            for page in range(pages)
            for token in v848.exact_click_coverage_page(grid, page)
        }
        coords = {
            (payload["x"], payload["y"])
            for token in tokens
            for action, payload in (unpack_action_choice(token),)
            if action == 6 and payload is not None
        }
        self.assertEqual(len(tokens), 100)
        self.assertEqual(coords, {(x, y) for y in range(10) for x in range(10)})
        self.assertLessEqual(
            max(len(v848.exact_click_coverage_page(grid, page)) for page in range(pages)),
            64,
        )

    def test_coverage_includes_background_and_large_component_interior(self):
        grid = np.zeros((12, 12), dtype=np.int64)
        grid[2:10, 2:10] = 7
        target = pack_action_choice(6, 5, 5)
        background = pack_action_choice(6, 11, 11)
        exposed = {
            token
            for page in range(v848._coverage_page_count(grid))
            for token in v848.exact_click_coverage_page(grid, page)
        }
        self.assertIn(target, exposed)
        self.assertIn(background, exposed)

    def test_real_click_games_expose_one_center_per_selected_cell_color(self):
        from collections import Counter

        from v7.environment.arc_adapter import ArcGridEnvironment
        from v8.action_targeting_v810 import native_action_id

        expected = {
            ("gp01", 0): (2, 4),
            ("gp01", 1): (5, 60),
            ("gp02", 0): (2, 4),
            ("gp02", 1): (11, 60),
        }
        for (game_id, seed), (color, count) in expected.items():
            env = ArcGridEnvironment(game_id=game_id, seed=seed)
            try:
                frame = env.observe()
                clicks = tuple(
                    action
                    for action in env.available_actions()
                    if native_action_id(action) == 6 and v848._is_exact_click_token(action)
                )
                decoded = [unpack_action_choice(action)[1] for action in clicks]
                observed = Counter(int(frame[row["y"], row["x"]]) for row in decoded)
                self.assertEqual(len(clicks), count)
                self.assertEqual(observed, Counter({color: count}))
                self.assertTrue(
                    all(row["x"] % 8 == 4 and row["y"] % 8 == 4 for row in decoded)
                )
            finally:
                close = getattr(getattr(env, "env", None), "close", None)
                if callable(close):
                    close()


class TargetSpecificSelectionTests(unittest.TestCase):
    def test_distinct_click_targets_survive_score_grouping(self):
        click_a = pack_action_choice(6, 1, 1)
        click_b = pack_action_choice(6, 2, 2)
        rows = (
            ActionScore(click_a, 5, 0.8, 1),
            ActionScore(click_b, 0, 0.0, 0),
            ActionScore(1, 3, 0.4, 1),
        )
        grouped = v848._one_score_per_target_v848(rows)
        self.assertEqual({row.action_id for row in grouped}, {1, click_a, click_b})

    def test_one_positive_click_does_not_hide_unseen_click_target(self):
        click_a = pack_action_choice(6, 1, 1)
        click_b = pack_action_choice(6, 2, 2)
        rows = (
            ActionScore(click_a, 5, 0.8, 1),
            ActionScore(click_b, 0, 0.0, 0),
            ActionScore(1, 0, 0.0, 0),
        )
        stabilized = v848._prefer_persisted_scores_v848(rows)
        by_action = {row.action_id: row for row in stabilized}
        self.assertEqual(by_action[click_b].support_count, 0)
        self.assertEqual(by_action[1].support_count, 1)

    def test_distinct_click_targets_survive_plan_grouping(self):
        click_a = pack_action_choice(6, 1, 1)
        click_b = pack_action_choice(6, 2, 2)
        zero = MemoryUid.zero()
        rows = (
            PlannedAction(click_a, zero, MemoryUid(1, 1), 0.8),
            PlannedAction(click_b, zero, MemoryUid(1, 2), 0.7),
        )
        grouped = v848._one_plan_per_target_v848(rows)
        self.assertEqual({row.action_id for row in grouped}, {click_a, click_b})


class ClickNoopFrontierTests(unittest.TestCase):
    def test_click_capable_sequence_selects_click_before_movement_noops(self):
        sampler = PortfolioSampler("click-fixture", seed=1)
        sampler.begin_lease(1)
        click_a = pack_action_choice(6, 3, 3)
        click_b = pack_action_choice(6, 4, 4)

        portfolio._set_mode("SEQUENCE")
        try:
            action = sampler.discovery_action(
                level=0,
                context=101,
                actions=(1, 2, 3, 4, click_b, click_a),
                history=(),
            )
        finally:
            portfolio._set_mode(None)

        self.assertEqual(action, click_a)
        self.assertEqual(sampler.base.current.kind, "SEQUENCE")

    def test_movement_only_sequence_keeps_environment_neutral_ordering(self):
        sampler = PortfolioSampler("movement-fixture", seed=1)
        sampler.begin_lease(1)

        portfolio._set_mode("SEQUENCE")
        try:
            action = sampler.discovery_action(
                level=0,
                context=202,
                actions=(4, 2, 3, 1),
                history=(),
            )
        finally:
            portfolio._set_mode(None)

        self.assertEqual(action, 1)

    def test_click_noop_does_not_create_latent_frontier(self):
        sampler = PortfolioSampler("click-fixture", seed=1)
        click = pack_action_choice(6, 3, 3)
        source = frontier._register_current_v847(
            sampler,
            level=1,
            context=101,
            actions=(click,),
            history=(),
        )
        destination = frontier._record_expansion_v847(
            sampler,
            source_node_id=source.node_id,
            action=click,
            before_level=1,
            before_context=101,
            after_level=1,
            after_context=101,
            after_actions=(click,),
            history_after=(click,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertIsNone(destination)
        self.assertFalse(any(node.latent for node in frontier._ensure_state_v847(sampler).values()))
        self.assertEqual(source.failures, 1)

    def test_nonclick_noop_keeps_v847_hidden_state_behavior(self):
        sampler = PortfolioSampler("movement-fixture", seed=1)
        source = frontier._register_current_v847(
            sampler,
            level=1,
            context=202,
            actions=(1,),
            history=(),
        )
        destination = frontier._record_expansion_v847(
            sampler,
            source_node_id=source.node_id,
            action=1,
            before_level=1,
            before_context=202,
            after_level=1,
            after_context=202,
            after_actions=(1,),
            history_after=(1,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertIsNotNone(destination)
        self.assertTrue(destination.latent)

    def test_restored_click_latent_is_not_selected_for_expansion(self):
        sampler = PortfolioSampler("click-migration", seed=1)
        click = pack_action_choice(6, 4, 4)
        latent = frontier.EvidencePrefixNode(
            node_id=frontier._latent_id(1, 303, (click,)),
            level=1,
            context=303,
            anchor=(click,),
            available_actions={click},
            latent=True,
            novel=True,
        )
        frontier._upsert_node_v847(sampler, latent)
        self.assertIsNone(frontier._best_expansion_v847(sampler))


class ClickAwareAllocationTests(unittest.TestCase):
    def test_unsolved_click_game_gets_branching_factor_weight(self):
        measurements = {
            "move": (False, 4),
            "click": (True, 100),
        }
        with patch.object(v848, "_probe_game_action_space", side_effect=lambda game: measurements[game]):
            coordinator = AdaptiveLearningCoordinator()
            coordinator.register_games(("move", "click"))
        self.assertAlmostEqual(coordinator.sampling_weight("move"), 1.0)
        self.assertAlmostEqual(coordinator.sampling_weight("click"), math.sqrt(25.0))

    def test_runtime_stack_installs_v848_last(self):
        from v8 import runtime_stack_v88

        self.assertEqual(runtime_stack_v88._LAYERS[-1], "click_exploration_v848")


if __name__ == "__main__":
    unittest.main()
