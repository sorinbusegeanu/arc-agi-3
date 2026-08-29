from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import v8
from v8 import click_transition_graph_v861 as graph
from v8 import click_transition_exploration_v860 as v860
from v8.learning_blockers_v055 import pack_action_choice
from v8.sampling_portfolio_v831 import PortfolioSampler
from v8.decision_point_sampling_v821 import Intervention


class ClickTransitionGraphV861Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = pack_action_choice(6, 2, 1)
        self.sampler = PortfolioSampler("gp03-v861", seed=0)
        self.sampler._v847_actor_id = 1
        graph._ensure_state(self.sampler)

    def test_local_cell_transition_is_recorded_from_live_grids(self) -> None:
        before = np.zeros((4, 4), dtype=np.int16)
        after = before.copy()
        after[1, 2] = 3
        self.sampler._v861_before_observation = before
        self.sampler._v861_after_observation = after
        intervention = Intervention("CLICK_SCAN", (0, 10), self.action, ())
        graph._record_transition(
            self.sampler,
            intervention,
            dict(
                before_level=0,
                before_context=10,
                action=self.action,
                after_level=0,
                after_context=11,
                changed_cells=1,
                terminal_state="",
                level_advanced=False,
            ),
        )
        self.assertIn((0, 2, 1, 0, 3), self.sampler._v861_local_cells)
        self.assertIn((0, 10, self.action, 11), self.sampler._v861_local_edges)

    def test_graph_selects_known_progress_edge(self) -> None:
        edge = graph.TransitionEdge(0, 10, self.action, 11, attempts=1, changed_cells=1, progress=True)
        self.sampler._v861_shared_edges = {edge.key(): edge}
        selected = graph._graph_action(
            self.sampler,
            level=0,
            context=10,
            actions=(self.action,),
        )
        self.assertEqual(selected, self.action)

    def test_graph_selects_edge_that_reduces_distance_to_progress(self) -> None:
        a2 = pack_action_choice(6, 3, 1)
        first = graph.TransitionEdge(0, 10, self.action, 20, attempts=1, changed_cells=1)
        second = graph.TransitionEdge(0, 20, a2, 30, attempts=1, changed_cells=1, progress=True)
        self.sampler._v861_shared_edges = {first.key(): first, second.key(): second}
        selected = graph._graph_action(
            self.sampler,
            level=0,
            context=10,
            actions=(self.action,),
        )
        self.assertEqual(selected, self.action)

    def test_local_model_reuses_productive_coordinate_across_global_contexts(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int16)
        grid[1, 2] = 4
        self.sampler._v861_current_observation = grid
        cell = graph.LocalTransition(0, 2, 1, 4, 5, attempts=2, changed_cells=1)
        self.sampler._v861_shared_cells = {cell.key(): cell}
        selected = graph._graph_action(
            self.sampler,
            level=0,
            context=999,
            actions=(self.action,),
        )
        self.assertEqual(selected, self.action)

    def test_peer_files_merge_during_same_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sampling_frontier_v847"
            root.mkdir(parents=True)
            self.sampler._v847_state_root = root
            self.sampler._v847_actor_id = 1
            peer = PortfolioSampler("gp03-v861", seed=1)
            peer._v847_state_root = root
            peer._v847_actor_id = 2
            graph._ensure_state(peer)
            edge = graph.TransitionEdge(0, 70, self.action, 71, attempts=3, changed_cells=2)
            peer._v861_local_edges[edge.key()] = edge
            peer._v861_dirty_observations = 1
            graph._save_local(peer)

            graph._refresh_shared(self.sampler)
            self.assertIn(edge.key(), self.sampler._v861_shared_edges)
            self.assertEqual(self.sampler._v861_shared_edges[edge.key()].attempts, 3)
            self.assertGreaterEqual(self.sampler._v861_peer_refreshes, 1)

    def test_terminal_failure_edge_is_not_guided(self) -> None:
        edge = graph.TransitionEdge(
            0, 10, self.action, 11,
            attempts=2,
            changed_cells=1,
            progress=False,
            terminal_failures=2,
        )
        self.sampler._v861_shared_edges = {edge.key(): edge}
        self.assertIsNone(graph._graph_action(self.sampler, level=0, context=10, actions=(self.action,)))

    def test_v860_begin_lease_clears_episode_scoped_characterization(self) -> None:
        self.sampler._v860_pending_action = self.action
        self.sampler._v860_repeat_depth = {self.action: 4}
        self.sampler._v860_seen_transitions = {(self.action, 1, 2)}
        self.sampler._v860_transition_counts = {(self.action, 1, 2): 3}
        with patch.object(v860, "_BASE_BEGIN_LEASE", return_value=None):
            v860._begin_lease_v860(self.sampler, 0)
        self.assertIsNone(self.sampler._v860_pending_action)
        self.assertEqual(self.sampler._v860_repeat_depth, {})
        self.assertEqual(self.sampler._v860_seen_transitions, set())
        self.assertEqual(self.sampler._v860_transition_counts, {})

    def test_prepare_exposes_live_grid_for_local_transition_learning(self) -> None:
        grid = np.arange(16, dtype=np.int16).reshape(4, 4)
        env = SimpleNamespace(_last_grid=grid)
        with patch.object(v860, "_BASE_PREPARE_STEP", return_value=False):
            self.assertFalse(v860._prepare_step_v860(self.sampler, env))
        self.assertIs(self.sampler._v861_env, env)
        self.assertIs(self.sampler._v861_before_observation, grid)
        self.assertIs(self.sampler._v861_current_observation, grid)

    def test_runtime_stack_installs_v861(self) -> None:
        from v8 import runtime_stack_v88 as stack
        self.assertIn("click_transition_graph_v861", Path(graph.__file__).stem)
        self.assertTrue(stack._INSTALLED)


if __name__ == "__main__":
    unittest.main()
