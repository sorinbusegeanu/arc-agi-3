import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v5_0.contact.policy import dedupe_contact_trajectories
from v5_0.contact.service import run_controlled_contact_for_episode, run_controlled_contact_multi_reset
from v5_0.contracts.avatar_types import ContactPolicy
from v5_0.runtime.run_avatar_bootstrap import (
    run_frontier_level_from_live_session,
    run_full_campaign_analysis,
)


class TestContactDedup(unittest.TestCase):
    def _policy(self, policy_id, actions):
        return ContactPolicy(
            policy_id=policy_id,
            poi_id="p1",
            episode_index=0,
            planned_actions=tuple(actions),
            max_steps=8,
            stop_on_contact=True,
            stop_on_screen_change=True,
            stop_on_terminal=True,
        )

    def test_duplicate_action_sequence_collapses_to_one(self):
        p1 = self._policy("a", ("RIGHT", "RIGHT"))
        p2 = self._policy("b", ("RIGHT", "RIGHT"))
        out = dedupe_contact_trajectories((p1, p2))
        self.assertEqual(len(out), 1)

    def test_same_displacement_and_first_move_keeps_shortest(self):
        short = self._policy("short", ("RIGHT", "DOWN"))
        long = self._policy("long", ("RIGHT", "DOWN", "STAY"))
        out = dedupe_contact_trajectories((long, short))
        self.assertEqual(len(out), 2)

    @patch("v5_0.contact.service.classify_contact_outcome")
    @patch("v5_0.contact.service.run_contact_policy")
    @patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi")
    def test_per_episode_logs_preserve_generated_and_attempted_trajectory_evidence(self, build_traj, run_policy, classify_outcome):
        build_traj.return_value = (self._policy("a", ("RIGHT",)), self._policy("b", ("RIGHT",)))
        classify_outcome.return_value = SimpleNamespace(
            outcome_type="no_effect",
            confidence=0.5,
            contact_step_index=None,
            level_transition=False,
            terminal=False,
            reward_change_step_indices=tuple(),
            object_removed=False,
            new_object_appeared=False,
            hud_change_only=False,
        )
        run_policy.return_value = SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=self._policy("a", ("RIGHT",)),
            steps=(
                SimpleNamespace(
                    blocked_action=False,
                    invalid_action=False,
                    screen_changed=False,
                    avatar_bbox_before=(0, 0, 0, 0),
                    avatar_bbox_after=(1, 0, 1, 0),
                    poi_bbox_before=(2, 0, 2, 0),
                    poi_bbox_after=(2, 0, 2, 0),
                ),
            ),
            outcome=SimpleNamespace(),
            initial_poi_bbox=(1, 1, 1, 1),
            final_poi_bbox=(1, 1, 1, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(0, 0, 0, 0),
        )
        selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0))
        poi1 = SimpleNamespace(poi_id="p1", confidence=0.9, bbox=(1, 1, 1, 1), area=4, ambiguity_flags=())
        poi2 = SimpleNamespace(poi_id="p2", confidence=0.8, bbox=(4, 4, 4, 4), area=4, ambiguity_flags=())
        probe_episode = SimpleNamespace(episode_index=0, transitions=tuple())
        report = SimpleNamespace(candidates=(poi1, poi2))
        result = run_controlled_contact_for_episode(
            probe_episode=probe_episode,
            poi_report=report,
            selected_avatar=selected_avatar,
            plan=SimpleNamespace(),
            seed=0,
            render_terminal=False,
            env_factory=None,
            max_pois_to_test=2,
        )
        self.assertGreaterEqual(run_policy.call_count, 1)
        self.assertEqual(build_traj.call_count, 2)
        self.assertTrue(result)
        route_evidence = getattr(result[0], "route_evidence", None)
        self.assertIsInstance(route_evidence, dict)
        self.assertIn("attempted_route_ids", route_evidence)
        self.assertIn("generated_trajectories", route_evidence)
        self.assertIn("attempted_trajectories", route_evidence)

    @patch("v5_0.contact.service.classify_contact_outcome")
    @patch("v5_0.contact.service.run_contact_policy")
    @patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi")
    def test_max_pois_to_test_honored_exactly(self, build_traj, run_policy, classify_outcome):
        build_traj.return_value = (self._policy("a", ("RIGHT",)),)
        classify_outcome.return_value = SimpleNamespace(
            outcome_type="no_effect",
            confidence=0.1,
            contact_step_index=None,
            level_transition=False,
            terminal=False,
            reward_change_step_indices=tuple(),
            object_removed=False,
            new_object_appeared=False,
            hud_change_only=False,
        )
        run_policy.return_value = SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=self._policy("a", ("RIGHT",)),
            steps=tuple(),
            outcome=SimpleNamespace(),
            initial_poi_bbox=(1, 1, 1, 1),
            final_poi_bbox=(1, 1, 1, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(0, 0, 0, 0),
        )
        selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0))
        pois = tuple(
            SimpleNamespace(poi_id=f"p{i}", confidence=1.0 - i * 0.1, bbox=(i, 1, i, 1), area=4, ambiguity_flags=())
            for i in range(4)
        )
        probe_episode = SimpleNamespace(episode_index=0, transitions=tuple())
        report = SimpleNamespace(candidates=pois)
        run_controlled_contact_for_episode(
            probe_episode=probe_episode,
            poi_report=report,
            selected_avatar=selected_avatar,
            plan=SimpleNamespace(),
            seed=0,
            render_terminal=False,
            env_factory=None,
            max_pois_to_test=2,
        )
        self.assertEqual(build_traj.call_count, 2)
    @patch("v5_0.contact.service.classify_contact_outcome")
    @patch("v5_0.contact.service.run_contact_policy")
    @patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi")
    def test_multi_reset_honors_max_pois_to_test_one(self, build_traj, run_policy, classify_outcome):
        build_traj.return_value = (self._policy("a", ("RIGHT",)), self._policy("b", ("DOWN",)))
        classify_outcome.return_value = SimpleNamespace(
            outcome_type="no_effect",
            confidence=0.1,
            contact_step_index=None,
            level_transition=False,
            terminal=False,
            reward_change_step_indices=tuple(),
            object_removed=False,
            new_object_appeared=False,
            hud_change_only=False,
        )
        run_policy.return_value = SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=self._policy("a", ("RIGHT",)),
            steps=tuple(),
            outcome=SimpleNamespace(),
            initial_poi_bbox=(1, 1, 1, 1),
            final_poi_bbox=(1, 1, 1, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(0, 0, 0, 0),
        )
        selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0), selected_bbox=(0, 0, 0, 0))
        pois = tuple(
            SimpleNamespace(poi_id=f"p{i}", confidence=1.0 - i * 0.1, bbox=(i, 1, i, 1), area=4, ambiguity_flags=())
            for i in range(3)
        )
        avatar_multi_report = SimpleNamespace(
            episodes=(SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=selected_avatar), transitions=tuple()),),
            selected=selected_avatar,
            diagnostics=SimpleNamespace(stable_avatar_found=True),
        )
        poi_multi_bundle = {"episodes": (SimpleNamespace(episode_index=0, poi_report=SimpleNamespace(candidates=pois), transitions=tuple()),)}
        result = run_controlled_contact_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_multi_bundle,
            plan=SimpleNamespace(level_id="L0"),
            base_seed=0,
            render_terminal=False,
            env_factory=None,
            max_pois_to_test=1,
        )
        self.assertEqual(build_traj.call_count, 1)
        self.assertEqual(len(result.episodes[0].tested_pois), 1)

    @patch("v5_0.contact.service.classify_contact_outcome")
    @patch("v5_0.contact.service.run_contact_policy")
    @patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi")
    def test_multi_reset_honors_max_pois_to_test_two(self, build_traj, run_policy, classify_outcome):
        build_traj.return_value = (self._policy("a", ("RIGHT",)), self._policy("b", ("DOWN",)))
        classify_outcome.return_value = SimpleNamespace(
            outcome_type="no_effect",
            confidence=0.1,
            contact_step_index=None,
            level_transition=False,
            terminal=False,
            reward_change_step_indices=tuple(),
            object_removed=False,
            new_object_appeared=False,
            hud_change_only=False,
        )
        run_policy.return_value = SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=self._policy("a", ("RIGHT",)),
            steps=tuple(),
            outcome=SimpleNamespace(),
            initial_poi_bbox=(1, 1, 1, 1),
            final_poi_bbox=(1, 1, 1, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(0, 0, 0, 0),
        )
        selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0), selected_bbox=(0, 0, 0, 0))
        pois = tuple(
            SimpleNamespace(poi_id=f"p{i}", confidence=1.0 - i * 0.1, bbox=(i, 1, i, 1), area=4, ambiguity_flags=())
            for i in range(3)
        )
        avatar_multi_report = SimpleNamespace(
            episodes=(SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=selected_avatar), transitions=tuple()),),
            selected=selected_avatar,
            diagnostics=SimpleNamespace(stable_avatar_found=True),
        )
        poi_multi_bundle = {"episodes": (SimpleNamespace(episode_index=0, poi_report=SimpleNamespace(candidates=pois), transitions=tuple()),)}
        result = run_controlled_contact_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_multi_bundle,
            plan=SimpleNamespace(level_id="L0"),
            base_seed=0,
            render_terminal=False,
            env_factory=None,
            max_pois_to_test=2,
        )
        self.assertEqual(build_traj.call_count, 2)
        self.assertEqual(len(result.episodes[0].tested_pois), 2)

    @patch("v5_0.contact.service.classify_contact_outcome")
    @patch("v5_0.contact.service.run_contact_policy")
    @patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi")
    def test_multi_reset_prefers_cross_reset_candidates_over_episode_local_pois(self, build_traj, run_policy, classify_outcome):
        build_traj.return_value = (self._policy("a", ("RIGHT",)),)
        classify_outcome.return_value = SimpleNamespace(
            outcome_type="no_effect",
            confidence=0.1,
            contact_step_index=None,
            level_transition=False,
            terminal=False,
            reward_change_step_indices=tuple(),
            object_removed=False,
            new_object_appeared=False,
            hud_change_only=False,
        )
        run_policy.return_value = SimpleNamespace(
            poi_id="cross_poi_000",
            episode_index=0,
            policy=self._policy("a", ("RIGHT",)),
            steps=tuple(),
            outcome=SimpleNamespace(),
            initial_poi_bbox=(1, 1, 1, 1),
            final_poi_bbox=(1, 1, 1, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(0, 0, 0, 0),
        )
        selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0), selected_bbox=(0, 0, 0, 0))
        local_poi = SimpleNamespace(poi_id="poi_003", confidence=0.95, bbox=(1, 1, 1, 1), area=4, ambiguity_flags=())
        cross_a = SimpleNamespace(poi_id="cross_poi_000", confidence=0.9, bbox=(2, 1, 2, 1), area=4, ambiguity_flags=())
        cross_b = SimpleNamespace(poi_id="cross_poi_001", confidence=0.8, bbox=(3, 1, 3, 1), area=4, ambiguity_flags=())
        avatar_multi_report = SimpleNamespace(
            episodes=(SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=selected_avatar), transitions=tuple()),),
            selected=selected_avatar,
            diagnostics=SimpleNamespace(stable_avatar_found=True),
        )
        poi_multi_bundle = {
            "report": SimpleNamespace(candidates=(cross_a, cross_b)),
            "episodes": (SimpleNamespace(episode_index=0, poi_report=SimpleNamespace(candidates=(local_poi,)), transitions=tuple()),),
        }
        result = run_controlled_contact_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_multi_bundle,
            plan=SimpleNamespace(level_id="L0"),
            base_seed=0,
            render_terminal=False,
            env_factory=None,
            max_pois_to_test=2,
        )
        tested_ids = tuple(item.poi_id for item in result.episodes[0].tested_pois)
        self.assertEqual(tested_ids, ("cross_poi_000", "cross_poi_000"))
        called_poi_ids = tuple(call.kwargs["poi_candidate"].poi_id for call in build_traj.call_args_list)
        self.assertEqual(called_poi_ids, ("cross_poi_000", "cross_poi_001"))



    @patch("v5_0.runtime.run_avatar_bootstrap.run_adaptive_solve_on_live_session", return_value=SimpleNamespace(episodes=tuple(), solved=False, failure_reason="no_progress"))
    @patch("v5_0.runtime.run_avatar_bootstrap.extract_verified_frontier_trace", return_value=None)
    @patch("v5_0.runtime.run_avatar_bootstrap.build_level_solution_from_adaptive_report")
    @patch("v5_0.runtime.run_avatar_bootstrap.write_level_solution_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_adaptive_solve_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_hud_hint_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_hud_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_poi_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_multi_reset_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap._build_multi_reset_avatar_report")
    @patch("v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier")
    @patch("v5_0.runtime.run_avatar_bootstrap.discover_pois_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.detect_hud_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.interpret_hud_hints_multi_reset")
    def test_contact_skipped_when_skip_gate_true(self, interp, detect_hud, discover_pois, run_probe, build_multi, *_rest):
        build_multi.return_value = SimpleNamespace(
            selected=SimpleNamespace(failure_reason=None),
            diagnostics=SimpleNamespace(successful_episode_count=2, cross_reset_ambiguous=False),
        )
        discover_pois.return_value = {"report": SimpleNamespace(candidates=(SimpleNamespace(poi_id="p1", confidence=0.9, bbox=(1, 1, 1, 1), area=4),)), "cross_reset_evidence": tuple(), "episodes": tuple()}
        detect_hud.return_value = {"report": SimpleNamespace(failure_reason=None), "cross_reset_evidence": tuple(), "episodes": tuple(), "value_samples": {}}
        interp.return_value = SimpleNamespace(selected=SimpleNamespace(selected_poi_id="p1", ambiguous=False, failure_reason=None, ranked_poi_ids=("p1",)))
        run_probe.return_value = ((),)
        with patch("v5_0.runtime.run_avatar_bootstrap._should_skip_redundant_frontier_analysis", return_value=True), patch(
            "v5_0.runtime.run_avatar_bootstrap.run_controlled_contact_multi_reset"
        ) as contact_mock:
            run_frontier_level_from_live_session(game_id="ez01", frontier_level_id="L1", session=object(), prefix_traces=tuple(), episode_count=1)
        contact_mock.assert_not_called()

    def test_frontier_bootstrap_uses_one_episode_default(self):
        self.assertEqual(inspect.signature(run_frontier_level_from_live_session).parameters["episode_count"].default, 1)
        self.assertEqual(inspect.signature(run_full_campaign_analysis).parameters["episode_count"].default, 1)


if __name__ == "__main__":
    unittest.main()
