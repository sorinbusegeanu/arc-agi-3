import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v5_0.contact.frame_tracker import (
    find_best_component_match_in_frame,
    reacquire_avatar_bbox_in_frame,
    reacquire_poi_bbox_in_frame,
)
from v5_0.solve.service import reanchor_frontier_objects, run_adaptive_solve_on_live_session
from v5_0.runtime.campaign_state import CampaignLevelState
from v5_0.runtime.run_avatar_bootstrap import run_frontier_level_from_live_session, run_full_campaign_analysis


def _frame(w, h, cells):
    rows = [[0 for _ in range(w)] for _ in range(h)]
    for (x, y), value in cells.items():
        rows[y][x] = value
    return tuple(tuple(int(v) for v in row) for row in rows)


class TestFrontierReanchor(unittest.TestCase):
    def test_strict_reanchor_success(self):
        frame = _frame(8, 8, {(1, 1): 1, (4, 4): 2})
        avatar = SimpleNamespace(selected_bbox=(1, 1, 1, 1), value_histogram={1: 1})
        poi = SimpleNamespace(bbox=(4, 4, 4, 4), value_histogram={2: 1})
        a, p = reanchor_frontier_objects(frame=frame, selected_avatar=avatar, initial_poi=poi)
        self.assertEqual(a, (1, 1, 1, 1))
        self.assertEqual(p, (4, 4, 4, 4))

    def test_relaxed_avatar_reacquire_success_when_overlap_gone(self):
        frame = _frame(8, 8, {(4, 1): 1, (6, 6): 2})
        out = reacquire_avatar_bbox_in_frame(frame, (1, 1, 1, 1), {1: 1})
        self.assertEqual(out, (4, 1, 4, 1))

    def test_relaxed_poi_reacquire_success_when_overlap_gone(self):
        frame = _frame(10, 10, {(6, 3): 2, (1, 1): 1})
        out = reacquire_poi_bbox_in_frame(frame, (2, 3, 2, 3), {2: 1})
        self.assertEqual(out, (6, 3, 6, 3))

    def test_near_plausible_component_beats_far_unrelated_component(self):
        frame = _frame(20, 20, {(5, 5): 3, (18, 18): 3})
        out = find_best_component_match_in_frame(
            frame=frame,
            reference_bbox=(4, 4, 4, 4),
            reference_histogram={3: 1},
            for_poi=True,
        )
        self.assertEqual(out, (5, 5, 5, 5))

    def test_missing_frame_returns_none(self):
        self.assertIsNone(reacquire_avatar_bbox_in_frame(None, (1, 1, 1, 1), {1: 1}))
        self.assertIsNone(reacquire_poi_bbox_in_frame(None, (1, 1, 1, 1), {2: 1}))

    def test_reanchor_frontier_objects_relaxed_success_prevents_immediate_failure(self):
        frame = _frame(10, 10, {(2, 2): 1, (4, 4): 2})
        avatar = SimpleNamespace(selected_bbox=(2, 2, 2, 2), value_histogram={1: 1})
        # Large reference poi box makes strict poi match reject; relaxed re-acquire should still find (4,4)
        poi = SimpleNamespace(bbox=(0, 0, 8, 8), value_histogram={2: 1})
        a, p = reanchor_frontier_objects(frame=frame, selected_avatar=avatar, initial_poi=poi)
        self.assertIsNotNone(a)
        self.assertIsNotNone(p)

    def test_unsolved_adaptive_run_emits_structured_trajectory_logs(self):
        class _Session:
            environment_metadata = SimpleNamespace(coordinate_action_id=-1, coordinate_bounds=None)

        class _Obs:
            def __init__(self, frame, levels_completed=0):
                self.frame = (frame,)
                self.levels_completed = levels_completed
                self.available_actions = (0, 1, 2, 3)
                self.raw_payload = {"reward": 0.0}

        class _SessionAdapter:
            def __init__(self):
                self._frame = _frame(8, 8, {(1, 1): 1, (5, 1): 2})

            def get_current_observation(self, _session):
                return _Obs(self._frame, levels_completed=0)

            def execute_action_prefix(self, _session, _translated, _tokens):
                return SimpleNamespace(
                    terminal_status="running",
                    step_results=(SimpleNamespace(action_legal=True),),
                )

        class _ActionAdapter:
            def translate_token(self, _token, _ctx):
                return 0

        selected_avatar = SimpleNamespace(
            failure_reason=None,
            selected_bbox=(1, 1, 1, 1),
            selected_center=(1.0, 1.0),
            value_histogram={1: 1},
        )
        ranked = (
            SimpleNamespace(
                poi_id="p1",
                bbox=(5, 1, 5, 1),
                center=(5.0, 1.0),
                confidence=1.0,
                area=1,
                value_histogram={2: 1},
                ambiguity_flags=tuple(),
            ),
        )
        hud = SimpleNamespace(selected=SimpleNamespace(selected_poi_id="p1", ambiguous=False, failure_reason=None))
        report = run_adaptive_solve_on_live_session(
            session=_Session(),
            selected_avatar=selected_avatar,
            ranked_poi_candidates=ranked,
            hud_targeting_report=hud,
            contact_experiment_report=None,
            game_id="ez01",
            level_id="L0",
            max_steps=2,
            session_adapter=_SessionAdapter(),
            action_adapter=_ActionAdapter(),
        )
        self.assertTrue(hasattr(report, "generated_trajectories"))
        self.assertTrue(hasattr(report, "attempted_trajectories"))
        self.assertGreaterEqual(len(tuple(getattr(report, "generated_trajectories", ()))), 1)
        self.assertGreaterEqual(len(tuple(getattr(report, "attempted_trajectories", ()))), 1)
        self.assertFalse(report.solved)
        self.assertTrue(hasattr(report, "trajectory_stats"))
        self.assertEqual(getattr(report.trajectory_stats, "level_id", ""), "L0")

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0", "L1"))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
    @patch(
        "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
        return_value={
            "L0": CampaignLevelState("ez01", "L0", "pending", False, None, None, 0),
            "L1": CampaignLevelState("ez01", "L1", "pending", False, None, None, 0),
        },
    )
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", "L1", None, None])
    @patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier")
    @patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_continuation_from_live_session")
    @patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session")
    @patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
    @patch("v5_0.runtime.run_avatar_bootstrap.finalize_solved_level_trace")
    def test_campaign_post_success_continues_in_same_live_session(
        self,
        finalize_trace,
        _update,
        run_frontier,
        run_continue,
        replay_prefix,
        _prefix,
        _current,
        _frontier,
        _load,
        _db,
        _seq,
        _init,
        _rebuild,
        _index,
    ):
        sess = object()
        replay_prefix.return_value = {"session": sess, "frontier_reached": True, "frontier_level_id": "L0", "divergence": False}
        run_frontier.return_value = {
            "solved": True,
            "solution": {
                "game_id": "ez01",
                "level_id": "L0",
                "solved": True,
                "step_count": 1,
                "action_trace": (
                    {"step_index": 0, "action": "RIGHT", "source": "frontier_solve", "pre_level_index": 0, "post_level_index": 1},
                ),
                "terminal": False,
                "level_transition": True,
                "failure_reason": None,
            },
            "saved_trace": None,
            "diagnostics": {},
            "failure_reason": None,
            "artifact_paths": {},
            "_selected_avatar_obj": SimpleNamespace(selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
        }
        run_continue.return_value = {
            "solved": False,
            "solution": {"game_id": "ez01", "level_id": "L1", "solved": False, "step_count": 0, "action_trace": (), "terminal": False, "level_transition": False, "failure_reason": "x"},
            "saved_trace": None,
            "diagnostics": {},
            "failure_reason": "x",
            "artifact_paths": {},
        }
        from v5_0.contracts.avatar_types import SavedLevelTrace
        finalize_trace.return_value = {
            "saved_trace": SavedLevelTrace("ez01", "L0", True, ("RIGHT",), 1, None, 1, True, action_sources=("frontier_solve",), trace_id="t1"),
            "replay_verified": True,
            "failure_reason": None,
            "trace_id": "t1",
        }
        with patch("v5_0.runtime.run_avatar_bootstrap.SessionAdapter") as sa:
            adapter = sa.return_value
            adapter.get_current_observation.return_value = SimpleNamespace(levels_completed=1)
            run_full_campaign_analysis(game_id="ez01", output_dir="runs_v5_0_test")
        self.assertEqual(replay_prefix.call_count, 1)
        self.assertEqual(run_frontier.call_count, 1)
        self.assertEqual(run_continue.call_count, 1)

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state", return_value={"L0": CampaignLevelState("ez01", "L0", "pending", False, None, None, 0)})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", return_value="L0")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier", return_value=((1,),))
    @patch("v5_0.runtime.run_avatar_bootstrap._build_multi_reset_avatar_report")
    @patch("v5_0.runtime.run_avatar_bootstrap.discover_pois_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.write_poi_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.detect_hud_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.write_hud_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.interpret_hud_hints_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.write_hud_hint_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.run_controlled_contact_multi_reset")
    @patch("v5_0.runtime.run_avatar_bootstrap.run_adaptive_solve_on_live_session")
    @patch("v5_0.runtime.run_avatar_bootstrap.write_contact_experiment_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_adaptive_solve_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_generated_trajectories", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_trajectory_attempts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_trajectory_stats", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_level_solution_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.write_multi_reset_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.finalize_solved_level_trace")
    def test_frontier_contact_receives_full_candidate_set_and_logs_diagnostics(
        self,
        finalize_trace,
        _write_multi_reset,
        _write_solution,
        _write_stats,
        _write_attempts,
        _write_generated,
        _write_adaptive,
        _write_contact,
        adaptive_mock,
        contact_mock,
        interp_mock,
        _write_hud,
        hud_mock,
        _write_hud_hint,
        _write_poi,
        pois_mock,
        avatar_mock,
        _probe,
        _prefix,
        _current,
        _frontier,
        _load,
        _db,
        _seq,
        _init,
        _rebuild,
        _index,
    ):
        avatar_selected = SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), selected_center=(1.0, 1.0), value_histogram={1: 1})
        avatar_mock.return_value = SimpleNamespace(
            selected=avatar_selected,
            diagnostics=SimpleNamespace(
                stable_avatar_found=True,
                episode_count=1,
                successful_episode_count=1,
                cross_reset_ambiguous=False,
            ),
            episodes=(SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=avatar_selected), transitions=tuple()),),
        )
        pois = (
            SimpleNamespace(poi_id="p1", confidence=0.9, bbox=(3, 3, 3, 3), area=4, ambiguity_flags=()),
            SimpleNamespace(poi_id="p2", confidence=0.8, bbox=(5, 5, 5, 5), area=4, ambiguity_flags=()),
        )
        pois_mock.return_value = {"report": SimpleNamespace(candidates=pois), "cross_reset_evidence": tuple(), "episodes": tuple()}
        _write_hud_hint.return_value = {"report": SimpleNamespace(failure_reason=None), "cross_reset_evidence": tuple(), "episodes": tuple(), "value_samples": {}}
        _write_hud.return_value = SimpleNamespace(selected=SimpleNamespace(selected_poi_id="p1", ambiguous=False, failure_reason=None, ranked_poi_ids=("p1", "p2")))
        hud_mock.return_value = {}
        interp_mock.return_value = {}
        captured = {}

        def _contact_side_effect(**kwargs):
            captured["candidate_count"] = len(tuple(kwargs["poi_multi_bundle"]["report"].candidates))
            captured["candidate_ids"] = tuple(item.poi_id for item in kwargs["poi_multi_bundle"]["report"].candidates)
            return SimpleNamespace(diagnostics={"generated_trajectories": tuple(), "attempted_trajectories": tuple(), "trajectory_stats_overall": {}}, episodes=tuple(), tested_pois=tuple())

        contact_mock.side_effect = _contact_side_effect
        adaptive_mock.return_value = SimpleNamespace(episodes=tuple(), solved=False, failure_reason="no_progress", generated_trajectories=tuple(), attempted_trajectories=tuple(), rejected_trajectories=tuple(), trajectory_stats=SimpleNamespace(to_dict=lambda: {"level_id": "L0"}))
        out = run_frontier_level_from_live_session(game_id="ez01", frontier_level_id="L0", session=object(), prefix_traces=tuple(), episode_count=1)
        self.assertEqual(captured["candidate_count"], 2)
        self.assertEqual(captured["candidate_ids"], ("p1", "p2"))
        self.assertEqual(out["diagnostics"]["contact_candidate_selection"]["selected_candidate_count"], 2)
        self.assertEqual(out["diagnostics"]["contact_candidate_selection"]["dropped_candidate_ids"], [])



if __name__ == "__main__":
    unittest.main()
