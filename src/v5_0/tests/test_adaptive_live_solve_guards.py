import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from v5_0.route.trajectory_enumerator import RouteCandidate
from v5_0.contracts.avatar_types import SavedLevelTrace
from v5_0.solve.service import (
    _stabilize_avatar_bbox_after_action,
    _stabilize_poi_bbox,
    _validate_adaptive_route,
    run_adaptive_solve_multi_reset,
    run_adaptive_solve_on_live_session,
)
from v5_0.solve.policy_builder import build_adaptive_policy_for_target


def _plane_frame(w=8, h=8, avatar=(1, 1), poi=(3, 3)):
    rows = [[0 for _ in range(w)] for _ in range(h)]
    rows[avatar[1]][avatar[0]] = 1
    rows[poi[1]][poi[0]] = 2
    plane = tuple(tuple(int(v) for v in row) for row in rows)
    return (plane,)


class _SessionAdapterStub:
    def __init__(self, observations, terminal_status="running", action_legal=True):
        self._observations = list(observations)
        self._idx = 0
        self._terminal_status = terminal_status
        self._action_legal = action_legal

    def get_current_observation(self, _session):
        if self._idx >= len(self._observations):
            return self._observations[-1]
        value = self._observations[self._idx]
        self._idx += 1
        return value

    def execute_action_prefix(self, _session, _translated, _raw):
        return SimpleNamespace(step_results=(SimpleNamespace(action_legal=bool(self._action_legal)),), terminal_status=self._terminal_status)


class _ActionAdapterStub:
    def translate_token(self, action, _context):
        return action


def _obs(frame, reward=0.0, levels=0):
    return SimpleNamespace(frame=frame, raw_payload={"reward": reward}, levels_completed=levels, available_actions=(0,))


class TestAdaptiveLiveSolveGuards(unittest.TestCase):
    def test_multi_reset_adaptive_solve_uses_canonical_selected_avatar_bbox(self):
        seen = {}

        def _run_episode(**kwargs):
            seen["selected_bbox"] = getattr(kwargs["selected_avatar"], "selected_bbox", None)
            return SimpleNamespace(
                steps=(),
                target_sequence=(),
                solved=False,
                failure_reason="no_progress",
            )

        avatar_multi_report = SimpleNamespace(
            selected=SimpleNamespace(selected_bbox=(24, 0, 31, 7), selected_center=(27.5, 3.5), failure_reason=None),
            diagnostics=SimpleNamespace(stable_avatar_found=True),
            episodes=(
                SimpleNamespace(
                    episode_index=0,
                    report=SimpleNamespace(
                        selected=SimpleNamespace(
                            selected_bbox=(24, 8, 31, 15),
                            selected_center=(27.5, 11.5),
                            failure_reason=None,
                        )
                    ),
                ),
            ),
        )
        poi_multi_bundle = {
            "report": SimpleNamespace(candidates=(SimpleNamespace(poi_id="p1", bbox=(24, 48, 31, 55)),)),
        }

        with patch("v5_0.solve.service.run_adaptive_solve_episode", side_effect=_run_episode):
            run_adaptive_solve_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_multi_bundle,
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez04",
                plan=SimpleNamespace(level_id="L2"),
                base_seed=0,
                render_terminal=False,
                env_factory=None,
                max_steps=5,
            )

        self.assertEqual(seen["selected_bbox"], (24, 0, 31, 7))

    def test_stabilize_avatar_bbox_after_action_rejects_orthogonal_reacquire(self):
        self.assertEqual(
            _stabilize_avatar_bbox_after_action(
                before_bbox=(40, 32, 47, 47),
                tracked_after_bbox=(40, 40, 47, 47),
                action="RIGHT",
                post_frame=None,
                blocked=False,
            ),
            (48, 32, 55, 47),
        )

    def test_stabilize_avatar_bbox_after_action_rejects_size_drift(self):
        self.assertEqual(
            _stabilize_avatar_bbox_after_action(
                before_bbox=(48, 32, 55, 47),
                tracked_after_bbox=(56, 29, 57, 39),
                action="RIGHT",
                post_frame=None,
                blocked=False,
            ),
            (56, 32, 63, 47),
        )

    def test_stabilize_poi_bbox_rejects_avatar_overlap(self):
        self.assertEqual(
            _stabilize_poi_bbox((56, 24, 63, 39), (56, 24, 63, 31), avatar_bbox=(56, 24, 63, 39)),
            (56, 24, 63, 31),
        )

    def test_overlap_without_meaningful_outcome_does_not_extend_useful_prefix(self):
        frame0 = _plane_frame(w=12, h=12, avatar=(5, 5), poi=(7, 5))
        frame1 = _plane_frame(w=12, h=12, avatar=(6, 5), poi=(7, 5))
        frame2 = _plane_frame(w=12, h=12, avatar=(7, 5), poi=(7, 5))
        session_adapter = _SessionAdapterStub(
            [
                _obs(frame0, reward=0.0, levels=0),
                _obs(frame0, reward=0.0, levels=0),
                _obs(frame1, reward=0.0, levels=0),
                _obs(frame1, reward=0.0, levels=0),
                _obs(frame2, reward=0.0, levels=0),
            ]
        )
        result = run_adaptive_solve_on_live_session(
            session=SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None)),
            selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(5, 5, 5, 5), value_histogram={1: 1}),
            ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(7, 5, 7, 5), center=(7.0, 5.0), value_histogram={2: 1}, confidence=1.0),),
            hud_targeting_report=SimpleNamespace(),
            contact_experiment_report=None,
            game_id="ez01",
            level_id="L0",
            max_steps=2,
            session_adapter=session_adapter,
            action_adapter=_ActionAdapterStub(),
        )
        self.assertFalse(result.solved)
        self.assertIn(result.failure_reason, {"no_valid_route_candidates", "route_exhausted_without_progress", "repeated_non_progress", "no_progress", "step_budget_exhausted"})

    def test_stabilize_poi_bbox_rejects_geometry_growth(self):
        self.assertEqual(
            _stabilize_poi_bbox((56, 24, 63, 39), (56, 24, 63, 31), avatar_bbox=(40, 40, 47, 47)),
            (56, 24, 63, 31),
        )

    def _run(self, observations, max_steps=5, ranked=None, session_adapter=None, terminal_status="running", action_legal=True):
        ranked = ranked or (SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),)
        if session_adapter is None:
            session_adapter = _SessionAdapterStub(observations, terminal_status=terminal_status, action_legal=action_legal)
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        selected_avatar = SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1})
        initial_target = SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True)
        with patch("v5_0.solve.service.select_initial_target", return_value=initial_target), patch(
            "v5_0.solve.service.build_adaptive_policy_for_target", return_value=("RIGHT",)
        ):
            return run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=selected_avatar,
                ranked_poi_candidates=ranked,
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=max_steps,
                session_adapter=session_adapter,
                action_adapter=_ActionAdapterStub(),
            )

    def test_repeated_identical_state_signature_triggers_repeated_non_progress(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        out = self._run(obs, max_steps=6)
        self.assertIn(out.failure_reason, {"repeated_non_progress", "route_exhausted_without_progress"})

    def test_same_action_no_progress_three_times_triggers_repeated_non_progress(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        out = self._run(obs, max_steps=6)
        self.assertIn(out.failure_reason, {"repeated_non_progress", "route_exhausted_without_progress"})

    def test_reward_change_prevents_no_progress_termination_for_that_step(self):
        frame = _plane_frame()
        obs = [
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=1.0, levels=0),
            _obs(frame, reward=1.0, levels=0),
            _obs(frame, reward=1.0, levels=0),
        ]
        out = self._run(obs, max_steps=3)
        self.assertNotEqual(out.failure_reason, "repeated_non_progress")

    def test_level_transition_still_returns_solved(self):
        frame = _plane_frame()
        obs = [
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=0),
            _obs(frame, reward=0.0, levels=1),
        ]
        out = self._run(obs, max_steps=2)
        self.assertTrue(out.solved)

    def test_relaxed_reacquire_used_before_frontier_geometry_mismatch(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 10
        with patch("v5_0.solve.service.track_avatar_bbox_in_frame", return_value=None), patch(
            "v5_0.solve.service.track_poi_bbox_in_frame", return_value=None
        ), patch("v5_0.solve.service.reacquire_avatar_bbox_in_frame", return_value=(1, 1, 1, 1)), patch(
            "v5_0.solve.service.reacquire_poi_bbox_in_frame", return_value=(3, 3, 3, 3)
        ):
            out = self._run(obs, max_steps=1)
        self.assertNotEqual(out.failure_reason, "frontier_geometry_mismatch")

    def test_retarget_path_uses_relaxed_poi_reacquire_before_failing(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        ranked = (
            SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),
            SimpleNamespace(poi_id="p2", bbox=(5, 5, 5, 5), value_histogram={2: 1}),
        )
        next_target = SimpleNamespace(target_poi_id="p2", source="fallback", confidence=0.8, attempt_count=1, last_outcome_type=None, active=True)
        with patch("v5_0.solve.service.select_next_target", return_value=next_target), patch(
            "v5_0.solve.service.track_poi_bbox_in_frame", side_effect=[(3, 3, 3, 3)] * 6 + [None] * 20
        ), patch("v5_0.solve.service.reacquire_poi_bbox_in_frame", return_value=(5, 5, 5, 5)):
            out = self._run(obs, max_steps=4, ranked=ranked, session_adapter=_SessionAdapterStub(obs, action_legal=False))
        self.assertNotEqual(out.failure_reason, "frontier_geometry_mismatch")

    def test_empty_route_set_does_not_fabricate_right_action(self):
        frame = _plane_frame()
        obs = _obs(frame, reward=0.0, levels=0)
        session_adapter = SimpleNamespace(
            get_current_observation=MagicMock(return_value=obs),
            execute_action_prefix=MagicMock(),
        )
        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service.build_adaptive_policy_for_target", return_value=tuple()
        ), patch("v5_0.solve.service.track_avatar_bbox_in_frame", return_value=(1, 1, 1, 1)), patch(
            "v5_0.solve.service.track_poi_bbox_in_frame", return_value=(3, 3, 3, 3)
        ):
            out = run_adaptive_solve_on_live_session(
                session=SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None)),
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=3,
                session_adapter=session_adapter,
                action_adapter=_ActionAdapterStub(),
            )
        self.assertIn(out.failure_reason, {"no_valid_route_candidates", "route_exhausted_without_progress"})
        self.assertEqual(session_adapter.execute_action_prefix.call_count, 0)
        self.assertFalse(any(step.action == "RIGHT" for episode in out.episodes for step in episode.steps))

    def test_action_space_short_route_is_not_rejected_as_impossible_displacement(self):
        plane = tuple(
            tuple(int(v) for v in row)
            for row in (
                (0, 0, 2, 2),
                (0, 0, 2, 2),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 1, 1, 0),
                (0, 1, 1, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            )
        )
        obs = _obs((plane,), reward=0.0, levels=0)
        session_adapter = SimpleNamespace(
            get_current_observation=MagicMock(return_value=obs),
            execute_action_prefix=MagicMock(return_value=SimpleNamespace(step_results=(SimpleNamespace(action_legal=True),), terminal_status="running")),
        )
        route = RouteCandidate(
            route_id="route_000",
            actions=("UP", "UP"),
            length=2,
            net_dx=0,
            net_dy=-2,
            first_action="UP",
            turn_count=0,
            axis_order="V_ONLY",
            waypoints=((0, 0), (0, -1), (0, -2)),
            score_components={},
        )
        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service.build_adaptive_policy_for_target", return_value=(route,)
        ), patch("v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 4, 2, 5), (2, 0, 3, 1))):
            out = run_adaptive_solve_on_live_session(
                session=SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None)),
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 4, 2, 5), selected_center=(1.5, 4.5), value_histogram={1: 4}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(2, 0, 3, 1), center=(2.5, 0.5), value_histogram={2: 4}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=1,
                session_adapter=session_adapter,
                action_adapter=_ActionAdapterStub(),
            )
        self.assertEqual(session_adapter.execute_action_prefix.call_count, 1)
        self.assertNotEqual(out.failure_reason, "no_valid_route_candidates")

    def test_touch_interaction_route_validates_against_its_planned_displacement(self):
        route = RouteCandidate(
            route_id="touch:route_000",
            actions=("LEFT", "LEFT"),
            length=2,
            net_dx=-2,
            net_dy=0,
            first_action="LEFT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (-1, 0), (-2, 0)),
            score_components={"interaction_touch": 1.0},
        )

        ok, reasons = _validate_adaptive_route(route, dx=-3, dy=0, hint_source=None)

        self.assertTrue(ok)
        self.assertNotIn("impossible_displacement", reasons)

    def test_hinted_overlap_route_validates_against_its_planned_displacement(self):
        route = RouteCandidate(
            route_id="hint:contact:0:cross_poi_000:overlap:route_000",
            actions=("DOWN", "DOWN", "RIGHT"),
            length=3,
            net_dx=1,
            net_dy=2,
            first_action="DOWN",
            turn_count=1,
            axis_order="MIXED",
            waypoints=((0, 0), (0, 1), (0, 2), (1, 2)),
            score_components={"interaction_overlap": 1.0},
        )

        ok, reasons = _validate_adaptive_route(route, dx=-3, dy=-3, hint_source=None)

        self.assertTrue(ok)
        self.assertNotIn("impossible_displacement", reasons)

    def test_policy_builder_uses_passed_refreshed_avatar_geometry(self):
        avatar = SimpleNamespace(selected_bbox=(24, 24, 31, 31), selected_center=(27.5, 27.5))
        poi = SimpleNamespace(bbox=(56, 0, 63, 7), center=(59.5, 3.5))

        with patch("v5_0.solve.policy_builder.enumerate_routes_between_points") as enumerate_routes:
            enumerate_routes.return_value = tuple()
            build_adaptive_policy_for_target(
                avatar,
                poi,
                current_frame=_plane_frame(),
                step_budget_remaining=8,
                route_hints=None,
            )

        kwargs = enumerate_routes.call_args.kwargs
        self.assertEqual(kwargs["start_center"], (27.5 / 8.0, 27.5 / 8.0))
        self.assertEqual(kwargs["target_center"], (59.5 / 8.0, 3.5 / 8.0))

    def test_same_level_continuation_replay_is_not_labeled_bootstrap_replay(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        out = self._run(obs, max_steps=2, session_adapter=_SessionAdapterStub(obs), ranked=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),))
        sources = tuple(str(getattr(step, "source", "")) for episode in out.episodes for step in episode.steps)
        self.assertNotIn("bootstrap_replay", sources)

    def test_world_change_contact_does_not_switch_targets_before_completion(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        ranked = (
            SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),
            SimpleNamespace(poi_id="p2", bbox=(5, 5, 5, 5), value_histogram={2: 1}),
        )
        session_adapter = _SessionAdapterStub(obs)
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("UP",),
            length=1,
            net_dx=0,
            net_dy=-1,
            first_action="UP",
            turn_count=0,
            axis_order="V_ONLY",
            waypoints=((0, 0), (0, -1)),
            score_components={},
        )
        route_results = iter(
            [
                {
                    "steps": tuple(),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": tuple(),
                    "avatar_bbox": (1, 1, 1, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                },
                {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=True,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_solve",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 1,
                    "executed_actions": ("UP",),
                    "avatar_bbox": (1, 1, 1, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                },
                {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=True,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_prefix_replay",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": True,
                    "useful_prefix_length": 1,
                    "executed_actions": ("UP",),
                    "avatar_bbox": (1, 1, 1, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                },
                {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=True,
                            outcome_type="level_transition",
                            blocked_action=False,
                            terminal=True,
                            levels_completed_before=0,
                            levels_completed_after=1,
                            source="frontier_solve",
                        ),
                    ),
                    "solved": True,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": True,
                    "useful_prefix_length": 1,
                    "executed_actions": ("UP",),
                    "avatar_bbox": (5, 5, 5, 5),
                    "target_bbox": (5, 5, 5, 5),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": True,
                },
            ]
        )

        def _execute_frontier_action_sequence(**_kwargs):
            return next(route_results)

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", return_value=(route,)), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (3, 3, 3, 3))
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            out = run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=ranked,
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=2,
                session_adapter=session_adapter,
                action_adapter=_ActionAdapterStub(),
            )

        self.assertTrue(out.solved)
        self.assertGreaterEqual(len(tuple(out.episodes)), 2)
        self.assertEqual(tuple(out.episodes[0].target_sequence)[0].target_poi_id, "p1")
        self.assertEqual(tuple(out.episodes[1].target_sequence)[0].target_poi_id, "p1")

    def test_nonterminal_world_change_prefix_is_promoted_into_next_attempt(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        prefix_calls = []
        call_state = {"route_calls": 0}

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                prefix_calls.append(tuple(kwargs["actions"]))
                return {
                    "steps": tuple(),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": tuple(kwargs["actions"]),
                    "avatar_bbox": kwargs["start_avatar_bbox"],
                    "target_bbox": kwargs["start_target_bbox"],
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            call_state["route_calls"] += 1
            if call_state["route_calls"] == 1:
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=False,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_solve",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": False,
                    "useful_prefix_length": 1,
                    "executed_actions": ("RIGHT",),
                    "avatar_bbox": (2, 1, 2, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": (1, 1, 1, 1),
                "target_bbox": (3, 3, 3, 3),
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", return_value=(route,)), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (3, 3, 3, 3))
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=2,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertGreaterEqual(len(prefix_calls), 2)
        self.assertEqual(prefix_calls[0], tuple())
        self.assertEqual(prefix_calls[1], ("RIGHT",))

    def test_prefix_replay_world_change_refreshes_live_anchor_before_next_route_build(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        route_call_state = {"count": 0}
        built_avatar_boxes = []
        built_target_boxes = []

        def _build_policy(route_avatar, route_poi, *_args, **_kwargs):
            built_avatar_boxes.append(getattr(route_avatar, "selected_bbox", None))
            built_target_boxes.append(getattr(route_poi, "bbox", None))
            return (route,)

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                actions = tuple(kwargs["actions"])
                if not actions:
                    return {
                        "steps": tuple(),
                        "solved": False,
                        "failure_reason": None,
                        "route_progress": False,
                        "route_closer": False,
                        "useful_prefix_length": 0,
                        "executed_actions": tuple(),
                        "avatar_bbox": kwargs["start_avatar_bbox"],
                        "target_bbox": kwargs["start_target_bbox"],
                        "recent_avatar_motion": None,
                        "recent_target_motion": None,
                        "level_transition": False,
                    }
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=False,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_prefix_replay",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": actions,
                    "avatar_bbox": kwargs["start_avatar_bbox"],
                    "target_bbox": kwargs["start_target_bbox"],
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            route_call_state["count"] += 1
            if route_call_state["count"] == 1:
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=False,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_solve",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": False,
                    "useful_prefix_length": 1,
                    "executed_actions": ("RIGHT",),
                    "avatar_bbox": (2, 1, 2, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": (4, 1, 4, 1),
                "target_bbox": (6, 6, 6, 6),
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        reanchor_results = iter((((4, 1, 4, 1), (6, 6, 6, 6)),))

        def _reanchor(**_kwargs):
            return next(reanchor_results)

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.reanchor_frontier_objects", side_effect=_reanchor
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=2,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertGreaterEqual(len(built_avatar_boxes), 2)
        self.assertEqual(built_avatar_boxes[0], (1, 1, 1, 1))
        self.assertEqual(built_target_boxes[0], (3, 3, 3, 3))
        self.assertEqual(built_avatar_boxes[1], (4, 1, 4, 1))
        self.assertEqual(built_target_boxes[1], (6, 6, 6, 6))

    def test_prefix_replay_prefers_post_prefix_avatar_bbox_over_stale_carried_bbox(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        built_avatar_boxes = []
        built_target_boxes = []
        route_call_state = {"count": 0}

        def _build_policy(route_avatar, route_poi, *_args, **_kwargs):
            built_avatar_boxes.append(getattr(route_avatar, "selected_bbox", None))
            built_target_boxes.append(getattr(route_poi, "bbox", None))
            return (route,)

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                actions = tuple(kwargs["actions"])
                if not actions:
                    return {
                        "steps": tuple(),
                        "solved": False,
                        "failure_reason": None,
                        "route_progress": False,
                        "route_closer": False,
                        "useful_prefix_length": 0,
                        "executed_actions": tuple(),
                        "avatar_bbox": kwargs["start_avatar_bbox"],
                        "target_bbox": kwargs["start_target_bbox"],
                        "recent_avatar_motion": None,
                        "recent_target_motion": None,
                        "level_transition": False,
                    }
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=False,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_prefix_replay",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": actions,
                    "avatar_bbox": (9, 9, 9, 9),
                    "target_bbox": (6, 6, 6, 6),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            route_call_state["count"] += 1
            if route_call_state["count"] == 1:
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=False,
                            outcome_type="world_change",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_solve",
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": False,
                    "useful_prefix_length": 1,
                    "executed_actions": ("RIGHT",),
                    "avatar_bbox": (2, 1, 2, 1),
                    "target_bbox": (3, 3, 3, 3),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": (9, 9, 9, 9),
                "target_bbox": (6, 6, 6, 6),
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=(None, None)
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=2,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertGreaterEqual(len(built_avatar_boxes), 2)
        self.assertEqual(built_avatar_boxes[1], (9, 9, 9, 9))

    def test_initial_frontier_attempt_validates_avatar_bbox_against_current_frame(self):
        frame = _plane_frame(w=12, h=12, avatar=(8, 8), poi=(10, 10))
        obs = [_obs(frame, reward=0.0, levels=0)] * 10
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        built_avatar_boxes = []

        def _build_policy(route_avatar, *_args, **_kwargs):
            built_avatar_boxes.append(getattr(route_avatar, "selected_bbox", None))
            return (route,)

        def _execute_frontier_action_sequence(**kwargs):
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.track_avatar_bbox_in_frame", return_value=(8, 8, 8, 8)
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(8, 0, 8, 0), selected_center=(8.0, 0.0), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(10, 10, 10, 10), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ul01",
                level_id="L1",
                max_steps=1,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertTrue(built_avatar_boxes)
        self.assertEqual(built_avatar_boxes[0], (8, 8, 8, 8))

    def test_initial_frontier_attempt_prefers_contact_hint_start_bbox_over_stale_selected_bbox(self):
        frame = _plane_frame(w=12, h=12, avatar=(8, 8), poi=(10, 10))
        obs = [_obs(frame, reward=0.0, levels=0)] * 10
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        built_avatar_boxes = []

        def _build_policy(route_avatar, *_args, **_kwargs):
            built_avatar_boxes.append(getattr(route_avatar, "selected_bbox", None))
            return (route,)

        def _execute_frontier_action_sequence(**kwargs):
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        contact_report = SimpleNamespace(
            tested_pois=(
                SimpleNamespace(
                    poi_id="p1",
                    outcome=SimpleNamespace(outcome_type="door_opens", contact_step_index=0, confidence=0.7, new_object_appeared=False),
                    policy=SimpleNamespace(planned_actions=("DOWN",)),
                    route_evidence={
                        "route_hint": {
                            "route_id": "contact:0:p1:overlap:route_000",
                            "actions": ("DOWN",),
                            "start_avatar_bbox": (8, 8, 8, 8),
                            "start_poi_bbox": (10, 10, 10, 10),
                            "last_avatar_bbox": (8, 9, 8, 9),
                            "last_poi_bbox": (10, 10, 10, 10),
                            "suggested_prefix_length": 1,
                        }
                    },
                ),
            ),
        )

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.track_avatar_bbox_in_frame", return_value=None
        ), patch("v5_0.solve.service.reacquire_avatar_bbox_in_frame", return_value=None), patch(
            "v5_0.solve.service.find_best_component_match_in_frame", return_value=None
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(8, 0, 8, 0), selected_center=(8.0, 0.0), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(10, 10, 10, 10), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=contact_report,
                game_id="ul01",
                level_id="L1",
                max_steps=1,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertTrue(built_avatar_boxes)
        self.assertEqual(built_avatar_boxes[0], (8, 8, 8, 8))

    def test_prefix_replay_uses_live_frontier_bbox_not_stale_selected_or_contact_hint_start_bbox(self):
        frame = _plane_frame(w=12, h=12, avatar=(0, 0), poi=(7, 0))
        obs = [_obs(frame, reward=0.0, levels=0)] * 10
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        built_avatar_boxes = []

        def _build_policy(route_avatar, *_args, **_kwargs):
            built_avatar_boxes.append(getattr(route_avatar, "selected_bbox", None))
            return (route,)

        def _execute_frontier_action_sequence(**kwargs):
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        contact_report = SimpleNamespace(
            tested_pois=(
                SimpleNamespace(
                    poi_id="p1",
                    outcome=SimpleNamespace(outcome_type="door_opens", contact_step_index=0, confidence=0.7, new_object_appeared=False),
                    policy=SimpleNamespace(planned_actions=("DOWN",)),
                    route_evidence={
                        "route_hint": {
                            "route_id": "contact:0:p1:overlap:route_000",
                            "actions": ("DOWN",),
                            "start_avatar_bbox": (8, 8, 8, 8),
                            "start_poi_bbox": (3, 3, 3, 3),
                            "last_avatar_bbox": (8, 9, 8, 9),
                            "last_poi_bbox": (3, 3, 3, 3),
                            "suggested_prefix_length": 1,
                        }
                    },
                ),
            ),
        )

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.track_avatar_bbox_in_frame", return_value=(0, 0, 0, 0)
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(8, 8, 8, 8), selected_center=(8.0, 8.0), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(7, 0, 7, 0), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=contact_report,
                game_id="ul01",
                level_id="L3",
                max_steps=1,
                initial_frontier_actions=("RIGHT",),
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertTrue(built_avatar_boxes)
        self.assertEqual(built_avatar_boxes[0], (0, 0, 0, 0))

    def test_switching_to_next_poi_does_not_reuse_previous_poi_bbox_as_target_or_avatar(self):
        frame = _plane_frame(w=64, h=64, avatar=(8, 8), poi=(24, 24))
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route = RouteCandidate(
            route_id="overlap:route_000",
            actions=("DOWN", "RIGHT"),
            length=2,
            net_dx=1,
            net_dy=1,
            first_action="DOWN",
            turn_count=1,
            axis_order="MIXED",
            waypoints=((0, 0), (0, 1), (1, 1)),
            score_components={},
        )
        built_pairs = []

        def _build_policy(route_avatar, route_poi, *_args, **_kwargs):
            built_pairs.append((getattr(route_avatar, "selected_bbox", None), getattr(route_poi, "bbox", None), getattr(route_poi, "poi_id", None)))
            return (route,)

        solve_call_count = {"count": 0}

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                return {
                    "steps": tuple(),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": tuple(kwargs["actions"]),
                    "avatar_bbox": kwargs["start_avatar_bbox"],
                    "target_bbox": kwargs["start_target_bbox"],
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            solve_call_count["count"] += 1
            if solve_call_count["count"] == 1:
                return {
                    "steps": (
                        SimpleNamespace(
                            contact_detected=True,
                            outcome_type="new_object_appeared",
                            blocked_action=False,
                            terminal=False,
                            levels_completed_before=0,
                            levels_completed_after=0,
                            source="frontier_solve",
                            avatar_bbox_before=(16, 24, 23, 31),
                            avatar_bbox_after=(24, 24, 31, 31),
                            target_bbox_before=(24, 24, 31, 31),
                            target_bbox_after=(24, 24, 31, 31),
                        ),
                    ),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": True,
                    "route_closer": True,
                    "useful_prefix_length": 1,
                    "executed_actions": tuple(kwargs["actions"]),
                    "avatar_bbox": (24, 24, 31, 31),
                    "target_bbox": (24, 24, 31, 31),
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            return {
                "steps": tuple(),
                "solved": False,
                "failure_reason": "blocked_action",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", side_effect=_build_policy), patch(
            "v5_0.solve.service.track_avatar_bbox_in_frame", return_value=(8, 8, 15, 15)
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(8, 8, 15, 15), selected_center=(11.5, 11.5), value_histogram={1: 1}),
                ranked_poi_candidates=(
                    SimpleNamespace(poi_id="p1", bbox=(24, 24, 31, 31), value_histogram={2: 1}, confidence=1.0),
                    SimpleNamespace(poi_id="p2", bbox=(48, 8, 55, 15), value_histogram={3: 1}, confidence=0.9),
                ),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ul01",
                level_id="L0",
                max_steps=2,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertGreaterEqual(len(built_pairs), 2)
        self.assertEqual(built_pairs[1], ((8, 8, 15, 15), (48, 8, 55, 15), "p2"))


    def test_non_progress_cutoff_waits_until_all_routes_for_state_are_tried(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 40
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        routes = tuple(
            RouteCandidate(
                route_id=f"route_{idx:03d}",
                actions=(action_a, action_b),
                length=2,
                net_dx=0,
                net_dy=0,
                first_action=action_a,
                turn_count=1,
                axis_order="MIXED",
                waypoints=((0, 0), (0, 0), (0, 0)),
                score_components={},
            )
            for idx, (action_a, action_b) in enumerate(
                (
                    ("RIGHT", "UP"),
                    ("UP", "RIGHT"),
                    ("RIGHT", "DOWN"),
                    ("DOWN", "RIGHT"),
                    ("LEFT", "UP"),
                    ("UP", "LEFT"),
                )
            )
        )
        attempted_actions = []

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                return {
                    "steps": tuple(),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": tuple(kwargs["actions"]),
                    "avatar_bbox": kwargs["start_avatar_bbox"],
                    "target_bbox": kwargs["start_target_bbox"],
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            attempted_actions.append(tuple(kwargs["actions"]))
            return {
                "steps": (
                    SimpleNamespace(
                        contact_detected=False,
                        outcome_type="no_effect",
                        blocked_action=False,
                        terminal=False,
                        levels_completed_before=0,
                        levels_completed_after=0,
                        source="frontier_solve",
                    ),
                ),
                "solved": False,
                "failure_reason": "no_progress",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(kwargs["actions"]),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", return_value=routes), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (3, 3, 3, 3))
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            out = run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=6,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertIn(out.failure_reason, {"route_exhausted_without_progress", "all_pois_exhausted", "repeated_non_progress", "step_budget_exhausted"})
        self.assertEqual(len(attempted_actions), 6)
        self.assertEqual(attempted_actions, [tuple(route.actions) for route in routes])

    def test_all_defined_routes_for_same_state_are_attempted_before_exhaustion(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))
        route_a = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        route_b = RouteCandidate(
            route_id="route_001",
            actions=("DOWN",),
            length=1,
            net_dx=0,
            net_dy=1,
            first_action="DOWN",
            turn_count=0,
            axis_order="V_ONLY",
            waypoints=((0, 0), (0, 1)),
            score_components={},
        )
        attempted_actions = []

        def _execute_frontier_action_sequence(**kwargs):
            if kwargs["source"] == "frontier_prefix_replay":
                return {
                    "steps": tuple(),
                    "solved": False,
                    "failure_reason": None,
                    "route_progress": False,
                    "route_closer": False,
                    "useful_prefix_length": 0,
                    "executed_actions": tuple(kwargs["actions"]),
                    "avatar_bbox": kwargs["start_avatar_bbox"],
                    "target_bbox": kwargs["start_target_bbox"],
                    "recent_avatar_motion": None,
                    "recent_target_motion": None,
                    "level_transition": False,
                }
            attempted_actions.append(tuple(kwargs["actions"]))
            return {
                "steps": (
                    SimpleNamespace(
                        contact_detected=False,
                        outcome_type="no_effect",
                        blocked_action=False,
                        terminal=False,
                        levels_completed_before=0,
                        levels_completed_after=0,
                        source="frontier_solve",
                    ),
                ),
                "solved": False,
                "failure_reason": "no_progress",
                "route_progress": False,
                "route_closer": False,
                "useful_prefix_length": 0,
                "executed_actions": tuple(kwargs["actions"]),
                "avatar_bbox": kwargs["start_avatar_bbox"],
                "target_bbox": kwargs["start_target_bbox"],
                "recent_avatar_motion": None,
                "recent_target_motion": None,
                "level_transition": False,
            }

        with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True, route_feasibility=True)), patch(
            "v5_0.solve.service._start_frontier_attempt_session", return_value=(session, False, None)
        ), patch("v5_0.solve.service.build_adaptive_policy_for_target", return_value=(route_a, route_b)), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (3, 3, 3, 3))
        ), patch("v5_0.solve.service._execute_frontier_action_sequence", side_effect=_execute_frontier_action_sequence):
            out = run_adaptive_solve_on_live_session(
                session=session,
                selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}),),
                hud_targeting_report=SimpleNamespace(),
                contact_experiment_report=None,
                game_id="ez01",
                level_id="L0",
                max_steps=2,
                session_adapter=_SessionAdapterStub(obs),
                action_adapter=_ActionAdapterStub(),
            )

        self.assertIn(out.failure_reason, {"route_exhausted_without_progress", "all_pois_exhausted", "repeated_non_progress", "step_budget_exhausted"})
        self.assertEqual(attempted_actions[:2], [("RIGHT",), ("DOWN",)])

    def test_skip_bootstrap_replay_in_final_solve_drops_bootstrap_sourced_prefix_trace(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 8
        bootstrap_trace = SavedLevelTrace(
            game_id="ez01",
            level_id="L0",
            solved=True,
            action_trace=("RIGHT",),
            step_count=1,
            source_run_id=None,
            trace_version=1,
            replay_verified=True,
            action_sources=("bootstrap_replay",),
            trace_id="bootstrap",
        )
        session_adapter = _SessionAdapterStub(obs)
        with patch("v5_0.solve.service.replay_prefix_traces_to_frontier") as replay:
            self._run(obs, max_steps=1, session_adapter=session_adapter)
            self.assertEqual(replay.call_count, 0)
            with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True)):
                run_adaptive_solve_on_live_session(
                    session=SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None)),
                    selected_avatar=SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
                    ranked_poi_candidates=(SimpleNamespace(poi_id="p1", bbox=(3, 3, 3, 3), value_histogram={2: 1}, confidence=1.0),),
                    hud_targeting_report=SimpleNamespace(),
                    contact_experiment_report=None,
                    game_id="ez01",
                    level_id="L0",
                    max_steps=1,
                    session_adapter=_SessionAdapterStub(obs),
                    action_adapter=_ActionAdapterStub(),
                    skip_bootstrap_replay_in_final_solve=True,
                    prefix_traces=(bootstrap_trace,),
                )
            self.assertEqual(replay.call_count, 0)

    def test_all_valid_routes_already_attempted_terminates_without_retry(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 30
        route = RouteCandidate(
            route_id="route_000",
            actions=("RIGHT",),
            length=1,
            net_dx=1,
            net_dy=0,
            first_action="RIGHT",
            turn_count=0,
            axis_order="H_ONLY",
            waypoints=((0, 0), (1, 0)),
            score_components={},
        )
        session_adapter = _SessionAdapterStub(obs)
        with patch("v5_0.solve.service.build_adaptive_policy_for_target", return_value=(route,)), patch(
            "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (3, 3, 3, 3))
        ):
            out = self._run(obs, max_steps=5, session_adapter=session_adapter)
        self.assertIn(out.failure_reason, {"route_exhausted_without_progress", "all_pois_exhausted"})
        self.assertLessEqual(session_adapter._idx, 30)


if __name__ == "__main__":
    unittest.main()
