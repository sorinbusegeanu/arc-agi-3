import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from v5_0.route.trajectory_enumerator import RouteCandidate
from v5_0.solve.service import run_adaptive_solve_on_live_session


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
        self.assertEqual(out.failure_reason, "repeated_non_progress")

    def test_same_action_no_progress_three_times_triggers_repeated_non_progress(self):
        frame = _plane_frame()
        obs = [_obs(frame, reward=0.0, levels=0)] * 20
        out = self._run(obs, max_steps=6)
        self.assertEqual(out.failure_reason, "repeated_non_progress")

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



if __name__ == "__main__":
    unittest.main()
