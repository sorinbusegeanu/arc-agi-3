from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from v5_0.solve.service import reanchor_frontier_objects, run_adaptive_solve_on_live_session


def _frame():
    return ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 2, 0), (0, 0, 0, 0))


def test_reanchor_succeeds_from_contact_hint_when_stored_bbox_fails():
    frame = _frame()
    selected = SimpleNamespace(selected_bbox=(0, 0, 0, 0), selected_center=(0.0, 0.0), value_histogram={1: 1})
    poi = SimpleNamespace(bbox=(3, 3, 3, 3), center=(3.0, 3.0), value_histogram={2: 1})
    hint = {"last_avatar_bbox": (1, 1, 1, 1), "last_poi_bbox": (2, 2, 2, 2)}
    with patch("v5_0.solve.service.track_avatar_bbox_in_frame", return_value=None), patch(
        "v5_0.solve.service.track_poi_bbox_in_frame", return_value=None
    ), patch(
        "v5_0.solve.service.reacquire_avatar_bbox_in_frame", return_value=(1, 1, 1, 1)
    ), patch(
        "v5_0.solve.service.reacquire_poi_bbox_in_frame", return_value=(2, 2, 2, 2)
    ):
        a, p = reanchor_frontier_objects(
            frame=frame,
            selected_avatar=selected,
            initial_poi=poi,
            contact_route_hint=hint,
        )
    assert a == (1, 1, 1, 1)
    assert p == (2, 2, 2, 2)


def test_local_preferred_center_match_beats_far_global_component():
    frame = ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 2, 0, 0))
    selected = SimpleNamespace(selected_bbox=(1, 1, 1, 1), selected_center=(1.0, 1.0), value_histogram={1: 1})
    poi = SimpleNamespace(bbox=(2, 4, 2, 4), center=(2.0, 4.0), value_histogram={2: 1})
    a, _ = reanchor_frontier_objects(
        frame=frame,
        selected_avatar=selected,
        initial_poi=poi,
        preferred_avatar_bbox=(1, 1, 1, 1),
    )
    assert a == (1, 1, 1, 1)


def test_solve_startup_avoids_frontier_geometry_mismatch_when_hint_recovery_succeeds():
    session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))

    class _SessionAdapter:
        def get_current_observation(self, _):
            return SimpleNamespace(frame=(_frame(),), available_actions=(0,), levels_completed=0, raw_payload={"reward": 0.0})

        def execute_action_prefix(self, *_args, **_kwargs):
            return SimpleNamespace(step_results=(SimpleNamespace(action_legal=True),), terminal_status="success")

    class _ActionAdapter:
        def translate_token(self, action, _ctx):
            return action

    selected_avatar = SimpleNamespace(failure_reason=None, selected_bbox=(0, 0, 0, 0), selected_center=(0.0, 0.0), value_histogram={1: 1})
    ranked = (SimpleNamespace(poi_id="p1", bbox=(2, 2, 2, 2), center=(2.0, 2.0), value_histogram={2: 1}),)
    hint = {"last_avatar_bbox": (1, 1, 1, 1), "last_poi_bbox": (2, 2, 2, 2), "actions": ("RIGHT",), "suggested_prefix_length": 1}

    with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True)), patch(
        "v5_0.solve.service.get_best_route_hint_for_poi", return_value=hint
    ), patch(
        "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (2, 2, 2, 2))
    ), patch(
        "v5_0.solve.service.build_adaptive_policy_for_target", return_value=(("LEFT",), ("RIGHT",))
    ):
        out = run_adaptive_solve_on_live_session(
            session=session,
            selected_avatar=selected_avatar,
            ranked_poi_candidates=ranked,
            hud_targeting_report=SimpleNamespace(hud_mask=None),
            contact_experiment_report=SimpleNamespace(tested_pois=tuple(), diagnostics={}),
            game_id="ez01",
            level_id="L0",
            max_steps=2,
            session_adapter=_SessionAdapter(),
            action_adapter=_ActionAdapter(),
        )
    assert out.failure_reason != "frontier_geometry_mismatch"


def test_solve_route_switches_after_repeated_no_progress_on_first_route():
    session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=None, coordinate_bounds=None))

    class _SessionAdapter:
        def __init__(self):
            self.actions = []
            self._obs = 0

        def get_current_observation(self, _):
            self._obs += 1
            return SimpleNamespace(frame=(_frame(),), available_actions=(0,), levels_completed=0, raw_payload={"reward": 0.0})

        def execute_action_prefix(self, _session, _translated, raw):
            self.actions.append(raw[0])
            return SimpleNamespace(step_results=(SimpleNamespace(action_legal=True),), terminal_status="running")

    class _ActionAdapter:
        def translate_token(self, action, _ctx):
            return action

    adapter = _SessionAdapter()
    selected_avatar = SimpleNamespace(failure_reason=None, selected_bbox=(1, 1, 1, 1), selected_center=(1.0, 1.0), value_histogram={1: 1})
    ranked = (SimpleNamespace(poi_id="p1", bbox=(2, 2, 2, 2), center=(2.0, 2.0), value_histogram={2: 1}),)
    with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True)), patch(
        "v5_0.solve.service.reanchor_frontier_objects", return_value=((1, 1, 1, 1), (2, 2, 2, 2))
    ), patch(
        "v5_0.solve.service.build_adaptive_policy_for_target", return_value=(("RIGHT", "DOWN"), ("DOWN", "RIGHT"))
    ), patch(
        "v5_0.solve.service.detect_screen_change", return_value=False
    ), patch(
        "v5_0.solve.service.detect_hud_only_change", return_value=False
    ), patch(
        "v5_0.solve.service.detect_screen_change_outside_hud_mask", return_value=False
    ), patch(
        "v5_0.solve.service.detect_contact", return_value=False
    ):
        run_adaptive_solve_on_live_session(
            session=session,
            selected_avatar=selected_avatar,
            ranked_poi_candidates=ranked,
            hud_targeting_report=SimpleNamespace(hud_mask=None),
            contact_experiment_report=None,
            game_id="ez01",
            level_id="L0",
            max_steps=4,
            session_adapter=adapter,
            action_adapter=_ActionAdapter(),
        )
    assert adapter.actions[:4] == ["RIGHT", "DOWN", "DOWN", "RIGHT"]
    assert "LEFT" not in adapter.actions
