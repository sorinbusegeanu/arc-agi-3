from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from v5_0.contact.policy import build_candidate_contact_trajectories_for_poi
from v5_0.contact.service import run_controlled_contact_for_episode
from v5_0.contracts.avatar_types import ContactPolicy
from v5_0.solve.policy_builder import build_adaptive_policy_for_target
from v5_0.solve.service import reanchor_frontier_objects, run_adaptive_solve_on_live_session


def _policy(pid: str, actions: tuple[str, ...]) -> ContactPolicy:
    return ContactPolicy(
        policy_id=pid,
        poi_id="p1",
        episode_index=0,
        planned_actions=actions,
        max_steps=max(1, len(actions) + 2),
        stop_on_contact=True,
        stop_on_screen_change=True,
        stop_on_terminal=True,
    )


def test_contact_policy_builder_returns_multiple_shortest_routes():
    selected_avatar = SimpleNamespace(selected_center=(0.0, 0.0))
    poi = SimpleNamespace(poi_id="p1", center=(2.0, 1.0), bbox=(2, 1, 2, 1), area=1, ambiguity_flags=())
    routes = build_candidate_contact_trajectories_for_poi(
        selected_avatar=selected_avatar,
        poi_candidate=poi,
        transitions=tuple(),
        episode_index=0,
    )
    shortest = [tuple(r.planned_actions) for r in routes if len(tuple(r.planned_actions)) == 3]
    assert len(shortest) >= 2
    assert ("DOWN", "RIGHT", "RIGHT") in shortest


def test_contact_service_tries_shortest_to_longest_and_selects_first_useful():
    selected_avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0))
    poi = SimpleNamespace(poi_id="p1", confidence=1.0, bbox=(2, 1, 2, 1), area=1, ambiguity_flags=())
    episode = SimpleNamespace(episode_index=0, transitions=tuple())
    report = SimpleNamespace(candidates=(poi,))

    p1 = _policy("r1", ("RIGHT", "RIGHT", "DOWN"))
    p2 = _policy("r2", ("DOWN", "RIGHT", "RIGHT"))

    def _partial(policy):
        return SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=policy,
            steps=tuple(),
            initial_poi_bbox=(2, 1, 2, 1),
            final_poi_bbox=(2, 1, 2, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=(1, 0, 1, 0),
        )

    with patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi", return_value=(p1, p2)), patch(
        "v5_0.contact.service.run_contact_policy",
        side_effect=lambda **kwargs: _partial(kwargs["policy"]),
    ), patch(
        "v5_0.contact.service.classify_contact_outcome",
        side_effect=[
            SimpleNamespace(
                outcome_type="no_effect",
                confidence=0.4,
                contact_step_index=None,
                reward_change_step_indices=tuple(),
                object_removed=False,
                new_object_appeared=False,
                level_transition=False,
                terminal=False,
                hud_change_only=False,
            ),
            SimpleNamespace(
                outcome_type="reward_change",
                confidence=0.9,
                contact_step_index=None,
                reward_change_step_indices=(0,),
                object_removed=False,
                new_object_appeared=False,
                level_transition=False,
                terminal=False,
                hud_change_only=False,
            ),
        ],
    ):
        tested = run_controlled_contact_for_episode(
            probe_episode=episode,
            poi_report=report,
            selected_avatar=selected_avatar,
            plan=SimpleNamespace(),
            seed=0,
            render_terminal=False,
            env_factory=None,
        )

    assert len(tested) == 1
    item = tested[0]
    assert item.policy.policy_id == "r2"
    assert item.route_evidence["attempted_route_ids"] == ("r1", "r2")
    assert item.route_evidence["winning_route_id"] == "r2"
    assert isinstance(item.route_evidence.get("route_hint"), dict)


def test_solve_policy_builder_prioritizes_contact_route_hint():
    selected_avatar = SimpleNamespace(selected_bbox=(0, 0, 0, 0), selected_center=(0.0, 0.0))
    poi = SimpleNamespace(center=(3.0, 0.0))
    hints = ({"actions": ("RIGHT", "RIGHT", "RIGHT"), "suggested_prefix_length": 2},)
    candidates = build_adaptive_policy_for_target(
        selected_avatar_result=selected_avatar,
        target_poi=poi,
        current_frame=((0, 0, 0, 0),),
        step_budget_remaining=6,
        route_hints=hints,
    )
    assert candidates
    assert tuple(getattr(candidates[0], "actions", ())) == ("RIGHT", "RIGHT")


def test_solve_service_switches_to_next_route_after_no_progress():
    class _SessionAdapter:
        def __init__(self):
            self.actions = []

        def get_current_observation(self, session):
            return SimpleNamespace(
                frame=(((0, 0), (0, 0)),),
                available_actions=("LEFT", "RIGHT"),
                levels_completed=0,
                raw_payload={"reward": 0.0},
            )

        def execute_action_prefix(self, session, translated, original):
            self.actions.append(original[0])
            return SimpleNamespace(step_results=(SimpleNamespace(action_legal=True),), terminal_status="running")

    class _ActionAdapter:
        def translate_token(self, action, context):
            return action

    session = SimpleNamespace(environment_metadata=SimpleNamespace(coordinate_action_id=0, coordinate_bounds=None))
    selected_avatar = SimpleNamespace(selected_bbox=(0, 0, 0, 0), selected_center=(0.0, 0.0), failure_reason=None, value_histogram={1: 1})
    ranked = (SimpleNamespace(poi_id="p1", bbox=(1, 1, 1, 1), center=(1.0, 1.0), value_histogram={2: 1}),)

    sess = _SessionAdapter()
    with patch("v5_0.solve.service.select_initial_target", return_value=SimpleNamespace(target_poi_id="p1", source="hud", confidence=1.0, attempt_count=0, last_outcome_type=None, active=True)), patch(
        "v5_0.solve.service.reanchor_frontier_objects",
        return_value=((0, 0, 0, 0), (1, 1, 1, 1)),
    ), patch(
        "v5_0.solve.service.build_adaptive_policy_for_target",
        return_value=(("RIGHT", "DOWN"), ("DOWN", "RIGHT")),
    ), patch(
        "v5_0.solve.service.track_avatar_bbox_in_frame",
        return_value=(0, 0, 0, 0),
    ), patch(
        "v5_0.solve.service.track_poi_bbox_in_frame",
        return_value=(1, 1, 1, 1),
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
            max_steps=3,
            session_adapter=sess,
            action_adapter=_ActionAdapter(),
        )

    assert sess.actions[:3] == ["RIGHT", "DOWN", "DOWN"]


def test_frontier_reanchor_uses_relaxed_local_recovery_before_failing():
    frame = ((0, 0, 0), (0, 1, 0), (0, 2, 0))
    selected_avatar = SimpleNamespace(selected_bbox=(0, 0, 0, 0), selected_center=(0.0, 0.0), value_histogram={1: 1})
    poi = SimpleNamespace(bbox=(2, 2, 2, 2), center=(2.0, 2.0), value_histogram={2: 1})
    with patch("v5_0.solve.service.track_avatar_bbox_in_frame", return_value=None), patch(
        "v5_0.solve.service.track_poi_bbox_in_frame", return_value=None
    ), patch(
        "v5_0.solve.service.reacquire_avatar_bbox_in_frame", return_value=(1, 1, 1, 1)
    ), patch(
        "v5_0.solve.service.reacquire_poi_bbox_in_frame", return_value=(1, 2, 1, 2)
    ):
        avatar_bbox, poi_bbox = reanchor_frontier_objects(
            frame=frame,
            selected_avatar=selected_avatar,
            initial_poi=poi,
        )
    assert avatar_bbox == (1, 1, 1, 1)
    assert poi_bbox == (1, 2, 1, 2)
