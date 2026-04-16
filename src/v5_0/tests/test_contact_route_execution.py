from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from v5_0.contact.policy_builder import build_contact_policies_for_poi
from v5_0.contact.service import (
    get_best_route_hint_for_poi,
    run_controlled_contact_for_episode,
    run_controlled_contact_multi_reset,
)
from v5_0.contracts.avatar_types import ContactOutcome


def _policy_result(policy_id: str, actions: tuple[str, ...], outcome: ContactOutcome):
    return SimpleNamespace(
        poi_id="p1",
        episode_index=0,
        policy=SimpleNamespace(policy_id=policy_id, planned_actions=actions),
        steps=tuple(),
        outcome=outcome,
        initial_poi_bbox=(2, 2, 2, 2),
        final_poi_bbox=(2, 2, 2, 2),
        initial_avatar_bbox=(0, 0, 0, 0),
        final_avatar_bbox=(1, 0, 1, 0),
        route_evidence={
            "route_hint": {
                "route_id": policy_id,
                "actions": actions,
                "length": len(actions),
                "contact_reached": outcome.contact_step_index is not None,
                "world_change_reached": outcome.outcome_type in {"reward_change", "object_removed", "door_opens", "level_transition", "terminal"},
                "outcome_type": outcome.outcome_type,
                "stop_step_index": outcome.contact_step_index,
                "suggested_prefix_length": (outcome.contact_step_index + 1 if outcome.contact_step_index is not None else len(actions)),
                "last_avatar_bbox": (1, 0, 1, 0),
                "last_poi_bbox": (2, 2, 2, 2),
                "best_post_frame_index": (outcome.contact_step_index if outcome.contact_step_index is not None else 0),
            }
        },
    )


def test_contact_policy_builder_returns_multiple_routes():
    avatar = SimpleNamespace(selected_center=(0.0, 0.0))
    poi = SimpleNamespace(poi_id="p1", center=(2.0, 1.0), bbox=(2, 1, 2, 1), area=1, ambiguity_flags=())
    policies = build_contact_policies_for_poi(avatar, poi, tuple(), 0)
    assert len(policies) >= 3


def test_contact_service_tries_routes_in_order_and_stops_on_first_useful():
    avatar = SimpleNamespace(failure_reason=None, selected_center=(0.0, 0.0))
    poi = SimpleNamespace(poi_id="p1", confidence=1.0, center=(2.0, 1.0), bbox=(2, 1, 2, 1), area=1, ambiguity_flags=())
    episode = SimpleNamespace(episode_index=0, transitions=tuple())
    report = SimpleNamespace(candidates=(poi,))

    p1 = SimpleNamespace(policy_id="p1", planned_actions=("RIGHT", "RIGHT", "DOWN"))
    p2 = SimpleNamespace(policy_id="p2", planned_actions=("DOWN", "RIGHT", "RIGHT"))
    order = []

    def _run_contact_policy(**kwargs):
        policy = kwargs["policy"]
        order.append(policy.policy_id)
        return SimpleNamespace(
            poi_id="p1",
            episode_index=0,
            policy=policy,
            steps=(
                SimpleNamespace(
                    blocked_action=False,
                    invalid_action=False,
                    screen_changed=True,
                    avatar_bbox_after=(1, 0, 1, 0),
                    poi_bbox_after=(2, 1, 2, 1),
                    avatar_reacquire_mode="reacquire",
                    poi_reacquire_mode="best_match",
                ),
            ),
            initial_poi_bbox=(2, 1, 2, 1),
            final_poi_bbox=(2, 1, 2, 1),
            initial_avatar_bbox=(0, 0, 0, 0),
            final_avatar_bbox=None,
        )

    with patch("v5_0.contact.service.build_candidate_contact_trajectories_for_poi", return_value=(p1, p2)), patch(
        "v5_0.contact.service.run_contact_policy", side_effect=_run_contact_policy
    ), patch(
        "v5_0.contact.service.classify_contact_outcome",
        side_effect=[
            ContactOutcome("no_effect", 0.2, None, tuple(), tuple(), False, False, False, False, False, tuple()),
            ContactOutcome("reward_change", 0.9, None, (0,), (0,), False, False, False, False, False, tuple()),
        ],
    ):
        tested = run_controlled_contact_for_episode(
            probe_episode=episode,
            poi_report=report,
            selected_avatar=avatar,
            plan=SimpleNamespace(),
            seed=0,
            render_terminal=False,
            env_factory=None,
        )

    assert order == ["p1", "p2"]
    assert len(tested) == 1
    assert tested[0].route_evidence["winning_route_id"] == "p2"
    hint = tested[0].route_evidence.get("route_hint")
    assert hint and hint.get("last_avatar_bbox") is not None and hint.get("last_poi_bbox") is not None
    attempted = tested[0].route_evidence.get("attempted_trajectories", [])
    assert attempted
    assert attempted[-1]["end_avatar_bbox"] is not None
    assert attempted[-1]["end_target_bbox"] is not None
    assert attempted[-1]["stop_reason"] in {"reward_change", "level_transition", "terminal", "contact_reached", "no_effect"}
    assert attempted[-1]["avatar_reacquire_mode"] in {"track", "reacquire", "best_match", "missing", None}
    assert attempted[-1]["target_reacquire_mode"] in {"track", "reacquire", "best_match", "missing", None}


def test_collapse_same_selected_poi_runs_single_real_contact_execution():
    avatar_multi = SimpleNamespace(
        episodes=(
            SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=SimpleNamespace(failure_reason=None))),
            SimpleNamespace(episode_index=1, report=SimpleNamespace(selected=SimpleNamespace(failure_reason=None))),
        )
    )
    poi = SimpleNamespace(poi_id="p1", confidence=1.0, bbox=(2, 1, 2, 1), area=1, ambiguity_flags=())
    poi_multi = {
        "episodes": (
            SimpleNamespace(episode_index=0, poi_report=SimpleNamespace(candidates=(poi,))),
            SimpleNamespace(episode_index=1, poi_report=SimpleNamespace(candidates=(poi,))),
        )
    }
    with patch("v5_0.contact.service.run_controlled_contact_for_episode", return_value=(
        _policy_result("r1", ("RIGHT",), ContactOutcome("reward_change", 0.8, None, (0,), (0,), False, False, False, False, False, tuple())),
    )) as run_one:
        report = run_controlled_contact_multi_reset(
            avatar_multi_report=avatar_multi,
            poi_multi_bundle=poi_multi,
            plan=SimpleNamespace(),
            base_seed=0,
            render_terminal=False,
            env_factory=None,
        )
    assert run_one.call_count == 1
    hint = get_best_route_hint_for_poi(report, "p1")
    assert hint is not None
    assert hint.get("last_avatar_bbox") is not None
    assert hint.get("last_poi_bbox") is not None
