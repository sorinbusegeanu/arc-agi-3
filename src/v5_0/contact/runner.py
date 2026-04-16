from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.contact.frame_tracker import (
    detect_contact,
    detect_hud_only_change,
    detect_screen_change,
    find_best_component_match_in_frame,
    reacquire_avatar_bbox_in_frame,
    reacquire_poi_bbox_in_frame,
    track_avatar_bbox_in_frame,
    track_poi_bbox_in_frame,
)
from v5_0.contracts.avatar_types import (
    ContactPolicy,
    ContactStepRecord,
    POICandidate,
    ProbePlan,
    TestedPOIResult,
)


def run_contact_policy(
    *,
    plan: ProbePlan,
    policy: ContactPolicy,
    seed: int,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
    selected_avatar,
    poi_candidate: POICandidate,
    session_adapter: SessionAdapter | None = None,
    action_adapter: ActionAdapter | None = None,
) -> TestedPOIResult:
    session_adapter = session_adapter or SessionAdapter()
    action_adapter = action_adapter or ActionAdapter()
    session = session_adapter.create_session(
        plan.game_id,
        seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )

    try:
        _replay_bootstrap(plan, session, session_adapter, action_adapter)
        avatar_bbox = selected_avatar.selected_bbox
        poi_bbox = poi_candidate.bbox

        steps: list[ContactStepRecord] = []
        for index, action in enumerate(policy.planned_actions[: policy.max_steps]):
            pre_obs = session_adapter.get_current_observation(session)
            pre_frame = _extract_frame(pre_obs.frame)
            avatar_before = track_avatar_bbox_in_frame(pre_frame, avatar_bbox, None)
            poi_before = track_poi_bbox_in_frame(pre_frame, poi_bbox, poi_candidate.value_histogram)
            reward_before = _extract_reward(pre_obs.raw_payload)

            translated, invalid = _translate_action(session, action, action_adapter, session_adapter)
            if invalid:
                record = ContactStepRecord(
                    step_index=index,
                    action=action,
                    pre_frame=pre_frame,
                    post_frame=None,
                    invalid_action=True,
                    blocked_action=False,
                    terminal=False,
                    levels_completed_before=int(pre_obs.levels_completed),
                    levels_completed_after=int(pre_obs.levels_completed),
                    reward_before=reward_before,
                    reward_after=reward_before,
                    avatar_bbox_before=avatar_before,
                    avatar_bbox_after=avatar_before,
                    poi_bbox_before=poi_before,
                    poi_bbox_after=poi_before,
                    screen_changed=False,
                    hud_changed_only=False,
                    contact_detected=detect_contact(avatar_before, poi_before),
                )
                steps.append(record)
                break

            executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
            post_obs = session_adapter.get_current_observation(session)
            post_frame = _extract_frame(post_obs.frame)
            reward_after = _extract_reward(post_obs.raw_payload)
            blocked = any(not item.action_legal for item in executed.step_results)
            terminal = executed.terminal_status in {"success", "failure"}

            avatar_after = track_avatar_bbox_in_frame(post_frame, avatar_before, None, frontier_reanchor=False)
            avatar_mode = "track" if avatar_after is not None else "missing"
            poi_after = track_poi_bbox_in_frame(post_frame, poi_before, poi_candidate.value_histogram, frontier_reanchor=False)
            poi_mode = "track" if poi_after is not None else "missing"
            if avatar_after is None:
                avatar_after = reacquire_avatar_bbox_in_frame(
                    post_frame,
                    avatar_before or avatar_bbox,
                    None,
                    preferred_center=_center_from_bbox(avatar_before or avatar_bbox),
                    recent_bbox=avatar_before,
                )
                if avatar_after is not None:
                    avatar_mode = "reacquire"
            if poi_after is None:
                poi_after = reacquire_poi_bbox_in_frame(
                    post_frame,
                    poi_before or poi_bbox,
                    poi_candidate.value_histogram,
                    preferred_center=_center_from_bbox(poi_before or poi_bbox),
                    recent_bbox=poi_before,
                )
                if poi_after is not None:
                    poi_mode = "reacquire"
            if avatar_after is None:
                avatar_after = find_best_component_match_in_frame(
                    frame=post_frame,
                    reference_bbox=avatar_before or avatar_bbox,
                    reference_histogram=None,
                    for_poi=False,
                    preferred_center=_center_from_bbox(avatar_before or avatar_bbox),
                    recent_bbox=avatar_before,
                )
                if avatar_after is not None:
                    avatar_mode = "best_match"
            if poi_after is None:
                poi_after = find_best_component_match_in_frame(
                    frame=post_frame,
                    reference_bbox=poi_before or poi_bbox,
                    reference_histogram=poi_candidate.value_histogram,
                    for_poi=True,
                    preferred_center=_center_from_bbox(poi_before or poi_bbox),
                    recent_bbox=poi_before,
                )
                if poi_after is not None:
                    poi_mode = "best_match"
            screen_changed = detect_screen_change(pre_frame, post_frame)
            hud_only = detect_hud_only_change(pre_frame, post_frame)
            contact = detect_contact(avatar_after, poi_after)

            record = ContactStepRecord(
                step_index=index,
                action=action,
                pre_frame=pre_frame,
                post_frame=post_frame,
                invalid_action=False,
                blocked_action=blocked,
                terminal=terminal,
                levels_completed_before=int(pre_obs.levels_completed),
                levels_completed_after=int(post_obs.levels_completed),
                reward_before=reward_before,
                reward_after=reward_after,
                avatar_bbox_before=avatar_before,
                avatar_bbox_after=avatar_after,
                poi_bbox_before=poi_before,
                poi_bbox_after=poi_after,
                screen_changed=screen_changed,
                hud_changed_only=hud_only,
                contact_detected=contact,
                avatar_reacquire_mode=avatar_mode,
                poi_reacquire_mode=poi_mode,
            )
            steps.append(record)

            avatar_bbox = avatar_after
            poi_bbox = poi_after
            if policy.stop_on_terminal and terminal:
                break
            if policy.stop_on_screen_change and screen_changed:
                break
            if policy.stop_on_contact and contact:
                break

        return TestedPOIResult(
            poi_id=poi_candidate.poi_id,
            episode_index=policy.episode_index,
            policy=policy,
            steps=tuple(steps),
            outcome=_placeholder_outcome(),
            initial_poi_bbox=poi_candidate.bbox,
            final_poi_bbox=poi_bbox,
            initial_avatar_bbox=selected_avatar.selected_bbox,
            final_avatar_bbox=avatar_bbox,
        )
    finally:
        session_adapter.close_session(session)


def _placeholder_outcome():
    from v5_0.contracts.avatar_types import ContactOutcome

    return ContactOutcome(
        outcome_type="no_effect",
        confidence=0.0,
        contact_step_index=None,
        screen_change_step_indices=(),
        reward_change_step_indices=(),
        object_removed=False,
        new_object_appeared=False,
        level_transition=False,
        terminal=False,
        hud_change_only=False,
        notes=(),
    )


def _replay_bootstrap(plan, session, session_adapter, action_adapter):
    for action in plan.action_sequence:
        translated, invalid = _translate_action(session, action, action_adapter, session_adapter)
        if invalid:
            break
        executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
        if executed.terminal_status in {"success", "failure"} or executed.level_transition:
            break


def _translate_action(session, action, action_adapter, session_adapter):
    observation = session_adapter.get_current_observation(session)
    context = ActionTranslationContext(
        available_action_ids=observation.available_actions,
        coordinate_action_id=session.environment_metadata.coordinate_action_id,
        coordinate_bounds=session.environment_metadata.coordinate_bounds,
    )
    try:
        translated = action_adapter.translate_token(action, context)
        return translated, False
    except ValueError:
        return None, True


def _extract_frame(frame):
    if not isinstance(frame, tuple) or not frame or not isinstance(frame[0], tuple):
        return None
    return tuple(tuple(int(v) for v in row) for row in frame[0])


def _extract_reward(payload):
    if not isinstance(payload, dict):
        return None
    value = payload.get("reward")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _center_from_bbox(bbox):
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
