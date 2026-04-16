from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.contact.frame_tracker import (
    detect_contact,
    detect_hud_only_change,
    detect_screen_change,
    track_avatar_bbox_in_frame,
    track_poi_bbox_in_frame,
)
from v5_0.contact.outcome_classifier import classify_contact_outcome, is_useful_world_change
from v5_0.contracts.avatar_types import (
    AdaptiveEpisodeResult,
    AdaptiveStepRecord,
    AdaptiveTargetState,
    SolveEpisodeResult,
    SolveStepRecord,
    SolveTargetState,
)
from v5_0.mechanics.decision import update_target_after_step
from v5_0.solve.policy_builder import build_adaptive_policy_for_target, build_solve_policy_for_target
from v5_0.solve.target_selector import select_initial_target, select_next_target


def run_solve_episode(
    *,
    game_id,
    plan,
    episode_index,
    initial_target_state,
    selected_avatar,
    ranked_poi_candidates,
    hud_targeting_report=None,
    contact_experiment_report=None,
    mechanic_report=None,
    seed,
    render_terminal,
    env_factory,
    max_steps,
    skip_bootstrap_replay_in_final_solve: bool = False,
) -> SolveEpisodeResult:
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    session = session_adapter.create_session(
        game_id,
        seed=int(seed),
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )

    steps: list[SolveStepRecord] = []
    initial_target = initial_target_state or SolveTargetState(
        target_poi_id=None,
        source="none",
        confidence=0.0,
        attempt_count=0,
        last_outcome_type=None,
        active=False,
    )
    current_target = initial_target
    solved = False
    failure_reason: str | None = None
    blocked_streak = 0

    poi_by_id = {item.poi_id: item for item in tuple(ranked_poi_candidates or ())}

    try:
        if not bool(skip_bootstrap_replay_in_final_solve):
            bootstrap_steps = _replay_bootstrap(
                plan,
                session,
                session_adapter,
                action_adapter,
            )
            if bootstrap_steps:
                steps.extend(list(bootstrap_steps))

        for frontier_index in range(max(0, int(max_steps))):
            step_index = len(steps)
            if current_target is None or current_target.target_poi_id is None:
                failure_reason = "no_valid_remaining_target"
                break

            target_poi = poi_by_id.get(str(current_target.target_poi_id))
            if target_poi is None:
                current_target = select_next_target(
                    current_target,
                    ranked_poi_candidates,
                    tuple(steps),
                    contact_experiment_report,
                    mechanic_report=mechanic_report,
                )
                if current_target is None:
                    failure_reason = "no_valid_remaining_target"
                    break
                target_poi = poi_by_id.get(str(current_target.target_poi_id))
                if target_poi is None:
                    failure_reason = "no_valid_remaining_target"
                    break

            pre_obs = session_adapter.get_current_observation(session)
            pre_frame = _extract_frame_plane(pre_obs.frame)
            reward_before = _extract_reward(pre_obs.raw_payload)
            avatar_before = track_avatar_bbox_in_frame(pre_frame, selected_avatar.selected_bbox, None)
            target_before = track_poi_bbox_in_frame(pre_frame, target_poi.bbox, target_poi.value_histogram)

            policy = build_solve_policy_for_target(
                selected_avatar,
                target_poi,
                pre_frame,
                int(max_steps) - int(frontier_index),
            )
            if not policy:
                failure_reason = "empty_policy"
                break
            action = policy[0]

            translated, invalid = _translate_action(session, action, action_adapter, session_adapter)
            if invalid:
                step = SolveStepRecord(
                    step_index=int(step_index),
                    action=str(action),
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
                    target_poi_id=current_target.target_poi_id,
                    target_bbox_before=target_before,
                    target_bbox_after=target_before,
                    contact_detected=False,
                    screen_changed=False,
                    hud_changed_only=False,
                    outcome_type="no_effect",
                    source="frontier_solve",
                )
                steps.append(step)
                blocked_streak = 0
            else:
                executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
                post_obs = session_adapter.get_current_observation(session)
                post_frame = _extract_frame_plane(post_obs.frame)
                reward_after = _extract_reward(post_obs.raw_payload)
                blocked = any(not item.action_legal for item in executed.step_results)
                terminal = executed.terminal_status in {"success", "failure"}

                avatar_after = track_avatar_bbox_in_frame(post_frame, avatar_before, None)
                target_after = track_poi_bbox_in_frame(post_frame, target_before, target_poi.value_histogram)
                screen_changed = detect_screen_change(pre_frame, post_frame)
                hud_only = detect_hud_only_change(pre_frame, post_frame)
                contact = detect_contact(avatar_after, target_after)

                partial = SimpleNamespace(
                    steps=(
                        SimpleNamespace(
                            step_index=int(step_index),
                            screen_changed=bool(screen_changed),
                            reward_before=reward_before,
                            reward_after=reward_after,
                            levels_completed_before=int(pre_obs.levels_completed),
                            levels_completed_after=int(post_obs.levels_completed),
                            terminal=bool(terminal),
                            hud_changed_only=bool(hud_only),
                            contact_detected=bool(contact),
                            poi_bbox_before=target_before,
                            poi_bbox_after=target_after,
                            pre_frame=pre_frame,
                            post_frame=post_frame,
                            avatar_bbox_after=avatar_after,
                        ),
                    )
                )
                outcome = classify_contact_outcome(partial, target_poi, selected_avatar)
                outcome_type = str(outcome.outcome_type)

                step = SolveStepRecord(
                    step_index=int(step_index),
                    action=str(action),
                    pre_frame=pre_frame,
                    post_frame=post_frame,
                    invalid_action=False,
                    blocked_action=bool(blocked),
                    terminal=bool(terminal),
                    levels_completed_before=int(pre_obs.levels_completed),
                    levels_completed_after=int(post_obs.levels_completed),
                    reward_before=reward_before,
                    reward_after=reward_after,
                    avatar_bbox_before=avatar_before,
                    avatar_bbox_after=avatar_after,
                    target_poi_id=current_target.target_poi_id,
                    target_bbox_before=target_before,
                    target_bbox_after=target_after,
                    contact_detected=bool(contact),
                    screen_changed=bool(screen_changed),
                    hud_changed_only=bool(hud_only),
                    outcome_type=outcome_type,
                    source="frontier_solve",
                )
                steps.append(step)

                if blocked:
                    blocked_streak += 1
                else:
                    blocked_streak = 0

                level_transition = int(post_obs.levels_completed) > int(pre_obs.levels_completed)
                terminal_success = terminal and str(executed.terminal_status) == "success"
                # Continue across level transitions; stop only when the game reaches a terminal success/failure.
                if terminal_success:
                    solved = True
                    failure_reason = None
                    break
                if terminal and not terminal_success:
                    failure_reason = "terminal_failure"
                    break

                useful_change = is_useful_world_change(outcome)
                repeated_no_effect = _tail_count(
                    [item for item in steps if item.target_poi_id == current_target.target_poi_id],
                    lambda s: s.outcome_type in {"no_effect", "hud_change_only"},
                ) >= 3
                target_disappeared = target_before is not None and target_after is None and not useful_change
                should_retarget = bool(target_disappeared or blocked_streak >= 2 or repeated_no_effect)
                if not should_retarget and useful_change and not level_transition and not terminal:
                    current_target = SolveTargetState(
                        target_poi_id=current_target.target_poi_id,
                        source=current_target.source,
                        confidence=current_target.confidence,
                        attempt_count=int(current_target.attempt_count) + 1,
                        last_outcome_type=outcome_type,
                        active=True,
                    )
                    continue
                if should_retarget:
                    mech_decision = update_target_after_step(
                        current_target.target_poi_id,
                        tuple(steps),
                        blocked_streak,
                    )
                    next_target = select_next_target(
                        current_target,
                        ranked_poi_candidates,
                        tuple(steps),
                        contact_experiment_report,
                        mechanic_report=mechanic_report,
                    )
                    if not bool(getattr(mech_decision, "retarget_required", False)):
                        next_target = current_target
                    if next_target is None:
                        failure_reason = "no_valid_remaining_target"
                        break
                    current_target = next_target
                    continue

            current_target = SolveTargetState(
                target_poi_id=current_target.target_poi_id,
                source=current_target.source,
                confidence=current_target.confidence,
                attempt_count=int(current_target.attempt_count) + 1,
                last_outcome_type=steps[-1].outcome_type if steps else None,
                active=True,
            )

        else:
            failure_reason = "step_budget_exhausted"

    finally:
        session_adapter.close_session(session)

    final_target = current_target or SolveTargetState(
        target_poi_id=None,
        source="none",
        confidence=0.0,
        attempt_count=0,
        last_outcome_type=None,
        active=False,
    )

    if not solved and failure_reason is None:
        failure_reason = "no_progress"

    return SolveEpisodeResult(
        episode_index=int(episode_index),
        initial_target=initial_target,
        steps=tuple(steps),
        final_target=final_target,
        solved=bool(solved),
        failure_reason=None if solved else failure_reason,
    )


def _replay_bootstrap_with_trace(
    *,
    plan,
    session,
    session_adapter,
    action_adapter,
    episode_index,
) -> tuple[SolveStepRecord, ...]:
    steps: list[SolveStepRecord] = []
    for action in plan.action_sequence:
        step_index = len(steps)
        pre_obs = session_adapter.get_current_observation(session)
        pre_frame = _extract_frame_plane(pre_obs.frame)
        reward_before = _extract_reward(pre_obs.raw_payload)
        translated, invalid = _translate_action(session, action, action_adapter, session_adapter)
        if invalid:
            steps.append(
                SolveStepRecord(
                    step_index=int(step_index),
                    action=str(action),
                    pre_frame=pre_frame,
                    post_frame=None,
                    invalid_action=True,
                    blocked_action=False,
                    terminal=False,
                    levels_completed_before=int(pre_obs.levels_completed),
                    levels_completed_after=int(pre_obs.levels_completed),
                    reward_before=reward_before,
                    reward_after=reward_before,
                    avatar_bbox_before=None,
                    avatar_bbox_after=None,
                    target_poi_id=None,
                    target_bbox_before=None,
                    target_bbox_after=None,
                    contact_detected=False,
                    screen_changed=False,
                    hud_changed_only=False,
                    outcome_type="bootstrap_replay",
                    source="bootstrap_replay",
                )
            )
            break
        executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
        post_obs = session_adapter.get_current_observation(session)
        post_frame = _extract_frame_plane(post_obs.frame)
        reward_after = _extract_reward(post_obs.raw_payload)
        blocked = any(not item.action_legal for item in tuple(getattr(executed, "step_results", ())))
        terminal = executed.terminal_status in {"success", "failure"}
        steps.append(
            SolveStepRecord(
                step_index=int(step_index),
                action=str(action),
                pre_frame=pre_frame,
                post_frame=post_frame,
                invalid_action=False,
                blocked_action=bool(blocked),
                terminal=bool(terminal),
                levels_completed_before=int(pre_obs.levels_completed),
                levels_completed_after=int(post_obs.levels_completed),
                reward_before=reward_before,
                reward_after=reward_after,
                avatar_bbox_before=None,
                avatar_bbox_after=None,
                target_poi_id=None,
                target_bbox_before=None,
                target_bbox_after=None,
                contact_detected=False,
                screen_changed=bool(pre_frame != post_frame),
                hud_changed_only=detect_hud_only_change(pre_frame, post_frame),
                outcome_type="bootstrap_replay",
                source="bootstrap_replay",
            )
        )
        if terminal or bool(getattr(executed, "level_transition", False)):
            break
    return tuple(steps)


def _replay_bootstrap(plan, session, session_adapter, action_adapter):
    return _replay_bootstrap_with_trace(
        plan=plan,
        session=session,
        session_adapter=session_adapter,
        action_adapter=action_adapter,
        episode_index=0,
    )


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


def _extract_frame_plane(frame: Any) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(frame, tuple) or not frame:
        return None
    plane = frame[0]
    if not isinstance(plane, tuple):
        return None
    rows: list[tuple[int, ...]] = []
    for row in plane:
        if not isinstance(row, tuple):
            return None
        rows.append(tuple(int(value) for value in row))
    return tuple(rows)


def _extract_reward(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("reward")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tail_count(items, predicate) -> int:
    count = 0
    for item in reversed(tuple(items)):
        if not predicate(item):
            break
        count += 1
    return count


def run_adaptive_solve_episode(
    *,
    game_id,
    plan,
    episode_index,
    selected_avatar,
    ranked_poi_candidates,
    hud_targeting_report,
    contact_experiment_report=None,
    seed,
    render_terminal,
    env_factory,
    max_steps,
    skip_bootstrap_replay_in_final_solve: bool = False,
) -> AdaptiveEpisodeResult:
    initial = select_initial_target(
        hud_targeting_report,
        ranked_poi_candidates,
        contact_experiment_report=contact_experiment_report,
    )
    if initial is None or initial.target_poi_id is None:
        return AdaptiveEpisodeResult(
            episode_index=int(episode_index),
            target_sequence=(),
            steps=(),
            solved=False,
            failure_reason="no_target_selected",
        )

    base = run_solve_episode(
        game_id=game_id,
        plan=plan,
        episode_index=episode_index,
        initial_target_state=initial,
        selected_avatar=selected_avatar,
        ranked_poi_candidates=ranked_poi_candidates,
        hud_targeting_report=hud_targeting_report,
        contact_experiment_report=contact_experiment_report,
        seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
        max_steps=max_steps,
        skip_bootstrap_replay_in_final_solve=bool(skip_bootstrap_replay_in_final_solve),
    )

    target_sequence: list[AdaptiveTargetState] = []
    seen_targets: list[str] = []
    for step in base.steps:
        pid = str(step.target_poi_id) if step.target_poi_id is not None else None
        if pid is None:
            continue
        if not seen_targets or seen_targets[-1] != pid:
            seen_targets.append(pid)
            target_sequence.append(
                AdaptiveTargetState(
                    target_poi_id=pid,
                    source="sequence",
                    confidence=1.0 if pid == initial.target_poi_id else 0.7,
                    attempt_count=sum(1 for s in base.steps if s.target_poi_id == pid),
                    last_outcome_type=next((s.outcome_type for s in reversed(base.steps) if s.target_poi_id == pid), None),
                    active=True,
                )
            )
    if not target_sequence:
        target_sequence.append(
            AdaptiveTargetState(
                target_poi_id=initial.target_poi_id,
                source=initial.source,
                confidence=float(initial.confidence),
                attempt_count=int(initial.attempt_count),
                last_outcome_type=initial.last_outcome_type,
                active=bool(initial.active),
            )
        )

    adaptive_steps: list[AdaptiveStepRecord] = []
    prev_target = None
    for step in base.steps:
        pid = str(step.target_poi_id) if step.target_poi_id is not None else None
        retargeted = bool(prev_target is not None and pid is not None and prev_target != pid)
        adaptive_steps.append(
            AdaptiveStepRecord(
                step_index=int(step.step_index),
                action=str(step.action),
                pre_frame=step.pre_frame,
                post_frame=step.post_frame,
                invalid_action=bool(step.invalid_action),
                blocked_action=bool(step.blocked_action),
                terminal=bool(step.terminal),
                levels_completed_before=int(step.levels_completed_before),
                levels_completed_after=int(step.levels_completed_after),
                reward_before=step.reward_before,
                reward_after=step.reward_after,
                avatar_bbox_before=step.avatar_bbox_before,
                avatar_bbox_after=step.avatar_bbox_after,
                target_poi_id=pid,
                target_bbox_before=step.target_bbox_before,
                target_bbox_after=step.target_bbox_after,
                contact_detected=bool(step.contact_detected),
                screen_changed=bool(step.screen_changed),
                hud_changed_only=bool(step.hud_changed_only),
                outcome_type=str(step.outcome_type),
                retargeted=retargeted,
                source=str(getattr(step, "source", "frontier_solve")),
            )
        )
        if pid is not None:
            prev_target = pid

    return AdaptiveEpisodeResult(
        episode_index=int(base.episode_index),
        target_sequence=tuple(target_sequence),
        steps=tuple(adaptive_steps),
        solved=bool(base.solved),
        failure_reason=base.failure_reason,
    )
