from __future__ import annotations

from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.contracts.avatar_types import ProbePlan, ProbeTransitionRecord
from v5_0.replay.player import replay_prefix_to_frontier, replay_prefix_traces_to_frontier


def run_probe_session(
    *,
    plan: ProbePlan,
    seed: int = 0,
    render_terminal: bool = False,
    session_adapter: SessionAdapter | None = None,
    action_adapter: ActionAdapter | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[ProbeTransitionRecord, ...]:
    session_adapter = session_adapter or SessionAdapter()
    action_adapter = action_adapter or ActionAdapter()
    session = session_adapter.create_session(
        plan.game_id,
        seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    records: list[ProbeTransitionRecord] = []
    try:
        for step_index, action in enumerate(plan.action_sequence):
            pre_observation = session_adapter.get_current_observation(session)
            reward_before = _extract_reward(pre_observation.raw_payload)
            context = ActionTranslationContext(
                available_action_ids=pre_observation.available_actions,
                coordinate_action_id=session.environment_metadata.coordinate_action_id,
                coordinate_bounds=session.environment_metadata.coordinate_bounds,
            )
            try:
                translated = action_adapter.translate_token(action, context)
            except ValueError:
                records.append(
                    ProbeTransitionRecord(
                        step_index=step_index,
                        action=action,
                        pre_frame=_extract_frame_plane(pre_observation.frame),
                        post_frame=None,
                        invalid_action=True,
                        blocked_action=False,
                        terminal=False,
                        levels_completed_before=int(pre_observation.levels_completed),
                        levels_completed_after=int(pre_observation.levels_completed),
                        reward_before=reward_before,
                        reward_after=reward_before,
                    )
                )
                break

            executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
            post_observation = session_adapter.get_current_observation(session)
            blocked = any(not item.action_legal for item in executed.step_results)
            reward_after = _extract_reward(post_observation.raw_payload)
            terminal = executed.terminal_status in {"success", "failure"}
            records.append(
                ProbeTransitionRecord(
                    step_index=step_index,
                    action=action,
                    pre_frame=_extract_frame_plane(pre_observation.frame),
                    post_frame=_extract_frame_plane(post_observation.frame),
                    invalid_action=False,
                    blocked_action=blocked,
                    terminal=terminal,
                    levels_completed_before=int(pre_observation.levels_completed),
                    levels_completed_after=int(post_observation.levels_completed),
                    reward_before=reward_before,
                    reward_after=reward_after,
                )
            )
            if terminal or executed.level_transition:
                break
    finally:
        session_adapter.close_session(session)
    return tuple(records)


def run_probe_episodes(
    *,
    plan: ProbePlan,
    episode_count: int,
    base_seed: int = 0,
    render_terminal: bool = False,
    session_adapter: SessionAdapter | None = None,
    action_adapter: ActionAdapter | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[tuple[ProbeTransitionRecord, ...], ...]:
    outputs: list[tuple[ProbeTransitionRecord, ...]] = []
    for episode_index in range(max(0, int(episode_count))):
        outputs.append(
            run_probe_session(
                plan=plan,
                seed=int(base_seed) + episode_index,
                render_terminal=render_terminal,
                session_adapter=session_adapter,
                action_adapter=action_adapter,
                env_factory=env_factory,
            )
        )
    return tuple(outputs)


def run_probe_session_on_live_session(
    *,
    plan: ProbePlan,
    session,
    session_adapter: SessionAdapter,
    action_adapter: ActionAdapter,
) -> tuple[ProbeTransitionRecord, ...]:
    records: list[ProbeTransitionRecord] = []
    for step_index, action in enumerate(plan.action_sequence):
        pre_observation = session_adapter.get_current_observation(session)
        reward_before = _extract_reward(pre_observation.raw_payload)
        context = ActionTranslationContext(
            available_action_ids=pre_observation.available_actions,
            coordinate_action_id=session.environment_metadata.coordinate_action_id,
            coordinate_bounds=session.environment_metadata.coordinate_bounds,
        )
        try:
            translated = action_adapter.translate_token(action, context)
        except ValueError:
            records.append(
                ProbeTransitionRecord(
                    step_index=step_index,
                    action=action,
                    pre_frame=_extract_frame_plane(pre_observation.frame),
                    post_frame=None,
                    invalid_action=True,
                    blocked_action=False,
                    terminal=False,
                    levels_completed_before=int(pre_observation.levels_completed),
                    levels_completed_after=int(pre_observation.levels_completed),
                    reward_before=reward_before,
                    reward_after=reward_before,
                )
            )
            break

        executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
        post_observation = session_adapter.get_current_observation(session)
        blocked = any(not item.action_legal for item in executed.step_results)
        reward_after = _extract_reward(post_observation.raw_payload)
        terminal = executed.terminal_status in {"success", "failure"}
        records.append(
            ProbeTransitionRecord(
                step_index=step_index,
                action=action,
                pre_frame=_extract_frame_plane(pre_observation.frame),
                post_frame=_extract_frame_plane(post_observation.frame),
                invalid_action=False,
                blocked_action=blocked,
                terminal=terminal,
                levels_completed_before=int(pre_observation.levels_completed),
                levels_completed_after=int(post_observation.levels_completed),
                reward_before=reward_before,
                reward_after=reward_after,
            )
        )
        if terminal or executed.level_transition:
            break
    return tuple(records)


def run_probe_episodes_at_frontier(
    *,
    plan: ProbePlan,
    prefix_traces,
    episode_count: int,
    base_seed: int = 0,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[tuple[ProbeTransitionRecord, ...], ...]:
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    expected_level_index = int(str(plan.level_id).lstrip("L") or 0)
    outputs: list[tuple[ProbeTransitionRecord, ...]] = []
    for _episode_index in range(max(0, int(episode_count))):
        replay = replay_prefix_traces_to_frontier(
            game_id=plan.game_id,
            prefix_traces=tuple(prefix_traces or ()),
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        session = replay.get("session")
        reached = bool(replay.get("frontier_reached", False))
        reached_level = int(str(replay.get("frontier_level_id", "L0")).lstrip("L") or 0)
        if session is None or not reached or reached_level < expected_level_index:
            outputs.append(tuple())
            continue
        try:
            outputs.append(
                run_probe_session_on_live_session(
                    plan=plan,
                    session=session,
                    session_adapter=session_adapter,
                    action_adapter=action_adapter,
                )
            )
        finally:
            session_adapter.close_session(session)
    return tuple(outputs)


def run_probe_session_after_prefix(
    *,
    plan: ProbePlan,
    prefix_traces,
    seed: int = 0,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[ProbeTransitionRecord, ...]:
    del seed
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    expected_level_index = int(str(plan.level_id).lstrip("L") or 0)
    replay = replay_prefix_to_frontier(
        game_id=plan.game_id,
        prefix_traces=tuple(prefix_traces or ()),
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    session = replay.get("session")
    if session is None or not bool(replay.get("frontier_reached", False)):
        return tuple()
    reached_level = int(str(replay.get("frontier_level_id", "L0")).lstrip("L") or 0)
    if reached_level < expected_level_index:
        session_adapter.close_session(session)
        return tuple()
    try:
        return run_probe_session_on_live_session(
            plan=plan,
            session=session,
            session_adapter=session_adapter,
            action_adapter=action_adapter,
        )
    finally:
        session_adapter.close_session(session)


def run_probe_episodes_after_prefix(
    *,
    plan: ProbePlan,
    prefix_traces,
    episode_count: int,
    base_seed: int = 0,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> tuple[tuple[ProbeTransitionRecord, ...], ...]:
    outputs: list[tuple[ProbeTransitionRecord, ...]] = []
    for episode_index in range(max(0, int(episode_count))):
        outputs.append(
            run_probe_session_after_prefix(
                plan=plan,
                prefix_traces=tuple(prefix_traces or ()),
                seed=int(base_seed) + episode_index,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
        )
    return tuple(outputs)


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
