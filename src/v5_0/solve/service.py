from __future__ import annotations

from collections import Counter
import hashlib
import time
from typing import Any, Callable

from v5_0.contracts.avatar_types import (
    AdaptiveDiagnostics,
    AdaptiveEpisodeResult,
    AdaptiveSolveReport,
    AdaptiveStepRecord,
    AdaptiveTargetState,
    SavedLevelTrace,
    LevelSolution,
    LevelSolveAction,
    SolveDiagnostics,
    SolveReport,
    SolveTargetState,
    TrajectoryAttemptRecord,
    TrajectoryCandidateRecord,
    TrajectoryStatsReport,
)
from v5_0.replay.player import replay_prefix_traces_to_frontier, replay_saved_trace
from v5_0.mechanics.service import build_mechanic_report
from v5_0.solve.policy_builder import build_adaptive_policy_for_target
from v5_0.solve.loop_runner import run_adaptive_solve_episode, run_solve_episode
from v5_0.solve.target_selector import select_initial_target, select_next_target
from v5_0.contact.frame_tracker import (
    detect_contact,
    detect_hud_only_change,
    detect_screen_change,
    detect_screen_change_outside_hud_mask,
    find_best_component_match_in_frame,
    reacquire_avatar_bbox_in_frame,
    reacquire_poi_bbox_in_frame,
    track_avatar_bbox_in_frame,
    track_poi_bbox_in_frame,
)
from v5_0.contact.outcome_classifier import is_useful_world_change
from v5_0.contact.service import get_best_route_hint_for_poi
from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.replay.player import replay_trace_at_frontier
from v5_0.memory.trace_store import save_trace_history_row, upsert_verified_best_trace
from v5_0.route.trajectory_enumerator import compute_action_space_delta, validate_route_actions_for_action_delta


def run_closed_loop_solve_multi_reset(
    *,
    avatar_multi_report,
    poi_multi_bundle,
    hud_targeting_report,
    contact_experiment_report=None,
    game_id,
    plan,
    base_seed,
    render_terminal,
    env_factory: Callable[[], Any] | None,
    max_steps,
) -> SolveReport:
    if getattr(avatar_multi_report.selected, "failure_reason", None) is not None or not bool(getattr(avatar_multi_report.diagnostics, "stable_avatar_found", False)):
        diagnostics = SolveDiagnostics(
            episode_count=0,
            solved_episode_count=0,
            failed_episode_count=0,
            failure_reason_counts={"no_stable_avatar": 1},
            retarget_count=0,
            level_transition_count=0,
            terminal_success_count=0,
            terminal_failure_count=0,
            step_budget_exhausted_count=0,
        )
        return SolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_stable_avatar")

    poi_report = poi_multi_bundle.get("report") if isinstance(poi_multi_bundle, dict) else None
    ranked_pois = tuple(getattr(poi_report, "candidates", ()))
    if not ranked_pois:
        diagnostics = SolveDiagnostics(
            episode_count=0,
            solved_episode_count=0,
            failed_episode_count=0,
            failure_reason_counts={"no_poi_candidate": 1},
            retarget_count=0,
            level_transition_count=0,
            terminal_success_count=0,
            terminal_failure_count=0,
            step_budget_exhausted_count=0,
        )
        return SolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_poi_candidate")

    mechanic_report = build_mechanic_report(
        ranked_pois,
        contact_experiment_report=contact_experiment_report,
        hud_targeting_report=hud_targeting_report,
        solve_report=None,
        previous_mechanic_memory=None,
    )

    initial_target = select_initial_target(
        hud_targeting_report,
        ranked_pois,
        contact_experiment_report=contact_experiment_report,
        mechanic_report=mechanic_report,
    )
    if initial_target is None or initial_target.target_poi_id is None:
        diagnostics = SolveDiagnostics(
            episode_count=0,
            solved_episode_count=0,
            failed_episode_count=0,
            failure_reason_counts={"no_target_selected": 1},
            retarget_count=0,
            level_transition_count=0,
            terminal_success_count=0,
            terminal_failure_count=0,
            step_budget_exhausted_count=0,
        )
        return SolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_target_selected")

    episodes = []
    failure_counts = Counter()
    retarget_count = 0
    level_transition_count = 0
    terminal_success_count = 0
    terminal_failure_count = 0
    step_budget_exhausted_count = 0

    for probe_episode in tuple(getattr(avatar_multi_report, "episodes", ())):
        selected_avatar = probe_episode.report.selected
        if selected_avatar.failure_reason is not None:
            continue
        result = run_solve_episode(
            game_id=game_id,
            plan=plan,
            episode_index=int(probe_episode.episode_index),
            initial_target_state=initial_target,
            selected_avatar=selected_avatar,
            ranked_poi_candidates=ranked_pois,
            hud_targeting_report=hud_targeting_report,
            contact_experiment_report=contact_experiment_report,
            mechanic_report=mechanic_report,
            seed=int(base_seed) + int(probe_episode.episode_index),
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )
        episodes.append(result)
        mechanic_report = build_mechanic_report(
            ranked_pois,
            contact_experiment_report=contact_experiment_report,
            hud_targeting_report=hud_targeting_report,
            solve_report=SolveReport(
                episodes=tuple(episodes),
                diagnostics=SolveDiagnostics(
                    episode_count=len(episodes),
                    solved_episode_count=sum(1 for ep in episodes if ep.solved),
                    failed_episode_count=sum(1 for ep in episodes if not ep.solved),
                    failure_reason_counts={},
                    retarget_count=0,
                    level_transition_count=0,
                    terminal_success_count=0,
                    terminal_failure_count=0,
                    step_budget_exhausted_count=0,
                ),
                selected_target_id=initial_target.target_poi_id,
                solved=any(ep.solved for ep in episodes),
                failure_reason=None,
            ),
            previous_mechanic_memory=getattr(mechanic_report, "memory", None),
        )

        if result.failure_reason is not None:
            failure_counts[str(result.failure_reason)] += 1
        step_budget_exhausted_count += 1 if result.failure_reason == "step_budget_exhausted" else 0
        terminal_failure_count += 1 if result.failure_reason == "terminal_failure" else 0
        retarget_count += sum(1 for step in result.steps if str(step.target_poi_id) != str(result.initial_target.target_poi_id))
        level_transition_count += sum(1 for step in result.steps if int(step.levels_completed_after) > int(step.levels_completed_before))
        terminal_success_count += sum(1 for step in result.steps if bool(step.terminal) and step.outcome_type in {"terminal", "level_transition"})

    solved_episode_count = sum(1 for item in episodes if item.solved)
    failed_episode_count = max(0, len(episodes) - solved_episode_count)
    solved = solved_episode_count > 0

    failure_reason = None
    if not episodes:
        failure_reason = "no_progress"
    elif not solved:
        if failure_counts:
            failure_reason = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        else:
            failure_reason = "no_progress"

    diagnostics = SolveDiagnostics(
        episode_count=len(episodes),
        solved_episode_count=solved_episode_count,
        failed_episode_count=failed_episode_count,
        failure_reason_counts=dict(sorted((str(k), int(v)) for k, v in failure_counts.items())),
        retarget_count=int(retarget_count),
        level_transition_count=int(level_transition_count),
        terminal_success_count=int(terminal_success_count),
        terminal_failure_count=int(terminal_failure_count),
        step_budget_exhausted_count=int(step_budget_exhausted_count),
    )
    return SolveReport(
        episodes=tuple(episodes),
        diagnostics=diagnostics,
        selected_target_id=str(initial_target.target_poi_id),
        solved=bool(solved),
        failure_reason=failure_reason,
    )


def run_adaptive_solve_multi_reset(
    *,
    avatar_multi_report,
    poi_multi_bundle,
    hud_targeting_report,
    contact_experiment_report=None,
    game_id,
    plan,
    base_seed,
    render_terminal,
    env_factory: Callable[[], Any] | None,
    max_steps,
) -> AdaptiveSolveReport:
    if getattr(avatar_multi_report.selected, "failure_reason", None) is not None or not bool(getattr(avatar_multi_report.diagnostics, "stable_avatar_found", False)):
        diagnostics = AdaptiveDiagnostics(
            episode_count=0,
            solved_episode_count=0,
            failed_episode_count=0,
            retarget_count=0,
            target_switch_count=0,
            useful_change_count=0,
            no_progress_count=0,
            level_transition_count=0,
            terminal_count=0,
            step_budget_exhausted_count=0,
            failure_reason_counts={"no_stable_avatar": 1},
        )
        return AdaptiveSolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_stable_avatar")

    poi_report = poi_multi_bundle.get("report") if isinstance(poi_multi_bundle, dict) else None
    ranked_pois = tuple(getattr(poi_report, "candidates", ()))
    if not ranked_pois:
        diagnostics = AdaptiveDiagnostics(
            episode_count=0,
            solved_episode_count=0,
            failed_episode_count=0,
            retarget_count=0,
            target_switch_count=0,
            useful_change_count=0,
            no_progress_count=0,
            level_transition_count=0,
            terminal_count=0,
            step_budget_exhausted_count=0,
            failure_reason_counts={"no_poi_candidate": 1},
        )
        return AdaptiveSolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_poi_candidate")

    episodes = []
    failure_counts = Counter()
    retarget_count = 0
    target_switch_count = 0
    useful_change_count = 0
    no_progress_count = 0
    level_transition_count = 0
    terminal_count = 0
    step_budget_exhausted_count = 0

    for probe_episode in tuple(getattr(avatar_multi_report, "episodes", ())):
        selected_avatar = probe_episode.report.selected
        if selected_avatar.failure_reason is not None:
            continue
        result = run_adaptive_solve_episode(
            game_id=game_id,
            plan=plan,
            episode_index=int(probe_episode.episode_index),
            selected_avatar=selected_avatar,
            ranked_poi_candidates=ranked_pois,
            hud_targeting_report=hud_targeting_report,
            contact_experiment_report=contact_experiment_report,
            seed=int(base_seed) + int(probe_episode.episode_index),
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )
        episodes.append(result)
        if result.failure_reason is not None:
            failure_counts[str(result.failure_reason)] += 1
            if result.failure_reason == "no_progress":
                no_progress_count += 1
            if result.failure_reason == "step_budget_exhausted":
                step_budget_exhausted_count += 1
        for step in result.steps:
            if bool(step.retargeted):
                retarget_count += 1
            if str(step.outcome_type) in {"reward_change", "object_removed", "door_opens", "level_transition", "terminal"}:
                useful_change_count += 1
            if int(step.levels_completed_after) > int(step.levels_completed_before):
                level_transition_count += 1
            if bool(step.terminal):
                terminal_count += 1
        target_switch_count += max(0, len(result.target_sequence) - 1)

    solved_episode_count = sum(1 for ep in episodes if ep.solved)
    failed_episode_count = max(0, len(episodes) - solved_episode_count)
    solved = solved_episode_count > 0
    failure_reason = None
    if not episodes:
        failure_reason = "no_progress"
    elif not solved:
        if failure_counts:
            failure_reason = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        else:
            failure_reason = "no_progress"

    selected_target_id = None
    if episodes and episodes[0].target_sequence:
        selected_target_id = episodes[0].target_sequence[0].target_poi_id
    diagnostics = AdaptiveDiagnostics(
        episode_count=len(episodes),
        solved_episode_count=solved_episode_count,
        failed_episode_count=failed_episode_count,
        retarget_count=int(retarget_count),
        target_switch_count=int(target_switch_count),
        useful_change_count=int(useful_change_count),
        no_progress_count=int(no_progress_count),
        level_transition_count=int(level_transition_count),
        terminal_count=int(terminal_count),
        step_budget_exhausted_count=int(step_budget_exhausted_count),
        failure_reason_counts=dict(sorted((str(k), int(v)) for k, v in failure_counts.items())),
    )
    return AdaptiveSolveReport(
        episodes=tuple(episodes),
        diagnostics=diagnostics,
        selected_target_id=selected_target_id,
        solved=bool(solved),
        failure_reason=failure_reason,
    )


def build_level_solution_from_adaptive_report(
    *,
    game_id: str,
    level_id: str,
    adaptive_report: AdaptiveSolveReport,
) -> LevelSolution:
    episodes = tuple(getattr(adaptive_report, "episodes", ()))
    solved_episode = next((ep for ep in episodes if bool(getattr(ep, "solved", False))), None)

    # Use solved episode if available, otherwise use first episode (original behavior for level solving)
    # For unsolved multi-POI cases, use last episode which has the cumulative trajectory
    if solved_episode is not None:
        source_episode = solved_episode
    elif episodes:
        # Check if ANY episode has level_transition
        has_level_transition = any(
            any(int(getattr(step, "levels_completed_after", 0)) > int(getattr(step, "levels_completed_before", 0))
                for step in tuple(getattr(ep, "steps", ())))
            for ep in episodes
        )
        if has_level_transition:
            # Use first episode with level transition
            for ep in episodes:
                for step in tuple(getattr(ep, "steps", ())):
                    if int(getattr(step, "levels_completed_after", 0)) > int(getattr(step, "levels_completed_before", 0)):
                        source_episode = ep
                        break
                if source_episode:
                    break
            if source_episode is None:
                source_episode = episodes[0]
        else:
            # No level transition - for multi-POI, use last episode (cumulative trajectory)
            source_episode = episodes[-1]
    else:
        source_episode = None

    trace: list[LevelSolveAction] = []
    terminal = False
    level_transition = False
    if source_episode is not None:
        trace = list(_extract_full_level_step_trace(tuple(getattr(source_episode, "steps", ()))))
        for step in tuple(getattr(source_episode, "steps", ())):
            pre_level = int(getattr(step, "levels_completed_before", 0))
            post_level = int(getattr(step, "levels_completed_after", 0))
            level_transition = level_transition or (post_level > pre_level)
            terminal = terminal or bool(getattr(step, "terminal", False))

    solved = bool(getattr(adaptive_report, "solved", False))
    if solved and not trace:
        fallback_episode = next((ep for ep in episodes if tuple(getattr(ep, "steps", ()))), None)
        if fallback_episode is not None:
            trace = list(_extract_full_level_step_trace(tuple(getattr(fallback_episode, "steps", ()))))
            for step in tuple(getattr(fallback_episode, "steps", ())):
                pre_level = int(getattr(step, "levels_completed_before", 0))
                post_level = int(getattr(step, "levels_completed_after", 0))
                level_transition = level_transition or (post_level > pre_level)
                terminal = terminal or bool(getattr(step, "terminal", False))

    # Level is complete if marked solved OR if level_transition occurred
    level_complete = solved or level_transition
    failure_reason = None if level_complete else getattr(adaptive_report, "failure_reason", "no_progress")
    return LevelSolution(
        game_id=str(game_id),
        level_id=str(level_id),
        solved=bool(level_complete),  # Mark as solved if level transitioned
        action_trace=tuple(trace),
        step_count=len(trace),
        terminal=bool(terminal),
        level_transition=bool(level_transition),
        failure_reason=failure_reason,
    )


def extract_replayable_level_trace(
    *,
    solved_level_result,
    executed_solve_steps,
    game_id: str,
    level_id: str,
    source_run_id: str | None = None,
    trace_version: int = 1,
) -> SavedLevelTrace:
    solved = bool(getattr(solved_level_result, "solved", False) if solved_level_result is not None else False)
    actions = tuple(str(getattr(step, "action", "")) for step in tuple(executed_solve_steps or ()))
    action_sources = tuple(str(getattr(step, "source", "frontier_solve")) for step in tuple(executed_solve_steps or ()))
    return SavedLevelTrace(
        game_id=str(game_id),
        level_id=str(level_id),
        solved=bool(solved),
        action_trace=actions,
        step_count=len(actions),
        source_run_id=source_run_id,
        trace_version=int(trace_version),
        replay_verified=False,
        action_sources=action_sources,
    )


def verify_level_trace_replay(
    *,
    game_id: str,
    level_id: str,
    saved_trace: SavedLevelTrace,
    prefix_traces=(),
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    replay = replay_trace_at_frontier(
        game_id=game_id,
        level_id=level_id,
        prefix_traces=tuple(prefix_traces or ()),
        frontier_trace=saved_trace,
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )
    return {
        "verified": bool(replay.get("success", False)) and bool(replay.get("level_solved", False)),
        "final_level_reached": int(replay.get("final_level_reached", 0)),
        "executed_action_count": int(replay.get("executed_action_count", 0)),
        "divergence": bool(replay.get("divergence", False)),
        "frontier_reached": bool(replay.get("frontier_reached", False)),
        "level_solved": bool(replay.get("level_solved", False)),
    }


def finalize_solved_level_trace(
    *,
    solved_level_result,
    executed_actions: tuple[str, ...] | list[str],
    game_id: str,
    level_id: str,
    prefix_traces=(),
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    trace_db_path: str | None = None,
) -> dict[str, Any]:
    solved = bool(getattr(solved_level_result, "solved", False) if solved_level_result is not None else False)
    raw_items = tuple(executed_actions or ())
    actions: list[str] = []
    action_sources: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            actions.append(str(item))
            action_sources.append("frontier_solve")
            continue
        action = getattr(item, "action", None)
        if action is None and isinstance(item, dict):
            action = item.get("action")
        source = getattr(item, "source", None)
        if source is None and isinstance(item, dict):
            source = item.get("source")
        if action is None:
            continue
        actions.append(str(action))
        action_sources.append(str(source or "frontier_solve"))
    actions_tuple = tuple(actions)
    sources_tuple = tuple(action_sources)
    if solved and not actions_tuple:
        trace = SavedLevelTrace(
            game_id=str(game_id),
            level_id=str(level_id),
            solved=True,
            action_trace=tuple(),
            step_count=0,
            source_run_id=None,
            trace_version=1,
            replay_verified=False,
            action_sources=tuple(),
            trace_id=None,
        )
        return {
            "saved_trace": trace,
            "replay_verified": False,
            "failure_reason": "solved_trace_missing_actions",
            "trace_id": None,
        }
    trace_id = f"{game_id}:{level_id}:{int(time.time() * 1000)}"
    trace = SavedLevelTrace(
        game_id=str(game_id),
        level_id=str(level_id),
        solved=bool(solved),
        action_trace=actions_tuple,
        step_count=len(actions_tuple),
        source_run_id=None,
        trace_version=1,
        replay_verified=False,
        action_sources=sources_tuple if sources_tuple else None,
        trace_id=trace_id,
    )
    replay = verify_level_trace_replay(
        game_id=game_id,
        level_id=level_id,
        saved_trace=trace,
        prefix_traces=tuple(prefix_traces or ()),
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )
    replay_verified = bool(replay.get("verified", False))
    finalized = SavedLevelTrace(
        game_id=trace.game_id,
        level_id=trace.level_id,
        solved=trace.solved,
        action_trace=trace.action_trace,
        step_count=trace.step_count,
        source_run_id=trace.source_run_id,
        trace_version=trace.trace_version,
        replay_verified=replay_verified,
        action_sources=trace.action_sources,
        trace_id=trace.trace_id,
    )
    persisted_trace_id = None
    if replay_verified:
        inserted, persisted_trace_id = upsert_verified_best_trace(db_path=trace_db_path, trace=finalized)
        if not inserted:
            save_trace_history_row(db_path=trace_db_path, trace=finalized, trace_id=trace.trace_id)
            persisted_trace_id = persisted_trace_id or trace.trace_id
        if not inserted and not persisted_trace_id:
            return {
                "saved_trace": finalized,
                "replay_verified": True,
                "failure_reason": "verified_trace_persist_failed",
                "trace_id": None,
            }
    return {
        "saved_trace": finalized,
        "replay_verified": replay_verified,
        "failure_reason": None if replay_verified else "trace_replay_verification_failed",
        "trace_id": persisted_trace_id if replay_verified else None,
    }


def _conservative_hud_only_change(
    pre_frame: tuple[tuple[int, ...], ...] | None,
    post_frame: tuple[tuple[int, ...], ...] | None,
) -> bool:
    return detect_hud_only_change(pre_frame, post_frame)


def _deterministic_frame_hash(frame: tuple[tuple[int, ...], ...] | None) -> str:
    if frame is None:
        return "none"
    payload = "|".join(",".join(str(int(v)) for v in row) for row in frame).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _build_state_signature(
    *,
    level_id: int,
    action: str,
    avatar_bbox_after,
    target_bbox_after,
    reward_after,
    post_frame: tuple[tuple[int, ...], ...] | None,
) -> tuple[str, str, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, float | None, str]:
    return (
        str(level_id),
        str(action),
        avatar_bbox_after,
        target_bbox_after,
        float(reward_after) if reward_after is not None else None,
        _deterministic_frame_hash(post_frame),
    )


def _bbox_center(bbox):
    if bbox is None:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _distance_between_bboxes(left, right) -> float | None:
    lc = _bbox_center(left)
    rc = _bbox_center(right)
    if lc is None or rc is None:
        return None
    dx = float(lc[0]) - float(rc[0])
    dy = float(lc[1]) - float(rc[1])
    return (dx * dx + dy * dy) ** 0.5


def _extract_route_hints_for_target(contact_experiment_report, target_poi_id: str | None) -> tuple[dict[str, object], ...]:
    if contact_experiment_report is None or target_poi_id is None:
        return tuple()
    hint = get_best_route_hint_for_poi(contact_experiment_report, str(target_poi_id))
    if isinstance(hint, dict):
        return (dict(hint),)
    return tuple()


def _normalize_route_candidates(routes) -> tuple[Any, ...]:
    if routes is None:
        return tuple()
    seq = tuple(routes)
    if not seq:
        return tuple()
    if all(hasattr(item, "actions") and hasattr(item, "route_id") for item in seq):
        return tuple(seq)
    if all(isinstance(item, str) for item in seq):
        return (_route_candidate_from_actions(route_id="route_0", actions=tuple(str(item) for item in seq)),)
    out: list[Any] = []
    for item in seq:
        if isinstance(item, str):
            out.append(_route_candidate_from_actions(route_id=f"route_{len(out)}", actions=(str(item),)))
            continue
        planned = tuple(str(v) for v in tuple(item))
        out.append(_route_candidate_from_actions(route_id=f"route_{len(out)}", actions=planned))
    return tuple(out)


def _route_candidate_from_actions(*, route_id: str, actions: tuple[str, ...]):
    dx = 0
    dy = 0
    turns = 0
    prev_axis = None
    waypoints = [(0, 0)]
    for action in tuple(actions):
        axis = "H" if action in {"LEFT", "RIGHT"} else ("V" if action in {"UP", "DOWN"} else prev_axis)
        if prev_axis is not None and axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
        if action == "LEFT":
            dx -= 1
        elif action == "RIGHT":
            dx += 1
        elif action == "UP":
            dy -= 1
        elif action == "DOWN":
            dy += 1
        waypoints.append((dx, dy))
    axis_order = "NONE" if not actions else ("H_ONLY" if all(a in {"LEFT", "RIGHT"} for a in actions) else ("V_ONLY" if all(a in {"UP", "DOWN"} for a in actions) else "MIXED"))
    from v5_0.route.trajectory_enumerator import RouteCandidate

    return RouteCandidate(
        route_id=str(route_id),
        actions=tuple(actions),
        length=int(len(actions)),
        net_dx=int(dx),
        net_dy=int(dy),
        first_action=(actions[0] if actions else None),
        turn_count=int(turns),
        axis_order=str(axis_order),
        waypoints=tuple(waypoints),
        score_components={"length": float(len(actions)), "turn_count": float(turns)},
    )


def _route_candidate_record(route, *, level_id: str, episode_index: int, target_poi_id: str | None, rank_index: int | None, selected_for_execution: bool) -> TrajectoryCandidateRecord:
    return TrajectoryCandidateRecord(
        trajectory_id=str(getattr(route, "route_id", f"route:{rank_index if rank_index is not None else 0}")),
        level_id=str(level_id),
        episode_index=int(episode_index),
        target_poi_id=(str(target_poi_id) if target_poi_id is not None else None),
        source="adaptive_solve",
        actions=tuple(str(a) for a in tuple(getattr(route, "actions", ()))),
        planned_length=int(getattr(route, "length", len(tuple(getattr(route, "actions", ())))) or 0),
        net_dx=int(getattr(route, "net_dx", 0)),
        net_dy=int(getattr(route, "net_dy", 0)),
        first_action=getattr(route, "first_action", None),
        turn_count=int(getattr(route, "turn_count", 0)),
        axis_order=str(getattr(route, "axis_order", "NONE")),
        waypoints=tuple(tuple(int(v) for v in point) for point in tuple(getattr(route, "waypoints", ()))),
        score_components={str(k): float(v) for k, v in dict(getattr(route, "score_components", {}) or {}).items()},
        rank_index=rank_index,
        selected_for_execution=bool(selected_for_execution),
    )


def _trajectory_attempt_record(
    *,
    route,
    level_id: str,
    episode_index: int,
    target_poi_id: str | None,
    action: str,
    step: AdaptiveStepRecord,
    route_step_offset: int,
) -> TrajectoryAttemptRecord:
    actions = tuple(str(a) for a in tuple(getattr(route, "actions", ())))
    completed = bool(actions) and int(route_step_offset + 1) >= int(len(actions))
    stop_reason = str(getattr(step, "outcome_type", "")) if (step.terminal or step.levels_completed_after > step.levels_completed_before or step.outcome_type != "no_effect") else None
    return TrajectoryAttemptRecord(
        trajectory_id=str(getattr(route, "route_id", "route_unknown")),
        level_id=str(level_id),
        episode_index=int(episode_index),
        target_poi_id=(str(target_poi_id) if target_poi_id is not None else None),
        source="adaptive_solve",
        actions=actions,
        planned_length=int(len(actions)),
        executed_step_count=1,
        completed_planned_route=bool(completed),
        stop_reason=stop_reason,
        outcome_type=str(getattr(step, "outcome_type", None)),
        solved=bool(step.levels_completed_after > step.levels_completed_before),
        terminal=bool(step.terminal),
        level_transition=bool(step.levels_completed_after > step.levels_completed_before),
        blocked_step_count=1 if bool(step.blocked_action) else 0,
        invalid_step_count=1 if bool(step.invalid_action) else 0,
        screen_changed_step_count=1 if bool(step.screen_changed) else 0,
        start_avatar_bbox=step.avatar_bbox_before,
        end_avatar_bbox=step.avatar_bbox_after,
        start_target_bbox=step.target_bbox_before,
        end_target_bbox=step.target_bbox_after,
    )


def _trajectory_stats(
    *,
    level_id: str,
    solved: bool,
    failure_reason: str | None,
    generated: tuple[TrajectoryCandidateRecord, ...],
    attempted: tuple[TrajectoryAttemptRecord, ...],
) -> TrajectoryStatsReport:
    attempted_steps = [int(item.executed_step_count) for item in tuple(attempted)]
    if attempted_steps:
        min_steps = min(attempted_steps)
        max_steps = max(attempted_steps)
        mean_steps = float(sum(attempted_steps)) / float(len(attempted_steps))
    else:
        min_steps = 0
        max_steps = 0
        mean_steps = 0.0
    return TrajectoryStatsReport(
        level_id=str(level_id),
        solved=bool(solved),
        failure_reason=failure_reason,
        generated_trajectory_count=int(len(tuple(generated))),
        attempted_trajectory_count=int(len(tuple(attempted))),
        completed_trajectory_count=int(sum(1 for item in tuple(attempted) if bool(item.completed_planned_route))),
        min_steps_per_attempted_trajectory=int(min_steps),
        max_steps_per_attempted_trajectory=int(max_steps),
        mean_steps_per_attempted_trajectory=float(mean_steps),
        total_executed_steps_across_attempted_trajectories=int(sum(attempted_steps)),
    )


def _validate_adaptive_route(route, *, dx: int | None, dy: int | None, hint_source: str | None = None) -> tuple[bool, tuple[str, ...]]:
    route_id = str(getattr(route, "route_id", ""))
    inferred_hint_source = hint_source or (route_id if route_id.startswith("hint:") else None)
    allow_exploratory = bool(getattr(route, "score_components", {}).get("reason_probe"))
    return validate_route_actions_for_action_delta(
        tuple(str(a) for a in tuple(getattr(route, "actions", ()))),
        dx=dx,
        dy=dy,
        budget=None,
        max_length=14,
        hint_source=inferred_hint_source,
        allow_exploratory=allow_exploratory,
    )


def _extract_saved_trace_actions(trace_like) -> tuple[str, ...]:
    actions = getattr(trace_like, "action_trace", ())
    return tuple(str(item) for item in tuple(actions or ()))


def _start_frontier_attempt_session(
    *,
    base_session,
    game_id: str,
    level_id: str,
    prefix_traces,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
    session_adapter,
):
    expected_level_index = int(str(level_id).lstrip("L") or 0)
    if tuple(prefix_traces or ()):
        replay = replay_prefix_traces_to_frontier(
            game_id=game_id,
            prefix_traces=tuple(prefix_traces or ()),
            render_terminal=bool(render_terminal),
            env_factory=env_factory,
        )
        session = replay.get("session")
        if session is None or not bool(replay.get("frontier_reached", False)) or bool(replay.get("divergence", False)):
            return None, True, "prefix_replay_failed"
        return session, True, None

    if base_session is not None:
        if not hasattr(session_adapter, "create_session"):
            return base_session, False, None
        try:
            obs = session_adapter.get_current_observation(base_session)
        except Exception:
            return None, False, "attempt_session_unavailable"
        if int(getattr(obs, "levels_completed", 0) or 0) != expected_level_index:
            return None, False, "prefix_replay_failed"
        return base_session, False, None

    if hasattr(session_adapter, "create_session"):
        session = session_adapter.create_session(
            game_id,
            seed=0,
            render_terminal=bool(render_terminal),
            env_factory=env_factory,
        )
        try:
            obs = session_adapter.get_current_observation(session)
        except Exception:
            try:
                session_adapter.close_session(session)
            except Exception:
                pass
            return None, True, "attempt_session_unavailable"
        if int(getattr(obs, "levels_completed", 0) or 0) != expected_level_index:
            try:
                session_adapter.close_session(session)
            except Exception:
                pass
            return None, True, "prefix_replay_failed"
        return session, True, None

    return None, False, "attempt_session_unavailable"


def _anchor_bbox_for_step(
    *,
    frame,
    avatar_bbox,
    avatar_hist,
    target_bbox,
    target_hist,
    recent_avatar_motion=None,
    recent_target_motion=None,
):
    avatar = track_avatar_bbox_in_frame(
        frame,
        avatar_bbox,
        avatar_hist,
        frontier_reanchor=False,
    )
    target = track_poi_bbox_in_frame(
        frame,
        target_bbox,
        target_hist,
        frontier_reanchor=False,
    )
    if avatar is None:
        avatar = reacquire_avatar_bbox_in_frame(
            frame,
            avatar_bbox,
            avatar_hist,
            previous_center=_bbox_center(avatar_bbox),
            recent_motion=recent_avatar_motion,
        )
    if target is None:
        target = reacquire_poi_bbox_in_frame(
            frame,
            target_bbox,
            target_hist,
            previous_center=_bbox_center(target_bbox),
            recent_motion=recent_target_motion,
        )
    if avatar is None:
        avatar = find_best_component_match_in_frame(
            frame=frame,
            reference_bbox=avatar_bbox,
            reference_histogram=avatar_hist,
            for_poi=False,
            previous_center=_bbox_center(avatar_bbox),
            recent_motion=recent_avatar_motion,
        )
    if target is None:
        target = find_best_component_match_in_frame(
            frame=frame,
            reference_bbox=target_bbox,
            reference_histogram=target_hist,
            for_poi=True,
            previous_center=_bbox_center(target_bbox),
            recent_motion=recent_target_motion,
        )
    return avatar, target


def _execute_frontier_action_sequence(
    *,
    session,
    session_adapter,
    action_adapter,
    selected_avatar,
    poi,
    actions: tuple[str, ...],
    start_step_index: int,
    start_avatar_bbox,
    start_target_bbox,
    hud_mask,
    source: str,
    stop_on_terminal: bool = True,
    stop_on_invalid_or_blocked: bool = True,
    max_actions: int | None = None,
    recent_avatar_motion=None,
    recent_target_motion=None,
    route_mode: str | None = None,  # "touch" or "overlap"
):
    steps: list[AdaptiveStepRecord] = []
    current_avatar_bbox = start_avatar_bbox
    current_target_bbox = start_target_bbox
    avatar_motion = recent_avatar_motion
    target_motion = recent_target_motion
    failure_reason: str | None = None
    solved = False
    level_transition = False
    route_progress = False
    route_closer = False
    useful_step_index: int | None = None
    executed_actions: list[str] = []
    action_limit = len(tuple(actions))
    if max_actions is not None:
        action_limit = min(action_limit, max(0, int(max_actions)))
    require_target_anchor = str(source) != "bootstrap_replay"
    for offset, action in enumerate(tuple(actions)[:action_limit]):
        obs = session_adapter.get_current_observation(session)
        pre_frame = _extract_frame_plane(obs.frame)
        reward_before = _extract_reward(obs.raw_payload)
        avatar_before, target_before = _anchor_bbox_for_step(
            frame=pre_frame,
            avatar_bbox=current_avatar_bbox,
            avatar_hist=getattr(selected_avatar, "value_histogram", None),
            target_bbox=current_target_bbox,
            target_hist=getattr(poi, "value_histogram", None),
            recent_avatar_motion=avatar_motion,
            recent_target_motion=target_motion,
        )
        if avatar_before is None and not require_target_anchor:
            avatar_before = current_avatar_bbox
        if target_before is None and not require_target_anchor:
            target_before = current_target_bbox
        # If tracking failed, fall back to the start bbox
        if avatar_before is None:
            avatar_before = start_avatar_bbox
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] Avatar tracking failed, using start_avatar_bbox={start_avatar_bbox}\n")
                f.flush()
        if target_before is None and require_target_anchor:
            target_before = start_target_bbox
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] Target tracking failed, using start_target_bbox={start_target_bbox}\n")
                f.flush()
        if avatar_before is None or (require_target_anchor and target_before is None):
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] Geometry mismatch in action loop: avatar_before={avatar_before}, target_before={target_before}, require_target={require_target_anchor}\n")
                f.flush()
            failure_reason = "frontier_geometry_mismatch"
            break
        context = ActionTranslationContext(
            available_action_ids=obs.available_actions,
            coordinate_action_id=session.environment_metadata.coordinate_action_id,
            coordinate_bounds=session.environment_metadata.coordinate_bounds,
        )
        try:
            translated = action_adapter.translate_token(str(action), context)
            executed = session_adapter.execute_action_prefix(session, (translated,), (str(action),))
            invalid = False
        except Exception:
            executed = None
            invalid = True

        if invalid:
            step = AdaptiveStepRecord(
                step_index=int(start_step_index) + len(steps),
                action=str(action),
                pre_frame=pre_frame,
                post_frame=None,
                invalid_action=True,
                blocked_action=False,
                terminal=False,
                levels_completed_before=int(obs.levels_completed),
                levels_completed_after=int(obs.levels_completed),
                reward_before=reward_before,
                reward_after=reward_before,
                avatar_bbox_before=avatar_before,
                avatar_bbox_after=avatar_before,
                target_poi_id=getattr(poi, "poi_id", None),
                target_bbox_before=target_before,
                target_bbox_after=target_before,
                contact_detected=False,
                screen_changed=False,
                hud_changed_only=False,
                outcome_type="no_effect",
                retargeted=False,
                source=str(source),
            )
            steps.append(step)
            failure_reason = "invalid_action"
            if stop_on_invalid_or_blocked:
                break
            continue

        post = session_adapter.get_current_observation(session)
        post_frame = _extract_frame_plane(post.frame)
        reward_after = _extract_reward(post.raw_payload)
        blocked = any(not item.action_legal for item in tuple(getattr(executed, "step_results", ())))
        terminal = str(getattr(executed, "terminal_status", "running")) in {"success", "failure"}
        level_transition = int(post.levels_completed) > int(obs.levels_completed)
        avatar_after, target_after = _anchor_bbox_for_step(
            frame=post_frame,
            avatar_bbox=avatar_before or current_avatar_bbox,
            avatar_hist=getattr(selected_avatar, "value_histogram", None),
            target_bbox=target_before or current_target_bbox,
            target_hist=getattr(poi, "value_histogram", None),
            recent_avatar_motion=avatar_motion,
            recent_target_motion=target_motion,
        )
        screen_changed = detect_screen_change(pre_frame, post_frame)
        contact_detected = detect_contact(avatar_after, target_after)
        hud_changed_only = detect_hud_only_change(pre_frame, post_frame) if hud_mask is not None else _conservative_hud_only_change(pre_frame, post_frame)
        world_change_outside_hud = detect_screen_change_outside_hud_mask(pre_frame, post_frame, hud_mask=hud_mask)
        reward_changed = reward_after != reward_before
        terminal_success = terminal and str(getattr(executed, "terminal_status", "running")) == "success"
        dist_before = _distance_between_bboxes(avatar_before, target_before)
        dist_after = _distance_between_bboxes(avatar_after, target_after)
        avatar_got_closer = bool(dist_before is not None and dist_after is not None and dist_after < dist_before)
        if level_transition:
            outcome_type = "level_transition"
        elif terminal:
            outcome_type = "terminal"
        elif reward_changed:
            outcome_type = "reward_change"
        elif hud_changed_only:
            outcome_type = "hud_change_only"
        elif world_change_outside_hud:
            outcome_type = "world_change"
        else:
            outcome_type = "no_effect"
        step = AdaptiveStepRecord(
            step_index=int(start_step_index) + len(steps),
            action=str(action),
            pre_frame=pre_frame,
            post_frame=post_frame,
            invalid_action=False,
            blocked_action=bool(blocked),
            terminal=bool(terminal),
            levels_completed_before=int(obs.levels_completed),
            levels_completed_after=int(post.levels_completed),
            reward_before=reward_before,
            reward_after=reward_after,
            avatar_bbox_before=avatar_before,
            avatar_bbox_after=avatar_after,
            target_poi_id=getattr(poi, "poi_id", None),
            target_bbox_before=target_before,
            target_bbox_after=target_after,
            contact_detected=bool(contact_detected),
            screen_changed=bool(screen_changed),
            hud_changed_only=bool(hud_changed_only),
            outcome_type=str(outcome_type),
            retargeted=False,
            source=str(source),
        )
        steps.append(step)
        executed_actions.append(str(action))
        current_avatar_bbox = avatar_after or avatar_before or current_avatar_bbox
        current_target_bbox = target_after or target_before or current_target_bbox
        with open("/tmp/v5_debug.log", "a") as f:
            f.write(f"[DEBUG] After action {action}: avatar_after={avatar_after}, target_after={target_after}, outcome={outcome_type}, contact={contact_detected}, blocked={blocked}, terminal={terminal}\n")
            f.flush()
        prev_avatar_center = _bbox_center(avatar_before)
        new_avatar_center = _bbox_center(current_avatar_bbox)
        avatar_motion = (
            (new_avatar_center[0] - prev_avatar_center[0], new_avatar_center[1] - prev_avatar_center[1])
            if prev_avatar_center is not None and new_avatar_center is not None
            else avatar_motion
        )
        prev_target_center = _bbox_center(target_before)
        new_target_center = _bbox_center(current_target_bbox)
        target_motion = (
            (new_target_center[0] - prev_target_center[0], new_target_center[1] - prev_target_center[1])
            if prev_target_center is not None and new_target_center is not None
            else target_motion
        )
        # Determine if this step made meaningful progress
        # Contact alone is NOT success - we need contact + meaningful outcome
        meaningful_outcome = reward_changed or world_change_outside_hud or level_transition

        if meaningful_outcome or avatar_got_closer:
            route_progress = True
            route_closer = route_closer or avatar_got_closer
            useful_step_index = len(executed_actions)

        if contact_detected:
            # Contact detected - check if it achieved something meaningful
            # Also check if this is true overlap (same bbox) vs just touch (adjacent/intersecting)
            is_overlap = (avatar_after is not None and target_after is not None and avatar_after == target_after)

            if meaningful_outcome:
                # Contact + meaningful change = success
                route_progress = True
                route_closer = True
                useful_step_index = len(executed_actions)
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Meaningful contact (outcome={outcome_type}, overlap={is_overlap}), stopping route\n")
                    f.flush()
            else:
                # Contact but no meaningful outcome (just touch, no effect)
                # This route didn't achieve the goal - mark as no progress
                route_progress = False
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Contact without meaningful outcome (outcome={outcome_type}, overlap={is_overlap}), marking as no progress\n")
                    f.flush()

            # Decide whether to break on contact:
            # - For "touch" routes: break on meaningful contact (goal is touch, not overlap)
            # - For "overlap" routes: only break on true overlap (same bbox)
            # - For unknown routes: break on overlap or meaningful outcome (conservative)
            should_break = False
            if route_mode == "touch":
                # Touch route: stop at first meaningful contact
                should_break = meaningful_outcome
            elif route_mode == "overlap":
                # Overlap route: only stop at true overlap (same bbox)
                should_break = is_overlap
            else:
                # Unknown mode: conservative - break on overlap or meaningful outcome
                should_break = is_overlap or meaningful_outcome

            if should_break:
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Breaking route (mode={route_mode}, overlap={is_overlap}, meaningful={meaningful_outcome})\n")
                    f.flush()
                break
        if terminal_success or level_transition:
            solved = True
            break
        if terminal:
            failure_reason = "terminal_failure"
            if stop_on_terminal:
                break
        if blocked:
            failure_reason = "blocked_action"
            if stop_on_invalid_or_blocked:
                break
    if not solved and failure_reason is None and steps and str(source) != "bootstrap_replay":
        failure_reason = "no_progress"
    return {
        "steps": tuple(steps),
        "executed_actions": tuple(executed_actions),
        "avatar_bbox": current_avatar_bbox,
        "target_bbox": current_target_bbox,
        "recent_avatar_motion": avatar_motion,
        "recent_target_motion": target_motion,
        "solved": bool(solved),
        "level_transition": bool(level_transition),
        "failure_reason": failure_reason,
        "route_progress": bool(route_progress),
        "route_closer": bool(route_closer),
        "useful_prefix_length": int(useful_step_index or 0),
    }


def _build_attempt_record_from_steps(
    *,
    route,
    level_id: str,
    episode_index: int,
    target_poi_id: str | None,
    steps,
    failure_reason: str | None,
):
    steps_seq = tuple(steps or ())
    start_step = steps_seq[0] if steps_seq else None
    end_step = steps_seq[-1] if steps_seq else None
    stop_reason = None
    if failure_reason is not None:
        stop_reason = str(failure_reason)
    elif end_step is not None:
        stop_reason = str(getattr(end_step, "outcome_type", None))
    return TrajectoryAttemptRecord(
        trajectory_id=str(getattr(route, "route_id", "route_unknown")),
        level_id=str(level_id),
        episode_index=int(episode_index),
        target_poi_id=(str(target_poi_id) if target_poi_id is not None else None),
        source="adaptive_solve",
        actions=tuple(str(a) for a in tuple(getattr(route, "actions", ()))),
        planned_length=int(getattr(route, "length", len(tuple(getattr(route, "actions", ())))) or 0),
        executed_step_count=int(len(steps_seq)),
        completed_planned_route=bool(len(steps_seq) >= int(getattr(route, "length", len(tuple(getattr(route, "actions", ())))) or 0)),
        stop_reason=stop_reason,
        outcome_type=(str(getattr(end_step, "outcome_type", None)) if end_step is not None else None),
        solved=bool(any(int(getattr(step, "levels_completed_after", 0)) > int(getattr(step, "levels_completed_before", 0)) for step in steps_seq)),
        terminal=bool(any(bool(getattr(step, "terminal", False)) for step in steps_seq)),
        level_transition=bool(any(int(getattr(step, "levels_completed_after", 0)) > int(getattr(step, "levels_completed_before", 0)) for step in steps_seq)),
        blocked_step_count=int(sum(1 for step in steps_seq if bool(getattr(step, "blocked_action", False)))),
        invalid_step_count=int(sum(1 for step in steps_seq if bool(getattr(step, "invalid_action", False)))),
        screen_changed_step_count=int(sum(1 for step in steps_seq if bool(getattr(step, "screen_changed", False)))),
        start_avatar_bbox=getattr(start_step, "avatar_bbox_before", None),
        end_avatar_bbox=getattr(end_step, "avatar_bbox_after", None) if end_step is not None else None,
        start_target_bbox=getattr(start_step, "target_bbox_before", None),
        end_target_bbox=getattr(end_step, "target_bbox_after", None) if end_step is not None else None,
    )


def _record_route_generation_failure(
    *,
    failure_reason: str,
    generated_trajectory_records: list[TrajectoryCandidateRecord],
    rejected_trajectory_records: list[TrajectoryCandidateRecord],
    route_generation_failures: list[dict[str, Any]],
    level_id: str,
    target_poi_id: str | None,
    route_hint_source: str | None = None,
) -> None:
    route_generation_failures.append(
        {
            "level_id": str(level_id),
            "target_poi_id": target_poi_id,
            "failure_reason": str(failure_reason),
            "generated_count": len(generated_trajectory_records),
            "rejected_count": len(rejected_trajectory_records),
            "hint_source": route_hint_source,
        }
    )


def _with_route_feasibility(target_state, route_feasibility: bool | None):
    if target_state is None:
        return None
    return target_state.__class__(
        target_poi_id=getattr(target_state, "target_poi_id", None),
        source=getattr(target_state, "source", "retained"),
        confidence=float(getattr(target_state, "confidence", 0.0)),
        attempt_count=int(getattr(target_state, "attempt_count", 0)),
        last_outcome_type=getattr(target_state, "last_outcome_type", None),
        active=bool(getattr(target_state, "active", True)),
        route_feasibility=route_feasibility,
    )


def _find_non_overlapping_component(
    *,
    frame: tuple[tuple[int, ...], ...] | None,
    reference_bbox: tuple[int, int, int, int] | None,
    reference_histogram: dict | None,
    exclude_bbox: tuple[int, int, int, int] | None,
    preferred_center: tuple[float, float] | None,
) -> tuple[int, int, int, int] | None:
    """
    Find a component that matches the reference but doesn't overlap with exclude_bbox.
    This helps distinguish between multiple similar-looking POIs.
    """
    if frame is None or reference_bbox is None:
        return None

    from v5_0.contact.frame_tracker import _candidate_components_for_reference, _bbox, _histogram_for_component, _hist_similarity, _bbox_area

    def _bbox_center_local(bbox):
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    # Get all candidate components
    components = _candidate_components_for_reference(frame, reference_histogram)
    if not components:
        return None

    ref_center = _bbox_center_local(reference_bbox)
    ref_area = _bbox_area(reference_bbox)
    pref_center = tuple(preferred_center) if preferred_center is not None else ref_center

    best_box = None
    best_score = -1.0

    for component in components:
        box = _bbox(component)

        # Skip if this overlaps significantly with the excluded bbox
        if exclude_bbox is not None and _bboxes_overlap(box, exclude_bbox):
            continue

        # Compute match score
        center = _bbox_center_local(box)
        dist = ((center[0] - ref_center[0]) ** 2 + (center[1] - ref_center[1]) ** 2) ** 0.5

        # Prioritize components near the reference location
        if dist > 20.0:  # Too far from reference
            continue

        proximity = max(0.0, 1.0 - dist / 20.0)

        # Check histogram similarity
        hist = _histogram_for_component(frame, component)
        hist_sim = _hist_similarity(hist, reference_histogram or {}) if reference_histogram else 0.5

        # Check size similarity
        area = _bbox_area(box)
        ratio = max(area, ref_area) / max(1.0, min(area, ref_area))
        area_similarity = max(0.0, 1.0 - (ratio - 1.0) / 3.5)

        # Weighted score - prioritize histogram match for POI disambiguation
        score = (0.50 * hist_sim) + (0.30 * proximity) + (0.20 * area_similarity)

        if score > best_score:
            best_score = score
            best_box = box

    # Only return if we have a reasonably confident match
    if best_score > 0.40:
        return best_box

    return None


def _bboxes_overlap(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes overlap."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Check if one box is to the left, right, above, or below the other
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True


def reanchor_frontier_objects(
    *,
    frame: tuple[tuple[int, ...], ...] | None,
    selected_avatar,
    initial_poi,
    contact_route_hint=None,
    preferred_avatar_bbox=None,
    preferred_target_bbox=None,
    previous_avatar_bbox=None,
    previous_target_bbox=None,
    previous_avatar_center=None,
    previous_target_center=None,
    recent_avatar_motion=None,
    recent_target_motion=None,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    with open("/tmp/v5_debug.log", "a") as f:
        poi_id = getattr(initial_poi, "poi_id", "unknown") if initial_poi else "None"
        f.write(f"[DEBUG] reanchor_frontier_objects called for POI: {poi_id}\n")
        f.flush()

    avatar_ref_bbox = previous_avatar_bbox or getattr(selected_avatar, "selected_bbox", None)
    avatar_ref_hist = getattr(selected_avatar, "value_histogram", None)
    poi_ref_bbox = previous_target_bbox or (getattr(initial_poi, "bbox", None) if initial_poi is not None else None)
    poi_ref_hist = getattr(initial_poi, "value_histogram", None) if initial_poi is not None else None
    avatar_prev_center = previous_avatar_center or _bbox_center(avatar_ref_bbox) or getattr(selected_avatar, "selected_center", None)
    poi_prev_center = previous_target_center or _bbox_center(poi_ref_bbox) or (getattr(initial_poi, "center", None) if initial_poi is not None else None)
    strict_avatar = track_avatar_bbox_in_frame(
        frame,
        avatar_ref_bbox,
        avatar_ref_hist,
        frontier_reanchor=False,
    )
    strict_target = track_poi_bbox_in_frame(
        frame,
        poi_ref_bbox,
        poi_ref_hist,
        frontier_reanchor=False,
    )
    if strict_avatar is not None and strict_target is not None:
        # FIRST CHECK: Collision detection on initial strict tracking
        if strict_avatar == strict_target:
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] WARNING: Initial strict tracking found same bbox: {strict_avatar}, trying frontier_reanchor\n")
                f.flush()
            # Continue to try frontier_reanchor mode instead of returning early
        else:
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] reanchor returning (first strict path): avatar={strict_avatar}, target={strict_target}\n")
                f.flush()
            return strict_avatar, strict_target

    if strict_avatar is None:
        strict_avatar = track_avatar_bbox_in_frame(
            frame,
            avatar_ref_bbox,
            avatar_ref_hist,
            frontier_reanchor=True,
        )
    if strict_target is None:
        strict_target = track_poi_bbox_in_frame(
            frame,
            poi_ref_bbox,
            poi_ref_hist,
            frontier_reanchor=True,
        )
    if strict_avatar is not None and strict_target is not None:
        # Check for collision even in strict tracking
        if strict_avatar == strict_target:
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] WARNING: Strict tracking found same bbox for avatar and target: {strict_avatar}\n")
                f.flush()
            # Don't return early - continue to relaxed search to try disambiguation
            strict_avatar = None
            strict_target = None
        else:
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] reanchor returning (strict path): avatar={strict_avatar}, target={strict_target}\n")
                f.flush()
            return strict_avatar, strict_target

    hint_avatar_bbox = None
    hint_target_bbox = None
    if isinstance(contact_route_hint, dict):
        hint_avatar_bbox = contact_route_hint.get("last_avatar_bbox")
        hint_target_bbox = contact_route_hint.get("last_poi_bbox")

    avatar_stage2_bbox = preferred_avatar_bbox or hint_avatar_bbox
    target_stage2_bbox = preferred_target_bbox or hint_target_bbox

    relaxed_avatar = strict_avatar or reacquire_avatar_bbox_in_frame(
        frame,
        avatar_stage2_bbox or avatar_ref_bbox,
        avatar_ref_hist,
        preferred_center=_bbox_center(avatar_stage2_bbox) if avatar_stage2_bbox is not None else avatar_prev_center,
        recent_bbox=avatar_stage2_bbox,
        previous_center=avatar_prev_center,
        recent_motion=recent_avatar_motion,
    )
    relaxed_target = strict_target or reacquire_poi_bbox_in_frame(
        frame,
        target_stage2_bbox or poi_ref_bbox,
        poi_ref_hist,
        preferred_center=_bbox_center(target_stage2_bbox) if target_stage2_bbox is not None else poi_prev_center,
        recent_bbox=target_stage2_bbox,
        previous_center=poi_prev_center,
        recent_motion=recent_target_motion,
    )

    if relaxed_avatar is None:
        relaxed_avatar = reacquire_avatar_bbox_in_frame(
            frame,
            avatar_ref_bbox,
            avatar_ref_hist,
            preferred_center=avatar_prev_center,
            recent_bbox=avatar_ref_bbox,
            previous_center=avatar_prev_center,
            recent_motion=recent_avatar_motion,
        )
    if relaxed_target is None:
        relaxed_target = reacquire_poi_bbox_in_frame(
            frame,
            poi_ref_bbox,
            poi_ref_hist,
            preferred_center=poi_prev_center,
            recent_bbox=poi_ref_bbox,
            previous_center=poi_prev_center,
            recent_motion=recent_target_motion,
        )

    if relaxed_avatar is None:
        relaxed_avatar = find_best_component_match_in_frame(
            frame=frame,
            reference_bbox=avatar_ref_bbox,
            reference_histogram=avatar_ref_hist,
            for_poi=False,
            preferred_center=avatar_prev_center,
            recent_bbox=avatar_ref_bbox,
            previous_center=avatar_prev_center,
            recent_motion=recent_avatar_motion,
        )
    if relaxed_target is None:
        relaxed_target = find_best_component_match_in_frame(
            frame=frame,
            reference_bbox=poi_ref_bbox,
            reference_histogram=poi_ref_hist,
            for_poi=True,
            preferred_center=poi_prev_center,
            recent_bbox=poi_ref_bbox,
            previous_center=poi_prev_center,
            recent_motion=recent_target_motion,
        )

    with open("/tmp/v5_debug.log", "a") as f:
        f.write(f"[DEBUG] About to check collision, relaxed_avatar={relaxed_avatar}, relaxed_target={relaxed_target}\n")
        f.flush()

    # CRITICAL: Check for collision - avatar and target must be distinct!
    with open("/tmp/v5_debug.log", "a") as f:
        f.write(f"[DEBUG] Checking collision: avatar={relaxed_avatar}, target={relaxed_target}\n")
        f.flush()

    if relaxed_avatar is not None and relaxed_target is not None:
        if relaxed_avatar == relaxed_target:
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] WARNING: Avatar and target have same bbox {relaxed_avatar}, attempting disambiguation\n")
                f.flush()

            # Try to find a different component for the target by being more strict with histogram matching
            # and excluding the avatar bbox region

            # Attempt 1: Use stricter histogram matching for POI
            if poi_ref_hist is not None and poi_ref_bbox is not None:
                # Search for components that match POI histogram but are NOT near avatar
                alternate_target = _find_non_overlapping_component(
                    frame=frame,
                    reference_bbox=poi_ref_bbox,
                    reference_histogram=poi_ref_hist,
                    exclude_bbox=relaxed_avatar,
                    preferred_center=poi_prev_center,
                )
                if alternate_target is not None:
                    relaxed_target = alternate_target
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Found alternate target: {relaxed_target}\n")
                        f.flush()
                else:
                    # Attempt 2: Fall back to original POI bbox from contact tests if reanchoring completely failed
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Reanchoring failed, using original POI bbox from contact tests\n")
                        f.flush()
                    relaxed_target = poi_ref_bbox

            # If still same, keep original reference bboxes (from contact phase) as best guess
            if relaxed_avatar == relaxed_target:
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Could not disambiguate, using original reference bboxes\n")
                    f.flush()
                relaxed_avatar = avatar_ref_bbox
                relaxed_target = poi_ref_bbox

    with open("/tmp/v5_debug.log", "a") as f:
        f.write(f"[DEBUG] reanchor final return: avatar={relaxed_avatar}, target={relaxed_target}\n")
        f.flush()
    return relaxed_avatar, relaxed_target


def run_adaptive_solve_on_live_session(
    *,
    session,
    selected_avatar,
    ranked_poi_candidates,
    hud_targeting_report,
    contact_experiment_report,
    game_id: str,
    level_id: str,
    max_steps: int,
    session_adapter: SessionAdapter | None = None,
    action_adapter: ActionAdapter | None = None,
    skip_bootstrap_replay_in_final_solve: bool = False,
    prefix_traces=(),
    initial_frontier_actions=(),
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> AdaptiveSolveReport:
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] run_adaptive_solve_on_live_session started\n"); f.flush()
    del skip_bootstrap_replay_in_final_solve
    session_adapter = session_adapter or SessionAdapter()
    action_adapter = action_adapter or ActionAdapter()
    mechanic_report = None
    ranked = tuple(ranked_poi_candidates or ())
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Have {len(ranked)} POI candidates\n"); f.flush()
    if selected_avatar is None or getattr(selected_avatar, "failure_reason", None) is not None:
        diagnostics = AdaptiveDiagnostics(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, {"no_stable_avatar": 1})
        return AdaptiveSolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_stable_avatar")
    if not ranked:
        diagnostics = AdaptiveDiagnostics(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, {"no_poi_candidate": 1})
        return AdaptiveSolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_poi_candidate")

    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Selecting initial target\n"); f.flush()
    initial = select_initial_target(
        hud_targeting_report,
        ranked,
        contact_experiment_report=contact_experiment_report,
    )
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Initial target selected: {getattr(initial, 'target_poi_id', None) if initial else None}\n"); f.flush()
    if initial is None or initial.target_poi_id is None:
        diagnostics = AdaptiveDiagnostics(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, {"no_target_selected": 1})
        return AdaptiveSolveReport(episodes=(), diagnostics=diagnostics, selected_target_id=None, solved=False, failure_reason="no_target_selected")

    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Getting initial observation\n"); f.flush()
    poi_by_id = {item.poi_id: item for item in ranked}
    initial_obs = session_adapter.get_current_observation(session)
    initial_frame = _extract_frame_plane(initial_obs.frame)
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Got initial frame\n"); f.flush()
    initial_poi = poi_by_id.get(str(initial.target_poi_id))
    best_contact_hint = get_best_route_hint_for_poi(contact_experiment_report, str(initial.target_poi_id)) if contact_experiment_report is not None else None
    initial_frontier_actions = tuple(str(item) for item in tuple(initial_frontier_actions or ()))
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] initial_frontier_actions: {initial_frontier_actions}\n"); f.flush()

    # OPTIMIZATION: Use POI positions from contact tests directly - they're known-good
    avatar_from_contact = getattr(selected_avatar, "selected_bbox", None)
    poi_from_contact = getattr(initial_poi, "bbox", None) if initial_poi is not None else None

    if initial_frontier_actions:
        avatar_anchor = avatar_from_contact
        target_anchor = poi_from_contact
    elif avatar_from_contact is not None and poi_from_contact is not None:
        # Use contact positions directly - skip expensive reanchoring for first attempt
        with open("/tmp/v5_debug.log", "a") as f:
            f.write(f"[DEBUG] Using known-good POI positions from contact tests: avatar={avatar_from_contact}, target={poi_from_contact}\n")
            f.flush()
        avatar_anchor = avatar_from_contact
        target_anchor = poi_from_contact
    else:
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Calling reanchor_frontier_objects (fallback)\n"); f.flush()
        avatar_anchor, target_anchor = reanchor_frontier_objects(
            frame=initial_frame,
            selected_avatar=selected_avatar,
            initial_poi=initial_poi,
            contact_route_hint=best_contact_hint,
            preferred_avatar_bbox=(best_contact_hint or {}).get("last_avatar_bbox") if isinstance(best_contact_hint, dict) else None,
            preferred_target_bbox=(best_contact_hint or {}).get("last_poi_bbox") if isinstance(best_contact_hint, dict) else None,
            previous_avatar_bbox=avatar_from_contact,
            previous_target_bbox=poi_from_contact,
            previous_avatar_center=getattr(selected_avatar, "selected_center", None),
            previous_target_center=getattr(initial_poi, "center", None) if initial_poi is not None else None,
        )
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] reanchor_frontier_objects returned: avatar={avatar_anchor}, target={target_anchor}\n"); f.flush()
    if avatar_anchor is None or target_anchor is None:
        diagnostics = AdaptiveDiagnostics(1, 0, 1, 0, 0, 0, 1, 0, 0, 0, {"frontier_geometry_mismatch": 1})
        return AdaptiveSolveReport(
            episodes=(),
            diagnostics=diagnostics,
            selected_target_id=initial.target_poi_id,
            solved=False,
            failure_reason="frontier_geometry_mismatch",
        )

    current = initial
    carried_prefix_actions: tuple[str, ...] = initial_frontier_actions
    carried_avatar_bbox = avatar_anchor
    carried_target_bbox = target_anchor
    recent_avatar_motion = None
    recent_target_motion = None
    selected_target_id = initial.target_poi_id
    generated_trajectory_records: list[TrajectoryCandidateRecord] = []
    rejected_trajectory_records: list[TrajectoryCandidateRecord] = []
    attempted_trajectory_records: list[TrajectoryAttemptRecord] = []
    route_generation_failures: list[dict[str, Any]] = []
    episodes: list[AdaptiveEpisodeResult] = []
    history_steps: list[AdaptiveStepRecord] = []
    attempted_route_actions: dict[tuple[str, tuple[str, ...]], set[tuple[str, ...]]] = {}
    consecutive_no_progress = 0
    blocked_streak = 0
    solved = False
    failure_reason: str | None = None
    is_very_first_attempt = True
    hud_mask = getattr(hud_targeting_report, "hud_mask", None)
    attempt_index = 0
    base_session = session

    # Track attempts per POI to enable proper multi-POI cycling
    poi_attempt_counts: dict[str, int] = {poi.poi_id: 0 for poi in ranked}
    max_attempts_per_poi = 10  # Try up to 10 routes per POI before switching
    max_total_attempts = 100  # Safety limit to prevent infinite loops
    poi_exhausted: set[str] = set()  # POIs that have been fully tried

    # Track which interaction modes (touch/overlap) have been tried per POI
    poi_tried_modes: dict[str, set[str]] = {poi.poi_id: set() for poi in ranked}

    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Starting main solve loop, max_steps={max_steps}, {len(ranked)} POIs to try\n"); f.flush()
    while int(sum(1 for ep in episodes for step in tuple(ep.steps) if str(getattr(step, "source", "")) == "frontier_solve")) < max(0, int(max_steps)) and attempt_index < max_total_attempts:
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Loop iteration {attempt_index}\n"); f.flush()
        remaining_budget = int(max_steps) - int(
            sum(1 for ep in episodes for step in tuple(ep.steps) if str(getattr(step, "source", "")) == "frontier_solve")
        )
        if remaining_budget <= 0:
            failure_reason = "step_budget_exhausted"
            break

        # Track attempts for current POI
        poi = poi_by_id.get(str(current.target_poi_id))
        if poi is None:
            failure_reason = "no_target_selected"
            break

        poi_attempt_counts[current.target_poi_id] = poi_attempt_counts.get(current.target_poi_id, 0) + 1
        current_poi_attempts = poi_attempt_counts[current.target_poi_id]

        with open("/tmp/v5_debug.log", "a") as f:
            f.write(f"[DEBUG] Attempt {attempt_index}, POI={current.target_poi_id}, poi_attempts={current_poi_attempts}/{max_attempts_per_poi}\n")
            f.flush()

        # If this POI has been tried too many times without success, mark it exhausted and switch
        if current_poi_attempts > max_attempts_per_poi and current.target_poi_id not in poi_exhausted:
            poi_exhausted.add(current.target_poi_id)
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] POI {current.target_poi_id} exhausted after {current_poi_attempts} attempts, switching target\n")
                f.flush()

            # Try to find a different POI that hasn't been exhausted
            nxt = select_next_target(
                _with_route_feasibility(current, False),  # Mark current as infeasible
                ranked,
                tuple(history_steps),
                contact_experiment_report,
                mechanic_report=mechanic_report,
            )

            if nxt is not None and nxt.target_poi_id != current.target_poi_id and nxt.target_poi_id not in poi_exhausted:
                current = nxt
                selected_target_id = current.target_poi_id
                consecutive_no_progress = 0
                carried_prefix_actions = tuple()  # Reset prefix for new POI
                attempt_index += 1
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Switched to new POI: {current.target_poi_id}\n")
                    f.flush()
                continue

            # All POIs exhausted
            if len(poi_exhausted) >= len(ranked):
                failure_reason = "all_pois_exhausted"
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] All {len(ranked)} POIs exhausted\n")
                    f.flush()
                break
        attempt_session, created_session, start_failure = _start_frontier_attempt_session(
            base_session=base_session,
            game_id=game_id,
            level_id=level_id,
            prefix_traces=prefix_traces,
            render_terminal=render_terminal,
            env_factory=env_factory,
            session_adapter=session_adapter,
        )
        if start_failure is not None or attempt_session is None:
            failure_reason = start_failure or "attempt_session_unavailable"
            break
        try:
            attempt_obs = session_adapter.get_current_observation(attempt_session)
            attempt_frame = _extract_frame_plane(attempt_obs.frame)
            route_hints = _extract_route_hints_for_target(contact_experiment_report, current.target_poi_id)

            # OPTIMIZATION: Use known-good positions from contact tests for current POI
            current_avatar_bbox = getattr(selected_avatar, "selected_bbox", None)
            current_poi_bbox = getattr(poi, "bbox", None) if poi is not None else None

            # If this is the VERY FIRST attempt (not just no carried actions) and we have good contact positions, use them directly
            if is_very_first_attempt and not carried_prefix_actions and current_avatar_bbox is not None and current_poi_bbox is not None:
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Using contact positions for POI {current.target_poi_id}: avatar={current_avatar_bbox}, target={current_poi_bbox}\n")
                    f.flush()
                avatar_before_attempt = current_avatar_bbox
                target_before_attempt = current_poi_bbox
                is_very_first_attempt = False
            else:
                # Otherwise use reanchoring
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Calling reanchoring with carried_avatar={carried_avatar_bbox}, carried_target={carried_target_bbox}, current_avatar={current_avatar_bbox}, current_poi={current_poi_bbox}\n")
                    f.flush()
                avatar_before_attempt, target_before_attempt = reanchor_frontier_objects(
                    frame=attempt_frame,
                    selected_avatar=selected_avatar,
                    initial_poi=poi,
                    contact_route_hint=(route_hints[0] if route_hints else None),
                    preferred_avatar_bbox=carried_avatar_bbox,
                    preferred_target_bbox=carried_target_bbox,
                    previous_avatar_bbox=carried_avatar_bbox or current_avatar_bbox,
                    previous_target_bbox=carried_target_bbox or current_poi_bbox,
                    previous_avatar_center=_bbox_center(carried_avatar_bbox) if carried_avatar_bbox else _bbox_center(current_avatar_bbox),
                    previous_target_center=_bbox_center(carried_target_bbox) if carried_target_bbox else _bbox_center(current_poi_bbox),
                    recent_avatar_motion=recent_avatar_motion,
                    recent_target_motion=recent_target_motion,
                )

            # Check for reanchoring failures (None or collision)
            if avatar_before_attempt is None or target_before_attempt is None or avatar_before_attempt == target_before_attempt:
                if avatar_before_attempt == target_before_attempt:
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Reanchoring collision in loop, falling back to contact positions\n")
                        f.flush()

                # Fall back to contact positions (known-good) regardless of carried actions
                avatar_before_attempt = current_avatar_bbox
                target_before_attempt = current_poi_bbox

                if avatar_before_attempt is None or target_before_attempt is None:
                    failure_reason = "frontier_geometry_mismatch"
                    episodes.append(
                        AdaptiveEpisodeResult(
                            episode_index=int(attempt_index),
                            target_sequence=(
                                AdaptiveTargetState(
                                    target_poi_id=current.target_poi_id,
                                    source=current.source,
                                    confidence=float(current.confidence),
                                    attempt_count=int(current.attempt_count),
                                    last_outcome_type=current.last_outcome_type,
                                    active=bool(current.active),
                                ),
                            ),
                            steps=tuple(),
                            solved=False,
                            failure_reason="frontier_geometry_mismatch",
                        )
                    )
                    break

            # ALWAYS replay prefix actions to build cumulative trajectory
            # Even when reusing base_session, we need to replay previous POIs' successful actions
            prefix_actions_to_replay = tuple(carried_prefix_actions)
            prefix_replay = _execute_frontier_action_sequence(
                session=attempt_session,
                session_adapter=session_adapter,
                action_adapter=action_adapter,
                selected_avatar=selected_avatar,
                poi=poi,
                actions=prefix_actions_to_replay,
                start_step_index=0,
                start_avatar_bbox=avatar_before_attempt,
                start_target_bbox=target_before_attempt,
                hud_mask=hud_mask,
                source="bootstrap_replay",
                stop_on_terminal=True,
                stop_on_invalid_or_blocked=True,
                recent_avatar_motion=recent_avatar_motion,
                recent_target_motion=recent_target_motion,
            )
            prefix_steps = tuple(prefix_replay["steps"])
            attempt_avatar_bbox = prefix_replay["avatar_bbox"] or avatar_before_attempt
            attempt_target_bbox = prefix_replay["target_bbox"] or target_before_attempt
            attempt_recent_avatar_motion = prefix_replay["recent_avatar_motion"] or recent_avatar_motion
            attempt_recent_target_motion = prefix_replay["recent_target_motion"] or recent_target_motion
            if prefix_replay["failure_reason"] is not None:
                failure_reason = "partial_prefix_replay_failed"
                episodes.append(
                    AdaptiveEpisodeResult(
                        episode_index=int(attempt_index),
                        target_sequence=(
                            AdaptiveTargetState(
                                target_poi_id=current.target_poi_id,
                                source=current.source,
                                confidence=float(current.confidence),
                                attempt_count=int(current.attempt_count),
                                last_outcome_type=current.last_outcome_type,
                                active=bool(current.active),
                            ),
                        ),
                        steps=prefix_steps,
                        solved=False,
                        failure_reason="partial_prefix_replay_failed",
                    )
                )
                break

            route_frame_obs = session_adapter.get_current_observation(attempt_session)
            route_frame = _extract_frame_plane(route_frame_obs.frame)
            raw_routes = _normalize_route_candidates(
                build_adaptive_policy_for_target(
                    selected_avatar,
                    poi,
                    route_frame,
                    max(1, int(remaining_budget)),
                    route_hints=route_hints,
                )
            )
            with open("/tmp/v5_debug.log", "a") as f:
                route_ids = [str(getattr(r, "route_id", "")) for r in raw_routes[:5]]
                f.write(f"[DEBUG] Generated {len(raw_routes)} routes, first 5 IDs: {route_ids}\n")
                f.flush()
            target_center = getattr(poi, "center", None)
            avatar_center = _bbox_center(attempt_avatar_bbox)
            dx = dy = None
            if target_center is not None and avatar_center is not None:
                dx, dy, _ = compute_action_space_delta(
                    start_center=avatar_center,
                    target_center=target_center,
                    start_bbox=attempt_avatar_bbox,
                    target_bbox=getattr(poi, "bbox", None),
                )
            valid_routes_all: list[Any] = []
            valid_routes_untried: list[Any] = []
            attempted_key = (str(current.target_poi_id), tuple(carried_prefix_actions))
            seen_for_key = attempted_route_actions.setdefault(attempted_key, set())
            for rank_index, route in enumerate(tuple(raw_routes or ())):
                ok, reasons = _validate_adaptive_route(route, dx=dx, dy=dy, hint_source=None)
                record = _route_candidate_record(
                    route,
                    level_id=str(level_id),
                    episode_index=int(attempt_index),
                    target_poi_id=current.target_poi_id,
                    rank_index=rank_index,
                    selected_for_execution=False,
                )
                if ok:
                    generated_trajectory_records.append(record)
                    valid_routes_all.append(route)
                    if tuple(getattr(route, "actions", ())) not in seen_for_key:
                        valid_routes_untried.append(route)
                else:
                    rejected_trajectory_records.append(
                        TrajectoryCandidateRecord(
                            trajectory_id=record.trajectory_id,
                            level_id=record.level_id,
                            episode_index=record.episode_index,
                            target_poi_id=record.target_poi_id,
                            source=record.source,
                            actions=record.actions,
                            planned_length=record.planned_length,
                            net_dx=record.net_dx,
                            net_dy=record.net_dy,
                            first_action=record.first_action,
                            turn_count=record.turn_count,
                            axis_order=record.axis_order,
                            waypoints=record.waypoints,
                            score_components=record.score_components,
                            rank_index=record.rank_index,
                            selected_for_execution=False,
                            validation_passed=False,
                            rejection_reasons=tuple(reasons),
                            plausibility_flags=("adaptive_rejected",),
                            hint_source=None,
                            start_avatar_center=record.start_avatar_center,
                            target_center=record.target_center,
                        )
                    )
            if not valid_routes_all:
                _record_route_generation_failure(
                    failure_reason="no_valid_route_candidates",
                    generated_trajectory_records=generated_trajectory_records,
                    rejected_trajectory_records=rejected_trajectory_records,
                    route_generation_failures=route_generation_failures,
                    level_id=str(level_id),
                    target_poi_id=current.target_poi_id,
                )
                # Mark current POI as having routing issues and try another
                poi_exhausted.add(current.target_poi_id)
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] No valid routes for POI {current.target_poi_id}, marking exhausted\n")
                    f.flush()

                current = _with_route_feasibility(current, False)
                nxt = select_next_target(
                    current,
                    ranked,
                    tuple(history_steps),
                    contact_experiment_report,
                    mechanic_report=mechanic_report,
                )
                if nxt is not None and nxt.target_poi_id != current.target_poi_id and nxt.target_poi_id not in poi_exhausted:
                    current = nxt
                    selected_target_id = current.target_poi_id
                    carried_prefix_actions = tuple()  # Reset prefix for new POI
                    carried_avatar_bbox = None  # Reset to force reanchoring
                    carried_target_bbox = None
                    attempt_index += 1
                    consecutive_no_progress = 0
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Switched to POI {current.target_poi_id} after routing failure\n")
                        f.flush()
                    continue

                # All POIs exhausted or no other target available
                if len(poi_exhausted) >= len(ranked):
                    failure_reason = "all_pois_exhausted"
                else:
                    failure_reason = "no_valid_route_candidates"
                break

            if valid_routes_untried:
                active_route = valid_routes_untried[0]
            elif consecutive_no_progress >= 3:
                # All routes for this POI tried, try switching to another POI
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] All routes tried for POI {current.target_poi_id}, attempting switch\n")
                    f.flush()

                nxt = select_next_target(
                    _with_route_feasibility(current, False),
                    ranked,
                    tuple(history_steps),
                    contact_experiment_report,
                    mechanic_report=mechanic_report,
                )

                if nxt is not None and nxt.target_poi_id != current.target_poi_id and nxt.target_poi_id not in poi_exhausted:
                    poi_exhausted.add(current.target_poi_id)  # Mark current as exhausted
                    current = nxt
                    selected_target_id = current.target_poi_id
                    carried_prefix_actions = tuple()
                    consecutive_no_progress = 0
                    attempt_index += 1
                    continue
                else:
                    failure_reason = "repeated_non_progress"
                    break
            else:
                # Retry a previously tried route
                active_route = valid_routes_all[0]
            seen_for_key.add(tuple(getattr(active_route, "actions", ())))
            generated_trajectory_records.append(
                _route_candidate_record(
                    active_route,
                    level_id=str(level_id),
                    episode_index=int(attempt_index),
                    target_poi_id=current.target_poi_id,
                    rank_index=0,
                    selected_for_execution=True,
                )
            )
            route_actions = tuple(str(a) for a in tuple(getattr(active_route, "actions", ())))
            route_id = str(getattr(active_route, "route_id", ""))
            # Determine route mode from route_id
            route_mode = "overlap"  # default
            if route_id.startswith("touch:"):
                route_mode = "touch"
            elif route_id.startswith("overlap:"):
                route_mode = "overlap"
            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] Executing route_id={route_id} (mode={route_mode}) with {len(route_actions)} actions: {route_actions}\n")
                f.flush()
            route_result = _execute_frontier_action_sequence(
                session=attempt_session,
                session_adapter=session_adapter,
                action_adapter=action_adapter,
                selected_avatar=selected_avatar,
                poi=poi,
                actions=route_actions,
                start_step_index=len(prefix_steps),
                start_avatar_bbox=attempt_avatar_bbox,
                start_target_bbox=attempt_target_bbox,
                hud_mask=hud_mask,
                source="frontier_solve",
                stop_on_terminal=True,
                stop_on_invalid_or_blocked=True,
                max_actions=remaining_budget,
                recent_avatar_motion=attempt_recent_avatar_motion,
                recent_target_motion=attempt_recent_target_motion,
                route_mode=route_mode,
            )
            route_steps = tuple(route_result["steps"])
            attempted_trajectory_records.append(
                _build_attempt_record_from_steps(
                    route=active_route,
                    level_id=str(level_id),
                    episode_index=int(attempt_index),
                    target_poi_id=current.target_poi_id,
                    steps=route_steps,
                    failure_reason=route_result["failure_reason"],
                )
            )
            episode_steps = prefix_steps + route_steps
            episodes.append(
                AdaptiveEpisodeResult(
                    episode_index=int(attempt_index),
                    target_sequence=(
                        AdaptiveTargetState(
                            target_poi_id=current.target_poi_id,
                            source=current.source,
                            confidence=float(current.confidence),
                            attempt_count=int(current.attempt_count),
                            last_outcome_type=current.last_outcome_type,
                            active=bool(current.active),
                        ),
                    ),
                    steps=episode_steps,
                    solved=bool(route_result["solved"]),
                    failure_reason=None if route_result["solved"] else route_result["failure_reason"],
                )
            )
            history_steps.extend(route_steps)
            carried_avatar_bbox = route_result["avatar_bbox"] or attempt_avatar_bbox
            carried_target_bbox = route_result["target_bbox"] or attempt_target_bbox
            recent_avatar_motion = route_result["recent_avatar_motion"] or attempt_recent_avatar_motion
            recent_target_motion = route_result["recent_target_motion"] or attempt_recent_target_motion
            blocked_streak = blocked_streak + 1 if any(bool(getattr(step, "blocked_action", False)) for step in route_steps) else 0

            if route_result["solved"]:
                solved = True
                failure_reason = None
                break

            # Check if this attempt achieved contact and what type of outcome
            achieved_contact = False
            achieved_meaningful_contact = False
            for step in route_steps:
                step_contact = bool(getattr(step, "contact_detected", False))
                step_outcome = str(getattr(step, "outcome_type", ""))
                if step_contact:
                    achieved_contact = True
                    if step_outcome in ("world_change", "level_transition", "reward_change"):
                        achieved_meaningful_contact = True
                        break

            # Track which interaction mode was used in this route
            route_mode = "overlap"  # default
            route_id = str(getattr(active_route, "route_id", ""))
            if route_id.startswith("touch:"):
                route_mode = "touch"
            elif route_id.startswith("overlap:"):
                route_mode = "overlap"
            poi_tried_modes[current.target_poi_id].add(route_mode)

            with open("/tmp/v5_debug.log", "a") as f:
                f.write(f"[DEBUG] Route mode={route_mode}, contact={achieved_contact}, meaningful={achieved_meaningful_contact}, tried_modes={poi_tried_modes[current.target_poi_id]}\n")
                f.flush()

            useful_prefix_length = int(route_result["useful_prefix_length"] or 0)
            if useful_prefix_length > 0:
                carried_prefix_actions = tuple(carried_prefix_actions) + tuple(route_result["executed_actions"][:useful_prefix_length])
                consecutive_no_progress = 0
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Useful prefix of {useful_prefix_length}, reset no_progress counter\n")
                    f.flush()
            elif route_result["route_closer"] or route_result["route_progress"]:
                carried_prefix_actions = tuple(carried_prefix_actions) + tuple(route_result["executed_actions"])
                consecutive_no_progress = 0
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Route progress detected, reset no_progress counter\n")
                    f.flush()
            else:
                consecutive_no_progress += 1
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] No progress, counter now {consecutive_no_progress}/5\n")
                    f.flush()

            # Decision logic for POI completion:
            # 1. If meaningful contact achieved → mark POI complete, switch to next
            # 2. If contact but not meaningful, and haven't tried overlap yet → keep trying (try overlap)
            # 3. If tried both touch and overlap without meaningful contact → mark POI complete, switch to next

            should_switch_poi = False
            if achieved_meaningful_contact:
                # Success! Mark complete and switch
                should_switch_poi = True
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] POI {current.target_poi_id}: Meaningful contact achieved, will switch\n")
                    f.flush()
            elif achieved_contact and not achieved_meaningful_contact:
                # Contact but no meaningful outcome - check if we should try overlap
                if route_mode == "touch" and "overlap" not in poi_tried_modes[current.target_poi_id]:
                    # Touch failed, haven't tried overlap yet - keep trying this POI
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] POI {current.target_poi_id}: Touch contact but not meaningful, will try overlap\n")
                        f.flush()
                    should_switch_poi = False
                else:
                    # Either: overlap also failed, or touch was the only mode - mark complete
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] POI {current.target_poi_id}: Tried both modes without meaningful outcome, will switch\n")
                        f.flush()
                    should_switch_poi = True

            if should_switch_poi:
                poi_exhausted.add(current.target_poi_id)
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] Meaningful contact achieved for POI {current.target_poi_id}, marking complete and switching\n")
                    f.flush()

                # Find next POI that hasn't been exhausted
                next_poi_candidate = None
                for poi_candidate in ranked:
                    if poi_candidate.poi_id not in poi_exhausted:
                        next_poi_candidate = poi_candidate
                        break

                if next_poi_candidate is not None:
                    # Create a new SolveTargetState for the next POI
                    current = SolveTargetState(
                        target_poi_id=next_poi_candidate.poi_id,
                        source="meaningful_contact_switch",
                        confidence=0.9,
                        attempt_count=0,
                        last_outcome_type="world_change",
                        active=True,
                        route_feasibility=None,
                    )
                    selected_target_id = current.target_poi_id

                    # IMPORTANT: Keep the successful actions from previous POI(s) as prefix
                    # This builds a cumulative trajectory: [POI1 actions] + [POI2 actions] + ...
                    # carried_prefix_actions is already updated with successful actions from this attempt
                    # DON'T reset it - we want to replay the full trajectory so far

                    # Keep carried avatar/target positions from the successful route
                    # The prefix replay will update these positions naturally
                    # carried_avatar_bbox and carried_target_bbox remain as set by route_result

                    consecutive_no_progress = 0
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Switched to new POI {current.target_poi_id} after meaningful contact, carrying {len(carried_prefix_actions)} actions: {carried_prefix_actions}\n")
                        f.flush()
                    attempt_index += 1
                    continue
                else:
                    # All POIs exhausted after meaningful contact - consider this success!
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] All POIs achieved meaningful contact, considering level complete\n")
                        f.flush()
                    failure_reason = None
                    break

            # Instead of breaking on consecutive_no_progress, try switching POIs
            if consecutive_no_progress >= 5:  # More lenient than before
                with open("/tmp/v5_debug.log", "a") as f:
                    f.write(f"[DEBUG] {consecutive_no_progress} consecutive no-progress attempts, trying to switch POI\n")
                    f.flush()

                # Try to switch to a different POI
                nxt = select_next_target(
                    _with_route_feasibility(current, False),
                    ranked,
                    tuple(history_steps),
                    contact_experiment_report,
                    mechanic_report=mechanic_report,
                )

                if nxt is not None and nxt.target_poi_id != current.target_poi_id and nxt.target_poi_id not in poi_exhausted:
                    current = nxt
                    selected_target_id = current.target_poi_id
                    carried_prefix_actions = tuple()  # Reset prefix
                    carried_avatar_bbox = None  # Reset to force reanchoring
                    carried_target_bbox = None
                    consecutive_no_progress = 0
                    with open("/tmp/v5_debug.log", "a") as f:
                        f.write(f"[DEBUG] Switched to POI {current.target_poi_id} after no progress\n")
                        f.flush()
                    attempt_index += 1
                    continue
                else:
                    # Can't switch, give up
                    failure_reason = "repeated_non_progress"
                    break

            nxt = select_next_target(
                current if useful_prefix_length <= 0 and not route_result["route_progress"] else _with_route_feasibility(current, True),
                ranked,
                tuple(history_steps),
                contact_experiment_report,
                mechanic_report=mechanic_report,
            )
            if nxt is not None and nxt.target_poi_id not in poi_exhausted:
                current = nxt
                selected_target_id = current.target_poi_id
                if nxt.target_poi_id != selected_target_id:  # Actually switched
                    carried_prefix_actions = tuple()  # Reset prefix for new POI
                    consecutive_no_progress = 0
            if route_result["failure_reason"] in {"invalid_action", "blocked_action", "terminal_failure"} and nxt is None:
                failure_reason = route_result["failure_reason"]
                break
            attempt_index += 1
        finally:
            if created_session:
                try:
                    session_adapter.close_session(attempt_session)
                except Exception:
                    pass

    if not solved and failure_reason is None:
        executed_frontier_steps = int(
            sum(1 for ep in episodes for step in tuple(ep.steps) if str(getattr(step, "source", "")) == "frontier_solve")
        )
        failure_reason = "step_budget_exhausted" if executed_frontier_steps >= int(max_steps) else "no_progress"
    diagnostics = AdaptiveDiagnostics(
        episode_count=int(len(episodes)),
        solved_episode_count=int(sum(1 for ep in episodes if bool(ep.solved))),
        failed_episode_count=int(sum(1 for ep in episodes if not bool(ep.solved))),
        retarget_count=0,
        target_switch_count=max(0, len({str(getattr(ep.target_sequence[0], "target_poi_id", "")) for ep in episodes if ep.target_sequence}) - 1),
        useful_change_count=int(sum(1 for ep in episodes for s in tuple(ep.steps) if str(getattr(s, "outcome_type", "")) in {"reward_change", "world_change", "level_transition", "terminal"})),
        no_progress_count=int(sum(1 for ep in episodes if str(getattr(ep, "failure_reason", "")) in {"no_progress", "blocked_action", "invalid_action"})),
        level_transition_count=int(sum(1 for ep in episodes for s in tuple(ep.steps) if int(getattr(s, "levels_completed_after", 0)) > int(getattr(s, "levels_completed_before", 0)))),
        terminal_count=int(sum(1 for ep in episodes for s in tuple(ep.steps) if bool(getattr(s, "terminal", False)))),
        step_budget_exhausted_count=1 if (not solved and failure_reason == "step_budget_exhausted") else 0,
        failure_reason_counts=dict(Counter(str(ep.failure_reason) for ep in episodes if ep.failure_reason)),
    )
    report = AdaptiveSolveReport(
        episodes=tuple(episodes),
        diagnostics=diagnostics,
        selected_target_id=selected_target_id,
        solved=bool(solved),
        failure_reason=None if solved else failure_reason,
    )
    object.__setattr__(report, "generated_trajectories", tuple(generated_trajectory_records))
    object.__setattr__(report, "rejected_trajectories", tuple(rejected_trajectory_records))
    object.__setattr__(report, "attempted_trajectories", tuple(attempted_trajectory_records))
    if not solved:
        object.__setattr__(
            report,
            "trajectory_stats",
            _trajectory_stats(
                level_id=str(level_id),
                solved=False,
                failure_reason=failure_reason,
                generated=tuple(generated_trajectory_records),
                attempted=tuple(attempted_trajectory_records),
            ),
        )
    return report


def extract_verified_frontier_trace(
    *,
    game_id: str,
    level_id: str,
    adaptive_report: AdaptiveSolveReport,
    source_run_id: str | None = None,
    trace_version: int = 1,
) -> SavedLevelTrace:
    solved_episode = next((ep for ep in adaptive_report.episodes if ep.solved), None)
    source = solved_episode if solved_episode is not None else (adaptive_report.episodes[0] if adaptive_report.episodes else None)
    actions = tuple(str(step.action) for step in tuple(getattr(source, "steps", ())))
    sources = tuple(str(getattr(step, "source", "frontier_solve")) for step in tuple(getattr(source, "steps", ())))
    return SavedLevelTrace(
        game_id=game_id,
        level_id=level_id,
        solved=bool(adaptive_report.solved),
        action_trace=actions,
        step_count=len(actions),
        source_run_id=source_run_id,
        trace_version=int(trace_version),
        replay_verified=False,
        action_sources=sources if sources else None,
    )


def _extract_full_level_step_trace(steps) -> tuple[LevelSolveAction, ...]:
    trace: list[LevelSolveAction] = []
    for idx, step in enumerate(tuple(steps or ())):
        pre_level = int(getattr(step, "levels_completed_before", 0))
        post_level = int(getattr(step, "levels_completed_after", 0))
        trace.append(
            LevelSolveAction(
                step_index=int(getattr(step, "step_index", idx)),
                action=str(getattr(step, "action", "")),
                target_poi_id=getattr(step, "target_poi_id", None),
                reason=str(getattr(step, "outcome_type", "")) if getattr(step, "outcome_type", None) is not None else None,
                pre_level_index=pre_level,
                post_level_index=post_level,
                source=str(getattr(step, "source", "frontier_solve")),
                pre_frame=getattr(step, "pre_frame", None),
                post_frame=getattr(step, "post_frame", None),
                invalid_action=bool(getattr(step, "invalid_action", False)),
                blocked_action=bool(getattr(step, "blocked_action", False)),
                terminal=bool(getattr(step, "terminal", False)),
                reward_before=getattr(step, "reward_before", None),
                reward_after=getattr(step, "reward_after", None),
            )
        )
    return tuple(trace)


def _extract_full_executed_action_trace(steps) -> tuple[LevelSolveAction, ...]:
    return _extract_full_level_step_trace(steps)


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
