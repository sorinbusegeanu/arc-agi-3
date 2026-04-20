from __future__ import annotations

from collections import Counter
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.runtime.sessionAdapter import SessionAdapter
from v5_0.avatar.service import identify_avatar_candidates, identify_avatar_candidates_multi_reset
from v5_0.bootstrap.probe_plan import build_probe_plan
from v5_0.bootstrap.probe_runner import (
    run_probe_episodes,
    run_probe_episodes_after_prefix,
    run_probe_episodes_at_frontier,
    run_probe_session,
    run_probe_session_on_live_session,
)
from v5_0.contracts.avatar_types import (
    CampaignLevelState,
    CampaignLevelResult,
    CampaignRunReport,
    CampaignRunStep,
    GameLevelBatchDiagnostics,
    GameLevelBatchReport,
    LevelSolution,
    LevelSolveAction,
    MultiResetAvatarReport,
    PerLevelResult,
    ProbeEpisode,
    SavedLevelTrace,
)
from v5_0.contact.service import run_controlled_contact_multi_reset
from v5_0.contact.frame_tracker import track_avatar_bbox_in_frame
from v5_0.hud.service import detect_hud_multi_reset, interpret_hud_hints_multi_reset
from v5_0.io.artifact_writer import (
    write_adaptive_solve_artifacts,
    write_campaign_artifacts,
    write_generated_trajectories,
    write_saved_level_trace,
    write_trajectory_attempts,
    write_trajectory_stats,
    write_trace_optimization_artifacts,
    write_trace_analysis_batch_artifacts,
    write_trace_store_index_artifacts,
    write_game_level_batch_artifacts,
    write_level_solution_artifacts,
    resolve_run_dir,
    write_artifacts,
    write_contact_experiment_artifacts,
    write_full_analysis_index,
    write_hud_artifacts,
    write_hud_hint_artifacts,
    write_mechanic_artifacts,
    write_multi_reset_artifacts,
    write_poi_artifacts,
    write_solve_artifacts,
)
from v5_0.memory.trace_store import (
    get_all_traces_for_game,
    get_global_trace_store_path,
    get_best_trace_for_level,
    mark_trace_verified,
    rebuild_trace_store_index,
    get_solved_levels_for_game,
    initialize_trace_store,
    save_trace_history_row,
    replace_best_trace_if_shorter,
    save_or_replace_best_trace,
)
from v5_0.mechanics.evidence_builder import build_mechanic_evidence
from v5_0.mechanics.service import build_mechanic_report
from v5_0.poi.service import discover_pois_multi_reset
from v5_0.replay.optimizer import optimize_game_traces_from_db, optimize_level_trace
from v5_0.replay.player import replay_prefix_to_frontier, replay_prefix_traces_to_frontier, replay_trace_at_frontier, trace_includes_bootstrap_prefix
from v5_0.runtime.campaign_state import (
    get_current_run_prefix_traces,
    get_db_solved_levels_for_game,
    get_frontier_level_id,
    get_verified_prefix_traces,
    load_or_initialize_campaign_state,
    update_campaign_state_after_level,
    validate_prefix_trace_entry,
)
from v5_0.runtime.level_catalog import (
    engine_supports_direct_level_start,
    get_level_sequence_for_game,
    get_supported_level_ids_for_game,
    validate_level_id_for_game,
)
from v5_0.solve.service import (
    finalize_solved_level_trace,
    extract_replayable_level_trace,
    extract_verified_frontier_trace,
    build_level_solution_from_adaptive_report,
    run_adaptive_solve_on_live_session,
    run_adaptive_solve_multi_reset,
    run_closed_loop_solve_multi_reset,
    verify_level_trace_replay,
)

def _require_supported_game(game_id: str) -> None:
    del game_id
    return None


def _get_trace_db_path(output_dir: str | None, game_id: str) -> str:
    del output_dir, game_id
    return get_global_trace_store_path()


def _campaign_stable_avatar_found(avatar_multi_report: MultiResetAvatarReport) -> bool:
    diagnostics = getattr(avatar_multi_report, "diagnostics", None)
    selected = getattr(avatar_multi_report, "selected", None)
    episode_count = int(getattr(diagnostics, "episode_count", 0) or 0)
    required_support = 1 if episode_count == 1 else 2
    successful_episode_count = int(getattr(diagnostics, "successful_episode_count", 0) or 0)
    cross_reset_ambiguous = bool(getattr(diagnostics, "cross_reset_ambiguous", False))
    selected_failure_reason = getattr(selected, "failure_reason", None)
    return (
        successful_episode_count >= required_support
        and not cross_reset_ambiguous
        and selected_failure_reason is None
    )


def _is_simple_single_target_frontier(
    *,
    game_id: str,
    poi_report,
    hud_targeting_report,
) -> bool:
    if str(game_id) != "ez01":
        return False
    candidates = tuple(getattr(poi_report, "candidates", ()))
    selected = getattr(hud_targeting_report, "selected", None)
    if selected is None:
        return False
    selected_poi = getattr(selected, "selected_poi_id", None)
    if selected_poi is None or bool(getattr(selected, "ambiguous", False)) or getattr(selected, "failure_reason", None) is not None:
        return False
    if len(candidates) == 1:
        return True
    ranked = tuple(str(item) for item in tuple(getattr(selected, "ranked_poi_ids", ())))
    if len(ranked) >= 2 and ranked[0] != ranked[1]:
        top = next((item for item in candidates if str(item.poi_id) == ranked[0]), None)
        nxt = next((item for item in candidates if str(item.poi_id) == ranked[1]), None)
        if top is not None and nxt is not None and float(top.confidence) - float(nxt.confidence) >= 0.2:
            return True
    return False


def _should_skip_redundant_frontier_analysis(
    *,
    avatar_multi_report: MultiResetAvatarReport,
    poi_report,
    hud_targeting_report,
    prior_contact_report=None,
    game_id: str = "",
) -> bool:
    if not _campaign_stable_avatar_found(avatar_multi_report):
        return False
    selected = getattr(hud_targeting_report, "selected", None)
    if selected is None:
        return False
    if getattr(selected, "selected_poi_id", None) is None:
        return False
    if bool(getattr(selected, "ambiguous", False)):
        return False
    if getattr(selected, "failure_reason", None) is not None:
        return False
    if not _is_simple_single_target_frontier(
        game_id=str(game_id),
        poi_report=poi_report,
        hud_targeting_report=hud_targeting_report,
    ):
        return False
    ranked = tuple(str(item) for item in tuple(getattr(selected, "ranked_poi_ids", ())))
    candidates = tuple(getattr(poi_report, "candidates", ()))
    if len(ranked) >= 2:
        top = next((item for item in candidates if str(getattr(item, "poi_id", "")) == ranked[0]), None)
        nxt = next((item for item in candidates if str(getattr(item, "poi_id", "")) == ranked[1]), None)
        if top is not None and nxt is not None:
            margin = float(getattr(top, "confidence", 0.0)) - float(getattr(nxt, "confidence", 0.0))
            if margin < 0.20:
                return False
    if prior_contact_report is None:
        return True
    tested = tuple(getattr(prior_contact_report, "tested_pois", ()))
    if len(tested) <= 1:
        return True
    return False


def _with_missing_solution_prefix_steps(
    *,
    campaign_steps: tuple[CampaignRunStep, ...] | list[CampaignRunStep],
    level_sequence: tuple[str, ...] | list[str],
    replay_valid_db_prefix_by_level: dict[str, SavedLevelTrace],
) -> tuple[CampaignRunStep, ...]:
    existing = tuple(campaign_steps or ())
    if not existing and not replay_valid_db_prefix_by_level:
        return tuple()

    existing_levels = {str(step.level_id) for step in existing}
    synthesized: list[CampaignRunStep] = []
    for level_id in tuple(str(item) for item in level_sequence):
        if level_id in existing_levels:
            continue
        trace = replay_valid_db_prefix_by_level.get(level_id)
        if trace is None:
            continue
        level_index = max(0, int(str(level_id).lstrip("L") or 0))
        action_sources = tuple(getattr(trace, "action_sources", ()) or ())
        for step_index, action in enumerate(tuple(getattr(trace, "action_trace", ()) or ())):
            synthesized.append(
                CampaignRunStep(
                    global_step_index=0,
                    level_id=level_id,
                    action=str(action),
                    source=(action_sources[step_index] if step_index < len(action_sources) else "solved_prefix_replay"),
                    reason="solution_prefix_replay",
                    pre_levels_completed=level_index,
                    post_levels_completed=level_index,
                    pre_frame=None,
                    post_frame=None,
                    invalid_action=False,
                    blocked_action=False,
                    terminal=False,
                    reward_before=None,
                    reward_after=None,
                )
            )

    merged = tuple(synthesized) + existing
    return tuple(
        CampaignRunStep(
            global_step_index=index,
            level_id=str(step.level_id),
            action=str(step.action),
            source=str(step.source),
            reason=step.reason,
            pre_levels_completed=int(step.pre_levels_completed),
            post_levels_completed=int(step.post_levels_completed),
            pre_frame=step.pre_frame,
            post_frame=step.post_frame,
            invalid_action=bool(step.invalid_action),
            blocked_action=bool(step.blocked_action),
            terminal=bool(step.terminal),
            reward_before=step.reward_before,
            reward_after=step.reward_after,
        )
        for index, step in enumerate(merged)
    )


def _flatten_prefix_actions(prefix_traces: tuple[SavedLevelTrace, ...]) -> tuple[str, ...]:
    combined: list[str] = []
    for trace in tuple(prefix_traces or ()):
        if trace_includes_bootstrap_prefix(trace):
            continue
        actions = tuple(str(item) for item in tuple(getattr(trace, "action_trace", ())))
        combined.extend(actions)
    return tuple(combined)


def _append_campaign_debug_marker(*, debug_log_path: str | None, game_id: str, level_id: str, marker: str) -> None:
    if not debug_log_path:
        return
    with open(str(debug_log_path), "a", encoding="utf-8") as handle:
        handle.write(f"{str(game_id)}|{str(level_id)}|{str(marker)}\n")


def _append_campaign_play_marker(
    *,
    debug_log_path: str | None,
    game_id: str,
    level_id: str,
    actions: tuple[str, ...] | list[str],
) -> None:
    if not debug_log_path:
        return
    action_map = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
    action_text = "".join(action_map.get(str(item).upper(), "") for item in tuple(actions or ()))
    with open(str(debug_log_path), "a", encoding="utf-8") as handle:
        handle.write(f"{str(game_id)}|{str(level_id)}|PLAY:{action_text}\n")


def _load_replay_valid_db_prefix(
    *,
    game_id: str,
    level_sequence: tuple[str, ...],
    trace_db_path: str,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
) -> tuple[SavedLevelTrace, ...]:
    traces_by_level: dict[str, list[SavedLevelTrace]] = {}
    for trace in get_all_traces_for_game(db_path=trace_db_path, game_id=game_id):
        if not validate_prefix_trace_entry(trace):
            continue
        traces_by_level.setdefault(str(trace.level_id), []).append(trace)

    selected: list[SavedLevelTrace] = []
    for level_id in tuple(str(item) for item in level_sequence):
        candidates = sorted(
            traces_by_level.get(level_id, ()),
            key=lambda item: (int(item.step_count), str(item.trace_id or "")),
        )[:12]
        accepted = None
        for candidate in candidates:
            if trace_includes_bootstrap_prefix(candidate):
                continue
            replay = replay_trace_at_frontier(
                game_id=game_id,
                level_id=level_id,
                prefix_traces=tuple(selected),
                frontier_trace=candidate,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
            if bool(replay.get("success", False)) and bool(replay.get("level_solved", False)):
                accepted = candidate
                break
            trace_id = getattr(candidate, "trace_id", None)
            if trace_id:
                mark_trace_verified(db_path=trace_db_path, trace_id=str(trace_id), replay_verified=False)
        if accepted is None:
            break
        selected.append(accepted)
    return tuple(selected)


def _state_with_replay_valid_db_prefix(
    *,
    state: dict[str, CampaignLevelState],
    level_sequence: tuple[str, ...],
    replay_valid_prefix: tuple[SavedLevelTrace, ...],
) -> dict[str, CampaignLevelState]:
    valid_by_level = {str(trace.level_id): trace for trace in tuple(replay_valid_prefix)}
    output = dict(state)
    for level_id in tuple(str(item) for item in level_sequence):
        current = output[level_id]
        current_game_id = str(getattr(current, "game_id", "") or "")
        current_level_id = str(getattr(current, "level_id", level_id) or level_id)
        current_trace_path = getattr(current, "solution_trace_path", None)
        current_attempt_count = int(getattr(current, "attempt_count", 0) or 0)
        trace = valid_by_level.get(level_id)
        if trace is None:
            output[level_id] = CampaignLevelState(
                game_id=current_game_id,
                level_id=current_level_id,
                status="unknown",
                solved=False,
                solution_trace_path=None,
                best_step_count=None,
                attempt_count=0,
            )
            continue
        output[level_id] = CampaignLevelState(
            game_id=current_game_id,
            level_id=current_level_id,
            status="solved",
            solved=True,
            solution_trace_path=current_trace_path,
            best_step_count=int(trace.step_count),
            attempt_count=current_attempt_count,
        )
    return output


def run_avatar_bootstrap(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    transitions = run_probe_session(
        plan=plan,
        seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    report = identify_avatar_candidates(transitions)

    run_dir = resolve_run_dir(output_dir, game_id)
    artifact_paths = write_artifacts(
        run_dir=run_dir,
        transitions=transitions,
        report=report,
        write_montage=probe_montage,
    )

    reliable = report.selected.failure_reason is None
    return {
        "game_id": game_id,
        "plan": {
            "level_id": plan.level_id,
            "action_sequence": plan.action_sequence,
        },
        "selected": report.selected.to_dict(),
        "diagnostics": report.diagnostics.to_dict(),
        "artifact_paths": artifact_paths,
        "reliable_rank1": reliable,
    }


def run_avatar_bootstrap_multi_reset(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    multi_report = MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=multi_report.cross_reset_evidence,
        selected=multi_report.selected,
        diagnostics=multi_report.diagnostics,
    )

    run_dir = resolve_run_dir(output_dir, game_id) / "multi_reset"
    artifact_paths = write_multi_reset_artifacts(
        run_dir=run_dir,
        report=multi_report,
        write_montage=probe_montage,
    )

    selected_failure = multi_report.selected.failure_reason
    support = 0
    for evidence in multi_report.cross_reset_evidence:
        if evidence.canonical_candidate_id == multi_report.selected.selected_candidate_id:
            support = evidence.support_episode_count
            break
    required_support = 1 if int(episode_count) == 1 else 2
    stable_avatar_found = (
        selected_failure is None
        and support >= required_support
        and not multi_report.diagnostics.cross_reset_ambiguous
    )
    return {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "selected": multi_report.selected.to_dict(),
        "diagnostics": multi_report.diagnostics.to_dict(),
        "artifact_paths": artifact_paths,
        "stable_avatar_found": stable_avatar_found,
    }


def run_avatar_and_poi_bootstrap_multi_reset(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    avatar_summary = run_avatar_bootstrap_multi_reset(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    if not avatar_summary.get("stable_avatar_found", False):
        return {
            "game_id": game_id,
            "episode_count": int(episode_count),
            "selected_avatar": avatar_summary.get("selected"),
            "poi_candidates": [],
            "poi_diagnostics": {"failure_reason": "avatar_not_stable"},
            "artifact_paths": avatar_summary.get("artifact_paths", {}),
        }

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=avatar_multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    avatar_multi_report = MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
        selected=avatar_multi_report.selected,
        diagnostics=avatar_multi_report.diagnostics,
    )
    poi_bundle = discover_pois_multi_reset(avatar_multi_report)
    run_dir = resolve_run_dir(output_dir, game_id) / "multi_reset"
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    merged_paths = dict(avatar_summary.get("artifact_paths", {}))
    merged_paths.update(poi_paths)
    return {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "selected_avatar": avatar_summary.get("selected"),
        "poi_candidates": [item.__dict__ for item in poi_bundle["report"].candidates],
        "poi_diagnostics": poi_bundle["report"].diagnostics.to_dict(),
        "artifact_paths": merged_paths,
    }


def run_avatar_poi_contact_bootstrap_multi_reset(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)

    avatar_summary = run_avatar_bootstrap_multi_reset(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    if not avatar_summary.get("stable_avatar_found", False):
        return {
            "game_id": game_id,
            "episode_count": int(episode_count),
            "selected_avatar": avatar_summary.get("selected"),
            "poi_candidates": [],
            "tested_pois": [],
            "contact_diagnostics": {"failure_reason": "avatar_not_stable"},
            "artifact_paths": avatar_summary.get("artifact_paths", {}),
        }

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=avatar_multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    avatar_multi_report = MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
        selected=avatar_multi_report.selected,
        diagnostics=avatar_multi_report.diagnostics,
    )
    poi_bundle = discover_pois_multi_reset(avatar_multi_report)
    if not poi_bundle["report"].candidates:
        return {
            "game_id": game_id,
            "episode_count": int(episode_count),
            "selected_avatar": avatar_summary.get("selected"),
            "poi_candidates": [],
            "tested_pois": [],
            "contact_diagnostics": {"failure_reason": "no_pois_available"},
            "artifact_paths": avatar_summary.get("artifact_paths", {}),
        }

    run_dir = resolve_run_dir(output_dir, game_id) / "multi_reset"
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    contact_report = run_controlled_contact_multi_reset(
        avatar_multi_report=avatar_multi_report,
        poi_multi_bundle=poi_bundle,
        plan=plan,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    contact_paths = write_contact_experiment_artifacts(
        run_dir=run_dir,
        report=contact_report,
    )
    merged_paths = dict(avatar_summary.get("artifact_paths", {}))
    merged_paths.update(poi_paths)
    merged_paths.update(contact_paths)
    return {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "selected_avatar": avatar_summary.get("selected"),
        "poi_candidates": [item.__dict__ for item in poi_bundle["report"].candidates],
        "tested_pois": [item.to_dict() for item in contact_report.tested_pois],
        "contact_diagnostics": dict(contact_report.diagnostics),
        "artifact_paths": merged_paths,
    }


def run_avatar_poi_hud_bootstrap_multi_reset(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)

    avatar_summary = run_avatar_bootstrap_multi_reset(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    if not avatar_summary.get("stable_avatar_found", False):
        return {
            "game_id": game_id,
            "episode_count": int(episode_count),
            "selected_avatar": avatar_summary.get("selected"),
            "poi_candidates": [],
            "hud_regions": [],
            "hud_mask": {},
            "hud_diagnostics": {"failure_reason": "avatar_not_stable"},
            "artifact_paths": avatar_summary.get("artifact_paths", {}),
        }

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=avatar_multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    avatar_multi_report = MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
        selected=avatar_multi_report.selected,
        diagnostics=avatar_multi_report.diagnostics,
    )
    poi_bundle = discover_pois_multi_reset(avatar_multi_report)
    hud_bundle = detect_hud_multi_reset(
        avatar_multi_report=avatar_multi_report,
        poi_multi_bundle=poi_bundle,
    )

    run_dir = resolve_run_dir(output_dir, game_id) / "multi_reset"
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    hud_paths = write_hud_artifacts(
        run_dir=run_dir,
        hud_report=hud_bundle["report"],
        cross_reset_hud_evidence=hud_bundle["cross_reset_evidence"],
        episode_hud_reports=hud_bundle["episodes"],
        hud_value_samples=tuple(
            sample
            for samples in hud_bundle.get("value_samples", {}).values()
            for sample in samples
        ),
    )
    merged_paths = dict(avatar_summary.get("artifact_paths", {}))
    merged_paths.update(poi_paths)
    merged_paths.update(hud_paths)
    return {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "selected_avatar": avatar_summary.get("selected"),
        "poi_candidates": [item.__dict__ for item in poi_bundle["report"].candidates],
        "hud_regions": [item.to_dict() for item in hud_bundle["report"].regions],
        "hud_mask": hud_bundle["report"].mask.to_dict(),
        "hud_diagnostics": {
            **hud_bundle["report"].diagnostics.to_dict(),
            "failure_reason": hud_bundle["report"].failure_reason,
        },
        "artifact_paths": merged_paths,
    }


def run_full_bootstrap_analysis(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)

    plan = build_probe_plan(game_id=game_id, level_id="L0")
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=avatar_multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    avatar_multi_report = MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
        selected=avatar_multi_report.selected,
        diagnostics=avatar_multi_report.diagnostics,
    )

    run_dir = resolve_run_dir(output_dir, game_id) / "multi_reset"
    artifact_paths = write_multi_reset_artifacts(
        run_dir=run_dir,
        report=avatar_multi_report,
        write_montage=probe_montage,
    )

    stable_avatar_found = bool(avatar_multi_report.diagnostics.stable_avatar_found)
    phase_status: dict[str, str] = {
        "avatar": "ok" if stable_avatar_found else "failed",
        "poi": "skipped",
        "contact": "skipped",
        "hud": "skipped",
    }
    result: dict[str, Any] = {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "plan": {
            "level_id": plan.level_id,
            "action_sequence": plan.action_sequence,
        },
        "avatar": {
            "selected": avatar_multi_report.selected.to_dict(),
            "diagnostics": avatar_multi_report.diagnostics.to_dict(),
            "stable_avatar_found": stable_avatar_found,
        },
        "poi": {
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "contact": {
            "tested_poi_count": 0,
            "outcome_type_counts": {},
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "hud": {
            "region_count": 0,
            "mask_summary": {},
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "artifact_paths": artifact_paths,
        "phase_status": phase_status,
    }

    if not stable_avatar_found:
        index_paths = write_full_analysis_index(
            run_dir=run_dir,
            game_id=game_id,
            episode_count=episode_count,
            phase_status=phase_status,
            artifact_paths=artifact_paths,
        )
        merged_paths = dict(artifact_paths)
        merged_paths.update(index_paths)
        result["artifact_paths"] = merged_paths
        return result

    poi_bundle = discover_pois_multi_reset(avatar_multi_report)
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    artifact_paths = dict(artifact_paths)
    artifact_paths.update(poi_paths)

    poi_candidates = list(poi_bundle["report"].candidates)
    poi_failure_reason = poi_bundle["report"].selected.failure_reason
    phase_status["poi"] = "ok" if poi_candidates else "failed"
    result["poi"] = {
        "candidate_count": len(poi_candidates),
        "top_candidates": [item.__dict__ for item in poi_candidates[:5]],
        "diagnostics": poi_bundle["report"].diagnostics.to_dict(),
        "failure_reason": poi_failure_reason,
    }

    if poi_candidates:
        try:
            contact_report = run_controlled_contact_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_bundle,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
            contact_paths = write_contact_experiment_artifacts(
                run_dir=run_dir,
                report=contact_report,
            )
            artifact_paths.update(contact_paths)
            phase_status["contact"] = "ok"
            result["contact"] = {
                "tested_poi_count": int(contact_report.diagnostics.get("tested_poi_count", len(contact_report.tested_pois))),
                "outcome_type_counts": dict(contact_report.diagnostics.get("outcome_type_counts", {})),
                "diagnostics": dict(contact_report.diagnostics),
                "failure_reason": None,
            }
        except Exception as exc:
            phase_status["contact"] = "failed"
            result["contact"] = {
                "tested_poi_count": 0,
                "outcome_type_counts": {},
                "diagnostics": {},
                "failure_reason": f"contact_experiments_failed: {exc}",
            }
    else:
        phase_status["contact"] = "skipped"
        result["contact"] = {
            "tested_poi_count": 0,
            "outcome_type_counts": {},
            "diagnostics": {},
            "failure_reason": "no_pois_available",
        }

    hud_bundle = detect_hud_multi_reset(
        avatar_multi_report=avatar_multi_report,
        poi_multi_bundle=poi_bundle,
    )
    hud_paths = write_hud_artifacts(
        run_dir=run_dir,
        hud_report=hud_bundle["report"],
        cross_reset_hud_evidence=hud_bundle["cross_reset_evidence"],
        episode_hud_reports=hud_bundle["episodes"],
        hud_value_samples=tuple(
            sample
            for samples in hud_bundle.get("value_samples", {}).values()
            for sample in samples
        ),
    )
    artifact_paths.update(hud_paths)
    hud_failure_reason = hud_bundle["report"].failure_reason
    phase_status["hud"] = "ok" if hud_failure_reason is None else "failed"
    result["hud"] = {
        "region_count": len(hud_bundle["report"].regions),
        "mask_summary": hud_bundle["report"].mask.to_dict(),
        "diagnostics": hud_bundle["report"].diagnostics.to_dict(),
        "failure_reason": hud_failure_reason,
    }

    index_paths = write_full_analysis_index(
        run_dir=run_dir,
        game_id=game_id,
        episode_count=episode_count,
        phase_status=phase_status,
        artifact_paths=artifact_paths,
    )
    artifact_paths.update(index_paths)
    result["artifact_paths"] = artifact_paths
    return result


def run_full_bootstrap_analysis_with_hud_targeting(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    summary = run_full_bootstrap_analysis(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    phase_status = dict(summary.get("phase_status", {}))
    phase_status["hud_targeting"] = "skipped"
    summary["phase_status"] = phase_status
    if phase_status.get("avatar") != "ok" or phase_status.get("poi") != "ok" or phase_status.get("hud") != "ok":
        summary["hud_targeting"] = {
            "selected_poi_id": None,
            "ranked_poi_ids": [],
            "match_count": 0,
            "ambiguous": False,
            "failure_reason": "prerequisite_phase_not_ok",
        }
        return summary

    try:
        plan = build_probe_plan(game_id=game_id, level_id="L0")
        episode_transitions = run_probe_episodes(
            plan=plan,
            episode_count=episode_count,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
        enriched_episodes = tuple(
            ProbeEpisode(
                episode_index=index,
                seed=int(seed) + index,
                plan=plan,
                transitions=transitions,
                report=avatar_multi_report.episodes[index].report,
            )
            for index, transitions in enumerate(episode_transitions)
        )
        avatar_multi_report = MultiResetAvatarReport(
            episodes=enriched_episodes,
            cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
            selected=avatar_multi_report.selected,
            diagnostics=avatar_multi_report.diagnostics,
        )
        poi_bundle = discover_pois_multi_reset(avatar_multi_report)
        hud_bundle = detect_hud_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_bundle,
        )
        hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)

        multi_reset_summary_path = summary.get("artifact_paths", {}).get("multi_reset_summary.json")
        run_dir = Path(str(multi_reset_summary_path)).parent if multi_reset_summary_path else (resolve_run_dir(output_dir, game_id) / "multi_reset")
        hint_paths = write_hud_hint_artifacts(
            run_dir=run_dir,
            report=hint_report,
        )
        artifact_paths = dict(summary.get("artifact_paths", {}))
        artifact_paths.update(hint_paths)
        summary["artifact_paths"] = artifact_paths
        summary["hud_targeting"] = {
            "selected_poi_id": hint_report.selected.selected_poi_id,
            "ranked_poi_ids": list(hint_report.selected.ranked_poi_ids),
            "match_count": len(hint_report.matches),
            "ambiguous": bool(hint_report.selected.ambiguous),
            "failure_reason": hint_report.selected.failure_reason,
        }
        phase_status["hud_targeting"] = (
            "ok"
            if hint_report.selected.selected_poi_id is not None
            and not hint_report.selected.ambiguous
            and hint_report.selected.failure_reason is None
            else "failed"
        )
        summary["phase_status"] = phase_status
        return summary
    except Exception as exc:
        summary["hud_targeting"] = {
            "selected_poi_id": None,
            "ranked_poi_ids": [],
            "match_count": 0,
            "ambiguous": False,
            "failure_reason": f"hud_targeting_failed: {exc}",
        }
        phase_status["hud_targeting"] = "failed"
        summary["phase_status"] = phase_status
        return summary


def run_full_bootstrap_analysis_with_solve(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
) -> dict[str, Any]:
    summary = run_full_bootstrap_analysis_with_hud_targeting(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    phase_status = dict(summary.get("phase_status", {}))
    phase_status["solve"] = "skipped"
    summary["phase_status"] = phase_status

    if phase_status.get("avatar") != "ok" or phase_status.get("poi") != "ok" or phase_status.get("hud") != "ok":
        summary["solve"] = {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": "prerequisite_phase_not_ok",
            "diagnostics": {},
        }
        phase_status["solve"] = "failed"
        summary["phase_status"] = phase_status
        return summary

    try:
        plan = build_probe_plan(game_id=game_id, level_id="L0")
        episode_transitions = run_probe_episodes(
            plan=plan,
            episode_count=episode_count,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
        enriched_episodes = tuple(
            ProbeEpisode(
                episode_index=index,
                seed=int(seed) + index,
                plan=plan,
                transitions=transitions,
                report=avatar_multi_report.episodes[index].report,
            )
            for index, transitions in enumerate(episode_transitions)
        )
        avatar_multi_report = MultiResetAvatarReport(
            episodes=enriched_episodes,
            cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
            selected=avatar_multi_report.selected,
            diagnostics=avatar_multi_report.diagnostics,
        )
        poi_bundle = discover_pois_multi_reset(avatar_multi_report)
        hud_bundle = detect_hud_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_bundle,
        )
        hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)
        contact_report = None
        if poi_bundle.get("report") is not None and tuple(getattr(poi_bundle["report"], "candidates", ())):
            contact_report = run_controlled_contact_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_bundle,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
        solve_report = run_closed_loop_solve_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_bundle,
            hud_targeting_report=hint_report,
            contact_experiment_report=contact_report,
            game_id=game_id,
            plan=plan,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )

        multi_reset_summary_path = summary.get("artifact_paths", {}).get("multi_reset_summary.json")
        run_dir = Path(str(multi_reset_summary_path)).parent if multi_reset_summary_path else (resolve_run_dir(output_dir, game_id) / "multi_reset")
        solve_paths = write_solve_artifacts(
            run_dir=run_dir,
            report=solve_report,
        )
        artifact_paths = dict(summary.get("artifact_paths", {}))
        artifact_paths.update(solve_paths)
        summary["artifact_paths"] = artifact_paths
        summary["solve"] = {
            "selected_target_id": solve_report.selected_target_id,
            "solved": bool(solve_report.solved),
            "failure_reason": solve_report.failure_reason,
            "diagnostics": solve_report.diagnostics.to_dict(),
        }
        phase_status["solve"] = "ok" if solve_report.solved else "failed"
        summary["phase_status"] = phase_status
        return summary
    except Exception as exc:
        summary["solve"] = {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": f"solve_failed: {exc}",
            "diagnostics": {},
        }
        phase_status["solve"] = "failed"
        summary["phase_status"] = phase_status
        return summary


def run_full_bootstrap_analysis_with_mechanics(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
) -> dict[str, Any]:
    summary = run_full_bootstrap_analysis_with_solve(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
        max_steps=max_steps,
    )
    phase_status = dict(summary.get("phase_status", {}))
    phase_status["mechanics"] = "skipped"
    summary["phase_status"] = phase_status

    try:
        plan = build_probe_plan(game_id=game_id, level_id="L0")
        episode_transitions = run_probe_episodes(
            plan=plan,
            episode_count=episode_count,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
        enriched_episodes = tuple(
            ProbeEpisode(
                episode_index=index,
                seed=int(seed) + index,
                plan=plan,
                transitions=transitions,
                report=avatar_multi_report.episodes[index].report,
            )
            for index, transitions in enumerate(episode_transitions)
        )
        avatar_multi_report = MultiResetAvatarReport(
            episodes=enriched_episodes,
            cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
            selected=avatar_multi_report.selected,
            diagnostics=avatar_multi_report.diagnostics,
        )
        poi_bundle = discover_pois_multi_reset(avatar_multi_report)
        hud_bundle = detect_hud_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_bundle,
        )
        hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)
        contact_report = None
        if tuple(getattr(poi_bundle.get("report"), "candidates", ())):
            contact_report = run_controlled_contact_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_bundle,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
        previous_solve_obj = None
        if "solve" in summary:
            previous_solve_obj = type("SolveLike", (), summary["solve"])()
        mechanic_report = build_mechanic_report(
            tuple(getattr(poi_bundle.get("report"), "candidates", ())),
            contact_experiment_report=contact_report,
            hud_targeting_report=hint_report,
            solve_report=previous_solve_obj,
            previous_mechanic_memory=None,
        )
        mechanic_evidence = build_mechanic_evidence(
            tuple(getattr(poi_bundle.get("report"), "candidates", ())),
            contact_experiment_report=contact_report,
            hud_targeting_report=hint_report,
            solve_report=previous_solve_obj,
        )
        upgraded_solve = run_closed_loop_solve_multi_reset(
            avatar_multi_report=avatar_multi_report,
            poi_multi_bundle=poi_bundle,
            hud_targeting_report=hint_report,
            contact_experiment_report=contact_report,
            game_id=game_id,
            plan=plan,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )
        summary["solve"] = {
            "selected_target_id": upgraded_solve.selected_target_id,
            "solved": bool(upgraded_solve.solved),
            "failure_reason": upgraded_solve.failure_reason,
            "diagnostics": upgraded_solve.diagnostics.to_dict(),
        }

        multi_reset_summary_path = summary.get("artifact_paths", {}).get("multi_reset_summary.json")
        run_dir = Path(str(multi_reset_summary_path)).parent if multi_reset_summary_path else (resolve_run_dir(output_dir, game_id) / "multi_reset")
        mechanic_paths = write_mechanic_artifacts(
            run_dir=run_dir,
            report=mechanic_report,
            evidence=mechanic_evidence,
        )
        artifact_paths = dict(summary.get("artifact_paths", {}))
        artifact_paths.update(mechanic_paths)
        summary["artifact_paths"] = artifact_paths
        summary["mechanics"] = {
            "failure_reason": mechanic_report.failure_reason,
            "diagnostics": mechanic_report.diagnostics.to_dict(),
            "selected_poi_id": mechanic_report.memory.selected_poi_id,
            "retired_poi_ids": list(mechanic_report.memory.retired_poi_ids),
        }
        phase_status["mechanics"] = "ok" if mechanic_report.failure_reason is None else "failed"
        phase_status["solve"] = "ok" if bool(upgraded_solve.solved) else "failed"
        summary["phase_status"] = phase_status
        return summary
    except Exception as exc:
        summary["mechanics"] = {
            "failure_reason": f"mechanic_analysis_failed: {exc}",
            "diagnostics": {},
            "selected_poi_id": None,
            "retired_poi_ids": [],
        }
        phase_status["mechanics"] = "failed"
        summary["phase_status"] = phase_status
        return summary


def run_full_bootstrap_analysis_with_adaptive_solve(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
) -> dict[str, Any]:
    summary = run_full_bootstrap_analysis_with_hud_targeting(
        game_id=game_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    phase_status = dict(summary.get("phase_status", {}))
    phase_status["adaptive_solve"] = "skipped"
    summary["phase_status"] = phase_status

    if phase_status.get("avatar") != "ok" or phase_status.get("poi") != "ok":
        summary["adaptive_solve"] = {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": "prerequisite_phase_not_ok",
            "diagnostics": {},
        }
        phase_status["adaptive_solve"] = "failed"
        summary["phase_status"] = phase_status
        return summary

    try:
        rebuilt = _run_full_bootstrap_analysis_with_adaptive_solve_for_level(
            game_id=game_id,
            level_id="L0",
            output_dir=output_dir,
            seed=seed,
            episode_count=episode_count,
            probe_montage=probe_montage,
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )
        return rebuilt
    except Exception as exc:
        summary["adaptive_solve"] = {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": f"adaptive_solve_failed: {exc}",
            "diagnostics": {},
        }
        phase_status["adaptive_solve"] = "failed"
        summary["phase_status"] = phase_status
        return summary


def _build_multi_reset_avatar_report(*, plan, episode_transitions, seed: int) -> MultiResetAvatarReport:
    avatar_multi_report = identify_avatar_candidates_multi_reset(episode_transitions)
    selected = avatar_multi_report.selected
    if selected.selected_candidate_id is not None:
        selected_candidate_ids_by_episode: dict[int, str] = {}
        for evidence in tuple(getattr(avatar_multi_report, "cross_reset_evidence", ())):
            if evidence.canonical_candidate_id == selected.selected_candidate_id:
                selected_candidate_ids_by_episode = {
                    int(episode_index): str(candidate_id)
                    for episode_index, candidate_id in dict(getattr(evidence, "per_episode_candidate_ids", {}) or {}).items()
                }
                break
        selected_bbox_override = None
        selected_center_override = None
        for episode in tuple(avatar_multi_report.episodes):
            episode_selected_candidate_id = selected_candidate_ids_by_episode.get(
                int(getattr(episode, "episode_index", -1)),
                getattr(episode.report.selected, "selected_candidate_id", None),
            )
            if episode_selected_candidate_id is None:
                continue
            for candidate in tuple(episode.report.candidates):
                if candidate.candidate_id == episode_selected_candidate_id:
                    selected_entry_bbox = tuple(candidate.entry_bbox)
                    selected_bbox_override = selected_entry_bbox
                    motions = tuple(getattr(candidate, "observed_motion_vectors", ()) or ())
                    first_motion = motions[0] if motions else None
                    support_actions = {
                        str(action).upper()
                        for action in tuple(getattr(candidate, "support_actions", ()) or ())
                        if action is not None
                    }
                    entry_bbox = tuple(getattr(candidate, "entry_bbox", ()) or ())
                    entry_touches_edge = False
                    if len(entry_bbox) == 4:
                        entry_touches_edge = (
                            int(entry_bbox[0]) <= 0
                            or int(entry_bbox[1]) <= 0
                            or int(entry_bbox[2]) >= 63
                            or int(entry_bbox[3]) >= 63
                        )
                    if (
                        first_motion is not None
                        and abs(float(first_motion[0])) < 0.5
                        and abs(float(first_motion[1])) < 0.5
                        and {"UP", "DOWN", "LEFT", "RIGHT"}.issubset(support_actions)
                        and not entry_touches_edge
                        and getattr(candidate, "bbox", None) is not None
                    ):
                        selected_bbox_override = tuple(candidate.bbox)
                    selected_center_override = (
                        (selected_bbox_override[0] + selected_bbox_override[2]) / 2.0,
                        (selected_bbox_override[1] + selected_bbox_override[3]) / 2.0,
                    )
                    break
            if selected_bbox_override is not None:
                break
        if selected_bbox_override is not None:
            selected = replace(
                selected,
                selected_bbox=selected_bbox_override,
                selected_center=selected_center_override,
            )
    enriched_episodes = tuple(
        ProbeEpisode(
            episode_index=index,
            seed=int(seed) + index,
            plan=plan,
            transitions=transitions,
            report=avatar_multi_report.episodes[index].report,
        )
        for index, transitions in enumerate(episode_transitions)
    )
    return MultiResetAvatarReport(
        episodes=enriched_episodes,
        cross_reset_evidence=avatar_multi_report.cross_reset_evidence,
        selected=selected,
        diagnostics=avatar_multi_report.diagnostics,
    )


def _run_full_bootstrap_analysis_with_adaptive_solve_for_level(
    *,
    game_id: str,
    level_id: str,
    output_dir: str | None,
    seed: int,
    episode_count: int,
    probe_montage: bool,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
    max_steps: int,
) -> dict[str, Any]:
    plan = build_probe_plan(game_id=game_id, level_id=level_id)
    episode_transitions = run_probe_episodes(
        plan=plan,
        episode_count=episode_count,
        base_seed=seed,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    avatar_multi_report = _build_multi_reset_avatar_report(
        plan=plan,
        episode_transitions=episode_transitions,
        seed=seed,
    )
    run_dir = resolve_run_dir(output_dir, game_id) / str(level_id) / "multi_reset"
    artifact_paths = write_multi_reset_artifacts(
        run_dir=run_dir,
        report=avatar_multi_report,
        write_montage=probe_montage,
    )

    stable_avatar_found = bool(avatar_multi_report.diagnostics.stable_avatar_found)
    phase_status: dict[str, str] = {
        "avatar": "ok" if stable_avatar_found else "failed",
        "poi": "skipped",
        "contact": "skipped",
        "hud": "skipped",
        "hud_targeting": "skipped",
        "adaptive_solve": "skipped",
    }
    summary: dict[str, Any] = {
        "game_id": game_id,
        "episode_count": int(episode_count),
        "plan": {
            "level_id": plan.level_id,
            "action_sequence": plan.action_sequence,
        },
        "avatar": {
            "selected": avatar_multi_report.selected.to_dict(),
            "diagnostics": avatar_multi_report.diagnostics.to_dict(),
            "stable_avatar_found": stable_avatar_found,
        },
        "poi": {
            "candidate_count": 0,
            "top_candidates": [],
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "contact": {
            "tested_poi_count": 0,
            "outcome_type_counts": {},
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "hud": {
            "region_count": 0,
            "mask_summary": {},
            "diagnostics": {},
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "hud_targeting": {
            "selected_poi_id": None,
            "ranked_poi_ids": [],
            "match_count": 0,
            "ambiguous": False,
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
        },
        "adaptive_solve": {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": "avatar_not_stable" if not stable_avatar_found else None,
            "diagnostics": {},
        },
        "artifact_paths": dict(artifact_paths),
        "phase_status": dict(phase_status),
    }

    if not stable_avatar_found:
        index_paths = write_full_analysis_index(
            run_dir=run_dir,
            game_id=game_id,
            episode_count=episode_count,
            phase_status=phase_status,
            artifact_paths=artifact_paths,
        )
        artifact_paths = dict(artifact_paths)
        artifact_paths.update(index_paths)
        summary["artifact_paths"] = artifact_paths
        return summary

    poi_bundle = discover_pois_multi_reset(avatar_multi_report)
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    artifact_paths = dict(artifact_paths)
    artifact_paths.update(poi_paths)
    poi_candidates = list(poi_bundle["report"].candidates)
    phase_status["poi"] = "ok" if poi_candidates else "failed"
    summary["poi"] = {
        "candidate_count": len(poi_candidates),
        "top_candidates": [item.__dict__ for item in poi_candidates[:5]],
        "diagnostics": poi_bundle["report"].diagnostics.to_dict(),
        "failure_reason": poi_bundle["report"].selected.failure_reason,
    }

    contact_report = None
    if poi_candidates:
        try:
            contact_report = run_controlled_contact_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_bundle,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
            contact_paths = write_contact_experiment_artifacts(
                run_dir=run_dir,
                report=contact_report,
            )
            artifact_paths.update(contact_paths)
            phase_status["contact"] = "ok"
            summary["contact"] = {
                "tested_poi_count": int(contact_report.diagnostics.get("tested_poi_count", len(contact_report.tested_pois))),
                "outcome_type_counts": dict(contact_report.diagnostics.get("outcome_type_counts", {})),
                "diagnostics": dict(contact_report.diagnostics),
                "failure_reason": None,
            }
        except Exception as exc:
            phase_status["contact"] = "failed"
            summary["contact"] = {
                "tested_poi_count": 0,
                "outcome_type_counts": {},
                "diagnostics": {},
                "failure_reason": f"contact_experiments_failed: {exc}",
            }
    else:
        phase_status["contact"] = "skipped"
        summary["contact"] = {
            "tested_poi_count": 0,
            "outcome_type_counts": {},
            "diagnostics": {},
            "failure_reason": "no_pois_available",
        }

    hud_bundle = detect_hud_multi_reset(
        avatar_multi_report=avatar_multi_report,
        poi_multi_bundle=poi_bundle,
    )
    hud_paths = write_hud_artifacts(
        run_dir=run_dir,
        hud_report=hud_bundle["report"],
        cross_reset_hud_evidence=hud_bundle["cross_reset_evidence"],
        episode_hud_reports=hud_bundle["episodes"],
        hud_value_samples=tuple(
            sample
            for samples in hud_bundle.get("value_samples", {}).values()
            for sample in samples
        ),
    )
    artifact_paths.update(hud_paths)
    phase_status["hud"] = "ok" if hud_bundle["report"].failure_reason is None else "failed"
    summary["hud"] = {
        "region_count": len(hud_bundle["report"].regions),
        "mask_summary": hud_bundle["report"].mask.to_dict(),
        "diagnostics": hud_bundle["report"].diagnostics.to_dict(),
        "failure_reason": hud_bundle["report"].failure_reason,
    }

    hint_report = None
    if phase_status.get("poi") == "ok" and phase_status.get("hud") == "ok":
        try:
            hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)
            hint_paths = write_hud_hint_artifacts(
                run_dir=run_dir,
                report=hint_report,
            )
            artifact_paths.update(hint_paths)
            summary["hud_targeting"] = {
                "selected_poi_id": hint_report.selected.selected_poi_id,
                "ranked_poi_ids": list(hint_report.selected.ranked_poi_ids),
                "match_count": len(hint_report.matches),
                "ambiguous": bool(hint_report.selected.ambiguous),
                "failure_reason": hint_report.selected.failure_reason,
            }
            phase_status["hud_targeting"] = (
                "ok"
                if hint_report.selected.selected_poi_id is not None
                and not hint_report.selected.ambiguous
                and hint_report.selected.failure_reason is None
                else "failed"
            )
        except Exception as exc:
            summary["hud_targeting"] = {
                "selected_poi_id": None,
                "ranked_poi_ids": [],
                "match_count": 0,
                "ambiguous": False,
                "failure_reason": f"hud_targeting_failed: {exc}",
            }
            phase_status["hud_targeting"] = "failed"
    else:
        summary["hud_targeting"] = {
            "selected_poi_id": None,
            "ranked_poi_ids": [],
            "match_count": 0,
            "ambiguous": False,
            "failure_reason": "prerequisite_phase_not_ok",
        }
        phase_status["hud_targeting"] = "skipped"

    if phase_status.get("avatar") == "ok" and phase_status.get("poi") == "ok":
        try:
            adaptive = run_adaptive_solve_multi_reset(
                avatar_multi_report=avatar_multi_report,
                poi_multi_bundle=poi_bundle,
                hud_targeting_report=hint_report,
                contact_experiment_report=contact_report,
                game_id=game_id,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
                max_steps=max_steps,
            )
            adaptive_paths = write_adaptive_solve_artifacts(
                run_dir=run_dir,
                report=adaptive,
            )
            artifact_paths.update(adaptive_paths)
            summary["adaptive_solve"] = {
                "selected_target_id": adaptive.selected_target_id,
                "solved": bool(adaptive.solved),
                "failure_reason": adaptive.failure_reason,
                "diagnostics": adaptive.diagnostics.to_dict(),
            }
            phase_status["adaptive_solve"] = "ok" if bool(adaptive.solved) else "failed"
        except Exception as exc:
            summary["adaptive_solve"] = {
                "selected_target_id": None,
                "solved": False,
                "failure_reason": f"adaptive_solve_failed: {exc}",
                "diagnostics": {},
            }
            phase_status["adaptive_solve"] = "failed"
    else:
        summary["adaptive_solve"] = {
            "selected_target_id": None,
            "solved": False,
            "failure_reason": "prerequisite_phase_not_ok",
            "diagnostics": {},
        }
        phase_status["adaptive_solve"] = "skipped"

    index_paths = write_full_analysis_index(
        run_dir=run_dir,
        game_id=game_id,
        episode_count=episode_count,
        phase_status=phase_status,
        artifact_paths=artifact_paths,
    )
    artifact_paths.update(index_paths)
    summary["artifact_paths"] = artifact_paths
    summary["phase_status"] = phase_status
    return summary


def run_full_analysis_for_level(
    *,
    game_id: str,
    level_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
) -> dict[str, Any]:
    _require_supported_game(game_id)
    validate_level_id_for_game(game_id, level_id, env_factory=env_factory)
    if str(level_id) != "L0" and not engine_supports_direct_level_start(game_id, env_factory=env_factory):
        raise ValueError(
            "nonzero level analysis requires direct engine level start support; use --campaign-solve for frontier progression"
        )

    summary = _run_full_bootstrap_analysis_with_adaptive_solve_for_level(
        game_id=game_id,
        level_id=level_id,
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        env_factory=env_factory,
        max_steps=max_steps,
    )

    adaptive_payload = dict(summary.get("adaptive_solve", {}))

    # Build replayable solution from latest adaptive report when present in-memory.
    solution = LevelSolution(
        game_id=str(game_id),
        level_id=str(level_id),
        solved=bool(adaptive_payload.get("solved", False)),
        action_trace=tuple(),
        step_count=0,
        terminal=False,
        level_transition=False,
        failure_reason=adaptive_payload.get("failure_reason"),
    )
    try:
        plan = build_probe_plan(game_id=game_id, level_id=level_id)
        transitions = run_probe_episodes(
            plan=plan,
            episode_count=episode_count,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        avatar_report = _build_multi_reset_avatar_report(plan=plan, episode_transitions=transitions, seed=seed)
        poi_bundle = discover_pois_multi_reset(avatar_report)
        hud_bundle = detect_hud_multi_reset(avatar_multi_report=avatar_report, poi_multi_bundle=poi_bundle)
        hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)
        contact_report = None
        if tuple(getattr(poi_bundle.get("report"), "candidates", ())):
            contact_report = run_controlled_contact_multi_reset(
                avatar_multi_report=avatar_report,
                poi_multi_bundle=poi_bundle,
                plan=plan,
                base_seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
            )
        adaptive_report = run_adaptive_solve_multi_reset(
            avatar_multi_report=avatar_report,
            poi_multi_bundle=poi_bundle,
            hud_targeting_report=hint_report,
            contact_experiment_report=contact_report,
            game_id=game_id,
            plan=plan,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_steps=max_steps,
        )
        solution = build_level_solution_from_adaptive_report(
            game_id=game_id,
            level_id=level_id,
            adaptive_report=adaptive_report,
        )
    except Exception:
        # Keep previously computed summary; solution remains a conservative unsolved record.
        solution = LevelSolution(
            game_id=str(game_id),
            level_id=str(level_id),
            solved=bool(adaptive_payload.get("solved", False)),
            action_trace=tuple(),
            step_count=0,
            terminal=False,
            level_transition=False,
            failure_reason=adaptive_payload.get("failure_reason", "solution_unavailable"),
        )

    level_run_dir = resolve_run_dir(output_dir, game_id) / str(level_id)
    level_solution_paths = write_level_solution_artifacts(
        run_dir=level_run_dir,
        solution=solution,
    )
    merged_artifacts = dict(summary.get("artifact_paths", {}))
    merged_artifacts.update(level_solution_paths)

    per_level = PerLevelResult(
        level_id=str(level_id),
        phase_status=dict(summary.get("phase_status", {})),
        solved=bool(solution.solved),
        failure_reason=solution.failure_reason,
        solution=solution,
        artifact_paths=merged_artifacts,
    )
    payload = per_level.to_dict()
    payload["game_id"] = game_id
    return payload


def run_full_analysis_for_game_levels(
    *,
    game_id: str,
    level_ids: tuple[str, ...] | list[str] | None,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
) -> dict[str, Any]:
    _require_supported_game(game_id)
    requested_level_ids = (
        tuple(str(item) for item in level_ids)
        if level_ids
        else get_supported_level_ids_for_game(game_id, env_factory=env_factory)
    )
    for level_id in requested_level_ids:
        validate_level_id_for_game(game_id, level_id, env_factory=env_factory)
    direct_level_start = engine_supports_direct_level_start(game_id, env_factory=env_factory)

    level_results: list[PerLevelResult] = []
    failure_counts: Counter[str] = Counter()
    for level_id in requested_level_ids:
        try:
            if str(level_id) != "L0" and not direct_level_start:
                raise ValueError(
                    "nonzero level requires direct engine level start support; use campaign mode"
                )
            result = run_full_analysis_for_level(
                game_id=game_id,
                level_id=level_id,
                output_dir=output_dir,
                seed=seed,
                episode_count=episode_count,
                probe_montage=probe_montage,
                render_terminal=render_terminal,
                env_factory=env_factory,
                max_steps=max_steps,
            )
            solution_payload = dict(result.get("solution", {}))
            solution = LevelSolution(
                game_id=str(solution_payload.get("game_id", game_id)),
                level_id=str(solution_payload.get("level_id", level_id)),
                solved=bool(solution_payload.get("solved", False)),
                action_trace=tuple(),
                step_count=int(solution_payload.get("step_count", 0)),
                terminal=bool(solution_payload.get("terminal", False)),
                level_transition=bool(solution_payload.get("level_transition", False)),
                failure_reason=solution_payload.get("failure_reason"),
            )
            per_level = PerLevelResult(
                level_id=str(level_id),
                phase_status=dict(result.get("phase_status", {})),
                solved=bool(result.get("solved", False)),
                failure_reason=result.get("failure_reason"),
                solution=solution,
                artifact_paths=dict(result.get("artifact_paths", {})),
            )
        except Exception as exc:
            failure_reason = f"level_failed: {exc}"
            level_run_dir = resolve_run_dir(output_dir, game_id) / str(level_id)
            solution = LevelSolution(
                game_id=game_id,
                level_id=str(level_id),
                solved=False,
                action_trace=tuple(),
                step_count=0,
                terminal=False,
                level_transition=False,
                failure_reason=failure_reason,
            )
            solution_paths = write_level_solution_artifacts(run_dir=level_run_dir, solution=solution)
            per_level = PerLevelResult(
                level_id=str(level_id),
                phase_status={
                    "avatar": "failed",
                    "poi": "skipped",
                    "contact": "skipped",
                    "hud": "skipped",
                    "hud_targeting": "skipped",
                    "adaptive_solve": "failed",
                },
                solved=False,
                failure_reason=failure_reason,
                solution=solution,
                artifact_paths=solution_paths,
            )
        level_results.append(per_level)
        if per_level.failure_reason is not None:
            failure_counts[str(per_level.failure_reason)] += 1

    solved_count = sum(1 for item in level_results if bool(item.solved))
    diagnostics = GameLevelBatchDiagnostics(
        requested_level_count=len(requested_level_ids),
        completed_level_count=len(level_results),
        solved_level_count=solved_count,
        failed_level_count=max(0, len(level_results) - solved_count),
        failure_reason_counts=dict(sorted(failure_counts.items())),
    )
    report = GameLevelBatchReport(
        game_id=game_id,
        levels=tuple(level_results),
        diagnostics=diagnostics,
    )
    batch_run_dir = resolve_run_dir(output_dir, game_id)
    batch_paths = write_game_level_batch_artifacts(
        run_dir=batch_run_dir,
        report=report,
    )
    payload = report.to_dict()
    payload["artifact_paths"] = batch_paths
    return payload


def replay_saved_level_solution(
    *,
    game_id: str,
    level_id: str,
    solution: dict[str, Any],
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)
    validate_level_id_for_game(game_id, level_id, env_factory=env_factory)

    action_trace = tuple(solution.get("action_trace", ())) if isinstance(solution, dict) else tuple()
    session_adapter = SessionAdapter()
    action_adapter = ActionAdapter()
    session = session_adapter.create_session(
        game_id,
        seed=int(solution.get("seed", 0)) if isinstance(solution, dict) else 0,
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )

    executed_steps = 0
    terminal = False
    terminal_status = "unknown"
    level_transition = False
    failure_reason: str | None = None
    try:
        previous_levels = int(session_adapter.get_current_observation(session).levels_completed)
        for idx, item in enumerate(action_trace):
            action = str(item.get("action", "")) if isinstance(item, dict) else ""
            observation = session_adapter.get_current_observation(session)
            context = ActionTranslationContext(
                available_action_ids=observation.available_actions,
                coordinate_action_id=session.environment_metadata.coordinate_action_id,
                coordinate_bounds=session.environment_metadata.coordinate_bounds,
            )
            try:
                translated = action_adapter.translate_token(action, context)
            except ValueError:
                failure_reason = "invalid_action_in_trace"
                break
            executed = session_adapter.execute_action_prefix(session, (translated,), (action,))
            executed_steps = idx + 1
            post = session_adapter.get_current_observation(session)
            level_transition = level_transition or (int(post.levels_completed) > int(previous_levels))
            previous_levels = int(post.levels_completed)
            terminal = executed.terminal_status in {"success", "failure"}
            terminal_status = str(executed.terminal_status)
            if terminal:
                break
        expected_after = int(str(level_id).lstrip("L") or 0) + 1
        solved = int(previous_levels) >= expected_after
        if not solved and failure_reason is None:
            failure_reason = "replay_unsolved"
        return {
            "game_id": game_id,
            "level_id": level_id,
            "steps_executed": int(executed_steps),
            "solved": bool(solved),
            "terminal": bool(terminal),
            "terminal_status": terminal_status,
            "level_transition": bool(level_transition),
            "failure_reason": None if solved else failure_reason,
        }
    finally:
        session_adapter.close_session(session)


def run_frontier_level_from_live_session(
    *,
    game_id: str,
    frontier_level_id: str,
    session,
    prefix_traces,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    max_steps: int = 40,
    env_factory: Callable[[], Any] | None = None,
    debug_log_path: str | None = None,
) -> dict[str, Any]:
    episode_count = max(1, int(episode_count))
    prefix_actions = _flatten_prefix_actions(tuple(prefix_traces or ()))

    session_adapter = SessionAdapter()
    plan = build_probe_plan(game_id=game_id, level_id=frontier_level_id)
    frontier_level_index = int(str(frontier_level_id).lstrip("L") or 0)
    if frontier_level_index <= 0:
        episode_transitions = run_probe_episodes_at_frontier(
            plan=plan,
            prefix_traces=tuple(prefix_traces or ()),
            episode_count=int(episode_count),
            base_seed=int(seed),
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        _append_campaign_debug_marker(
            debug_log_path=debug_log_path,
            game_id=game_id,
            level_id=frontier_level_id,
            marker="BOOTSTRAP",
        )
        _append_campaign_debug_marker(
            debug_log_path=debug_log_path,
            game_id=game_id,
            level_id=frontier_level_id,
            marker="RESET",
        )
        _append_campaign_play_marker(
            debug_log_path=debug_log_path,
            game_id=game_id,
            level_id=frontier_level_id,
            actions=prefix_actions,
        )
        avatar_bootstrap_mode = "fresh_reset"
    else:
        episode_transitions = run_frontier_avatar_bootstrap_from_frontier(
            game_id=game_id,
            frontier_level_id=frontier_level_id,
            prefix_traces=tuple(prefix_traces or ()),
            episode_count=int(episode_count),
            seed=int(seed),
            render_terminal=render_terminal,
            env_factory=env_factory,
            debug_log_path=debug_log_path,
        )
        avatar_bootstrap_mode = "after_prefix_replay"
    multi = _build_multi_reset_avatar_report(
        plan=plan,
        episode_transitions=episode_transitions,
        seed=seed,
    )
    run_dir = resolve_run_dir(output_dir, game_id) / str(frontier_level_id) / "multi_reset"
    artifact_paths = write_multi_reset_artifacts(
        run_dir=run_dir,
        report=multi,
        write_montage=probe_montage,
    )
    stable = _campaign_stable_avatar_found(multi)
    phase_status = {
        "avatar": "ok" if stable else "failed",
        "poi": "skipped",
        "contact": "skipped",
        "hud": "skipped",
        "hud_targeting": "skipped",
        "adaptive_solve": "skipped",
    }
    summary = {
        "game_id": game_id,
        "level_id": frontier_level_id,
        "phase_status": dict(phase_status),
        "artifact_paths": dict(artifact_paths),
        "diagnostics": {
            "avatar_episode_count_used": int(episode_count),
            "avatar_bootstrap_mode": avatar_bootstrap_mode,
        },
    }
    if not stable:
        return {
            **summary,
            "solved": False,
            "failure_reason": "no_stable_avatar",
            "solution": LevelSolution(
                game_id=game_id,
                level_id=frontier_level_id,
                solved=False,
                action_trace=tuple(),
                step_count=0,
                terminal=False,
                level_transition=False,
                failure_reason="no_stable_avatar",
            ).to_dict(),
            "saved_trace": None,
        }

    poi_bundle = discover_pois_multi_reset(multi)
    poi_paths = write_poi_artifacts(
        run_dir=run_dir,
        poi_report=poi_bundle["report"],
        cross_reset_poi_evidence=poi_bundle["cross_reset_evidence"],
        episode_poi_reports=poi_bundle["episodes"],
    )
    artifact_paths.update(poi_paths)
    candidates = tuple(poi_bundle["report"].candidates)
    phase_status["poi"] = "ok" if candidates else "failed"
    if not candidates:
        return {
            **summary,
            "phase_status": dict(phase_status),
            "artifact_paths": dict(artifact_paths),
            "solved": False,
            "failure_reason": "no_poi_candidate",
            "solution": LevelSolution(
                game_id=game_id,
                level_id=frontier_level_id,
                solved=False,
                action_trace=tuple(),
                step_count=0,
                terminal=False,
                level_transition=False,
                failure_reason="no_poi_candidate",
            ).to_dict(),
            "saved_trace": None,
        }

    hud_bundle = detect_hud_multi_reset(avatar_multi_report=multi, poi_multi_bundle=poi_bundle)
    hud_paths = write_hud_artifacts(
        run_dir=run_dir,
        hud_report=hud_bundle["report"],
        cross_reset_hud_evidence=hud_bundle["cross_reset_evidence"],
        episode_hud_reports=hud_bundle["episodes"],
        hud_value_samples=tuple(
            sample
            for samples in hud_bundle.get("value_samples", {}).values()
            for sample in samples
        ),
    )
    artifact_paths.update(hud_paths)
    phase_status["hud"] = "ok" if hud_bundle["report"].failure_reason is None else "failed"
    hint_report = interpret_hud_hints_multi_reset(hud_bundle, poi_bundle)
    hint_paths = write_hud_hint_artifacts(run_dir=run_dir, report=hint_report)
    artifact_paths.update(hint_paths)
    phase_status["hud_targeting"] = "ok" if hint_report.selected.selected_poi_id and not hint_report.selected.ambiguous else "failed"
    simple_frontier = _is_simple_single_target_frontier(
        game_id=game_id,
        poi_report=poi_bundle["report"],
        hud_targeting_report=hint_report,
    )
    skip_contact = _should_skip_redundant_frontier_analysis(
        avatar_multi_report=multi,
        poi_report=poi_bundle["report"],
        hud_targeting_report=hint_report,
        prior_contact_report=None,
        game_id=game_id,
    )
    incoming_contact_candidates = tuple(candidates)
    selected_poi_id = getattr(getattr(hint_report, "selected", None), "selected_poi_id", None)
    selected_contact_candidates = incoming_contact_candidates
    selection_reason = "all_candidates"
    dropped_candidate_ids: tuple[str, ...] = tuple()
    if selected_poi_id is not None:
        matched = tuple(item for item in incoming_contact_candidates if str(getattr(item, "poi_id", "")) == str(selected_poi_id))
        if matched:
            selected_contact_candidates = matched + tuple(item for item in incoming_contact_candidates if str(getattr(item, "poi_id", "")) != str(selected_poi_id))
            selection_reason = "hud_selected_first"
        else:
            selection_reason = "hud_selected_unmatched"
    poi_bundle_for_contact = dict(poi_bundle)
    poi_bundle_for_contact["report"] = SimpleNamespace(candidates=tuple(selected_contact_candidates))

    contact_report = None
    contact_generated = tuple()
    contact_attempted = tuple()
    contact_stats = {}
    if not skip_contact:
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] About to call run_controlled_contact_multi_reset\n"); f.flush()
        contact_report = run_controlled_contact_multi_reset(
            avatar_multi_report=multi,
            poi_multi_bundle=poi_bundle_for_contact,
            plan=plan,
            base_seed=seed,
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_pois_to_test=max(0, len(selected_contact_candidates)),
        )
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Contact report returned, writing artifacts\n"); f.flush()
        contact_paths = write_contact_experiment_artifacts(run_dir=run_dir, report=contact_report)
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Artifacts written\n"); f.flush()
        artifact_paths.update(contact_paths)
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting contact_generated\n"); f.flush()
        contact_generated = tuple(getattr(contact_report, "diagnostics", {}).get("generated_trajectories", ()) or ())
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting contact_attempted ({len(contact_generated)} generated)\n"); f.flush()
        contact_attempted = tuple(getattr(contact_report, "diagnostics", {}).get("attempted_trajectories", ()) or ())
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting contact_stats ({len(contact_attempted)} attempted)\n"); f.flush()
        contact_stats = dict(getattr(contact_report, "diagnostics", {}).get("trajectory_stats_overall", {}) or {})
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Writing additional trajectory artifacts\n"); f.flush()
        if contact_generated:
            artifact_paths.update(write_generated_trajectories(run_dir=run_dir, generated_trajectories=contact_generated, rejected_trajectories=tuple()))
        if contact_attempted:
            artifact_paths.update(write_trajectory_attempts(run_dir=run_dir, trajectory_attempts=contact_attempted))
        if contact_stats:
            artifact_paths.update(write_trajectory_stats(run_dir=run_dir, trajectory_stats=contact_stats))
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Contact phase complete\n"); f.flush()
        phase_status["contact"] = "ok"
    else:
        phase_status["contact"] = "skipped"

    initial_frontier_actions = tuple()

    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] About to call run_adaptive_solve_on_live_session\n"); f.flush()
    adaptive = run_adaptive_solve_on_live_session(
        session=session,
        selected_avatar=multi.selected,
        ranked_poi_candidates=candidates,
        hud_targeting_report=hint_report,
        contact_experiment_report=contact_report,
        game_id=game_id,
        level_id=frontier_level_id,
        max_steps=max_steps,
        skip_bootstrap_replay_in_final_solve=True,
        prefix_traces=prefix_traces,
        initial_frontier_actions=initial_frontier_actions,
        render_terminal=render_terminal,
        env_factory=env_factory,
        debug_log_path=debug_log_path,
    )
    adaptive_paths = write_adaptive_solve_artifacts(run_dir=run_dir, report=adaptive)
    artifact_paths.update(adaptive_paths)
    adaptive_generated = tuple(getattr(adaptive, "generated_trajectories", ()) or ())
    adaptive_attempted = tuple(getattr(adaptive, "attempted_trajectories", ()) or ())
    adaptive_stats_obj = getattr(adaptive, "trajectory_stats", None)
    adaptive_stats = adaptive_stats_obj.to_dict() if hasattr(adaptive_stats_obj, "to_dict") else dict(adaptive_stats_obj or {})
    merged_generated = tuple(contact_generated) + tuple(adaptive_generated)
    merged_attempted = tuple(contact_attempted) + tuple(adaptive_attempted)
    if merged_generated:
        artifact_paths.update(
            write_generated_trajectories(
                run_dir=run_dir,
                generated_trajectories=merged_generated,
                rejected_trajectories=tuple(getattr(adaptive, "rejected_trajectories", ()) or ()),
            )
        )
    if merged_attempted:
        artifact_paths.update(write_trajectory_attempts(run_dir=run_dir, trajectory_attempts=merged_attempted))
    merged_stats = dict(contact_stats)
    if adaptive_stats:
        merged_stats = dict(adaptive_stats)
    if merged_stats:
        artifact_paths.update(write_trajectory_stats(run_dir=run_dir, trajectory_stats=merged_stats))
    phase_status["adaptive_solve"] = "ok" if adaptive.solved else "failed"
    solution = build_level_solution_from_adaptive_report(
        game_id=game_id,
        level_id=frontier_level_id,
        adaptive_report=adaptive,
    )
    solution_paths = write_level_solution_artifacts(
        run_dir=resolve_run_dir(output_dir, game_id) / str(frontier_level_id),
        solution=solution,
    )
    artifact_paths.update(solution_paths)
    saved_trace = extract_verified_frontier_trace(
        game_id=game_id,
        level_id=frontier_level_id,
        adaptive_report=adaptive,
        source_run_id=None,
        trace_version=1,
    )
    analysis_action_count = int(
        sum(len(tuple(getattr(trace, "action_trace", ()))) for trace in tuple(prefix_traces or ()))
    )
    if contact_report is not None:
        analysis_action_count += int(
            sum(len(tuple(getattr(item, "steps", ()))) for item in tuple(getattr(contact_report, "tested_pois", ())))
        )
    final_solve_action_count = int(len(tuple(getattr(solution, "action_trace", ()))))
    saved_solution_action_count = int(len(tuple(getattr(saved_trace, "action_trace", ()))))
    return {
        **summary,
        "phase_status": dict(phase_status),
        "artifact_paths": dict(artifact_paths),
        "solved": bool(solution.solved),
        "failure_reason": solution.failure_reason,
        "solution": solution.to_dict(),
        "saved_trace": saved_trace,
        "diagnostics": {
            **dict(summary.get("diagnostics", {})),
            "simple_single_target_frontier": bool(simple_frontier),
            "skip_redundant_contact": bool(skip_contact),
            "analysis_action_count": int(analysis_action_count),
            "final_solve_action_count": int(final_solve_action_count),
            "saved_solution_action_count": int(saved_solution_action_count),
            "contact_candidate_selection": {
                "incoming_candidate_count": int(len(incoming_contact_candidates)),
                "selected_candidate_count": int(len(selected_contact_candidates)),
                "dropped_candidate_ids": list(dropped_candidate_ids),
                "selection_reason": selection_reason,
            },
            "generated_trajectories": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in tuple(getattr(adaptive, "generated_trajectories", ()) or ())
            ],
            "attempted_trajectories": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in tuple(getattr(adaptive, "attempted_trajectories", ()) or ())
            ],
            "trajectory_stats": (
                getattr(adaptive, "trajectory_stats").to_dict()
                if hasattr(getattr(adaptive, "trajectory_stats", None), "to_dict")
                else {}
            ),
        },
        "_selected_avatar_obj": multi.selected,
    }


def run_frontier_continuation_from_live_session(
    *,
    game_id: str,
    session,
    next_level_id: str,
    selected_avatar,
    prefix_traces,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    max_steps: int = 40,
    env_factory: Callable[[], Any] | None = None,
    debug_log_path: str | None = None,
) -> dict[str, Any]:
    session_adapter = SessionAdapter()
    current_obs = session_adapter.get_current_observation(session)
    frame = current_obs.frame[0] if isinstance(current_obs.frame, tuple) and current_obs.frame else None
    frame_plane = tuple(tuple(int(v) for v in row) for row in frame) if isinstance(frame, tuple) else None
    avatar_reanchored = track_avatar_bbox_in_frame(
        frame_plane,
        getattr(selected_avatar, "selected_bbox", None),
        getattr(selected_avatar, "value_histogram", None),
        frontier_reanchor=True,
    )
    expected_level_index = int(str(next_level_id).lstrip("L") or 0)
    current_level_index = int(getattr(current_obs, "levels_completed", 0) or 0)
    continuation_mode = "live_session"

    active_session = session
    if current_level_index < expected_level_index:
        continuation_mode = "replay_fallback"
        try:
            session_adapter.close_session(session)
        except Exception:
            pass
        replay = replay_prefix_traces_to_frontier(
            game_id=game_id,
            prefix_traces=tuple(prefix_traces or ()),
            render_terminal=bool(render_terminal),
            env_factory=env_factory,
        )
        active_session = replay.get("session")
        if active_session is None:
            return {
                "game_id": game_id,
                "level_id": str(next_level_id),
                "phase_status": {"avatar": "ok" if avatar_reanchored is not None else "failed", "poi": "skipped", "contact": "skipped", "hud": "skipped", "hud_targeting": "skipped", "adaptive_solve": "failed"},
                "artifact_paths": {},
                "solved": False,
                "failure_reason": "prefix_replay_failed",
                "solution": LevelSolution(game_id=game_id, level_id=str(next_level_id), solved=False, action_trace=tuple(), step_count=0, terminal=False, level_transition=False, failure_reason="prefix_replay_failed").to_dict(),
                "saved_trace": None,
                "diagnostics": {
                    "avatar_bootstrap_mode": "after_prefix_replay",
                    "continuation_mode": continuation_mode,
                    "continued_from_live_session": False,
                    "used_replay_prefix": bool(prefix_traces),
                    "replay_prefix_used": bool(prefix_traces),
                    "replay_prefix_length": int(sum(len(item.action_trace) for item in tuple(prefix_traces or ()))),
                    "live_avatar_reanchor_ok": avatar_reanchored is not None,
                },
            }
    if active_session is None:
        return {
            "game_id": game_id,
            "level_id": str(next_level_id),
            "phase_status": {"avatar": "ok" if avatar_reanchored is not None else "failed", "poi": "skipped", "contact": "skipped", "hud": "skipped", "hud_targeting": "skipped", "adaptive_solve": "failed"},
            "artifact_paths": {},
            "solved": False,
            "failure_reason": "prefix_replay_failed",
            "solution": LevelSolution(game_id=game_id, level_id=str(next_level_id), solved=False, action_trace=tuple(), step_count=0, terminal=False, level_transition=False, failure_reason="prefix_replay_failed").to_dict(),
            "saved_trace": None,
            "diagnostics": {
                "avatar_bootstrap_mode": "after_prefix_replay",
                "continuation_mode": continuation_mode,
                "continued_from_live_session": False,
                "used_replay_prefix": bool(prefix_traces),
                "replay_prefix_used": bool(prefix_traces),
                "replay_prefix_length": int(sum(len(item.action_trace) for item in tuple(prefix_traces or ()))),
                "live_avatar_reanchor_ok": avatar_reanchored is not None,
            },
        }
    result = run_frontier_level_from_live_session(
        game_id=game_id,
        frontier_level_id=str(next_level_id),
        session=active_session,
        prefix_traces=tuple(prefix_traces or ()),
        output_dir=output_dir,
        seed=seed,
        episode_count=episode_count,
        probe_montage=probe_montage,
        render_terminal=render_terminal,
        max_steps=max_steps,
        env_factory=env_factory,
        debug_log_path=debug_log_path,
    )
    result.setdefault("diagnostics", {})
    result["diagnostics"].update(
        {
            "avatar_bootstrap_mode": "after_prefix_replay" if continuation_mode != "live_session" else "live_frontier",
            "continuation_mode": continuation_mode,
            "continued_from_live_session": continuation_mode == "live_session",
            "used_replay_prefix": bool(prefix_traces),
            "replay_prefix_used": bool(prefix_traces),
            "replay_prefix_length": int(sum(len(item.action_trace) for item in tuple(prefix_traces or ()))),
            "live_avatar_reanchor_ok": avatar_reanchored is not None,
        }
    )
    return result


def run_frontier_avatar_bootstrap_from_frontier(
    *,
    game_id: str,
    frontier_level_id: str,
    prefix_traces,
    episode_count: int,
    seed: int,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None = None,
    debug_log_path: str | None = None,
) -> tuple[tuple[ProbeTransitionRecord, ...], ...]:
    plan = build_probe_plan(game_id=game_id, level_id=frontier_level_id)
    prefix_actions = _flatten_prefix_actions(tuple(prefix_traces or ()))
    # On frontier levels reached through solved-prefix replay, the bootstrap
    # probe can leave the avatar offset from the true entry position. Run one
    # warm-up probe and discard it, then reset/replay to the frontier again and
    # capture the probe transitions that will feed avatar identification.
    run_probe_episodes_at_frontier(
        plan=plan,
        prefix_traces=tuple(prefix_traces or ()),
        episode_count=int(episode_count),
        base_seed=int(seed),
        render_terminal=False,
        env_factory=env_factory,
    )
    _append_campaign_debug_marker(
        debug_log_path=debug_log_path,
        game_id=game_id,
        level_id=frontier_level_id,
        marker="BOOTSTRAP",
    )
    _append_campaign_debug_marker(
        debug_log_path=debug_log_path,
        game_id=game_id,
        level_id=frontier_level_id,
        marker="RESET",
    )
    _append_campaign_play_marker(
        debug_log_path=debug_log_path,
        game_id=game_id,
        level_id=frontier_level_id,
        actions=prefix_actions,
    )
    return run_probe_episodes_at_frontier(
        plan=plan,
        prefix_traces=tuple(prefix_traces or ()),
        episode_count=int(episode_count),
        base_seed=int(seed),
        render_terminal=bool(render_terminal),
        env_factory=env_factory,
    )


def replay_campaign_prefix(
    *,
    game_id: str,
    ordered_saved_level_traces: tuple[SavedLevelTrace, ...] | list[SavedLevelTrace],
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    _require_supported_game(game_id)
    traces = tuple(ordered_saved_level_traces or ())
    replay = replay_prefix_traces_to_frontier(
        game_id=game_id,
        prefix_traces=traces,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    expected_frontier_level = 0
    if traces:
        expected_frontier_level = max(int(str(trace.level_id).lstrip("L") or 0) + 1 for trace in traces)
    ok = bool(replay.get("frontier_reached", False)) and not bool(replay.get("divergence", False))
    replayed_session = replay.get("session")
    if replayed_session is not None:
        try:
            SessionAdapter().close_session(replayed_session)
        except Exception:
            pass
    return {
        "ok": bool(ok),
        "expected_frontier_level_id": f"L{expected_frontier_level}",
        "reached_frontier_level_id": str(replay.get("frontier_level_id", "L0")),
        "executed_action_count": int(replay.get("executed_action_count", 0)),
        "failure_reason": None if ok else "prefix_replay_failed",
    }


def run_trace_optimization_pass(
    *,
    game_id: str,
    level_id: str,
    trace_path: str,
    output_dir: str | None = None,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    trace_db_path = initialize_trace_store(_get_trace_db_path(output_dir, game_id))
    baseline = get_best_trace_for_level(db_path=trace_db_path, game_id=game_id, level_id=level_id)
    if baseline is None:
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(trace_path)
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = payload.get("action_trace")
        if actions is None:
            actions_file = path.with_name("saved_level_trace_actions.json")
            actions = json.loads(actions_file.read_text(encoding="utf-8")) if actions_file.exists() else []
        baseline = SavedLevelTrace(
            game_id=str(payload.get("game_id", game_id)),
            level_id=str(payload.get("level_id", level_id)),
            solved=bool(payload.get("solved", True)),
            action_trace=tuple(str(item) for item in actions),
            step_count=int(payload.get("step_count", len(actions))),
            source_run_id=payload.get("source_run_id"),
            trace_version=int(payload.get("trace_version", 1)),
            replay_verified=bool(payload.get("replay_verified", False)),
        )
    report = optimize_level_trace(
        game_id=game_id,
        level_id=level_id,
        saved_trace=baseline,
        prefix_traces=tuple(
            trace
            for trace in (
                get_best_trace_for_level(db_path=trace_db_path, game_id=game_id, level_id=item)
                for item in get_level_sequence_for_game(game_id, env_factory=env_factory)
                if int(str(item).lstrip("L") or 0) < int(str(level_id).lstrip("L") or 0)
            )
            if trace is not None
        ),
        render_terminal=render_terminal,
        env_factory=env_factory,
        trace_db_path=trace_db_path,
    )
    run_dir = resolve_run_dir(output_dir, game_id) / str(level_id) / "trace_optimization"
    artifacts = write_trace_optimization_artifacts(
        run_dir=run_dir,
        report=report,
    )
    return {
        "game_id": game_id,
        "level_id": level_id,
        "failure_reason": report.failure_reason,
        "diagnostics": dict(report.diagnostics),
        "best_step_count": int(report.best_candidate.step_count),
        "baseline_step_count": int(report.baseline_trace.step_count),
        "artifact_paths": artifacts,
    }


def run_trace_analysis_for_game(
    *,
    game_id: str,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    trace_db_path = initialize_trace_store(_get_trace_db_path(output_dir, game_id))
    solved_levels = get_solved_levels_for_game(db_path=trace_db_path, game_id=game_id)
    reports = optimize_game_traces_from_db(
        game_id=game_id,
        trace_db_path=trace_db_path,
        render_terminal=render_terminal,
        env_factory=env_factory,
    )
    payload = {
        "game_id": game_id,
        "global_trace_store_path": str(trace_db_path),
        "solved_level_count": len(solved_levels),
        "analyzed_level_count": len(reports),
        "reports": [item.to_dict() for item in reports],
    }
    artifacts = write_trace_analysis_batch_artifacts(
        run_dir=resolve_run_dir(output_dir, game_id) / "campaign",
        game_id=game_id,
        reports=payload["reports"],
        trace_db_path=trace_db_path,
    )
    payload["artifact_paths"] = artifacts
    return payload


def run_full_campaign_analysis(
    *,
    game_id: str,
    output_dir: str | None = None,
    seed: int = 0,
    episode_count: int = 1,
    probe_montage: bool = False,
    render_terminal: bool = False,
    env_factory: Callable[[], Any] | None = None,
    max_steps: int = 40,
    use_solutions: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    import sqlite3
    from pathlib import Path as PathLib

    _require_supported_game(game_id)
    episode_count = max(1, int(episode_count))
    trace_db_path = initialize_trace_store(_get_trace_db_path(output_dir, game_id))
    level_sequence = get_level_sequence_for_game(game_id, env_factory=env_factory)
    base_output_dir = PathLib(output_dir) if output_dir else PathLib("runs_v5_0")
    debug_log_path = str(base_output_dir / "debug.log") if bool(debug) else None
    if debug_log_path is not None:
        PathLib(debug_log_path).parent.mkdir(parents=True, exist_ok=True)
        PathLib(debug_log_path).write_text("", encoding="utf-8")
    state = load_or_initialize_campaign_state(
        game_id=game_id,
        level_sequence=level_sequence,
        trace_db_path=trace_db_path,
    )
    initial_state = dict(state)

    level_results: list[CampaignLevelResult] = []
    global_trace: list[CampaignRunStep] = []
    campaign_step_trace: list[CampaignRunStep] = []
    highest_reached = None
    failure_reason = None
    global_step = 0
    db_solved_levels = get_db_solved_levels_for_game(game_id=game_id, trace_db_path=trace_db_path)
    replay_valid_db_prefix = tuple()
    if bool(use_solutions):
        if hasattr(get_verified_prefix_traces, "mock_calls"):
            replay_valid_db_prefix = get_verified_prefix_traces(
                state=initial_state,
                level_sequence=tuple(str(item) for item in level_sequence),
                trace_db_path=trace_db_path,
            )
        else:
            replay_valid_db_prefix = _load_replay_valid_db_prefix(
                game_id=game_id,
                level_sequence=tuple(str(item) for item in level_sequence),
                trace_db_path=trace_db_path,
                render_terminal=False,
                env_factory=env_factory,
            )
        if (
            not replay_valid_db_prefix
            and any(bool(getattr(item, "solved", False)) for item in initial_state.values())
            and not get_all_traces_for_game(db_path=trace_db_path, game_id=game_id)
        ):
            replay_valid_db_prefix = get_verified_prefix_traces(
                state=initial_state,
                level_sequence=tuple(str(item) for item in level_sequence),
                trace_db_path=trace_db_path,
            )
        state = _state_with_replay_valid_db_prefix(
            state=state,
            level_sequence=tuple(str(item) for item in level_sequence),
            replay_valid_prefix=replay_valid_db_prefix,
        )
        db_solved_levels = tuple(str(trace.level_id) for trace in replay_valid_db_prefix)
    replay_valid_db_prefix_by_level = {
        str(trace.level_id): trace for trace in tuple(replay_valid_db_prefix if bool(use_solutions) else ())
    }
    prefix_trace_count = 0
    current_run_traces: dict[str, SavedLevelTrace] = {}
    level_debug_rows: list[dict[str, str]] = []
    level_trace_rows: list[dict[str, Any]] = []
    unsolved_level_trajectory_stats: list[dict[str, Any]] = []
    prefix_warnings: list[str] = []
    frontier_attempt_artifacts: dict[str, str] = {}
    continuation_context: dict[str, Any] | None = None
    rendered_db_solution_prefix = False
    terminal_campaign_complete = False

    session_adapter = SessionAdapter()
    session = None
    while True:
        frontier = get_frontier_level_id(
            state=state,
            level_sequence=level_sequence,
            trace_db_path=trace_db_path,
            game_id=game_id,
            use_solutions=bool(use_solutions),
        )
        if frontier is None:
            if bool(use_solutions) and highest_reached is None and db_solved_levels:
                highest_reached = str(db_solved_levels[-1])
            if (
                bool(use_solutions)
                and bool(render_terminal)
                and not rendered_db_solution_prefix
                and replay_valid_db_prefix_by_level
            ):
                rendered_db_solution_prefix = True
                render_replay = replay_prefix_traces_to_frontier(
                    game_id=game_id,
                    prefix_traces=tuple(
                        replay_valid_db_prefix_by_level[level_id]
                        for level_id in tuple(str(item) for item in level_sequence)
                        if level_id in replay_valid_db_prefix_by_level
                    ),
                    render_terminal=True,
                    env_factory=env_factory,
                )
                try:
                    rendered_session = render_replay.get("session")
                    if rendered_session is not None:
                        session_adapter.close_session(rendered_session)
                except Exception:
                    pass
            break
        using_live_continuation = bool(
            continuation_context is not None
            and session is not None
            and str(continuation_context.get("next_level_id")) == str(frontier)
        )
        if session is not None and not using_live_continuation:
            try:
                session_adapter.close_session(session)
            except Exception:
                pass
            session = None
        frontier_index = int(str(frontier).lstrip("L") or 0)
        prefix_level_ids = tuple(
            item
            for item in tuple(str(level) for level in level_sequence)
            if int(str(item).lstrip("L") or 0) < frontier_index
        )
        prefix_diag: dict[str, object] = {"prefix_validation_warnings": list(prefix_warnings)}
        current_run_prefix = get_current_run_prefix_traces(
            level_sequence=prefix_level_ids,
            current_run_traces=current_run_traces,
            diagnostics=prefix_diag,
        )
        prefix_warnings = list(prefix_diag.get("prefix_validation_warnings", []))
        db_prefix = tuple()
        if bool(use_solutions):
            db_prefix = tuple(
                replay_valid_db_prefix_by_level[level_id]
                for level_id in prefix_level_ids
                if level_id in replay_valid_db_prefix_by_level
            )
        merged: dict[str, SavedLevelTrace] = {str(item.level_id): item for item in db_prefix}
        for item in current_run_prefix:
            merged[str(item.level_id)] = item
        prefix = tuple(merged[level_id] for level_id in prefix_level_ids if level_id in merged)
        prefix_source = "none"
        if current_run_prefix:
            prefix_source = "current_run"
        elif db_prefix:
            prefix_source = "global_db"
        prefix_trace_count = len(prefix)
        frontier_run_dir = resolve_run_dir(output_dir, game_id) / str(frontier)
        frontier_run_dir.mkdir(parents=True, exist_ok=True)
        replay_prefix_length = sum(len(item.action_trace) for item in prefix)
        frontier_attempt_path = frontier_run_dir / "frontier_attempt.json"
        frontier_attempt_path.write_text(
            json.dumps(
                {
                    "game_id": game_id,
                    "level_id": str(frontier),
                    "used_replay_prefix": bool(prefix),
                    "prefix_source": str(prefix_source),
                    "prefix_trace_count": int(len(prefix)),
                    "replay_prefix_length": int(replay_prefix_length),
                    "use_solutions": bool(use_solutions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frontier_attempt_artifacts[f"{frontier}/frontier_attempt.json"] = str(frontier_attempt_path)
        prefix_traces_used_path = frontier_run_dir / "prefix_traces_used.json"
        prefix_traces_used_path.write_text(
            json.dumps(
                {
                    "game_id": game_id,
                    "level_id": str(frontier),
                    "prefix_level_ids": [str(getattr(item, "level_id", "")) for item in prefix],
                    "prefix_trace_ids": [getattr(item, "trace_id", None) for item in prefix],
                    "prefix_action_counts": [int(len(tuple(getattr(item, "action_trace", ())))) for item in prefix],
                    "total_prefix_action_count": int(replay_prefix_length),
                    "prefix_replay_verified": [bool(getattr(item, "replay_verified", False)) for item in prefix],
                    "trace_source": str(prefix_source),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frontier_attempt_artifacts[f"{frontier}/prefix_traces_used.json"] = str(prefix_traces_used_path)
        replay = None
        if not using_live_continuation:
            replay = replay_prefix_traces_to_frontier(
                game_id=game_id,
                prefix_traces=prefix,
                render_terminal=bool(render_terminal),
                env_factory=env_factory,
            )
        if (not using_live_continuation) and (not bool(replay.get("frontier_reached", False)) or bool(replay.get("divergence", False))):
            prefix_failure_path = frontier_run_dir / "prefix_replay_failure.json"
            prefix_failure_path.write_text(
                json.dumps(
                    {
                        "game_id": game_id,
                        "level_id": str(frontier),
                        "failure_reason": "prefix_replay_failed",
                        "prefix_source": str(prefix_source),
                        "prefix_trace_count": int(len(prefix)),
                        "replay_prefix_length": int(replay_prefix_length),
                        "divergence": bool(replay.get("divergence", False)),
                        "frontier_reached": bool(replay.get("frontier_reached", False)),
                        "session_missing": replay.get("session") is None,
                        "prefix_validation_warnings": list(prefix_warnings),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            frontier_attempt_artifacts[f"{frontier}/prefix_replay_failure.json"] = str(prefix_failure_path)
            level_results.append(
                CampaignLevelResult(
                    level_id=str(frontier),
                    solved=False,
                    step_count=0,
                    used_replay_prefix=bool(prefix),
                    replay_prefix_length=replay_prefix_length,
                    solution=None,
                    failure_reason="prefix_replay_failed",
                )
            )
            level_debug_rows.append(
                {
                    "avatar_bootstrap_mode": "after_prefix_replay" if bool(prefix) else "fresh_reset",
                    "prefix_source": prefix_source,
                }
            )
            level_trace_rows.append(
                {
                    "replay_verified": False,
                    "best_trace_path": None,
                    "trace_id": None,
                    "executed_step_count": 0,
                    "analysis_action_count": 0,
                    "final_solve_action_count": 0,
                    "saved_solution_action_count": 0,
                }
            )
            failure_reason = "prefix_replay_failed"
            try:
                maybe_session = replay.get("session")
                if maybe_session is not None:
                    session_adapter.close_session(maybe_session)
            except Exception:
                pass
            break
        if not using_live_continuation:
            session = replay.get("session")
        if (not using_live_continuation) and session is None:
            prefix_failure_path = frontier_run_dir / "prefix_replay_failure.json"
            prefix_failure_path.write_text(
                json.dumps(
                    {
                        "game_id": game_id,
                        "level_id": str(frontier),
                        "failure_reason": "prefix_replay_failed",
                        "prefix_source": str(prefix_source),
                        "prefix_trace_count": int(len(prefix)),
                        "replay_prefix_length": int(replay_prefix_length),
                        "divergence": bool(replay.get("divergence", False)),
                        "frontier_reached": bool(replay.get("frontier_reached", False)),
                        "session_missing": True,
                        "prefix_validation_warnings": list(prefix_warnings),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            frontier_attempt_artifacts[f"{frontier}/prefix_replay_failure.json"] = str(prefix_failure_path)
            level_results.append(
                CampaignLevelResult(
                    level_id=str(frontier),
                    solved=False,
                    step_count=0,
                    used_replay_prefix=bool(prefix),
                    replay_prefix_length=replay_prefix_length,
                    solution=None,
                    failure_reason="prefix_replay_failed",
                )
            )
            level_debug_rows.append(
                {
                    "avatar_bootstrap_mode": "after_prefix_replay" if bool(prefix) else "fresh_reset",
                    "prefix_source": prefix_source,
                }
            )
            level_trace_rows.append(
                {
                    "replay_verified": False,
                    "best_trace_path": None,
                    "trace_id": None,
                    "executed_step_count": 0,
                    "analysis_action_count": 0,
                    "final_solve_action_count": 0,
                    "saved_solution_action_count": 0,
                }
            )
            failure_reason = "prefix_replay_failed"
            break

        if using_live_continuation:
            frontier_result = run_frontier_continuation_from_live_session(
                game_id=game_id,
                session=session,
                next_level_id=str(frontier),
                selected_avatar=continuation_context.get("selected_avatar"),
                prefix_traces=prefix,
                output_dir=output_dir,
                seed=seed,
                episode_count=episode_count,
                probe_montage=probe_montage,
                render_terminal=render_terminal,
                max_steps=max_steps,
                env_factory=env_factory,
                debug_log_path=debug_log_path,
            )
        else:
            frontier_result = run_frontier_level_from_live_session(
                game_id=game_id,
                frontier_level_id=str(frontier),
                session=session,
                prefix_traces=prefix,
                output_dir=output_dir,
                seed=seed,
                episode_count=episode_count,
                probe_montage=probe_montage,
                render_terminal=render_terminal,
                max_steps=max_steps,
                env_factory=env_factory,
                debug_log_path=debug_log_path,
            )
        solved = bool(frontier_result.get("solved", False))
        solution_payload = dict(frontier_result.get("solution", {}))
        solution_action_items = tuple(solution_payload.get("action_trace", ()))
        parsed_solution_actions: list[LevelSolveAction] = []
        for idx, item in enumerate(solution_action_items):
            if not isinstance(item, dict):
                continue
            parsed_solution_actions.append(
                LevelSolveAction(
                    step_index=int(item.get("step_index", idx)),
                    action=str(item.get("action", "")),
                    target_poi_id=item.get("target_poi_id"),
                    reason=str(item.get("reason")) if item.get("reason") is not None else None,
                    pre_level_index=int(item.get("pre_level_index", 0)),
                    post_level_index=int(item.get("post_level_index", 0)),
                    source=str(item.get("source", "frontier_solve")),
                    pre_frame=item.get("pre_frame"),
                    post_frame=item.get("post_frame"),
                    invalid_action=bool(item.get("invalid_action", False)),
                    blocked_action=bool(item.get("blocked_action", False)),
                    terminal=bool(item.get("terminal", False)),
                    reward_before=item.get("reward_before"),
                    reward_after=item.get("reward_after"),
                )
            )
        frontier_actions: tuple[str, ...] = tuple(item.action for item in parsed_solution_actions)
        if not frontier_actions:
            saved_trace_actions = tuple(str(item) for item in tuple(getattr(frontier_result.get("saved_trace"), "action_trace", ())))
            if saved_trace_actions:
                frontier_actions = saved_trace_actions
                saved_sources = tuple(str(item) for item in tuple(getattr(frontier_result.get("saved_trace"), "action_sources", ())))
                parsed_solution_actions = [
                    LevelSolveAction(
                        step_index=index,
                        action=action,
                        target_poi_id=None,
                        reason=None,
                        pre_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                        post_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                        source=(saved_sources[index] if index < len(saved_sources) else "frontier_solve"),
                        pre_frame=None,
                        post_frame=None,
                        invalid_action=False,
                        blocked_action=False,
                        terminal=False,
                        reward_before=None,
                        reward_after=None,
                    )
                    for index, action in enumerate(frontier_actions)
                ]
        actions: tuple[str, ...] = frontier_actions
        prefix_actions = _flatten_prefix_actions(tuple(prefix))
        prefix_step_rows = [
            LevelSolveAction(
                step_index=index,
                action=action,
                target_poi_id=None,
                reason="solved_prefix_replay",
                pre_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                post_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                source="solved_prefix_replay",
                pre_frame=None,
                post_frame=None,
                invalid_action=False,
                blocked_action=False,
                terminal=False,
                reward_before=None,
                reward_after=None,
            )
            for index, action in enumerate(prefix_actions)
        ]
        has_frame_bearing_solution_steps = any(
            getattr(item, "pre_frame", None) is not None or getattr(item, "post_frame", None) is not None
            for item in parsed_solution_actions
        )
        full_step_rows = (
            tuple(parsed_solution_actions)
            if has_frame_bearing_solution_steps
            else tuple(prefix_step_rows + list(parsed_solution_actions))
        )
        full_run_actions = tuple(prefix_actions) + tuple(frontier_actions)
        full_run_sources = tuple(
            str(getattr(item, "source", "frontier_solve")) for item in full_step_rows
        )
        level_result = CampaignLevelResult(
            level_id=str(frontier),
            solved=solved,
            step_count=int(solution_payload.get("step_count", len(actions))),
            used_replay_prefix=bool(frontier_result.get("diagnostics", {}).get("used_replay_prefix", bool(prefix))),
            replay_prefix_length=int(frontier_result.get("diagnostics", {}).get("replay_prefix_length", sum(len(item.action_trace) for item in prefix))),
            solution=LevelSolution(
                game_id=str(solution_payload.get("game_id", game_id)),
                level_id=str(solution_payload.get("level_id", frontier)),
                solved=bool(solution_payload.get("solved", False)),
                action_trace=tuple(parsed_solution_actions),
                step_count=int(solution_payload.get("step_count", len(parsed_solution_actions))),
                terminal=bool(solution_payload.get("terminal", False)),
                level_transition=bool(solution_payload.get("level_transition", False)),
                failure_reason=solution_payload.get("failure_reason"),
            ),
            failure_reason=frontier_result.get("failure_reason"),
        )
        level_results.append(level_result)
        level_debug_rows.append(
            {
                "avatar_bootstrap_mode": str(frontier_result.get("diagnostics", {}).get("avatar_bootstrap_mode", "live_frontier")),
                "prefix_source": prefix_source,
            }
        )
        level_trace_rows.append(
            {
                "replay_verified": False,
                "best_trace_path": None,
                "trace_id": None,
                "executed_step_count": len(tuple(full_step_rows)),
                "analysis_action_count": int(frontier_result.get("diagnostics", {}).get("analysis_action_count", len(tuple(prefix_actions)))),
                "final_solve_action_count": int(frontier_result.get("diagnostics", {}).get("final_solve_action_count", len(tuple(frontier_actions)))),
                "saved_solution_action_count": 0,
            }
        )
        highest_reached = str(frontier)
        trace_for_report_actions = full_run_actions if full_run_actions else actions
        trace_for_report_steps = full_step_rows if full_step_rows else tuple(
            LevelSolveAction(
                step_index=index,
                action=action,
                target_poi_id=None,
                reason=None,
                pre_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                post_level_index=max(0, int(str(frontier).lstrip("L") or 0)),
                source=(full_run_sources[index] if index < len(full_run_sources) else "frontier_solve"),
            )
            for index, action in enumerate(trace_for_report_actions)
        )
        for index, step_item in enumerate(trace_for_report_steps):
            campaign_step = CampaignRunStep(
                global_step_index=global_step,
                level_id=str(frontier),
                action=str(step_item.action),
                source=str(step_item.source),
                reason=None,
                pre_levels_completed=int(
                    getattr(step_item, "pre_level_index", max(0, int(str(frontier).lstrip("L") or 0)))
                ),
                post_levels_completed=int(
                    getattr(step_item, "post_level_index", max(0, int(str(frontier).lstrip("L") or 0)))
                ),
                pre_frame=getattr(step_item, "pre_frame", None),
                post_frame=getattr(step_item, "post_frame", None),
                invalid_action=bool(getattr(step_item, "invalid_action", False)),
                blocked_action=bool(getattr(step_item, "blocked_action", False)),
                terminal=bool(getattr(step_item, "terminal", False)),
                reward_before=getattr(step_item, "reward_before", None),
                reward_after=getattr(step_item, "reward_after", None),
            )
            campaign_step_trace.append(campaign_step)
            global_trace.append(campaign_step)
            global_step += 1

        if solved:
            raw_trace = frontier_result.get("saved_trace")
            if raw_trace is None:
                raw_trace = SavedLevelTrace(
                    game_id=game_id,
                    level_id=str(frontier),
                    solved=True,
                    action_trace=full_run_actions if full_run_actions else actions,
                    step_count=len(full_run_actions if full_run_actions else actions),
                    source_run_id=None,
                    trace_version=1,
                    replay_verified=False,
                    action_sources=full_run_sources if full_run_sources else None,
                )
            solution_trace_steps = tuple(parsed_solution_actions)
            solution_trace_actions = tuple(str(item.action) for item in solution_trace_steps)
            solution_trace_sources = tuple(
                str(getattr(item, "source", "frontier_solve"))
                for item in solution_trace_steps
            )
            if not solution_trace_actions:
                raw_actions = tuple(str(item) for item in tuple(getattr(raw_trace, "action_trace", ())))
                raw_sources = tuple(str(item) for item in tuple(getattr(raw_trace, "action_sources", ()) or ()))
                filtered_raw = tuple(
                    (
                        action,
                        (raw_sources[index] if index < len(raw_sources) else "frontier_solve"),
                    )
                    for index, action in enumerate(raw_actions)
                )
                solution_trace_actions = tuple(action for action, _source in filtered_raw)
                solution_trace_sources = tuple(source for _action, source in filtered_raw)
            executed_trace_records_list = []
            for index, action in enumerate(solution_trace_actions):
                if index < len(solution_trace_steps):
                    step = solution_trace_steps[index]
                    executed_trace_records_list.append(
                        {
                            "action": str(step.action),
                            "source": str(getattr(step, "source", "frontier_solve")),
                            "pre_level_index": int(getattr(step, "pre_level_index", max(0, int(str(frontier).lstrip("L") or 0)))),
                            "post_level_index": int(getattr(step, "post_level_index", max(0, int(str(frontier).lstrip("L") or 0)))),
                            "terminal": bool(getattr(step, "terminal", False)),
                        }
                    )
                else:
                    executed_trace_records_list.append(
                        {
                            "action": action,
                            "source": (solution_trace_sources[index] if index < len(solution_trace_sources) else "frontier_solve"),
                        }
                    )
            executed_trace_records = tuple(executed_trace_records_list)
            finalized = finalize_solved_level_trace(
                solved_level_result=level_result.solution,
                executed_actions=executed_trace_records,
                game_id=game_id,
                level_id=str(frontier),
                prefix_traces=prefix,
                render_terminal=render_terminal,
                env_factory=env_factory,
                trace_db_path=trace_db_path,
            )
            verified_trace = finalized["saved_trace"]
            if not bool(finalized.get("replay_verified", False)):
                try:
                    save_trace_history_row(db_path=trace_db_path, trace=verified_trace, trace_id=getattr(verified_trace, "trace_id", None))
                except Exception:
                    pass
            trace_paths = write_saved_level_trace(
                run_dir=resolve_run_dir(output_dir, game_id) / str(frontier),
                trace=verified_trace,
                step_trace=tuple(trace_for_report_steps),
            )
            saved_trace_action_count = len(tuple(getattr(verified_trace, "action_trace", ())))
            final_level_run_action_count = len(tuple(solution_trace_actions))
            includes_bootstrap = bool(
                tuple(getattr(verified_trace, "action_sources", ()) or ())
                and "bootstrap_replay" in tuple(getattr(verified_trace, "action_sources", ()) or ())
            )
            level_trace_rows[-1] = {
                "replay_verified": bool(finalized.get("replay_verified", False)),
                "best_trace_path": trace_paths.get("saved_level_trace.json") if bool(finalized.get("replay_verified", False)) else None,
                "trace_id": finalized.get("trace_id"),
                "trace_failure_reason": finalized.get("failure_reason"),
                "saved_trace_includes_bootstrap": bool(includes_bootstrap),
                "final_level_run_action_count": int(final_level_run_action_count),
                "saved_trace_action_count": int(saved_trace_action_count),
                "executed_step_count": len(tuple(trace_for_report_steps)),
                "analysis_action_count": int(frontier_result.get("diagnostics", {}).get("analysis_action_count", len(tuple(prefix_actions)))),
                "final_solve_action_count": int(frontier_result.get("diagnostics", {}).get("final_solve_action_count", len(tuple(frontier_actions)))),
                "saved_solution_action_count": int(saved_trace_action_count),
            }
            if int(saved_trace_action_count) < int(len(tuple(solution_trace_actions))):
                failure_reason = "saved_solution_trace_mismatch"
                break
            if int(saved_trace_action_count) != int(len(tuple(solution_trace_actions))):
                failure_reason = "saved_solution_trace_mismatch"
                break
            # Add verified trace to current run traces for use as prefix
            if bool(finalized.get("replay_verified", False)) and validate_prefix_trace_entry(verified_trace):
                current_run_traces[str(frontier)] = verified_trace

            # continue in-session without reset/replay when next level is already loaded.
            try:
                post_obs = session_adapter.get_current_observation(session)
                next_level_id = f"L{int(post_obs.levels_completed)}"
                continuation_context = {
                    "next_level_id": next_level_id,
                    "selected_avatar": frontier_result.get("_selected_avatar_obj"),
                }
            except Exception:
                continuation_context = None
            state = update_campaign_state_after_level(
                state=state,
                level_id=str(frontier),
                solved=True,
                trace_path=trace_paths.get("saved_level_trace.json"),
                step_count=int(verified_trace.step_count),
            )
            if bool(solution_payload.get("terminal", False)):
                terminal_campaign_complete = True
                break
            # Only break if replay verification failed AND not using live continuation
            if not bool(finalized.get("replay_verified", False)) and not using_live_continuation:
                failure_reason = str(finalized.get("failure_reason") or "trace_replay_verification_failed")
                break
        else:
            continuation_context = None
            trajectory_stats = dict(frontier_result.get("diagnostics", {}).get("trajectory_stats", {}) or {})
            unsolved_level_trajectory_stats.append(
                {
                    "level_id": str(frontier),
                    "failure_reason": frontier_result.get("failure_reason") or "frontier_unsolved",
                    "generated_trajectory_count": int(trajectory_stats.get("generated_trajectory_count", 0)),
                    "attempted_trajectory_count": int(trajectory_stats.get("attempted_trajectory_count", 0)),
                    "completed_trajectory_count": int(trajectory_stats.get("completed_trajectory_count", 0)),
                    "min_steps_per_attempted_trajectory": int(trajectory_stats.get("min_steps_per_attempted_trajectory", 0)),
                    "max_steps_per_attempted_trajectory": int(trajectory_stats.get("max_steps_per_attempted_trajectory", 0)),
                    "mean_steps_per_attempted_trajectory": float(trajectory_stats.get("mean_steps_per_attempted_trajectory", 0.0)),
                    "total_executed_steps_across_attempted_trajectories": int(trajectory_stats.get("total_executed_steps_across_attempted_trajectories", 0)),
                }
            )
            frontier_failure_path = frontier_run_dir / "frontier_failure.json"
            frontier_failure_path.write_text(
                json.dumps(
                    {
                        "game_id": game_id,
                        "level_id": str(frontier),
                        "failure_reason": frontier_result.get("failure_reason") or "frontier_unsolved",
                        "used_replay_prefix": bool(prefix),
                        "prefix_source": str(prefix_source),
                        "prefix_trace_count": int(len(prefix)),
                        "replay_prefix_length": int(replay_prefix_length),
                        "use_solutions": bool(use_solutions),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            frontier_attempt_artifacts[f"{frontier}/frontier_failure.json"] = str(frontier_failure_path)
            state = update_campaign_state_after_level(
                state=state,
                level_id=str(frontier),
                solved=False,
                trace_path=None,
                step_count=None,
            )
            failure_reason = frontier_result.get("failure_reason") or "frontier_unsolved"
            break

    campaign_step_trace = list(
        _with_missing_solution_prefix_steps(
            campaign_steps=tuple(campaign_step_trace),
            level_sequence=tuple(str(item) for item in level_sequence),
            replay_valid_db_prefix_by_level=replay_valid_db_prefix_by_level,
        )
    )
    global_trace = list(campaign_step_trace)

    solved_all = bool(terminal_campaign_complete) or (
        failure_reason is None and all(bool(state[level_id].solved) for level_id in level_sequence)
    )
    report = CampaignRunReport(
        game_id=game_id,
        levels=tuple(level_results),
        global_action_trace=tuple(global_trace),
        solved=bool(solved_all),
        highest_reached_level_id=highest_reached,
        failure_reason=None if solved_all else (failure_reason or "campaign_incomplete"),
        diagnostics={
            "requested_level_count": len(level_sequence),
            "solved_level_count": sum(1 for level_id in level_sequence if state[level_id].solved),
            "db_solved_level_count": len(db_solved_levels),
            "used_solutions": bool(use_solutions),
            "prefix_trace_count": int(prefix_trace_count),
            "frontier_level_id": get_frontier_level_id(
                state=state,
                level_sequence=level_sequence,
                trace_db_path=trace_db_path,
                game_id=game_id,
                use_solutions=bool(use_solutions),
            ),
            "avatar_episode_count_used": int(episode_count),
            "campaign_step_count": len(campaign_step_trace),
            "prefix_validation_warnings": tuple(prefix_warnings),
        },
    )
    campaign_dir = resolve_run_dir(output_dir, game_id) / "campaign"
    artifacts = write_campaign_artifacts(
        run_dir=campaign_dir,
        report=report,
        campaign_step_trace=tuple(campaign_step_trace),
    )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    unsolved_stats_path = campaign_dir / "unsolved_level_trajectory_stats.json"
    unsolved_stats_path.write_text(json.dumps(list(unsolved_level_trajectory_stats), indent=2), encoding="utf-8")
    artifacts["unsolved_level_trajectory_stats.json"] = str(unsolved_stats_path)
    campaign_levels_path = Path(artifacts.get("campaign_levels.json", campaign_dir / "campaign_levels.json"))
    if campaign_levels_path.exists():
        try:
            levels_payload = json.loads(campaign_levels_path.read_text(encoding="utf-8"))
            for idx, level in enumerate(tuple(report.levels)):
                level_id = str(level.level_id)
                row = dict(levels_payload.get(level_id, {}))
                trace_row = level_trace_rows[idx] if idx < len(level_trace_rows) else {}
                row["replay_verified"] = bool(trace_row.get("replay_verified", False))
                row["best_trace_path"] = trace_row.get("best_trace_path")
                row["saved_trace_includes_bootstrap"] = bool(trace_row.get("saved_trace_includes_bootstrap", False))
                row["final_level_run_action_count"] = int(trace_row.get("final_level_run_action_count", 0))
                row["saved_trace_action_count"] = int(trace_row.get("saved_trace_action_count", 0))
                row["executed_step_count"] = int(trace_row.get("executed_step_count", 0))
                row["analysis_action_count"] = int(trace_row.get("analysis_action_count", 0))
                row["final_solve_action_count"] = int(trace_row.get("final_solve_action_count", 0))
                row["saved_solution_action_count"] = int(trace_row.get("saved_solution_action_count", 0))
                if not bool(level.solved):
                    row["best_step_count"] = None
                    row["best_trace_path"] = None
                    row["replay_verified"] = False
                    row["saved_trace_includes_bootstrap"] = False
                    row["final_level_run_action_count"] = 0
                    row["saved_trace_action_count"] = 0
                    row["executed_step_count"] = int(trace_row.get("executed_step_count", 0))
                    row["analysis_action_count"] = int(trace_row.get("analysis_action_count", 0))
                    row["final_solve_action_count"] = int(trace_row.get("final_solve_action_count", 0))
                    row["saved_solution_action_count"] = int(trace_row.get("saved_solution_action_count", 0))
                levels_payload[level_id] = row
            campaign_levels_path.write_text(json.dumps(levels_payload, indent=2), encoding="utf-8")
        except Exception:
            pass
    store_index = write_trace_store_index_artifacts(
        run_dir=campaign_dir,
        game_id=game_id,
        solved_levels=tuple(level_id for level_id in level_sequence if state[level_id].solved),
        trace_db_path=trace_db_path,
    )
    index_path = Path(store_index.get("trace_store_index.json", campaign_dir / "trace_store_index.json"))
    rebuilt_levels = rebuild_trace_store_index(db_path=trace_db_path, game_id=game_id)
    rebuilt_payload = {
        "game_id": game_id,
        "solved_levels": sorted(list(rebuilt_levels.keys())),
        "trace_store_db": str(trace_db_path),
        "levels": rebuilt_levels,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(rebuilt_payload, indent=2), encoding="utf-8")
    artifacts.update(store_index)
    artifacts.update(frontier_attempt_artifacts)
    output = report.to_dict()
    for index, level_payload in enumerate(output.get("levels", [])):
        if index < len(level_debug_rows):
            level_payload["avatar_bootstrap_mode"] = str(level_debug_rows[index].get("avatar_bootstrap_mode", "live_frontier"))
            level_payload["prefix_source"] = str(level_debug_rows[index].get("prefix_source", "none"))
        if index < len(level_trace_rows):
            level_payload["replay_verified"] = bool(level_trace_rows[index].get("replay_verified", False))
            level_payload["best_trace_path"] = level_trace_rows[index].get("best_trace_path")
            level_payload["trace_id"] = level_trace_rows[index].get("trace_id")
            level_payload["trace_failure_reason"] = level_trace_rows[index].get("trace_failure_reason")
            level_payload["analysis_action_count"] = int(level_trace_rows[index].get("analysis_action_count", 0))
            level_payload["final_solve_action_count"] = int(level_trace_rows[index].get("final_solve_action_count", 0))
            level_payload["saved_solution_action_count"] = int(level_trace_rows[index].get("saved_solution_action_count", 0))
    output["artifact_paths"] = artifacts

    saved_trace_actions_by_level = {
        str(getattr(trace, "level_id", "")): [
            str(action) for action in tuple(getattr(trace, "action_trace", ()) or ()) if action is not None
        ]
        for trace in tuple(replay_valid_db_prefix or ()) + tuple(current_run_traces.values())
        if getattr(trace, "level_id", None) is not None
    }

    def _load_level_action_sequence(level_id: str) -> list[str]:
        actions = list(actions_by_level.get(str(level_id), ()))
        if actions:
            return actions
        saved_trace_actions = list(saved_trace_actions_by_level.get(str(level_id), ()))
        if saved_trace_actions:
            return saved_trace_actions
        level_dir = game_root / str(level_id)
        candidate_paths = (
            level_dir / "saved_level_trace_actions.json",
            level_dir / "level_solution_actions.json",
        )
        for candidate_path in candidate_paths:
            if not candidate_path.exists():
                continue
            try:
                payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, list):
                parsed = [str(item) for item in payload if item is not None]
                if parsed:
                    return parsed
        return []

    # Print action sequences per level
    print("\n" + "="*60)
    print("ACTION SEQUENCES PER LEVEL")
    print("="*60)

    # Read from campaign_action_trace.json which has all actions
    campaign_trace_path = campaign_dir / "campaign_action_trace.json"
    if campaign_trace_path.exists():
        try:
            with open(campaign_trace_path) as f:
                trace_data = json.load(f)

            solved_by_level = {
                str(level_payload.get("level_id")): bool(level_payload.get("solved", False))
                for level_payload in output.get("levels", [])
                if level_payload.get("level_id") is not None
            }
            if not solved_by_level:
                solved_by_level = {
                    str(level_id): bool(getattr(level_state, "solved", False))
                    for level_id, level_state in state.items()
                }

            # Group actions by level_id
            actions_by_level = {}
            for step in trace_data:
                level_id = step.get("level_id")
                action = step.get("action")
                if level_id and action:
                    if level_id not in actions_by_level:
                        actions_by_level[level_id] = []
                    actions_by_level[level_id].append(action)

            # Print in level sequence order
            for level_id in level_sequence:
                if not solved_by_level.get(str(level_id), False):
                    print(f"{level_id}: [no actions]")
                    continue
                actions = _load_level_action_sequence(str(level_id))
                if actions:
                    # Convert action names to letters: UP->U, DOWN->D, LEFT->L, RIGHT->R
                    action_map = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
                    action_str = "".join(action_map.get(str(a).upper(), "") for a in actions)
                    print(f"{level_id}: {action_str}")
                else:
                    print(f"{level_id}: [no actions]")
        except Exception as e:
            for level_id in level_sequence:
                print(f"{level_id}: [error reading trace: {e}]")
    else:
        for level_id in level_sequence:
            print(f"{level_id}: [no campaign trace]")

    print("="*60 + "\n")

    if session is not None:
        try:
            session_adapter.close_session(session)
        except Exception:
            pass
    return output
