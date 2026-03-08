from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.controller.controller import select_instruction
from codex_baseline_v2.executor.online_executor import run_online_execution
from codex_baseline_v2.memory.store import append_round_report, load_blackboard_typed, save_blackboard
from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.session_manager import SessionManagerV2
from codex_baseline_v2.runtime.trajectory_collector import CollectionConfigV2, TrajectoryCollectorV2
from codex_baseline_v2.shared.config import V2Config
from codex_baseline_v2.shared.logging_utils import log_event
from codex_baseline_v2.shared.metrics import compute_round_metrics
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories
from codex_baseline_v2.trajectory_analysis.avatar_hypothesis_debug import export_avatar_debug


def run_autonomous_rounds(cfg: V2Config, env_factory: Any, env_factory_path: Optional[str] = None, workers: int = 1) -> None:
    storage = StoragePathsV2(cfg.memory.storage_dir)
    session_mgr = SessionManagerV2(cfg.memory.storage_dir)
    state = session_mgr.init_or_resume(cfg.game_id, resume_if_exists=bool(cfg.runtime.resume_if_exists))

    max_rounds = int(cfg.runtime.max_rounds) if cfg.runtime else cfg.rounds
    last_report = None
    shared_env = None
    shared_session: Optional[EnvironmentSessionV2] = None
    if workers <= 1:
        shared_env = env_factory()
        shared_session = EnvironmentSessionV2(shared_env, cfg.game_id)
    try:
        for round_id in range(state.round_id, max_rounds):
            print(f"[v2] round_start game_id={cfg.game_id} round_id={round_id}", flush=True)
            storage.ensure_round_dirs(cfg.game_id, round_id)
            session = shared_session
            if session is None:
                env = env_factory()
                session = EnvironmentSessionV2(env, cfg.game_id)
            collector = TrajectoryCollectorV2(
                storage,
                CollectionConfigV2(
                    episodes=cfg.collection.initial_probe_episodes if round_id == 0 else cfg.collection.directed_probe_episodes,
                    max_steps_per_episode=cfg.collection.max_steps_per_episode,
                    max_steps_per_instruction=cfg.collection.max_steps_per_instruction,
                    seed=cfg.collection.seed,
                    action_repeat_limit=cfg.collection.action_repeat_limit,
                    keep_invalid_steps_for_debug=cfg.debug.keep_invalid_steps_for_debug,
                ),
            )
            if round_id == 0:
                if workers > 1 and env_factory_path and cfg.env.env_id and cfg.env.env_root:
                    episodes = collector.collect_round_parallel(
                        env_factory_path=env_factory_path,
                        env_id=cfg.env.env_id,
                        env_root=cfg.env.env_root,
                        mode="random_probe",
                        instruction=None,
                        round_id=round_id,
                        workers=workers,
                    )
                else:
                    episodes = collector.collect_round(session, "random_probe", None, round_id)
                analyzed = analyze_episodes(episodes, cfg.analyst)
                blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=round_id)
                collector.write_artifacts(cfg.game_id, round_id, analyzed)
                if cfg.debug.export_avatar_candidates:
                    paths = storage.ensure_round_dirs(cfg.game_id, round_id)
                    with open(f"{paths['analyst_outputs']}/avatar_candidates.json", "w", encoding="utf-8") as handle:
                        handle.write(json.dumps(export_avatar_debug(analyzed), sort_keys=True))
                controller_modes = ["random_probe"]
                target_progress = None
            else:
                prior_blackboard = load_blackboard_typed(storage, cfg.game_id)
                if prior_blackboard is None:
                    raise RuntimeError("Missing blackboard for directed round")
                instruction = select_instruction(prior_blackboard, cfg.controller, cfg.scoring, round_id)
                outcome, episode = run_online_execution(session, instruction, prior_blackboard, cfg.executor)
                episodes = [episode]
                remaining = max(0, int(cfg.collection.directed_probe_episodes) - 1)
                if remaining > 0 and workers > 1 and env_factory_path and cfg.env.env_id and cfg.env.env_root:
                    parallel_collector = TrajectoryCollectorV2(
                        storage,
                        CollectionConfigV2(
                            episodes=remaining,
                            max_steps_per_episode=cfg.collection.max_steps_per_episode,
                            max_steps_per_instruction=cfg.collection.max_steps_per_instruction,
                            seed=cfg.collection.seed,
                            action_repeat_limit=cfg.collection.action_repeat_limit,
                            keep_invalid_steps_for_debug=cfg.debug.keep_invalid_steps_for_debug,
                        ),
                    )
                    extra_eps = parallel_collector.collect_round_parallel(
                        env_factory_path=env_factory_path,
                        env_id=cfg.env.env_id,
                        env_root=cfg.env.env_root,
                        mode="unguided_probe",
                        instruction=None,
                        round_id=round_id,
                        workers=workers,
                    )
                    episodes.extend(extra_eps)
                analyzed = analyze_episodes(episodes, cfg.analyst)
                blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=round_id, prior_blackboard=prior_blackboard)
                blackboard = BlackboardStateV2(
                    schema_version=blackboard.schema_version,
                    game_id=blackboard.game_id,
                    round_id=blackboard.round_id,
                    palette=blackboard.palette,
                    poi_table=blackboard.poi_table,
                    reachability_table=blackboard.reachability_table,
                    consequence_table=blackboard.consequence_table + list(outcome.consequence_records),
                    avatar_hypotheses=blackboard.avatar_hypotheses,
                    traversable_map=blackboard.traversable_map,
                    unresolved_hypotheses=blackboard.unresolved_hypotheses,
                    falsified_hypotheses=blackboard.falsified_hypotheses,
                    metadata=dict(blackboard.metadata),
                )
                collector.write_artifacts(cfg.game_id, round_id, analyzed)
                if cfg.debug.export_avatar_candidates:
                    paths = storage.ensure_round_dirs(cfg.game_id, round_id)
                    with open(f"{paths['analyst_outputs']}/avatar_candidates.json", "w", encoding="utf-8") as handle:
                        handle.write(json.dumps(export_avatar_debug(analyzed), sort_keys=True))
                controller_modes = [instruction.mode]
                target_progress = outcome.target_progress
            reach_lookup = {r.poi_id: r.status for r in blackboard.reachability_table}
            metrics = compute_round_metrics(
                analyzed,
                blackboard.poi_table,
                reach_lookup,
                blackboard.consequence_table,
                controller_modes,
                len(blackboard.avatar_hypotheses),
                target_progress=target_progress,
            )
            diagnostics = []
            if metrics.states_observed > 0 and metrics.unique_states == 0:
                diagnostics.append("state_hash_missing")
            if metrics.candidate_avatar_count == 0:
                diagnostics.append("no_avatar_candidates")
            if metrics.candidate_poi_count > 0 and metrics.reachable_poi_count == 0:
                diagnostics.append("no_reachable_pois")
            targeted_run = instruction.mode not in {"random_probe", "unguided_probe"} if round_id > 0 else False
            if targeted_run and metrics.target_progress_mean == 0:
                diagnostics.append("no_target_progress")
            if targeted_run and metrics.route_success_rate == 0:
                diagnostics.append("no_route_success")
            instruction_history = list(blackboard.metadata.get("instruction_history", [])) if isinstance(blackboard.metadata, dict) else []
            if round_id > 0:
                instruction_history.append(
                    {
                        "round_id": round_id,
                        "mode": instruction.mode,
                        "instruction_id": instruction.instruction_id,
                        "target_poi_id": instruction.target_poi_id,
                        "outcome": "progress" if any(c.distance_decreased or c.reached or c.contact for c in outcome.consequence_records) else "no_progress",
                    }
                )
            outcome_history = list(blackboard.metadata.get("executor_outcome_history", [])) if isinstance(blackboard.metadata, dict) else []
            if round_id > 0:
                outcome_history.append(outcome.to_dict())
            blackboard = BlackboardStateV2(
                schema_version=blackboard.schema_version,
                game_id=blackboard.game_id,
                round_id=blackboard.round_id,
                palette=blackboard.palette,
                poi_table=blackboard.poi_table,
                reachability_table=blackboard.reachability_table,
                consequence_table=blackboard.consequence_table,
                avatar_hypotheses=blackboard.avatar_hypotheses,
                traversable_map=blackboard.traversable_map,
                unresolved_hypotheses=blackboard.unresolved_hypotheses,
                falsified_hypotheses=blackboard.falsified_hypotheses,
                metadata={
                    **blackboard.metadata,
                    "diagnostics": diagnostics,
                    "metrics": metrics.to_dict(),
                    "instruction_history": instruction_history[-20:],
                    "executor_outcome_history": outcome_history[-10:],
                    "round_mode": controller_modes[0] if controller_modes else "unknown",
                },
            )
            save_blackboard(cfg.memory, storage, blackboard)
            report = {
                "round_id": round_id,
                "game_id": cfg.game_id,
                "poi_count": len(blackboard.poi_table),
                "metrics": metrics.to_dict(),
                "diagnostics": diagnostics,
                "invalid_target_link_count": sum(1 for c in blackboard.consequence_table if c.consequence_class == "invalid_target_link"),
            }
            append_round_report(storage, cfg.game_id, round_id, report)
            log_event(cfg.logging.log_dir, "v2_round_complete", report)
            print(f"[v2] round_complete game_id={cfg.game_id} round_id={round_id} poi_count={len(blackboard.poi_table)}", flush=True)
            last_report = report

        if last_report is not None:
            session_mgr.export_summary(cfg.game_id, last_report)
    finally:
        if shared_env is not None and hasattr(shared_env, "close"):
            try:
                shared_env.close()
            except Exception:
                pass
