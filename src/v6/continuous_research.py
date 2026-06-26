from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v6.evaluation.interaction_sampling import InteractionSamplingConfig, parse_v05c_games, parse_v05c_samplers, run_interaction_sampling_v05c
from v6.hypothesis_suite_report import run_hypothesis_suite_report
from v6.memory.compact_memory import CompactMemoryFoldConfig, build_memory_summary, ensure_memory_layout, fold_epoch_raw_into_compact_memory, load_memory_summary
from v6.memory.memory_cleanup import cleanup_epoch_artifacts, disk_usage_snapshot, stop_due_to_disk, validate_cleanup_safe


@dataclass(frozen=True)
class ContinuousResearchConfig:
    experiment_name: str
    games: str
    samplers: str
    seeds: str
    steps_per_epoch: int
    max_epochs: int
    horizon: int
    context_depth: int
    output_dir: str
    stop_if_disk_above_percent: float = 90.0
    stop_if_no_new_stable_contingencies_for: int = 2
    scan_all_dbs: bool = False
    max_db_files: int = 0
    max_rows: int = 1_000_000
    resume: bool = True
    cleanup: bool = True
    max_replay_queue_size: int = 50_000
    replay_retention_percent: int = 5
    fast_postprocessing: bool = True
    workers: int = 60
    initial_workers: int | None = None
    ram_ramp_threshold_percent: float = 85.0
    initial_worker_ramp_delay_seconds: float = 20.0
    per_worker_ramp_delay_seconds: float = 5.0
    env_root: str | None = None


def run_continuous_research(config: ContinuousResearchConfig) -> dict[str, Any]:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    stop_file = root / "STOP"
    manifest = _load_or_initialize_manifest(config, manifest_path)
    memory_dir = root / "memory"
    memory_paths = ensure_memory_layout(memory_dir)
    latest_status: dict[str, Any] | None = None
    consecutive_no_new = int(manifest.get("consecutive_no_new_stable_contingencies", 0) or 0)

    while int(manifest["current_epoch"]) < int(config.max_epochs):
        if stop_file.exists():
            manifest["stopped"] = True
            manifest["stop_reason"] = "manual stop file exists"
            break
        triggered, disk_before = stop_due_to_disk(root, threshold_percent=float(config.stop_if_disk_above_percent))
        if triggered:
            manifest["stopped"] = True
            manifest["stop_reason"] = "disk usage exceeded configured limit"
            manifest["latest_disk_snapshot"] = disk_before
            break

        epoch_number = int(manifest["current_epoch"]) + 1
        epoch_id = f"epoch_{epoch_number:04d}"
        epoch_dir = root / "epochs" / epoch_id
        raw_dir = epoch_dir / "raw"
        reports_dir = epoch_dir / "reports"
        cleanup_dir = epoch_dir / "cleanup"
        status_dir = epoch_dir / "status"
        raw_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)

        global_step_start = int(manifest.get("last_global_step_end", 0) or 0) + 1
        global_step_end = global_step_start + int(config.steps_per_epoch) - 1
        memory_before = build_memory_summary(memory_paths)
        memory_size_before = _tree_size(memory_dir)
        previous_summary_snapshot = load_memory_summary(memory_paths.summary_json)
        previous_peak_workers = int(manifest.get("last_sampling_peak_workers", config.workers) or config.workers)
        requested_workers = max(1, int(config.workers))
        max_epoch_workers = max(1, min(requested_workers, previous_peak_workers))
        initial_epoch_workers = max(1, min(int(config.initial_workers or 1), max_epoch_workers))
        initial_worker_ramp_delay_seconds = 0.0
        ram_snapshot_at_epoch_start = _system_ram_snapshot()
        epoch_start_payload = {
            "epoch_id": epoch_id,
            "global_step_start": global_step_start,
            "global_step_end": global_step_end,
            "requested_workers": requested_workers,
            "max_epoch_workers": max_epoch_workers,
            "initial_epoch_workers": initial_epoch_workers,
            "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
            "initial_worker_ramp_delay_seconds": float(initial_worker_ramp_delay_seconds),
            "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
            "ram_snapshot_at_epoch_start": ram_snapshot_at_epoch_start,
        }
        _write_epoch_start(status_dir, epoch_start_payload)
        print(_format_epoch_start(epoch_start_payload))

        sampling_rows = run_interaction_sampling_v05c(
            InteractionSamplingConfig(
                games=parse_v05c_games(config.games, env_root=config.env_root),
                samplers=parse_v05c_samplers(config.samplers),
                seeds=tuple(int(item) for item in config.seeds.split(",") if item.strip()),
                train_seeds=(0,),
                test_seed=0,
                steps=int(config.steps_per_epoch),
                horizon=int(config.horizon),
                context_depth=int(config.context_depth),
                output_dir=str(raw_dir),
                env_root=config.env_root,
                memory_input_dir=str(memory_dir),
                memory_output_dir=str(memory_dir),
                global_step_offset=global_step_start - 1,
                fast_postprocessing=bool(config.fast_postprocessing),
                workers=max_epoch_workers,
                initial_workers=initial_epoch_workers,
                enable_worker_ramp=True,
                ram_ramp_threshold_percent=float(config.ram_ramp_threshold_percent),
                initial_worker_ramp_delay_seconds=float(initial_worker_ramp_delay_seconds),
                per_worker_ramp_delay_seconds=float(config.per_worker_ramp_delay_seconds),
            )
        )
        worker_execution = _load_sampling_worker_execution(raw_dir)
        fold_summary = fold_epoch_raw_into_compact_memory(
            epoch_raw_dir=raw_dir,
            memory_dir=memory_dir,
            fold_config=CompactMemoryFoldConfig(
                global_step_start=global_step_start,
                global_step_end=global_step_end,
                max_replay_queue_size=int(config.max_replay_queue_size),
                replay_retention_percent=int(config.replay_retention_percent),
            ),
        )
        suite_summary = run_hypothesis_suite_report(
            run_dir=raw_dir,
            memory_dir=memory_dir,
            output_dir=reports_dir,
            scan_all_dbs=bool(config.scan_all_dbs),
            max_db_files=int(config.max_db_files),
            max_rows=int(config.max_rows),
            epoch_id=epoch_id,
            global_step_start=global_step_start,
            global_step_end=global_step_end,
            interactions_this_epoch=int(sum(int(row.get("total_interactions", 0) or 0) for row in sampling_rows)),
            total_interactions_seen=int(load_memory_summary(memory_paths.summary_json).get("total_interactions_seen", 0) or 0),
            memory_size_before_bytes=memory_size_before,
            memory_size_after_bytes=_tree_size(memory_dir),
        )
        memory_after = build_memory_summary(memory_paths)
        continuity_report = _write_memory_continuity_report(
            reports_dir=reports_dir,
            epoch_id=epoch_id,
            previous_memory_summary=previous_summary_snapshot,
            restored_memory_summary=memory_before,
            after_epoch_memory_summary=memory_after,
            memory_loaded_from_previous_epoch=epoch_number > 1,
        )
        validate_cleanup_safe(epoch_dir, memory_dir, required_reports=True)
        cleanup_summary = cleanup_epoch_artifacts(epoch_dir=epoch_dir, memory_dir=memory_dir) if bool(config.cleanup) else _no_cleanup_summary(epoch_dir, memory_dir)
        disk_after = disk_usage_snapshot(root)

        deltas = _compute_epoch_deltas(memory_before, memory_after, suite_summary, latest_status)
        if int(deltas["stable_contingency_count_delta"]) <= 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        h04_metrics = suite_summary.get("H04 core metrics", {}) or {}
        h05_metrics = suite_summary.get("H05 core metrics", {}) or {}
        h06_metrics = suite_summary.get("H06 core metrics", {}) or {}
        h07_metrics = suite_summary.get("H07 core metrics", {}) or {}
        h08_metrics = suite_summary.get("H08 core metrics", {}) or {}
        h09_metrics = suite_summary.get("H09 core metrics", {}) or {}
        h10_metrics = suite_summary.get("H10 core metrics", {}) or {}
        h11_metrics = suite_summary.get("H11 core metrics", {}) or {}
        status = {
            "epoch_id": epoch_id,
            "global_step_start": global_step_start,
            "global_step_end": global_step_end,
            "games": suite_summary.get("game_count"),
            "interactions_this_epoch": suite_summary.get("interactions_this_epoch"),
            "levels_successfully_completed_per_epoch": suite_summary.get("levels_successfully_completed_per_epoch"),
            "games_solved_per_epoch": suite_summary.get("games_solved_per_epoch"),
            "solved_games": suite_summary.get("solved_games"),
            "completed_levels_by_game": suite_summary.get("completed_levels_by_game"),
            "disk_before_cleanup_bytes": cleanup_summary["disk_before_cleanup_bytes"],
            "disk_after_cleanup_bytes": cleanup_summary["disk_after_cleanup_bytes"],
            "disk_used_percent": disk_after["disk_used_percent"],
            "H01": suite_summary.get("H01 decision"),
            "H02": suite_summary.get("H02 decision"),
            "H02A": (suite_summary.get("H02 core metrics") or {}).get("h02a_replay_attention_decision"),
            "H02B": (suite_summary.get("H02 core metrics") or {}).get("h02b_pre_carrier_timing_decision"),
            "H03": suite_summary.get("H03 decision"),
            "H04": suite_summary.get("H04 decision"),
            "H05": suite_summary.get("H05 decision"),
            "H06": suite_summary.get("H06 decision"),
            "H07": suite_summary.get("H07 decision"),
            "H08": suite_summary.get("H08 decision"),
            "H09": suite_summary.get("H09 decision"),
            "H10": suite_summary.get("H10 decision"),
            "H11": suite_summary.get("H11 decision"),
            "stable_contingencies": (suite_summary.get("H01 core metrics") or {}).get("stable_contingency_count"),
            "games_with_stable_contingencies": (suite_summary.get("H01 core metrics") or {}).get("games_with_stable_contingencies"),
            "replay_lift": (suite_summary.get("H02 core metrics") or {}).get("prediction_violation_replay_lift"),
            "direct_replay_evidence": "available" if (suite_summary.get("H02 core metrics") or {}).get("direct_replay_lift_available") else "unavailable",
            "h02_timing_note": h02_dir_note(suite_summary),
            "compression_ratio": (suite_summary.get("H03 core metrics") or {}).get("compression_ratio"),
            "singleton_ratio": (suite_summary.get("H03 core metrics") or {}).get("singleton_family_ratio"),
            "cross_context_families": (suite_summary.get("H03 core metrics") or {}).get("family_cross_context_count"),
            "carrier_candidates": h04_metrics.get("carrier_candidate_count"),
            "stable_carriers": h04_metrics.get("stable_carrier_count"),
            "role_candidates": h05_metrics.get("role_candidate_count"),
            "emergent_roles": h05_metrics.get("emergent_role_count"),
            "role_transfer_attempts": h06_metrics.get("transfer_attempt_count"),
            "role_transfer_success_rate": h06_metrics.get("transfer_success_rate"),
            "h06_role_mismatch_count": h06_metrics.get("role_mismatch_count"),
            "h06_mean_best_margin": h06_metrics.get("mean_best_margin"),
            "concept_candidates": h07_metrics.get("concept_candidate_count"),
            "promoted_concepts": h07_metrics.get("promoted_concept_count"),
            "h07_strong_transfer_successes": h07_metrics.get("concept_strong_transfer_success_count"),
            "world_model_components": h08_metrics.get("world_model_component_count"),
            "coherent_world_model_components": h08_metrics.get("coherent_world_model_component_count"),
            "candidate_only_world_model_components": h08_metrics.get("candidate_only_world_model_component_count"),
            "future_option_events": h09_metrics.get("future_option_event_count"),
            "future_option_motifs": h09_metrics.get("future_option_motif_count"),
            "emergent_future_option_motifs": h09_metrics.get("emergent_future_option_motif_count"),
            "option_attention_lift": h10_metrics.get("option_attention_lift"),
            "high_option_change_attention_rate": h10_metrics.get("high_option_change_attention_rate"),
            "h10_replay_attention_count": h10_metrics.get("replay_attention_count"),
            "h10_contradiction_attention_count": h10_metrics.get("contradiction_attention_count"),
            "future_option_transfer_links": h11_metrics.get("future_option_transfer_link_count"),
            "motifs_with_strong_transfer": h11_metrics.get("motifs_with_strong_transfer_count"),
            "motifs_with_promoted_concepts": h11_metrics.get("motifs_with_promoted_concept_count"),
            "h11_emergent_motif_transfer_links": h11_metrics.get("emergent_motif_transfer_link_count"),
            "h11_emergent_motifs_with_strong_transfer": h11_metrics.get("emergent_motifs_with_strong_transfer_count"),
            "h11_emergent_motifs_with_promoted_concepts": h11_metrics.get("emergent_motifs_with_promoted_concept_count"),
            "h11_non_emergent_motif_transfer_links": h11_metrics.get("non_emergent_motif_transfer_link_count"),
            "workers_requested": requested_workers,
            "workers_initial": initial_epoch_workers,
            "workers_max_epoch": max_epoch_workers,
            "worker_execution": worker_execution,
            "ram_snapshot_at_epoch_start": ram_snapshot_at_epoch_start,
            "initial_worker_ramp_delay_seconds": float(initial_worker_ramp_delay_seconds),
            "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
            "cleanup": cleanup_summary,
            "memory_continuity": continuity_report,
            "deltas": deltas,
            "next_action": f"continue {f'epoch_{epoch_number + 1:04d}'}",
        }
        _write_epoch_status(status_dir, status)
        print(_format_epoch_status(status))

        manifest["current_epoch"] = epoch_number
        manifest["updated_at"] = _now()
        manifest["last_global_step_end"] = global_step_end
        manifest["completed_epochs"] = int(manifest.get("completed_epochs", 0) or 0) + 1
        manifest["latest_status_path"] = str(status_dir / "epoch_status.json")
        manifest["consecutive_no_new_stable_contingencies"] = consecutive_no_new
        manifest["last_sampling_peak_workers"] = int(worker_execution.get("peak_workers", max_epoch_workers) or max_epoch_workers)
        manifest["total_levels_successfully_completed"] = int(manifest.get("total_levels_successfully_completed", 0) or 0) + int(
            suite_summary.get("levels_successfully_completed_per_epoch", 0) or 0
        )
        games_solved_by_epoch = dict(manifest.get("games_solved_by_epoch", {}) or {})
        games_solved_by_epoch[epoch_id] = list(suite_summary.get("solved_games", []) or [])
        manifest["games_solved_by_epoch"] = games_solved_by_epoch
        manifest["total_games_solved"] = len({game for games in games_solved_by_epoch.values() for game in games})
        manifest.setdefault("epochs", []).append(
            {
                "epoch_id": epoch_id,
                "global_step_start": global_step_start,
                "global_step_end": global_step_end,
                "epoch_dir": str(epoch_dir),
                "reports_dir": str(reports_dir),
                "cleanup_summary_path": str(cleanup_dir / "cleanup_summary.json"),
                "status_path": str(status_dir / "epoch_status.json"),
                "started_at": _now(),
                "finished_at": _now(),
                "status": "complete",
                "workers_requested": requested_workers,
                "workers_initial": initial_epoch_workers,
                "workers_max_epoch": max_epoch_workers,
                "worker_execution": worker_execution,
                "ram_snapshot_at_epoch_start": ram_snapshot_at_epoch_start,
                "initial_worker_ramp_delay_seconds": float(initial_worker_ramp_delay_seconds),
                "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
                "deltas": deltas,
            }
        )
        manifest["memory_paths"] = {
            "current_state": str(memory_paths.current_state),
            "graph": str(memory_paths.graph),
            "replay_queue": str(memory_paths.replay_queue),
            "memory_summary": str(memory_paths.summary_json),
        }
        latest_status = status

        stop_reason = _epoch_stop_reason(
            config=config,
            manifest=manifest,
            root=root,
            disk_snapshot=disk_after,
            consecutive_no_new=consecutive_no_new,
        )
        if stop_reason is not None:
            manifest["stopped"] = True
            manifest["stop_reason"] = stop_reason
            status["next_action"] = f"stopped: {stop_reason}"
            _write_epoch_status(status_dir, status)
            print(_format_epoch_status(status))
            break
        _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    return manifest


def _load_or_initialize_manifest(config: ContinuousResearchConfig, manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists() and bool(config.resume):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        memory_paths = manifest.get("memory_paths") or {}
        missing = [path for path in memory_paths.values() if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"resume requested but required memory files are missing: {missing}")
        return manifest
    manifest = {
        "experiment_name": config.experiment_name,
        "created_at": _now(),
        "updated_at": _now(),
        "current_epoch": 0,
        "max_epochs": int(config.max_epochs),
        "games": [item.strip() for item in config.games.split(",") if item.strip()] if config.games != "all" else ["all"],
        "samplers": [item.strip() for item in config.samplers.split(",") if item.strip()],
        "seeds": [int(item) for item in config.seeds.split(",") if item.strip()],
        "steps_per_epoch": int(config.steps_per_epoch),
        "horizon": int(config.horizon),
        "context_depth": int(config.context_depth),
        "fast_postprocessing": bool(config.fast_postprocessing),
        "workers": int(config.workers),
        "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
        "initial_worker_ramp_delay_seconds": float(config.initial_worker_ramp_delay_seconds),
        "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
        "output_dir": str(config.output_dir),
        "stop_if_disk_above_percent": float(config.stop_if_disk_above_percent),
        "stop_if_no_new_stable_contingencies_for": int(config.stop_if_no_new_stable_contingencies_for),
        "completed_epochs": 0,
        "stopped": False,
        "stop_reason": None,
        "memory_paths": {},
        "latest_status_path": None,
        "epochs": [],
        "last_global_step_end": 0,
        "consecutive_no_new_stable_contingencies": 0,
        "total_levels_successfully_completed": 0,
        "total_games_solved": 0,
        "games_solved_by_epoch": {},
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def _epoch_stop_reason(
    *,
    config: ContinuousResearchConfig,
    manifest: dict[str, Any],
    root: Path,
    disk_snapshot: dict[str, Any],
    consecutive_no_new: int,
) -> str | None:
    if int(manifest["current_epoch"]) >= int(config.max_epochs):
        return "reached max epochs"
    if float(disk_snapshot["disk_used_percent"]) >= float(config.stop_if_disk_above_percent):
        return "disk usage exceeded configured limit"
    if consecutive_no_new >= int(config.stop_if_no_new_stable_contingencies_for):
        return "no new stable contingencies for configured consecutive epochs"
    if (root / "STOP").exists():
        return "manual stop file exists"
    return None


def _compute_epoch_deltas(
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
    suite_summary: dict[str, Any],
    latest_status: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = latest_status or {}
    return {
        "stable_contingency_count_delta": int(memory_after.get("stable_contingency_count", 0) or 0) - int(memory_before.get("stable_contingency_count", 0) or 0),
        "transformation_family_count_delta": int(memory_after.get("transformation_family_count", 0) or 0) - int(memory_before.get("transformation_family_count", 0) or 0),
        "carrier_candidate_count_delta": int(memory_after.get("carrier_candidate_count", 0) or 0) - int(memory_before.get("carrier_candidate_count", 0) or 0),
        "graph_node_count_delta": int(memory_after.get("graph_node_count", 0) or 0) - int(memory_before.get("graph_node_count", 0) or 0),
        "graph_edge_count_delta": int(memory_after.get("graph_edge_count", 0) or 0) - int(memory_before.get("graph_edge_count", 0) or 0),
        "replay_lift_delta": _delta_value((suite_summary.get("H02 core metrics") or {}).get("prediction_violation_replay_lift"), previous.get("replay_lift")),
        "compression_ratio_delta": _delta_value((suite_summary.get("H03 core metrics") or {}).get("compression_ratio"), previous.get("compression_ratio")),
        "singleton_family_ratio_delta": _delta_value((suite_summary.get("H03 core metrics") or {}).get("singleton_family_ratio"), previous.get("singleton_ratio")),
        "memory_size_delta_bytes": int(suite_summary.get("memory_size_after_bytes", 0) or 0) - int(suite_summary.get("memory_size_before_bytes", 0) or 0),
    }


def _delta_value(current: Any, previous: Any) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def _write_epoch_status(status_dir: Path, status: dict[str, Any]) -> None:
    (status_dir / "epoch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (status_dir / "epoch_status.txt").write_text(_format_epoch_status(status), encoding="utf-8")


def _write_epoch_start(status_dir: Path, payload: dict[str, Any]) -> None:
    (status_dir / "epoch_start.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (status_dir / "epoch_start.txt").write_text(_format_epoch_start(payload), encoding="utf-8")


def _format_epoch_status(status: dict[str, Any]) -> str:
    cleanup = status.get("cleanup") or {}
    stable_count = status.get("stable_contingencies")
    stable_delta = ((status.get("deltas") or {}).get("stable_contingency_count_delta"))
    return (
        f"Epoch {status['epoch_id'].split('_')[-1]} complete\n"
        f"Global steps: {status['global_step_start']}-{status['global_step_end']}\n"
        f"Workers: requested={status.get('workers_requested')} initial={status.get('workers_initial')} max_epoch={status.get('workers_max_epoch')} peak={((status.get('worker_execution') or {}).get('peak_workers'))}\n"
        f"RAM at start: {float((status.get('ram_snapshot_at_epoch_start') or {}).get('ram_used_percent', 0.0) or 0.0):.2f}%\n"
        f"Games: {status.get('games')}\n"
        f"Interactions this epoch: {status.get('interactions_this_epoch')}\n"
        f"Levels successfully completed this epoch: {status.get('levels_successfully_completed_per_epoch')}\n"
        f"Games solved this epoch: {status.get('games_solved_per_epoch')}\n"
        f"Solved games: {', '.join(status.get('solved_games', []) or [])}\n"
        f"Disk before cleanup: {cleanup.get('disk_before_cleanup_bytes', 0) / (1024 ** 3):.3f} GB\n"
        f"Disk after cleanup: {cleanup.get('disk_after_cleanup_bytes', 0) / (1024 ** 3):.3f} GB\n"
        f"Disk used percent: {status.get('disk_used_percent', 0.0):.2f}\n\n"
        f"H01: {status.get('H01')}\n"
        f"stable contingencies: {stable_count} ({stable_delta:+d})\n"
        f"games with stable contingencies: {status.get('games_with_stable_contingencies')}\n\n"
        f"H02: {status.get('H02')}\n"
        f"H02A replay/attention: {status.get('H02A')}\n"
        f"H02B pre-carrier timing: {status.get('H02B')}\n"
        f"replay lift: {status.get('replay_lift')}\n"
        f"direct replay evidence: {status.get('direct_replay_evidence')}\n\n"
        f"timing note: {status.get('h02_timing_note')}\n\n"
        f"H03: {status.get('H03')}\n"
        f"compression ratio: {status.get('compression_ratio')}\n"
        f"singleton ratio: {status.get('singleton_ratio')}\n"
        f"cross-context families: {status.get('cross_context_families')}\n\n"
        f"H04: {status.get('H04')}\n"
        f"carrier candidates: {status.get('carrier_candidates')}\n"
        f"stable carriers: {status.get('stable_carriers')}\n\n"
        f"H05: {status.get('H05')}\n"
        f"role candidates: {status.get('role_candidates')}\n"
        f"emergent roles: {status.get('emergent_roles')}\n\n"
        f"H06: {status.get('H06')}\n"
        f"transfer attempts: {status.get('role_transfer_attempts')}\n"
        f"transfer success rate: {status.get('role_transfer_success_rate')}\n"
        f"role mismatch count: {status.get('h06_role_mismatch_count')}\n"
        f"mean best margin: {status.get('h06_mean_best_margin')}\n\n"
        f"H07: {status.get('H07')}\n"
        f"concept candidates: {status.get('concept_candidates')}\n"
        f"promoted concepts: {status.get('promoted_concepts')}\n"
        f"strong transfer successes: {status.get('h07_strong_transfer_successes')}\n\n"
        f"H08: {status.get('H08')}\n"
        f"world model components: {status.get('world_model_components')}\n"
        f"coherent components: {status.get('coherent_world_model_components')}\n"
        f"candidate-only components: {status.get('candidate_only_world_model_components')}\n\n"
        f"H09: {status.get('H09')}\n"
        f"future-option events: {status.get('future_option_events')}\n"
        f"future-option motifs: {status.get('future_option_motifs')}\n"
        f"emergent motifs: {status.get('emergent_future_option_motifs')}\n\n"
        f"H10: {status.get('H10')}\n"
        f"option-attention lift: {status.get('option_attention_lift')}\n"
        f"high-option-change attention rate: {status.get('high_option_change_attention_rate')}\n\n"
        f"replay attention count: {status.get('h10_replay_attention_count')}\n"
        f"contradiction attention count: {status.get('h10_contradiction_attention_count')}\n\n"
        f"H11: {status.get('H11')}\n"
        f"future-option transfer links: {status.get('future_option_transfer_links')}\n"
        f"motifs with strong transfer: {status.get('motifs_with_strong_transfer')}\n"
        f"motifs with promoted concepts: {status.get('motifs_with_promoted_concepts')}\n\n"
        f"emergent motif transfer links: {status.get('h11_emergent_motif_transfer_links')}\n"
        f"emergent motifs with strong transfer: {status.get('h11_emergent_motifs_with_strong_transfer')}\n"
        f"emergent motifs with promoted concepts: {status.get('h11_emergent_motifs_with_promoted_concepts')}\n"
        f"non-emergent motif transfer links: {status.get('h11_non_emergent_motif_transfer_links')}\n\n"
        f"Cleanup:\n"
        f"deleted raw files: {cleanup.get('raw_files_deleted_count')}\n"
        f"freed: {cleanup.get('disk_freed_bytes', 0) / (1024 ** 3):.3f} GB\n\n"
        f"Next action:\n"
        f"{status.get('next_action')}\n"
    )


def _format_epoch_start(payload: dict[str, Any]) -> str:
    ram = payload.get("ram_snapshot_at_epoch_start") or {}
    return (
        f"Epoch {payload['epoch_id'].split('_')[-1]} starting\n"
        f"Global steps: {payload['global_step_start']}-{payload['global_step_end']}\n"
        f"Workers: requested={payload.get('requested_workers')} initial={payload.get('initial_epoch_workers')} max_epoch={payload.get('max_epoch_workers')}\n"
        f"RAM used percent at start: {float(ram.get('ram_used_percent', 0.0) or 0.0):.2f}\n"
        f"RAM ramp threshold percent: {float(payload.get('ram_ramp_threshold_percent', 0.0) or 0.0):.2f}\n"
        f"Initial worker ramp delay seconds: {float(payload.get('initial_worker_ramp_delay_seconds', 0.0) or 0.0):.1f}\n"
        f"Per-worker ramp delay seconds: {float(payload.get('per_worker_ramp_delay_seconds', 0.0) or 0.0):.1f}\n"
    )


def _no_cleanup_summary(epoch_dir: Path, memory_dir: Path) -> dict[str, Any]:
    summary = {
        "epoch_id": epoch_dir.name,
        "disk_before_cleanup_bytes": _tree_size(epoch_dir),
        "disk_after_cleanup_bytes": _tree_size(epoch_dir),
        "disk_freed_bytes": 0,
        "raw_files_deleted_count": 0,
        "raw_bytes_deleted": 0,
        "temp_files_deleted_count": 0,
        "temp_bytes_deleted": 0,
        "memory_db_size_bytes": _tree_size(Path(memory_dir)),
        "graph_db_size_bytes": 0,
        "replay_queue_db_size_bytes": 0,
        "reports_size_bytes": _tree_size(epoch_dir / "reports"),
        "kept_files": [],
        "deleted_files_sample": [],
        "deletion_errors": [],
    }
    (epoch_dir / "cleanup" / "cleanup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_memory_continuity_report(
    *,
    reports_dir: Path,
    epoch_id: str,
    previous_memory_summary: dict[str, Any],
    restored_memory_summary: dict[str, Any],
    after_epoch_memory_summary: dict[str, Any],
    memory_loaded_from_previous_epoch: bool,
) -> dict[str, Any]:
    continuity_failures: list[str] = []
    continuity_valid = True
    if memory_loaded_from_previous_epoch:
        for key in ("stable_contingency_count", "transformation_family_count", "graph_node_count", "graph_edge_count"):
            if int(restored_memory_summary.get(key, 0) or 0) < int(previous_memory_summary.get(key, 0) or 0):
                continuity_failures.append(f"{key} decreased before epoch start")
        if int(previous_memory_summary.get("replay_queue_size", 0) or 0) > 0 and int(restored_memory_summary.get("replay_queue_size", 0) or 0) <= 0:
            continuity_failures.append("replay_queue_size was not restored")
        continuity_valid = not continuity_failures
    report = {
        "epoch_id": epoch_id,
        "memory_loaded_from_previous_epoch": memory_loaded_from_previous_epoch,
        "previous_memory_summary": previous_memory_summary,
        "restored_memory_summary": restored_memory_summary,
        "after_epoch_memory_summary": after_epoch_memory_summary,
        "continuity_valid": continuity_valid,
        "continuity_failures": continuity_failures,
    }
    (reports_dir / "epoch_memory_continuity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _system_ram_snapshot() -> dict[str, float | int]:
    meminfo: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                key, _sep, remainder = line.partition(":")
                value = remainder.strip().split()[0]
                meminfo[key] = int(value) * 1024
    except Exception:
        return {
            "ram_total_bytes": 0,
            "ram_available_bytes": 0,
            "ram_used_bytes": 0,
            "ram_used_percent": 0.0,
        }
    total = int(meminfo.get("MemTotal", 0))
    available = int(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)))
    used = max(0, total - available)
    used_percent = (float(used) / float(total) * 100.0) if total > 0 else 0.0
    return {
        "ram_total_bytes": total,
        "ram_available_bytes": available,
        "ram_used_bytes": used,
        "ram_used_percent": used_percent,
    }


def _load_sampling_worker_execution(raw_dir: Path) -> dict[str, Any]:
    report_path = raw_dir / "interaction_sampling_v05c_report.json"
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    worker_execution = payload.get("worker_execution")
    return worker_execution if isinstance(worker_execution, dict) else {}


def h02_dir_note(suite_summary: dict[str, Any]) -> str | None:
    return ((suite_summary.get("H02 core metrics") or {}).get("carrier_timing_note"))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
