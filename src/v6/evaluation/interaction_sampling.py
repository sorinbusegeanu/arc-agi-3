from __future__ import annotations

import csv
import inspect
import json
import os
import multiprocessing
import re
import sqlite3
import sys
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from v6.environment.arc_adapter import registered_game_ids
from v6.environment.arc_adapter import ArcGridEnvironment
from v6.evaluation.broad_game_validation import family_for_game, game_passes, parse_game_selector
from v6.evaluation.failure_diagnostics import compute_run_diagnostics
from v6.evaluation.id_free_prefuture_validation import ID_FREE_FEATURE_SETS, evaluate_id_free_config
from v6.evaluation.prefuture_role_prediction import PREFUTURE_CLASSIFIERS, PrefutureExample, load_prefuture_examples
from v6.game_sets import load_game_set_manifest, parquet_games_present
from v6.main import V6Config, V6System
from v6.memory.live_memory_queue import (
    LiveMemoryReadCache,
    LiveMemoryWriterConfig,
    make_live_memory_delta_store,
    make_live_memory_queue,
    live_memory_manager_count,
    shutdown_live_memory_managers,
    start_live_memory_writer,
    stop_live_memory_writer,
)
from v6.memory.worker_snapshot import (
    SnapshotMemoryQueryEngine,
    WorkerMemorySnapshot,
    build_worker_memory_snapshot_from_directory,
    load_worker_memory_snapshot_artifact,
    write_worker_memory_snapshot_artifact,
    WORKER_MEMORY_SNAPSHOT_SCHEMA_VERSION,
)
from v6.memory.direct_streaming_fold import (
    DirectStreamingFoldConfig,
    DirectStreamingFoldJob,
    DirectStreamingFoldWriter,
    direct_streaming_manifest_exists,
    load_direct_streamed_job_metrics,
    load_direct_streamed_temporal_milestones,
    load_direct_streamed_validation_payloads,
    record_direct_fold_postprocessing_timings,
)
from v6.sampling import make_sampler, sampler_registry
from v6.storage.migration import migrate_sqlite_to_parquet


FAILED_REPRESENTATIVES = ("tt01", "pb02", "fs02", "tp02", "gr01")
PASSING_REFERENCES = ("va02", "mo01")
DEFAULT_V05C_GAMES = FAILED_REPRESENTATIVES + PASSING_REFERENCES
FOUNDATION_GAMES = (
    "ez01", "ez02", "ez03", "ez04",
    "ul01", "tt01", "pb01", "fs01", "ic01", "va01",
)
TRANSFORMATION_GAMES = (
    "pb01", "pb02", "pb03",
    "sk01", "sk02", "sk03",
    "ci01", "op01", "rz01", "mb01", "tk01",
    "ic01", "ic02", "ic03",
    "fs01", "fs02", "fs03",
    "tp01", "tp02", "tp03",
    "ml01", "ml02", "ml03",
    "tb01", "tb02", "tb03",
    "cr01", "rn01", "wl01", "dr01", "dg01", "mx01",
)
CONTEXT_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "tp01", "tp02", "tp03",
    "ic01", "ic02", "ic03",
    "nw01", "nw02", "nw03",
    "rs01", "rs02", "rs03",
    "zq01", "zq02", "zq03",
    "ex01", "ex02", "ex03",
    "va01", "va02", "va03",
    "bd01", "hm01", "gl01", "tr01", "vp01", "cf01",
)
ROLE_TRANSFER_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "co01",
    "ex01", "ex02", "ex03",
    "tb01", "tb02", "tb03",
    "cr01", "rn01", "wl01",
    "fi01", "fw01", "hd01", "bp01", "dd01", "as01",
)
FUTURE_ENABLE_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "co01",
    "tb01", "tb02", "tb03",
    "cr01", "rn01", "wl01",
    "fi01", "fw01", "mx01", "dr01", "dg01",
    "ex01", "ex02", "ex03",
    "bp01",
)
FUTURE_BLOCK_GAMES = (
    "pb01", "pb02", "pb03",
    "sk01", "sk02", "sk03",
    "ci01", "rz01", "op01", "mb01", "fb01",
    "va02", "bd01", "hm01", "vp01", "gl01", "tr01",
    "cf01", "in01", "wk01", "lf01", "rh01", "hz01",
    "zq01", "zq02", "zq03",
)
FUTURE_REVERSIBLE_GAMES = (
    "pb01", "fs01",
    "tp01", "tp02",
    "ic01", "ic02", "ic03",
    "nw01", "nw02", "nw03",
    "rs01", "rs02", "rs03",
    "ex01", "ex02", "ex03",
    "rc01", "bl01", "sw01", "dv01", "dp01", "wr01",
)
FUTURE_TERMINATE_GAMES = (
    "tt01", "tt02", "tt03",
    "wm01", "wm02", "wm03",
    "sv01", "sv02", "sv03",
    "st01", "tg01", "hs01", "sc01", "vi01",
    "fw01", "fi01", "fb01", "av01", "sb01",
    "rh01", "lf01", "hz01",
    "zq01", "zq02", "zq03",
)
BRIDGE_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "co01",
    "tb01", "tb02", "tb03",
    "cr01", "rn01", "wl01",
    "fi01", "fw01",
    "pb01", "pb02", "pb03",
    "sk01", "sk02", "sk03",
    "ci01",
    "ic01", "ic02", "ic03",
    "tp01", "tp02", "tp03",
    "ex01", "ex02", "ex03",
    "bp01", "dd01", "as01",
)
TRANSFER_VALIDATION_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "co01",
    "pb01", "pb02", "pb03",
    "sk01", "sk02", "sk03",
    "ci01",
    "tb01", "tb02", "tb03",
    "cr01", "rn01", "wl01",
    "ex01", "ex02", "ex03",
    "tp01", "tp02", "tp03",
    "ml01", "ml02", "ml03",
    "bp01", "dd01", "as01",
    "fi01", "fw01",
)
FALSIFICATION_GAMES = (
    "ul01", "ul02", "ul03",
    "fs01", "fs02", "fs03",
    "co01",
    "pb01", "pb03",
    "sk02", "sk03",
    "ci01",
    "tp02", "tp03",
    "tb02", "tb03",
    "cr01", "rn01",
    "ex02", "ex03",
    "ml03", "nw03", "rs03", "zq03",
    "dr01", "dg01", "mx01",
    "fb01", "vi01",
)
DIVERSE_GAMES = (
    "ez01", "ul01", "pb01", "fs01", "tp01",
    "ic01", "tb01", "ex01", "bp01", "fw01",
)
DEFAULT_V05C_SAMPLERS = (
    "random_baseline",
    "action_balance",
    "no_change_avoidance",
    "low_confidence",
    "novelty_delta",
    "mixed",
    "reset_aware_mixed",
)
V05C_GAME_PRESETS = {
    "failed_representatives": FAILED_REPRESENTATIVES,
    "passing_references": PASSING_REFERENCES,
    "broad": DEFAULT_V05C_GAMES,
    "foundation": FOUNDATION_GAMES,
    "transformation": TRANSFORMATION_GAMES,
    "context": CONTEXT_GAMES,
    "role_transfer": ROLE_TRANSFER_GAMES,
    "future_enable": FUTURE_ENABLE_GAMES,
    "future_block": FUTURE_BLOCK_GAMES,
    "future_reversible": FUTURE_REVERSIBLE_GAMES,
    "future_terminate": FUTURE_TERMINATE_GAMES,
    "bridge": BRIDGE_GAMES,
    "transfer_validation": TRANSFER_VALIDATION_GAMES,
    "falsification": FALSIFICATION_GAMES,
    "diverse": DIVERSE_GAMES,
}
DEFAULT_HIGH_REPLAY_PRIORITY_THRESHOLD = 0.70
DEFAULT_STABLE_TRANSFORMATION_FAMILY_SUPPORT = 5
ROOT_DIR = Path(__file__).resolve().parents[3]
GAMES_MD_CANDIDATES = (
    ROOT_DIR / "GAMES.md",
    ROOT_DIR / "other_repos" / "arc-interactive" / "GAMES.md",
)


@dataclass(frozen=True)
class InteractionSamplingConfig:
    games: tuple[str, ...] = DEFAULT_V05C_GAMES
    samplers: tuple[str, ...] = DEFAULT_V05C_SAMPLERS
    seeds: tuple[int, ...] = (0, 1, 2)
    train_seeds: tuple[int, ...] = (0, 1)
    test_seed: int = 2
    steps: int = 30000
    horizon: int = 10
    context_depth: int = 1
    adaptive_context_expansion: bool = False
    max_context_depth: int | None = None
    workers: int = 60
    max_tasks_per_child: int = 1
    commit_steps: int = 5000
    sqlite_synchronous: str = "normal"
    storage_backend: str = "sqlite"
    parquet_root: str = "runs/v6/storage_parquet"
    duckdb_path: str = "runs/v6/arc_agi3.duckdb"
    storage_batch_size: int = 50000
    compression: str = "zstd"
    output_dir: str = "runs/v6"
    env_root: str | None = None
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    only_missing_from_parquet_root: bool = False
    collect_only: bool = False
    memory_input_dir: str | None = None
    memory_output_dir: str | None = None
    global_step_offset: int = 0
    fast_postprocessing: bool = False
    initial_workers: int | None = None
    enable_worker_ramp: bool = False
    ram_ramp_threshold_percent: float = 85.0
    initial_worker_ramp_delay_seconds: float = 20.0
    per_worker_ramp_delay_seconds: float = 5.0
    shared_live_memory: str = "none"
    live_memory_refresh_steps: int = 250
    live_memory_queue_maxsize: int = 100_000
    live_memory_batch_size: int = 1000
    live_memory_flush_seconds: float = 2.0
    live_memory_delta_max_events: int = 100_000
    live_memory_delta_batch_limit: int = 5_000
    memory_snapshot_mode: str = "worker_local"
    memory_snapshot_max_bytes: int | None = None
    memory_snapshot_include_graph: bool = True
    memory_snapshot_include_substrate: bool = True
    memory_snapshot_max_ram_percent: float = 85.0
    direct_streaming_fold_enabled: bool = True
    direct_streaming_fold_workers: int = 8
    delete_raw_after_direct_streaming_fold: bool = True
    retain_raw_for_hypothesis_suite: bool = False
    direct_streaming_fold_retry_attempts: int = 5
    direct_streaming_fold_retry_initial_delay_seconds: float = 5.0
    direct_streaming_fold_busy_timeout_ms: int = 60000
    direct_streaming_fold_submit_delay_seconds: float = 0.0
    direct_streaming_shard_synchronous: str = "off"
    direct_streaming_checkpoint_every_merged_jobs: int = 25
    direct_streaming_merge_batch_size: int = 25
    delete_sidecars_after_fold: bool = True
    max_live_shard_bytes: int | None = None
    write_debug_sidecars: bool = False
    max_examples_per_contingency: int = 1
    max_examples_per_family: int = 1
    max_examples_per_carrier: int = 1
    max_examples_per_contradiction_cluster: int = 2
    fold_memory_substrate: bool = True
    fold_graph: bool = True
    max_graph_edges_per_fold: int = 1_000_000
    max_edges_per_source_node: int = 128
    max_edges_per_carrier: int = 32
    max_edges_per_family: int = 64
    enable_graph_edge_caps: bool = True
    use_set_based_merge: bool = True
    compact_finalize_mode: str = "full"
    full_finalize_every_epochs: int = 5
    memory_query_enabled: bool = False
    memory_action_selection_enabled: bool = False
    restore_compact_graph: bool = False
    restore_compact_substrate: bool = False


def resolve_game_ids(games_arg: str, env_root: str | None = None) -> list[str]:
    value = str(games_arg).strip()
    available = tuple(sorted(registered_game_ids(env_root)))
    if value == "all":
        if not available:
            raise ValueError("no registered game ids were found in the project environment registry")
        return [game_id for game_id in available if game_id != "gc01"]
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected:
        raise ValueError("games selection is empty; pass 'all' or a comma-separated list of valid game ids")
    if not available:
        return selected
    invalid = sorted({item for item in selected if item not in available})
    if invalid:
        raise ValueError(
            "invalid game id(s): "
            f"{', '.join(invalid)}. Valid game ids: {', '.join(available)}"
        )
    return selected


def resolve_game_preset(preset_name: str, env_root: str | None = None) -> tuple[str, ...]:
    games = tuple(dict.fromkeys(V05C_GAME_PRESETS[preset_name]))
    available = tuple(sorted(registered_game_ids(env_root)))
    if available:
        invalid = [game_id for game_id in games if game_id not in available]
        if invalid:
            raise ValueError(
                f"game preset '{preset_name}' contains invalid game ids: "
                f"{', '.join(invalid)}. Valid installed IDs: {', '.join(available)}"
            )
    return games


def parse_v05c_games(selector: str, env_root: str | None = None) -> tuple[str, ...]:
    value = selector.strip()
    if value in V05C_GAME_PRESETS:
        return resolve_game_preset(value, env_root)
    return tuple(resolve_game_ids(value, env_root))


def parse_v05c_samplers(selector: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in selector.split(",") if item.strip())
    registry = sampler_registry()
    unknown = [value for value in values if value not in registry]
    if unknown:
        raise ValueError(f"unknown samplers: {unknown}")
    return values


def run_interaction_sampling_v05c(config: InteractionSamplingConfig) -> list[dict]:
    config = resolve_interaction_sampling_scope(config)
    output = Path(config.output_dir)
    if bool(config.direct_streaming_fold_enabled) and not config.memory_output_dir:
        config = InteractionSamplingConfig(
            **{
                **config.__dict__,
                "memory_output_dir": str(output / "memory"),
            }
        )
    if str(config.shared_live_memory) != "none" and not config.memory_output_dir:
        config = InteractionSamplingConfig(
            **{
                **config.__dict__,
                "memory_output_dir": str(output / "memory"),
            }
        )
    sampling_root = output / ("sampling_v05c_sqlite_tmp" if config.storage_backend == "parquet" else "sampling_v05c")
    sampling_root.mkdir(parents=True, exist_ok=True)
    worker_execution = _generate_sampling_dbs(config, sampling_root)
    if config.collect_only:
        if config.storage_backend == "parquet":
            if not bool(config.direct_streaming_fold_enabled):
                _export_sampling_sqlite_to_parquet(config, sampling_root)
            shutil.rmtree(sampling_root, ignore_errors=True)
        return []
    direct_fold_started_at = worker_execution.pop("direct_fold_shutdown_started_at", None)
    direct_fold_timings = (
        dict(worker_execution.get("direct_fold_shutdown_timings", {}) or {})
        if isinstance(direct_fold_started_at, (int, float))
        else None
    )
    rows = _evaluate_sampling_runs(config, sampling_root, direct_fold_timings)
    diagnostics_started_at = time.perf_counter()
    comparison = sampler_comparison_rows(rows)
    best = best_by_game(rows)
    family_summary = summary_by_family(best)
    _add_direct_fold_timing(direct_fold_timings, "diagnostics", time.perf_counter() - diagnostics_started_at)
    milestones_started_at = time.perf_counter()
    temporal_milestones = _collect_temporal_milestones(config, sampling_root)
    _add_direct_fold_timing(direct_fold_timings, "milestones", time.perf_counter() - milestones_started_at)
    metrics_started_at = time.perf_counter()
    level_completion_records = _collect_level_completion_records(config, sampling_root)
    _add_direct_fold_timing(direct_fold_timings, "metrics", time.perf_counter() - metrics_started_at)
    epoch_completion = compute_epoch_completion_counters(level_completion_records)
    validation_started_at = time.perf_counter()
    validation = validation_summary(rows, comparison, best)
    _add_direct_fold_timing(direct_fold_timings, "validation", time.perf_counter() - validation_started_at)
    payload = {
        "runs": rows,
        "sampler_comparison": comparison,
        "best_by_game": best,
        "summary_by_family": family_summary,
        "validation": validation,
        "games": list(config.games),
        "samplers": list(config.samplers),
        "seeds": [int(seed) for seed in config.seeds],
        "steps": int(config.steps),
        "horizon": int(config.horizon),
        "context_depth": int(config.context_depth),
        "global_step_offset": int(config.global_step_offset),
        "global_step_start": int(config.global_step_offset) + 1,
        "global_step_end": int(config.global_step_offset) + int(config.steps),
        "memory_input_dir": config.memory_input_dir,
        "memory_output_dir": config.memory_output_dir,
        "memory_query_enabled": bool(config.memory_query_enabled),
        "memory_action_selection_enabled": bool(config.memory_action_selection_enabled),
        "restore_compact_graph": bool(config.restore_compact_graph),
        "restore_compact_substrate": bool(config.restore_compact_substrate),
        "worker_execution": worker_execution,
        "memory_metrics": worker_execution.get("memory_metrics", {}),
        "shared_live_memory": {
            "mode": str(config.shared_live_memory),
            "enabled": str(config.shared_live_memory) != "none",
            "writer_started": bool(worker_execution.get("live_memory_writer_started", False)),
            "writer_exitcode": worker_execution.get("live_memory_writer_exitcode"),
            "writer_forced_terminated": bool(worker_execution.get("live_memory_writer_forced_terminated", False)),
            "summary_path": worker_execution.get("live_memory_summary_path"),
            "live_memory_event_counts": worker_execution.get("live_memory_event_counts"),
        },
        "temporal_milestones": temporal_milestones,
        "level_completion_records": level_completion_records,
        **epoch_completion,
        "forbidden_features_used_during_sampling": False,
        "efficiency_diagnostics_enabled": not bool(config.fast_postprocessing),
        "efficiency_used_for_sampling": False,
        "efficiency_used_for_m2": False,
        "efficiency_used_for_m3": False,
        "efficiency_used_for_m4": False,
        "future_option_efficiency_posthoc_only": not bool(config.fast_postprocessing),
        "fast_postprocessing": bool(config.fast_postprocessing),
    }
    if direct_fold_timings is not None:
        worker_execution["direct_fold_postprocessing_timings"] = direct_fold_timings
        payload["direct_fold_postprocessing_timings"] = direct_fold_timings
    reports_started_at = time.perf_counter()
    write_interaction_sampling_reports(payload, output)
    _add_direct_fold_timing(direct_fold_timings, "reports", time.perf_counter() - reports_started_at)
    post_cleanup_started_at = time.perf_counter()
    if config.storage_backend == "parquet":
        if not bool(config.direct_streaming_fold_enabled):
            _export_sampling_sqlite_to_parquet(config, sampling_root)
        shutil.rmtree(sampling_root, ignore_errors=True)
        _add_direct_fold_timing(direct_fold_timings, "post_cleanup", time.perf_counter() - post_cleanup_started_at)
    if direct_fold_timings is not None:
        manifest_started_at = time.perf_counter()
        if config.memory_output_dir and direct_streaming_manifest_exists(config.memory_output_dir):
            record_direct_fold_postprocessing_timings(config.memory_output_dir, direct_fold_timings)
        _add_direct_fold_timing(direct_fold_timings, "reports", time.perf_counter() - manifest_started_at)
        direct_fold_timings["total"] = time.perf_counter() - float(direct_fold_started_at)
        worker_execution["direct_fold_postprocessing_timings"] = direct_fold_timings
        payload["direct_fold_postprocessing_timings"] = direct_fold_timings
        (output / "interaction_sampling_v05c_report.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(_format_direct_fold_cleanup_timing_line(direct_fold_timings), flush=True)
    return rows


def resolve_interaction_sampling_scope(config: InteractionSamplingConfig) -> InteractionSamplingConfig:
    selected_games = config.games
    if config.game_set_manifest or config.game_set_name:
        manifest = load_game_set_manifest(
            manifest_path=config.game_set_manifest,
            game_set_name=config.game_set_name,
            fallback_games=config.games,
        )
        if manifest.games:
            selected_games = manifest.games
    if config.only_missing_from_parquet_root:
        present = set(parquet_games_present(config.parquet_root))
        selected_games = tuple(game for game in selected_games if game not in present)
    available_seeds = tuple(int(seed) for seed in config.seeds)
    if available_seeds:
        requested_test_seed = int(config.test_seed)
        if requested_test_seed in available_seeds:
            resolved_test_seed = requested_test_seed
        else:
            resolved_test_seed = int(available_seeds[-1])
        resolved_train_seeds = tuple(int(seed) for seed in config.train_seeds if int(seed) in available_seeds and int(seed) != resolved_test_seed)
        if not resolved_train_seeds:
            resolved_train_seeds = (resolved_test_seed,)
    else:
        resolved_test_seed = int(config.test_seed)
        resolved_train_seeds = tuple(int(seed) for seed in config.train_seeds)
    return InteractionSamplingConfig(
        **{
            **config.__dict__,
            "games": tuple(dict.fromkeys(selected_games)),
            "train_seeds": resolved_train_seeds,
            "test_seed": resolved_test_seed,
        }
    )


def _generate_sampling_dbs(config: InteractionSamplingConfig, sampling_root: Path) -> dict[str, object]:
    aggregated_worker_execution = _default_worker_execution_stats(config)
    if config.collect_only and config.storage_backend == "parquet":
        for game in config.games:
            game_jobs = []
            order = 0
            for sampler_name in config.samplers:
                for seed in config.seeds:
                    db_path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                    if _sampling_db_ready(db_path):
                        continue
                    if db_path.exists():
                        db_path.unlink()
                    game_jobs.append(
                        {
                            "order": order,
                            "game": game,
                            "sampler_name": sampler_name,
                            "seed": int(seed),
                            "steps": int(config.steps),
                            "horizon": int(config.horizon),
                            "context_depth": int(config.context_depth),
                            "adaptive_context_expansion": bool(config.adaptive_context_expansion),
                            "max_context_depth": config.max_context_depth,
                            "commit_steps": int(config.commit_steps),
                            "sqlite_synchronous": str(config.sqlite_synchronous),
                            "memory_input_dir": config.memory_input_dir,
                            "memory_output_dir": config.memory_output_dir,
                            "global_step_offset": int(config.global_step_offset),
                            "fast_postprocessing": bool(config.fast_postprocessing),
                            "shared_live_memory": str(config.shared_live_memory),
                            "live_memory_refresh_steps": int(config.live_memory_refresh_steps),
                            "live_memory_queue_maxsize": int(config.live_memory_queue_maxsize),
                            "live_memory_batch_size": int(config.live_memory_batch_size),
                            "live_memory_flush_seconds": float(config.live_memory_flush_seconds),
                            "live_memory_delta_max_events": int(config.live_memory_delta_max_events),
                            "live_memory_delta_batch_limit": int(config.live_memory_delta_batch_limit),
                            "memory_snapshot_mode": str(config.memory_snapshot_mode),
                            "memory_snapshot_max_bytes": config.memory_snapshot_max_bytes,
                            "memory_snapshot_include_graph": bool(config.memory_snapshot_include_graph),
                            "memory_snapshot_include_substrate": bool(config.memory_snapshot_include_substrate),
                            "memory_snapshot_max_ram_percent": float(config.memory_snapshot_max_ram_percent),
                            "max_tasks_per_child": int(config.max_tasks_per_child),
                            "direct_streaming_fold_enabled": bool(config.direct_streaming_fold_enabled),
                            "direct_streaming_fold_workers": int(config.direct_streaming_fold_workers),
                            "delete_raw_after_direct_streaming_fold": bool(
                                config.delete_raw_after_direct_streaming_fold and not config.retain_raw_for_hypothesis_suite
                            ),
                            "retain_raw_for_hypothesis_suite": bool(config.retain_raw_for_hypothesis_suite),
                            "direct_streaming_fold_retry_attempts": int(config.direct_streaming_fold_retry_attempts),
                            "direct_streaming_fold_retry_initial_delay_seconds": float(config.direct_streaming_fold_retry_initial_delay_seconds),
                            "direct_streaming_fold_busy_timeout_ms": int(config.direct_streaming_fold_busy_timeout_ms),
                            "direct_streaming_fold_submit_delay_seconds": float(config.direct_streaming_fold_submit_delay_seconds),
                            "direct_streaming_shard_synchronous": str(config.direct_streaming_shard_synchronous),
                            "collect_only": bool(config.collect_only),
                            "storage_backend": str(config.storage_backend),
                            "parquet_root": str(config.parquet_root),
                            "storage_batch_size": int(config.storage_batch_size),
                            "compression": str(config.compression),
                            "write_debug_sidecars": bool(config.write_debug_sidecars),
                            "direct_streaming_checkpoint_every_merged_jobs": int(config.direct_streaming_checkpoint_every_merged_jobs),
                            "direct_streaming_merge_batch_size": int(config.direct_streaming_merge_batch_size),
                            "delete_sidecars_after_fold": bool(config.delete_sidecars_after_fold),
                            "max_live_shard_bytes": config.max_live_shard_bytes,
                            "max_examples_per_contingency": int(config.max_examples_per_contingency),
                            "max_examples_per_family": int(config.max_examples_per_family),
                            "max_examples_per_carrier": int(config.max_examples_per_carrier),
                            "max_examples_per_contradiction_cluster": int(config.max_examples_per_contradiction_cluster),
                            "fold_memory_substrate": bool(config.fold_memory_substrate),
                            "fold_graph": bool(config.fold_graph),
                            "max_graph_edges_per_fold": int(config.max_graph_edges_per_fold),
                            "max_edges_per_source_node": int(config.max_edges_per_source_node),
                            "max_edges_per_carrier": int(config.max_edges_per_carrier),
                            "max_edges_per_family": int(config.max_edges_per_family),
                            "enable_graph_edge_caps": bool(config.enable_graph_edge_caps),
                            "use_set_based_merge": bool(config.use_set_based_merge),
                            "compact_finalize_mode": str(config.compact_finalize_mode),
                            "full_finalize_every_epochs": int(config.full_finalize_every_epochs),
                            "memory_query_enabled": bool(config.memory_query_enabled),
                            "memory_action_selection_enabled": bool(config.memory_action_selection_enabled),
                            "restore_compact_graph": bool(config.restore_compact_graph),
                            "restore_compact_substrate": bool(config.restore_compact_substrate),
                            "db_path": str(db_path),
                            "env_root": config.env_root,
                        }
                    )
                    order += 1
            if game_jobs:
                stats = _invoke_run_sampling_jobs(
                    game_jobs,
                    workers=config.workers,
                    initial_workers=config.initial_workers,
                    enable_worker_ramp=bool(config.enable_worker_ramp),
                    ram_ramp_threshold_percent=float(config.ram_ramp_threshold_percent),
                    initial_worker_ramp_delay_seconds=float(config.initial_worker_ramp_delay_seconds),
                    per_worker_ramp_delay_seconds=float(config.per_worker_ramp_delay_seconds),
                )
                aggregated_worker_execution = _merge_worker_execution_stats(aggregated_worker_execution, stats)
            if not bool(config.direct_streaming_fold_enabled):
                _export_sampling_sqlite_to_parquet(config, sampling_root, games=(game,))
            shutil.rmtree(sampling_root / game, ignore_errors=True)
        return aggregated_worker_execution

    jobs = []
    order = 0
    for game in config.games:
        for sampler_name in config.samplers:
            for seed in config.seeds:
                db_path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                if _sampling_db_ready(db_path):
                    continue
                if db_path.exists():
                    db_path.unlink()
                jobs.append(
                    {
                        "order": order,
                        "game": game,
                        "sampler_name": sampler_name,
                        "seed": int(seed),
                        "steps": int(config.steps),
                        "horizon": int(config.horizon),
                        "context_depth": int(config.context_depth),
                        "adaptive_context_expansion": bool(config.adaptive_context_expansion),
                        "max_context_depth": config.max_context_depth,
                        "commit_steps": int(config.commit_steps),
                        "sqlite_synchronous": str(config.sqlite_synchronous),
                        "memory_input_dir": config.memory_input_dir,
                        "memory_output_dir": config.memory_output_dir,
                        "global_step_offset": int(config.global_step_offset),
                        "fast_postprocessing": bool(config.fast_postprocessing),
                        "shared_live_memory": str(config.shared_live_memory),
                        "live_memory_refresh_steps": int(config.live_memory_refresh_steps),
                        "live_memory_queue_maxsize": int(config.live_memory_queue_maxsize),
                        "live_memory_batch_size": int(config.live_memory_batch_size),
                        "live_memory_flush_seconds": float(config.live_memory_flush_seconds),
                        "live_memory_delta_max_events": int(config.live_memory_delta_max_events),
                        "live_memory_delta_batch_limit": int(config.live_memory_delta_batch_limit),
                        "memory_snapshot_mode": str(config.memory_snapshot_mode),
                        "memory_snapshot_max_bytes": config.memory_snapshot_max_bytes,
                        "memory_snapshot_include_graph": bool(config.memory_snapshot_include_graph),
                        "memory_snapshot_include_substrate": bool(config.memory_snapshot_include_substrate),
                        "memory_snapshot_max_ram_percent": float(config.memory_snapshot_max_ram_percent),
                        "max_tasks_per_child": int(config.max_tasks_per_child),
                        "direct_streaming_fold_enabled": bool(config.direct_streaming_fold_enabled),
                        "direct_streaming_fold_workers": int(config.direct_streaming_fold_workers),
                        "delete_raw_after_direct_streaming_fold": bool(
                            config.delete_raw_after_direct_streaming_fold and not config.retain_raw_for_hypothesis_suite
                        ),
                        "retain_raw_for_hypothesis_suite": bool(config.retain_raw_for_hypothesis_suite),
                        "direct_streaming_fold_retry_attempts": int(config.direct_streaming_fold_retry_attempts),
                        "direct_streaming_fold_retry_initial_delay_seconds": float(config.direct_streaming_fold_retry_initial_delay_seconds),
                        "direct_streaming_fold_busy_timeout_ms": int(config.direct_streaming_fold_busy_timeout_ms),
                        "direct_streaming_fold_submit_delay_seconds": float(config.direct_streaming_fold_submit_delay_seconds),
                        "direct_streaming_shard_synchronous": str(config.direct_streaming_shard_synchronous),
                        "collect_only": bool(config.collect_only),
                        "storage_backend": str(config.storage_backend),
                        "parquet_root": str(config.parquet_root),
                        "storage_batch_size": int(config.storage_batch_size),
                        "compression": str(config.compression),
                        "write_debug_sidecars": bool(config.write_debug_sidecars),
                        "direct_streaming_checkpoint_every_merged_jobs": int(config.direct_streaming_checkpoint_every_merged_jobs),
                        "direct_streaming_merge_batch_size": int(config.direct_streaming_merge_batch_size),
                        "delete_sidecars_after_fold": bool(config.delete_sidecars_after_fold),
                        "max_live_shard_bytes": config.max_live_shard_bytes,
                        "max_examples_per_contingency": int(config.max_examples_per_contingency),
                        "max_examples_per_family": int(config.max_examples_per_family),
                        "max_examples_per_carrier": int(config.max_examples_per_carrier),
                        "max_examples_per_contradiction_cluster": int(config.max_examples_per_contradiction_cluster),
                        "fold_memory_substrate": bool(config.fold_memory_substrate),
                        "fold_graph": bool(config.fold_graph),
                        "max_graph_edges_per_fold": int(config.max_graph_edges_per_fold),
                        "max_edges_per_source_node": int(config.max_edges_per_source_node),
                        "max_edges_per_carrier": int(config.max_edges_per_carrier),
                        "max_edges_per_family": int(config.max_edges_per_family),
                        "enable_graph_edge_caps": bool(config.enable_graph_edge_caps),
                        "use_set_based_merge": bool(config.use_set_based_merge),
                        "compact_finalize_mode": str(config.compact_finalize_mode),
                        "full_finalize_every_epochs": int(config.full_finalize_every_epochs),
                        "memory_query_enabled": bool(config.memory_query_enabled),
                        "memory_action_selection_enabled": bool(config.memory_action_selection_enabled),
                        "restore_compact_graph": bool(config.restore_compact_graph),
                        "restore_compact_substrate": bool(config.restore_compact_substrate),
                        "db_path": str(db_path),
                        "env_root": config.env_root,
                    }
                )
                order += 1
    if not jobs:
        return aggregated_worker_execution
    stats = _invoke_run_sampling_jobs(
        jobs,
        workers=config.workers,
        initial_workers=config.initial_workers,
        enable_worker_ramp=bool(config.enable_worker_ramp),
        ram_ramp_threshold_percent=float(config.ram_ramp_threshold_percent),
        initial_worker_ramp_delay_seconds=float(config.initial_worker_ramp_delay_seconds),
        per_worker_ramp_delay_seconds=float(config.per_worker_ramp_delay_seconds),
    )
    return _merge_worker_execution_stats(aggregated_worker_execution, stats)


def _invoke_run_sampling_jobs(
    jobs: list[dict],
    *,
    workers: int,
    initial_workers: int | None = None,
    enable_worker_ramp: bool = False,
    ram_ramp_threshold_percent: float = 85.0,
    initial_worker_ramp_delay_seconds: float = 20.0,
    per_worker_ramp_delay_seconds: float = 5.0,
) -> dict[str, object]:
    params = inspect.signature(_run_sampling_jobs).parameters
    kwargs: dict[str, object] = {"workers": workers}
    if "initial_workers" in params:
        kwargs["initial_workers"] = initial_workers
    if "enable_worker_ramp" in params:
        kwargs["enable_worker_ramp"] = enable_worker_ramp
    if "ram_ramp_threshold_percent" in params:
        kwargs["ram_ramp_threshold_percent"] = ram_ramp_threshold_percent
    if "initial_worker_ramp_delay_seconds" in params:
        kwargs["initial_worker_ramp_delay_seconds"] = initial_worker_ramp_delay_seconds
    if "per_worker_ramp_delay_seconds" in params:
        kwargs["per_worker_ramp_delay_seconds"] = per_worker_ramp_delay_seconds
    stats = _run_sampling_jobs(jobs, **kwargs)
    if isinstance(stats, dict):
        return stats
    return {
        "requested_workers": int(workers),
        "initial_workers": int(initial_workers if initial_workers is not None else workers),
        "peak_workers": 0,
        "worker_ramp_enabled": bool(enable_worker_ramp),
        "ram_ramp_threshold_percent": float(ram_ramp_threshold_percent),
        "initial_worker_ramp_delay_seconds": float(initial_worker_ramp_delay_seconds),
        "per_worker_ramp_delay_seconds": float(per_worker_ramp_delay_seconds),
        "ram_used_percent_at_start": 0.0,
        "ramp_event_count": 0,
        "ramp_events": [],
    }


def _direct_streaming_fold_job_id(job: dict[str, Any]) -> str:
    db_path = Path(str(job.get("db_path", "")))
    epoch_id = "epoch_unknown"
    parts = db_path.parts
    if "epochs" in parts:
        try:
            epoch_id = str(parts[parts.index("epochs") + 1])
        except (IndexError, ValueError):
            epoch_id = "epoch_unknown"
    global_step_start = int(job.get("global_step_offset", 0) or 0) + 1
    global_step_end = global_step_start + int(job.get("steps", 0) or 0) - 1
    return (
        f"{epoch_id}:{job['game']}:{job['sampler_name']}:seed{job['seed']}:steps{job['steps']}"
        f":g{global_step_start}-{global_step_end}"
    )


def _format_direct_fold_cleanup_timing_line(timings: dict[str, float]) -> str:
    """Return the single compact stdout line for direct-fold post-processing."""
    phases = (
        ("wait", "wait_futures"),
        ("flush", "live_writer_flush"),
        ("shutdown", "manager_shutdown"),
        ("merge", "merge_shards"),
        ("finalize", "finalize_memory"),
        ("checkpoint", "checkpoint"),
        ("cleanup", "cleanup"),
        ("metrics", "metrics"),
        ("milestones", "milestones"),
        ("validation", "validation"),
        ("diagnostics", "diagnostics"),
        ("reports", "reports"),
        ("post_cleanup", "post_cleanup"),
    )
    parts = ["DF"]
    for label, key in phases:
        if key in timings:
            parts.append(f"{label}={max(0.0, float(timings[key])):.1f}s")
    parts.append(f"total={max(0.0, float(timings.get('total', 0.0))):.1f}s")
    return " ".join(parts)


def _run_sampling_jobs(
    jobs: list[dict],
    *,
    workers: int,
    initial_workers: int | None = None,
    enable_worker_ramp: bool = False,
    ram_ramp_threshold_percent: float = 85.0,
    initial_worker_ramp_delay_seconds: float = 20.0,
    per_worker_ramp_delay_seconds: float = 5.0,
) -> dict[str, object]:
    workers = max(1, min(int(workers), len(jobs)))
    initial = workers if initial_workers is None else max(1, min(int(initial_workers), workers))
    shared_live_memory_mode = str(jobs[0].get("shared_live_memory", "none") or "none") if jobs else "none"
    direct_streaming_fold_enabled = bool(jobs and jobs[0].get("direct_streaming_fold_enabled", True) and any(job.get("memory_output_dir") for job in jobs))
    ram_at_start = _system_ram_snapshot()
    print(
        f"running {len(jobs)} v0.5c sampling jobs with workers={workers}"
        f" initial_workers={initial}"
        f" direct_streaming_fold={'on' if direct_streaming_fold_enabled else 'off'}"
        f" worker_ramp={'on' if enable_worker_ramp else 'off'}"
        f" initial_ramp_delay_s={float(initial_worker_ramp_delay_seconds):.1f}"
        f" per_worker_ramp_delay_s={float(per_worker_ramp_delay_seconds):.1f}"
        f" ram_used_percent={ram_at_start['ram_used_percent']:.2f}",
        file=sys.stderr,
        flush=True,
    )
    progress = tqdm(
        total=len(jobs),
        desc="epoch sampling",
        unit="job",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    )
    pending_jobs = iter(jobs)
    active_futures = {}
    pending_jobs_remaining = len(jobs)
    job_started_at: dict[Any, float] = {}
    target_workers = initial
    peak_workers = 0
    ramp_events: list[dict[str, float | int]] = []
    sampling_refill_count = 0
    max_done_batch_size = 0
    seconds_spent_in_fold_submit_delay = 0.0
    sampling_pool_underfilled_seconds = 0.0
    worker_ramp_blocked_by_ram = False
    memory_job_metrics: list[dict[str, object]] = []
    ramp_start_time = time.monotonic()
    last_ramp_time = ramp_start_time
    last_stalled_progress_log = ramp_start_time
    main_memory_dir = next((str(job["memory_output_dir"]) for job in jobs if job.get("memory_output_dir")), None)
    snapshot_job = next(
        (job for job in jobs if str(job.get("memory_snapshot_mode", "none")) == "worker_local" and job.get("memory_input_dir")),
        None,
    )
    snapshot_artifact: Path | None = None
    snapshot_serialized_bytes = 0
    snapshot_build_seconds = 0.0
    snapshot_estimated_worker_bytes = 0
    if snapshot_job is not None:
        metadata: dict[str, object] = {
            "epoch_id": "worker_snapshot",
            "compact_memory_generation": None,
            "maximum_global_step": None,
            "schema_version": WORKER_MEMORY_SNAPSHOT_SCHEMA_VERSION,
            "created_at": time.time(),
        }
        summary_path = Path(str(snapshot_job["memory_input_dir"])) / "memory_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metadata["compact_memory_generation"] = summary.get("generation", summary.get("compact_memory_generation"))
                metadata["maximum_global_step"] = summary.get("maximum_global_step", summary.get("last_global_step"))
                metadata["epoch_id"] = summary.get("epoch_id", "worker_snapshot")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        snapshot_build_started = time.perf_counter()
        parent_snapshot = build_worker_memory_snapshot_from_directory(
            str(snapshot_job["memory_input_dir"]),
            include_graph=bool(snapshot_job.get("restore_compact_graph", False) and snapshot_job.get("memory_snapshot_include_graph", True)),
            include_substrate=bool(snapshot_job.get("restore_compact_substrate", False) and snapshot_job.get("memory_snapshot_include_substrate", True)),
            version_metadata=metadata,
        )
        snapshot_artifact = Path(main_memory_dir or str(snapshot_job["memory_input_dir"])) / "worker_memory_snapshot_v2.pkl"
        _, snapshot_serialized_bytes = write_worker_memory_snapshot_artifact(parent_snapshot, snapshot_artifact)
        snapshot_build_seconds = time.perf_counter() - snapshot_build_started
        snapshot_estimated_worker_bytes = int(snapshot_serialized_bytes)
        estimated_total = int(snapshot_estimated_worker_bytes) * int(workers)
        max_snapshot_bytes = snapshot_job.get("memory_snapshot_max_bytes")
        if max_snapshot_bytes is not None and estimated_total > int(max_snapshot_bytes):
            raise MemoryError(
                f"estimated worker-local memory snapshot footprint {estimated_total} bytes exceeds "
                f"--memory-snapshot-max-bytes={int(max_snapshot_bytes)}"
            )
        ram = _system_ram_snapshot()
        ram_limit = float(snapshot_job.get("memory_snapshot_max_ram_percent", 85.0) or 85.0)
        available = int(ram.get("ram_available_bytes", 0) or 0)
        if available and estimated_total > int(available * ram_limit / 100.0):
            print(
                f"warning: estimated worker-local memory snapshot footprint {estimated_total} bytes "
                f"exceeds {ram_limit:.1f}% of available RAM ({available} bytes)",
                file=sys.stderr,
                flush=True,
            )
        snapshot_metrics = {
            "memory_snapshot_serialized_bytes": int(snapshot_serialized_bytes),
            "memory_snapshot_build_seconds": float(snapshot_build_seconds),
            "memory_snapshot_estimated_ram_bytes": {
            str(count): int(snapshot_estimated_worker_bytes) * count for count in (10, 20, 40, 60)
            },
        }
        for job in jobs:
            job.update(snapshot_metrics)
    live_memory_queue = None
    live_memory_manager_start = live_memory_manager_count()
    live_memory_delta_store = None
    live_memory_writer = None
    direct_fold_writer: DirectStreamingFoldWriter | None = None
    direct_fold_summary: dict[str, Any] = {
        "direct_streaming_fold_enabled": bool(direct_streaming_fold_enabled),
        "direct_streaming_fold_job_count": 0,
        "direct_streaming_fold_success_count": 0,
        "direct_streaming_fold_failed_count": 0,
        "direct_streaming_fold_deleted_raw_count": 0,
        "direct_streaming_fold_manifest_path": str(Path(str(main_memory_dir or "")) / "direct_streaming_fold_manifest.sqlite") if main_memory_dir else None,
        "direct_streaming_fold_legacy_temp_cleanup_count": 0,
    }
    live_memory_writer_stop: dict[str, Any] = {
        "writer_exitcode": None,
        "writer_forced_terminated": False,
        "summary_path": None,
        "live_memory_event_counts": None,
    }
    max_tasks_per_child = int(jobs[0].get("max_tasks_per_child", 0) or 0) if jobs else 0
    live_memory_summary_path = None
    if shared_live_memory_mode != "none" and main_memory_dir is not None:
        live_memory_queue = make_live_memory_queue(int(jobs[0].get("live_memory_queue_maxsize", 100_000) or 100_000))
        if shared_live_memory_mode == "readwrite" and snapshot_job is not None:
            live_memory_delta_store = make_live_memory_delta_store(
                int(jobs[0].get("live_memory_delta_max_events", 100_000) or 100_000)
            )
        live_memory_writer = start_live_memory_writer(
            LiveMemoryWriterConfig(
                memory_dir=str(main_memory_dir),
                queue_maxsize=int(jobs[0].get("live_memory_queue_maxsize", 100_000) or 100_000),
                batch_size=int(jobs[0].get("live_memory_batch_size", 1000) or 1000),
                flush_seconds=float(jobs[0].get("live_memory_flush_seconds", 2.0) or 2.0),
                delta_store=live_memory_delta_store,
            ),
            live_memory_queue,
        )
        live_memory_summary_path = str(Path(str(main_memory_dir)) / "live_memory_summary.json")
        for job in jobs:
            job["live_memory_queue"] = live_memory_queue
            job["live_memory_delta_store"] = live_memory_delta_store
    if direct_streaming_fold_enabled and main_memory_dir is not None:
        direct_fold_writer = DirectStreamingFoldWriter(
            DirectStreamingFoldConfig(
                memory_dir=str(main_memory_dir),
                delete_raw_after_fold=bool(jobs[0].get("delete_raw_after_direct_streaming_fold", True)),
                fold_workers=int(jobs[0].get("direct_streaming_fold_workers", 8) or 8),
                max_tasks_per_child=int(jobs[0].get("max_tasks_per_child", 0) or 0),
                retry_attempts=int(jobs[0].get("direct_streaming_fold_retry_attempts", 5) or 5),
                retry_initial_delay_seconds=float(jobs[0].get("direct_streaming_fold_retry_initial_delay_seconds", 5.0) or 5.0),
                busy_timeout_ms=int(jobs[0].get("direct_streaming_fold_busy_timeout_ms", 60000) or 60000),
                shard_synchronous=str(jobs[0].get("direct_streaming_shard_synchronous", "off") or "off"),
                checkpoint_every_merged_jobs=int(jobs[0].get("direct_streaming_checkpoint_every_merged_jobs", 25) or 25),
                merge_batch_size=int(jobs[0].get("direct_streaming_merge_batch_size", 25) or 25),
                delete_sidecars_after_fold=bool(jobs[0].get("delete_sidecars_after_fold", True)),
                max_live_shard_bytes=None if jobs[0].get("max_live_shard_bytes") in (None, "") else int(jobs[0].get("max_live_shard_bytes")),
                max_examples_per_contingency=int(jobs[0].get("max_examples_per_contingency", 1) or 1),
                max_examples_per_family=int(jobs[0].get("max_examples_per_family", 1) or 1),
                max_examples_per_carrier=int(jobs[0].get("max_examples_per_carrier", 1) or 1),
                max_examples_per_contradiction_cluster=int(jobs[0].get("max_examples_per_contradiction_cluster", 2) or 2),
                fold_memory_substrate=bool(jobs[0].get("fold_memory_substrate", True)),
                fold_graph=bool(jobs[0].get("fold_graph", True)),
                max_graph_edges_per_fold=int(jobs[0].get("max_graph_edges_per_fold", 1_000_000) or 1_000_000),
                max_edges_per_source_node=int(jobs[0].get("max_edges_per_source_node", 128) or 128),
                max_edges_per_carrier=int(jobs[0].get("max_edges_per_carrier", 32) or 32),
                max_edges_per_family=int(jobs[0].get("max_edges_per_family", 64) or 64),
                enable_graph_edge_caps=bool(jobs[0].get("enable_graph_edge_caps", True)),
                use_set_based_merge=bool(jobs[0].get("use_set_based_merge", True)),
                compact_finalize_mode=str(jobs[0].get("compact_finalize_mode", "full") or "full"),
            ),
            sampling_config=SimpleNamespace(
                steps=int(jobs[0].get("steps", 0) or 0),
                horizon=int(jobs[0].get("horizon", 0) or 0),
            ),
        )
        direct_fold_writer.start()

    def _maybe_ramp() -> bool:
        nonlocal target_workers, last_ramp_time, worker_ramp_blocked_by_ram
        if not enable_worker_ramp or target_workers >= workers:
            return False
        if worker_ramp_blocked_by_ram:
            return False
        now = time.monotonic()
        required_delay = (
            float(initial_worker_ramp_delay_seconds)
            if target_workers <= initial
            else float(per_worker_ramp_delay_seconds)
        )
        if (now - last_ramp_time) < required_delay:
            return False
        snapshot = _system_ram_snapshot()
        if float(snapshot["ram_used_percent"]) >= float(ram_ramp_threshold_percent):
            worker_ramp_blocked_by_ram = True
            return False
        target_workers += 1
        last_ramp_time = now
        ramp_events.append(
            {
                "target_workers": int(target_workers),
                "ram_used_percent": float(snapshot["ram_used_percent"]),
                "seconds_since_start": float(now - ramp_start_time),
            }
        )
        print(
            f"ramped sampling workers to {target_workers}/{workers}"
            f" at ram_used_percent={snapshot['ram_used_percent']:.2f}"
            f" elapsed_s={now - ramp_start_time:.1f}",
            file=sys.stderr,
            flush=True,
        )
        return True

    def _submit_until_target(executor: ProcessPoolExecutor) -> None:
        nonlocal peak_workers, pending_jobs_remaining, sampling_refill_count
        submitted_now = 0
        while len(active_futures) < target_workers:
            try:
                job = next(pending_jobs)
            except StopIteration:
                break
            future = executor.submit(_run_sampling_job, job)
            active_futures[future] = job
            job_started_at[future] = time.monotonic()
            pending_jobs_remaining = max(0, pending_jobs_remaining - 1)
            submitted_now += 1
        if submitted_now > 0:
            sampling_refill_count += 1
        peak_workers = max(peak_workers, len(active_futures))

    def _log_stalled_progress() -> None:
        nonlocal last_stalled_progress_log
        # Once all jobs are submitted, repeated liveness lines add noise while
        # long-running samplers make ordinary progress internally.
        if pending_jobs_remaining <= 0:
            return
        now = time.monotonic()
        if now - last_stalled_progress_log < 10.0:
            return
        last_stalled_progress_log = now
        try:
            writer_queue_depth = int(live_memory_queue.qsize()) if live_memory_queue is not None else 0
        except (AttributeError, NotImplementedError, OSError):
            writer_queue_depth = -1
        delta_size = 0
        if live_memory_delta_store is not None:
            try:
                delta_size = int(live_memory_delta_store.stats().get("size", 0) or 0)
            except Exception:
                delta_size = -1
        oldest_seconds = max((now - started for started in job_started_at.values()), default=0.0)
        print(
            "sampling workers alive; "
            f"active_workers={len(active_futures)} pending_jobs={pending_jobs_remaining} "
            f"writer_queue_depth={writer_queue_depth} delta_log_size={delta_size} "
            f"oldest_running_job_seconds={oldest_seconds:.1f} "
            f"current_target_workers={target_workers} peak_workers={peak_workers}",
            file=sys.stderr,
            flush=True,
        )

    try:
        executor_kwargs = {"max_workers": workers}
        if max_tasks_per_child > 0:
            executor_kwargs["max_tasks_per_child"] = max_tasks_per_child
        if snapshot_job is not None:
            executor_kwargs["initializer"] = _initialize_worker_memory_snapshot
            executor_kwargs["initargs"] = (
                str(snapshot_artifact),
                bool(snapshot_job.get("restore_compact_graph", False)),
                bool(snapshot_job.get("restore_compact_substrate", False)),
                bool(snapshot_job.get("memory_snapshot_include_graph", True)),
                bool(snapshot_job.get("memory_snapshot_include_substrate", True)),
            )
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            _submit_until_target(executor)
            while active_futures:
                done, _pending = wait(active_futures, timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    _log_stalled_progress()
                    if _maybe_ramp():
                        _submit_until_target(executor)
                    continue
                done_jobs: list[dict[str, Any]] = []
                done_count = len(done)
                if done_count > max_done_batch_size:
                    max_done_batch_size = done_count
                underfilled_started_at = 0.0
                for future in done:
                    _result = future.result()
                    job = active_futures.pop(future, None)
                    job_started_at.pop(future, None)
                    if isinstance(_result, dict) and isinstance(_result.get("memory_access_metrics"), dict):
                        memory_job_metrics.append(dict(_result["memory_access_metrics"]))
                    if job is not None:
                        done_jobs.append(job)
                    progress.update(1)
                _maybe_ramp()
                _submit_until_target(executor)
                if pending_jobs_remaining > 0 and len(active_futures) < target_workers and underfilled_started_at:
                    sampling_pool_underfilled_seconds += max(0.0, time.monotonic() - underfilled_started_at)
                if direct_fold_writer is not None:
                    for job in done_jobs:
                        if not job.get("memory_output_dir"):
                            continue
                        direct_fold_writer.submit(
                            DirectStreamingFoldJob(
                                job_id=_direct_streaming_fold_job_id(job),
                                db_path=str(job["db_path"]),
                                game=str(job["game"]),
                                sampler=str(job["sampler_name"]),
                                seed=int(job["seed"]),
                                steps=int(job["steps"]),
                                horizon=int(job["horizon"]),
                                context_depth=int(job.get("context_depth", 0) or 0),
                                global_step_start=int(job.get("global_step_offset", 0) or 0) + 1,
                                global_step_end=int(job.get("global_step_offset", 0) or 0) + int(job.get("steps", 0) or 0),
                                memory_dir=str(job["memory_output_dir"]),
                                delete_raw_after_fold=bool(job.get("delete_raw_after_direct_streaming_fold", True)),
                                parquet_export_enabled=bool(job.get("collect_only", False) and str(job.get("storage_backend", "sqlite")) == "parquet"),
                                parquet_root=None if not job.get("parquet_root") else str(job.get("parquet_root")),
                                storage_batch_size=int(job.get("storage_batch_size", 50000) or 50000),
                                compression=str(job.get("compression", "zstd") or "zstd"),
                            )
                        )
    finally:
        progress.close()
        direct_fold_cleanup_started_at = time.perf_counter()
        direct_fold_cleanup_timings: dict[str, float] = {}
        if live_memory_queue is not None and live_memory_writer is not None:
            live_writer_flush_started_at = time.perf_counter()
            live_memory_writer_stop = stop_live_memory_writer(live_memory_queue, live_memory_writer)
            direct_fold_cleanup_timings["live_writer_flush"] = time.perf_counter() - live_writer_flush_started_at
            live_memory_writer_stop["summary_path"] = live_memory_summary_path
            if live_memory_summary_path and Path(live_memory_summary_path).exists():
                try:
                    live_memory_writer_stop["live_memory_event_counts"] = json.loads(
                        Path(live_memory_summary_path).read_text(encoding="utf-8")
                    )
                except Exception:
                    live_memory_writer_stop["live_memory_event_counts"] = None
        manager_shutdown_needed = live_memory_manager_count() > live_memory_manager_start
        manager_shutdown_started_at = time.perf_counter()
        shutdown_live_memory_managers(live_memory_manager_start)
        if manager_shutdown_needed:
            direct_fold_cleanup_timings["manager_shutdown"] = time.perf_counter() - manager_shutdown_started_at
        if direct_fold_writer is not None:
            direct_fold_summary = direct_fold_writer.close()
            close_timings = direct_fold_summary.pop("direct_streaming_fold_close_timings", {})
            if isinstance(close_timings, dict):
                direct_fold_cleanup_timings.update(
                    {
                        str(key): float(value)
                        for key, value in close_timings.items()
                        if isinstance(value, (int, float))
                    }
                )
            if int(direct_fold_summary.get("direct_streaming_fold_failed_count", 0) or 0) > 0:
                failed_ids = list(direct_fold_summary.get("direct_streaming_fold_failed_job_ids", []))[:3]
                failed_errors = list(direct_fold_summary.get("direct_streaming_fold_failed_errors", []))[:3]
                failed_attempts = list(direct_fold_summary.get("direct_streaming_fold_failed_retry_attempt_counts", []))[:3]
                details = ", ".join(
                    f"{job_id} attempts={attempts} error={error}"
                    for job_id, attempts, error in zip(failed_ids, failed_attempts, failed_errors)
                )
                raise RuntimeError(
                    "direct streaming fold failed for "
                    f"{int(direct_fold_summary.get('direct_streaming_fold_failed_count', 0) or 0)} job(s); "
                    f"manifest={direct_fold_summary.get('direct_streaming_fold_manifest_path')}; {details}"
                )
    return {
        "requested_workers": int(workers),
        "initial_workers": int(initial),
        "fold_workers": int(direct_fold_summary.get("direct_streaming_fold_worker_count", 0) or 0),
            "peak_workers": int(peak_workers),
        "sampling_refill_count": int(sampling_refill_count),
        "max_done_batch_size": int(max_done_batch_size),
        "seconds_spent_in_fold_submit_delay": float(seconds_spent_in_fold_submit_delay),
        "sampling_pool_underfilled_seconds": float(sampling_pool_underfilled_seconds),
        "max_tasks_per_child": int(max_tasks_per_child),
        "worker_ramp_blocked_by_ram": bool(worker_ramp_blocked_by_ram),
        "worker_ramp_enabled": bool(enable_worker_ramp),
        "ram_ramp_threshold_percent": float(ram_ramp_threshold_percent),
        "initial_worker_ramp_delay_seconds": float(initial_worker_ramp_delay_seconds),
        "per_worker_ramp_delay_seconds": float(per_worker_ramp_delay_seconds),
        "ram_used_percent_at_start": float(ram_at_start["ram_used_percent"]),
        "ramp_event_count": int(len(ramp_events)),
        "ramp_events": ramp_events,
        "parallel_sidecar_fold_enabled": False,
        "parallel_sidecar_fold_shard_count": 0,
        "direct_streaming_fold_enabled": bool(direct_streaming_fold_enabled),
        "direct_fold_shutdown_timings": direct_fold_cleanup_timings,
        "direct_fold_shutdown_started_at": (
            float(direct_fold_cleanup_started_at) if direct_fold_writer is not None else None
        ),
        "direct_streaming_fold_writer_count": 1 if direct_streaming_fold_enabled else 0,
        "direct_streaming_fold_worker_count": int(direct_fold_summary.get("direct_streaming_fold_worker_count", 0) or 0),
        "direct_streaming_fold_shard_count": int(direct_fold_summary.get("direct_streaming_fold_shard_count", 0) or 0),
        "direct_streaming_fold_shard_root": direct_fold_summary.get("direct_streaming_fold_shard_root"),
        "direct_streaming_fold_job_count": int(direct_fold_summary.get("direct_streaming_fold_job_count", 0) or 0),
        "direct_streaming_fold_success_count": int(direct_fold_summary.get("direct_streaming_fold_success_count", 0) or 0),
        "direct_streaming_fold_failed_count": int(direct_fold_summary.get("direct_streaming_fold_failed_count", 0) or 0),
        "direct_streaming_fold_deleted_raw_count": int(direct_fold_summary.get("direct_streaming_fold_deleted_raw_count", 0) or 0),
        "direct_streaming_fold_manifest_path": direct_fold_summary.get("direct_streaming_fold_manifest_path"),
        "direct_streaming_fold_shards_deleted": bool(direct_fold_summary.get("direct_streaming_fold_shards_deleted", False)),
        "direct_streaming_fold_merge_started_at": direct_fold_summary.get("direct_streaming_fold_merge_started_at"),
        "direct_streaming_fold_merge_finished_at": direct_fold_summary.get("direct_streaming_fold_merge_finished_at"),
        "direct_streaming_fold_merge_seconds": direct_fold_summary.get("direct_streaming_fold_merge_seconds"),
        "direct_streaming_fold_jobs_submitted": int(direct_fold_summary.get("direct_streaming_fold_jobs_submitted", 0) or 0),
        "direct_streaming_fold_jobs_completed": int(direct_fold_summary.get("direct_streaming_fold_jobs_completed", 0) or 0),
        "direct_streaming_fold_jobs_failed": int(direct_fold_summary.get("direct_streaming_fold_jobs_failed", 0) or 0),
        "direct_streaming_fold_raw_deleted_after_shard_fold_count": int(direct_fold_summary.get("direct_streaming_fold_raw_deleted_after_shard_fold_count", 0) or 0),
        "direct_streaming_fold_finalized_main_memory": bool(direct_fold_summary.get("direct_streaming_fold_finalized_main_memory", False)),
        "direct_streaming_fold_total_raw_bytes": int(direct_fold_summary.get("direct_streaming_fold_total_raw_bytes", 0) or 0),
        "direct_streaming_fold_total_shard_bytes_added": int(direct_fold_summary.get("direct_streaming_fold_total_shard_bytes_added", 0) or 0),
        "direct_streaming_fold_mean_job_seconds": float(direct_fold_summary.get("direct_streaming_fold_mean_job_seconds", 0.0) or 0.0),
        "direct_streaming_fold_mean_write_mb_per_second": float(direct_fold_summary.get("direct_streaming_fold_mean_write_mb_per_second", 0.0) or 0.0),
        "direct_streaming_shard_synchronous": direct_fold_summary.get("direct_streaming_shard_synchronous"),
        "legacy_sidecar_fold_removed": True,
        "no_backpressure": True,
        "shared_live_memory_enabled": bool(shared_live_memory_mode != "none"),
        "shared_live_memory_mode": shared_live_memory_mode,
        "live_memory_writer_started": bool(live_memory_writer is not None),
        "live_memory_writer_exitcode": live_memory_writer_stop.get("writer_exitcode"),
        "live_memory_writer_forced_terminated": bool(live_memory_writer_stop.get("writer_forced_terminated", False)),
        "live_memory_queue_maxsize": int(jobs[0].get("live_memory_queue_maxsize", 100_000) or 100_000) if jobs else 0,
        "live_memory_batch_size": int(jobs[0].get("live_memory_batch_size", 1000) or 1000) if jobs else 0,
        "live_memory_flush_seconds": float(jobs[0].get("live_memory_flush_seconds", 2.0) or 2.0) if jobs else 0.0,
        "live_memory_delta_max_events": int(jobs[0].get("live_memory_delta_max_events", 100_000) or 100_000) if jobs else 0,
        "live_memory_delta_batch_limit": int(jobs[0].get("live_memory_delta_batch_limit", 5_000) or 5_000) if jobs else 0,
        "live_memory_summary_path": live_memory_summary_path,
        "live_memory_event_counts": live_memory_writer_stop.get("live_memory_event_counts"),
        "live_memory_events_submitted": int(sum(int(row.get("live_memory_events_submitted", 0) or 0) for row in memory_job_metrics)),
        "live_memory_events_written": int((live_memory_writer_stop.get("live_memory_event_counts") or {}).get("events_written", 0) or 0),
        "live_memory_batches_written": int((live_memory_writer_stop.get("live_memory_event_counts") or {}).get("batches_written", 0) or 0),
        "live_memory_queue_peak_size": int(max((int(row.get("live_memory_queue_peak_size", 0) or 0) for row in memory_job_metrics), default=0)),
        "live_memory_queue_block_seconds": float(sum(float(row.get("live_memory_queue_block_seconds", 0.0) or 0.0) for row in memory_job_metrics)),
        "live_memory_events_dropped": int(sum(int(row.get("live_memory_events_dropped", 0) or 0) for row in memory_job_metrics)),
        "memory_metrics": _aggregate_memory_job_metrics(memory_job_metrics),
    }


def _export_sampling_sqlite_to_parquet(config: InteractionSamplingConfig, sampling_root: Path, *, games: tuple[str, ...] | None = None) -> None:
    parquet_root = Path(config.parquet_root)
    selected_games = games or config.games
    for game in selected_games:
        for sampler_name in config.samplers:
            for seed in config.seeds:
                sqlite_path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                if not sqlite_path.exists():
                    continue
                migrate_sqlite_to_parquet(
                    sqlite_path=sqlite_path,
                    parquet_root=parquet_root,
                    game=game,
                    sampler=sampler_name,
                    seed=int(seed),
                    steps=int(config.steps),
                    batch_size=int(config.storage_batch_size),
                    compression=config.compression,
                    run_summary={
                        "horizon": int(config.horizon),
                        "context_depth": int(config.context_depth),
                        "context_length": int(config.context_depth),
                        "storage_backend": "parquet",
                    },
                )


_WORKER_MEMORY_SNAPSHOT: WorkerMemorySnapshot | None = None
_WORKER_MEMORY_QUERY_ENGINE: SnapshotMemoryQueryEngine | None = None
_WORKER_MEMORY_INIT_METRICS: dict[str, object] = {}
_WORKER_MEMORY_SNAPSHOT_INIT_COUNT = 0
_WORKER_MEMORY_RESTORE_REPORTED = False


class _SnapshotEnvironment:
    """Construction-only environment used by a worker snapshot initializer."""


def _initialize_worker_memory_snapshot(
    snapshot_artifact: str | None,
    restore_graph: bool,
    restore_substrate: bool,
    include_graph: bool,
    include_substrate: bool,
) -> None:
    global _WORKER_MEMORY_SNAPSHOT, _WORKER_MEMORY_QUERY_ENGINE, _WORKER_MEMORY_INIT_METRICS, _WORKER_MEMORY_SNAPSHOT_INIT_COUNT, _WORKER_MEMORY_RESTORE_REPORTED
    if not snapshot_artifact:
        return
    del restore_graph, restore_substrate, include_graph, include_substrate
    restore_started = time.perf_counter()
    rss_before = _process_rss_bytes()
    snapshot = load_worker_memory_snapshot_artifact(snapshot_artifact)
    rss_after = _process_rss_bytes()
    snapshot = WorkerMemorySnapshot(
        **{
            **snapshot.__dict__,
            "restore_seconds": time.perf_counter() - restore_started,
            "graph_restore_seconds": 0.0,
            "substrate_restore_seconds": 0.0,
        }
    )
    _WORKER_MEMORY_SNAPSHOT = snapshot
    _WORKER_MEMORY_QUERY_ENGINE = SnapshotMemoryQueryEngine(snapshot)
    _WORKER_MEMORY_SNAPSHOT_INIT_COUNT += 1
    _WORKER_MEMORY_RESTORE_REPORTED = False
    _WORKER_MEMORY_INIT_METRICS = {
            "memory_snapshot_version": dict(snapshot.version_metadata),
            "memory_snapshot_bytes": int(snapshot.snapshot_bytes),
            "memory_snapshot_restore_seconds": float(snapshot.restore_seconds),
            "memory_graph_restore_seconds": float(snapshot.graph_restore_seconds),
            "memory_substrate_restore_seconds": float(snapshot.substrate_restore_seconds),
            "memory_snapshot_serialized_bytes": int(Path(snapshot_artifact).stat().st_size),
            "memory_snapshot_rss_increase_bytes": max(0, int(rss_after - rss_before)),
            **snapshot.index_metrics(),
    }


def _process_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return int(pages * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def _worker_memory_snapshot_probe(delay_seconds: float = 0.0) -> tuple[int, int, int]:
    """Test-only observable proving one process-local snapshot is reused."""
    if delay_seconds > 0.0:
        time.sleep(float(delay_seconds))
    return os.getpid(), id(_WORKER_MEMORY_SNAPSHOT), int(_WORKER_MEMORY_SNAPSHOT_INIT_COUNT)


def _run_sampling_job(job: dict) -> dict:
    global _WORKER_MEMORY_RESTORE_REPORTED
    db_path = Path(str(job["db_path"]))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler_name = str(job["sampler_name"])
    seed = int(job["seed"])
    sampler = make_sampler(sampler_name, seed=seed)
    env = ArcGridEnvironment(game_id=str(job["game"]), seed=seed, env_root=job["env_root"])
    system_config = V6Config(
        database_path=str(db_path),
        memory_input_dir=job.get("memory_input_dir"),
        memory_output_dir=job.get("memory_output_dir"),
        restore_compact_memory=bool(job.get("memory_input_dir")) and _WORKER_MEMORY_SNAPSHOT is None,
        persist_compact_memory_on_close=False,
        restore_compact_graph=bool(job.get("restore_compact_graph", False)),
        restore_compact_substrate=bool(job.get("restore_compact_substrate", False)),
        global_step_offset=int(job.get("global_step_offset", 0) or 0),
        random_seed=seed,
        context_length=int(job.get("context_depth", 3)),
        max_context_depth=job.get("max_context_depth"),
        adaptive_context_expansion=bool(job.get("adaptive_context_expansion", False)),
        database_commit_every=int(job.get("commit_steps", 5000)),
        sqlite_synchronous=str(job.get("sqlite_synchronous", "normal") or "normal"),
        memory_promotion_every=max(1000, int(job.get("steps", 0) or 0) + 1),
        memory_query_enabled=bool(job.get("memory_query_enabled", False)),
        memory_action_selection_enabled=bool(job.get("memory_action_selection_enabled", False)),
        shared_live_memory_mode=str(job.get("shared_live_memory", "none") or "none"),
        live_memory_worker_id=f"{job['game']}:{sampler_name}:seed{seed}",
        live_memory_refresh_steps=int(job.get("live_memory_refresh_steps", 250) or 250),
    )
    use_worker_snapshot = str(job.get("memory_snapshot_mode", "worker_local")) == "worker_local" and _WORKER_MEMORY_SNAPSHOT is not None
    if use_worker_snapshot and _WORKER_MEMORY_QUERY_ENGINE is not None:
        _WORKER_MEMORY_QUERY_ENGINE.reset_metrics()
    live_memory_cache = None
    if str(job.get("shared_live_memory", "none") or "none") == "readwrite" and job.get("memory_output_dir"):
        live_memory_cache = LiveMemoryReadCache(
            memory_dir=str(job["memory_output_dir"]),
            refresh_steps=int(job.get("live_memory_refresh_steps", 250) or 250),
            delta_store=job.get("live_memory_delta_store") if use_worker_snapshot else None,
            delta_batch_limit=int(job.get("live_memory_delta_batch_limit", 5_000) or 5_000),
        )
    try:
        system_kwargs = {
            "env": env,
            "config": system_config,
            "action_sampler": sampler,
            "live_memory_queue": job.get("live_memory_queue"),
            "live_memory_cache": live_memory_cache,
        }
        if use_worker_snapshot:
            system_kwargs["memory_query_engine"] = _WORKER_MEMORY_QUERY_ENGINE
            system_kwargs["worker_memory_snapshot"] = _WORKER_MEMORY_SNAPSHOT
        system = V6System(**system_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "live_memory_queue" not in message and "live_memory_cache" not in message:
            raise
        system = V6System(
            env=env,
            config=system_config,
            action_sampler=sampler,
        )
    edge_counts: dict[str, int] = {}
    contradiction_summary: dict[str, object] = {}
    carrier_candidates: list[object] = []
    memory_summary: dict[str, object] = {}
    replay_candidates: list[dict] = []
    efficiency_summary: dict[str, object] = {}
    adaptive_context_summary: dict[str, object] = {}
    sampler_memory_guided_summary: dict[str, object] = {}
    memory_access_summary: dict[str, object] = {}
    fast_postprocessing = bool(job.get("fast_postprocessing", False))
    try:
        system.run(steps=int(job["steps"]))
        graph = getattr(system, "graph", None)
        if graph is not None and hasattr(graph, "edge_type_counts"):
            edge_counts = graph.edge_type_counts()
        if graph is not None and hasattr(graph, "export_compact_rows"):
            db_path.with_name("live_graph_compact.json").write_text(
                json.dumps(graph.export_compact_rows(), separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
        tracker = getattr(system, "context_contradictions", None)
        if tracker is not None and hasattr(tracker, "summary"):
            contradiction_summary = tracker.summary()
        carrier_tracker = getattr(system, "carrier_tracker", None)
        if carrier_tracker is not None and hasattr(carrier_tracker, "build_candidates"):
            carrier_candidates = carrier_tracker.build_candidates()
        memory_lifecycle = getattr(system, "memory_lifecycle", None)
        if memory_lifecycle is not None and hasattr(memory_lifecycle, "summary"):
            memory_summary = memory_lifecycle.summary()
            if hasattr(memory_lifecycle, "get_replay_batch"):
                replay_candidates = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in memory_lifecycle.get_replay_batch(limit=1000)
                ]
        efficiency_tracker = getattr(system, "efficiency_tracker", None)
        if efficiency_tracker is not None and hasattr(efficiency_tracker, "summary"):
            efficiency_summary = efficiency_tracker.summary()
        if hasattr(system, "adaptive_context_summary"):
            adaptive_context_summary = system.adaptive_context_summary()
        if hasattr(sampler, "memory_guided_summary"):
            sampler_memory_guided_summary = dict(sampler.memory_guided_summary())
        if hasattr(system, "memory_access_metrics"):
            memory_access_summary = dict(system.memory_access_metrics())
        if use_worker_snapshot or memory_access_summary:
            memory_access_summary.update(
                {
                    "memory_snapshot_build_seconds": float(job.get("memory_snapshot_build_seconds", 0.0) or 0.0),
                    "memory_snapshot_serialized_bytes": int(job.get("memory_snapshot_serialized_bytes", 0) or 0),
                    "memory_snapshot_estimated_ram_bytes": dict(job.get("memory_snapshot_estimated_ram_bytes", {}) or {}),
                }
            )
        memory_access_summary.update(_WORKER_MEMORY_INIT_METRICS)
        if use_worker_snapshot and _WORKER_MEMORY_RESTORE_REPORTED:
            for key in (
                "memory_restore_seconds",
                "memory_graph_restore_seconds",
                "memory_substrate_restore_seconds",
                "memory_snapshot_restore_seconds",
            ):
                memory_access_summary[key] = 0.0
        if use_worker_snapshot:
            _WORKER_MEMORY_RESTORE_REPORTED = True
    finally:
        system.close()
    _write_sampling_metadata(
        db_path,
        game=str(job["game"]),
        sampler_name=sampler_name,
        seed=seed,
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
        context_depth=int(job.get("context_depth", 3)),
        context_length=int(job.get("context_depth", 3)),
        global_step_offset=int(job.get("global_step_offset", 0) or 0),
        global_step_start=int(job.get("global_step_offset", 0) or 0) + 1,
        global_step_end=int(job.get("global_step_offset", 0) or 0) + int(job["steps"]),
        memory_input_dir=job.get("memory_input_dir"),
        memory_output_dir=job.get("memory_output_dir"),
        compact_memory_loaded=bool(job.get("memory_input_dir")),
        compact_memory_restore_summary=getattr(system, "compact_memory_restore_summary", {}),
        memory_query_enabled=bool(job.get("memory_query_enabled", False)),
        memory_action_selection_enabled=bool(job.get("memory_action_selection_enabled", False)),
        restore_compact_graph=bool(job.get("restore_compact_graph", False)),
        restore_compact_substrate=bool(job.get("restore_compact_substrate", False)),
        memory_snapshot_mode=str(job.get("memory_snapshot_mode", "none")),
        memory_snapshot_max_bytes=job.get("memory_snapshot_max_bytes"),
        memory_snapshot_include_graph=bool(job.get("memory_snapshot_include_graph", True)),
        memory_snapshot_include_substrate=bool(job.get("memory_snapshot_include_substrate", True)),
        fast_postprocessing_enabled=fast_postprocessing,
        future_effects_postprocessing_skipped=True,
        future_effects_legacy_removed=True,
        legacy_future_effects_removed=True,
        future_option_memory_source="memory_substrate",
        adaptive_context_expansion_enabled=bool(adaptive_context_summary.get("adaptive_context_expansion_enabled", False)),
        base_context_depth=int(adaptive_context_summary.get("base_context_depth", job.get("context_depth", 3)) or 0),
        max_context_depth=int(adaptive_context_summary.get("max_context_depth", job.get("max_context_depth", job.get("context_depth", 3))) or 0),
        adaptive_context_expansion_count=int(adaptive_context_summary.get("adaptive_context_expansion_count", 0) or 0),
        adaptive_context_active_action_count=int(adaptive_context_summary.get("adaptive_context_active_action_count", 0) or 0),
        adaptive_context_max_depth_reached=int(adaptive_context_summary.get("adaptive_context_max_depth_reached", job.get("context_depth", 3)) or 0),
        contingency_support_threshold=int(system_config.contingency_support_threshold),
        contingency_confidence_threshold=float(system_config.contingency_confidence_threshold),
        transformation_family_stable_support=int(system_config.min_cluster_size),
        reset_count=int(getattr(env, "reset_count", 0)) + int(getattr(sampler, "reset_count", 0)),
        terminal_count=int(getattr(env, "skipped_terminal_steps", 0)),
        reset_unavailable=bool(getattr(sampler, "reset_unavailable", False)),
        context_contradiction_count=int(contradiction_summary.get("context_contradiction_count", 0) or 0),
        prediction_result_count=int(contradiction_summary.get("prediction_result_count", 0) or 0),
        prediction_error_positive_count=int(contradiction_summary.get("prediction_error_positive_count", 0) or 0),
        predicted_family_available_count=int(contradiction_summary.get("predicted_family_available_count", 0) or 0),
        actual_family_available_count=int(contradiction_summary.get("actual_family_available_count", 0) or 0),
        wrong_prediction_count=int(contradiction_summary.get("wrong_prediction_count", 0) or 0),
        confident_wrong_prediction_count=int(contradiction_summary.get("confident_wrong_prediction_count", 0) or 0),
        contradiction_event_count=int(contradiction_summary.get("contradiction_event_count", 0) or 0),
        contradiction_suppressed_missing_prediction_count=int(contradiction_summary.get("contradiction_suppressed_missing_prediction_count", 0) or 0),
        contradiction_suppressed_missing_actual_count=int(contradiction_summary.get("contradiction_suppressed_missing_actual_count", 0) or 0),
        contradiction_suppressed_low_confidence_count=int(contradiction_summary.get("contradiction_suppressed_low_confidence_count", 0) or 0),
        contradiction_suppressed_correct_or_unknown_count=int(contradiction_summary.get("contradiction_suppressed_correct_or_unknown_count", 0) or 0),
        contradicted_context_count=int(contradiction_summary.get("contradicted_context_count", 0) or 0),
        contradicted_context_action_count=int(contradiction_summary.get("contradicted_context_action_count", 0) or 0),
        repeated_contradiction_count=int(contradiction_summary.get("repeated_contradiction_count", 0) or 0),
        carrier_candidate_count=len(carrier_candidates),
        emergent_carrier_count=sum(1 for item in carrier_candidates if getattr(item, "status", "") == "emergent_carrier"),
        carrier_spatial_candidate_count=sum(1 for item in carrier_candidates if getattr(item, "carrier_source", "") == "spatial"),
        carrier_object_candidate_count=sum(1 for item in carrier_candidates if getattr(item, "carrier_source", "") == "object"),
        carrier_cell_candidate_count=sum(1 for item in carrier_candidates if getattr(item, "carrier_source", "") == "cell"),
        carrier_context_action_fallback_candidate_count=sum(
            1 for item in carrier_candidates if getattr(item, "carrier_source", "") == "context_action_fallback"
        ),
        emergent_spatial_carrier_count=sum(
            1
            for item in carrier_candidates
            if getattr(item, "status", "") == "emergent_carrier" and getattr(item, "carrier_source", "") == "spatial"
        ),
        emergent_object_carrier_count=sum(
            1
            for item in carrier_candidates
            if getattr(item, "status", "") == "emergent_carrier" and getattr(item, "carrier_source", "") == "object"
        ),
        emergent_cell_carrier_count=sum(
            1
            for item in carrier_candidates
            if getattr(item, "status", "") == "emergent_carrier" and getattr(item, "carrier_source", "") == "cell"
        ),
        emergent_context_action_fallback_count=0,
        carrier_event_count=sum(int(getattr(item, "support_count", 0) or 0) for item in carrier_candidates),
        carrier_max_support=max((int(getattr(item, "support_count", 0) or 0) for item in carrier_candidates), default=0),
        carrier_mean_support=float(np.mean([float(getattr(item, "support_count", 0) or 0.0) for item in carrier_candidates])) if carrier_candidates else 0.0,
        mean_carrier_prediction_lift=float(np.mean([float(getattr(item, "prediction_lift", 0.0) or 0.0) for item in carrier_candidates])) if carrier_candidates else 0.0,
        max_carrier_prediction_lift=max((float(getattr(item, "prediction_lift", 0.0) or 0.0) for item in carrier_candidates), default=0.0),
        mean_carrier_compression_gain=float(np.mean([float(getattr(item, "compression_gain", 0.0) or 0.0) for item in carrier_candidates])) if carrier_candidates else 0.0,
        max_carrier_compression_gain=max((float(getattr(item, "compression_gain", 0.0) or 0.0) for item in carrier_candidates), default=0.0),
        memory_record_count=int(memory_summary.get("memory_record_count", 0) or 0),
        memory_active_count=int(memory_summary.get("memory_active_count", 0) or 0),
        memory_protected_count=int(memory_summary.get("memory_protected_count", 0) or 0),
        memory_compressed_count=int(memory_summary.get("memory_compressed_count", 0) or 0),
        memory_forgotten_count=int(memory_summary.get("memory_forgotten_count", 0) or 0),
        memory_replay_candidate_count=int(memory_summary.get("memory_replay_candidate_count", 0) or 0),
        memory_max_replay_priority=float(memory_summary.get("memory_max_replay_priority", 0.0) or 0.0),
        memory_mean_replay_priority=float(memory_summary.get("memory_mean_replay_priority", 0.0) or 0.0),
        efficiency_event_count=int(efficiency_summary.get("efficiency_event_count", 0) or 0),
        efficiency_total_action_cost=float(efficiency_summary.get("total_action_cost", 0.0) or 0.0),
        efficiency_mean_action_cost=float(efficiency_summary.get("mean_action_cost", 0.0) or 0.0),
        efficiency_no_effect_action_count=int(efficiency_summary.get("no_effect_action_count", 0) or 0),
        efficiency_repeated_state_count=int(efficiency_summary.get("repeated_state_count", 0) or 0),
        efficiency_repeated_context_action_count=int(efficiency_summary.get("repeated_context_action_count", 0) or 0),
        efficiency_terminal_outcome_count=int(efficiency_summary.get("terminal_outcome_count", 0) or 0),
        efficiency_distinct_outcome_count=int(efficiency_summary.get("distinct_outcome_count", 0) or 0),
        efficiency_mean_normalized_solve_efficiency=efficiency_summary.get("mean_normalized_solve_efficiency"),
        efficiency_max_normalized_solve_efficiency=efficiency_summary.get("max_normalized_solve_efficiency"),
        efficiency_mean_equivalent_outcome_cost_gap=efficiency_summary.get("mean_equivalent_outcome_cost_gap"),
        efficiency_max_equivalent_outcome_cost_gap=efficiency_summary.get("max_equivalent_outcome_cost_gap"),
        efficiency_mean_future_option_gain_per_cost=efficiency_summary.get("mean_future_option_gain_per_cost"),
        **{f"graph_edge_{edge_type}_count": int(count) for edge_type, count in edge_counts.items()},
        graph_edge_total_count=int(sum(edge_counts.values())),
        final_state=getattr(env, "last_state_name", None),
        final_levels_completed=int(getattr(env, "last_levels_completed", 0) or 0),
        final_win_levels=int(getattr(env, "last_win_levels", 0) or 0),
        final_full_game_id=getattr(env, "last_full_game_id", None),
        shared_live_memory_mode=str(job.get("shared_live_memory", "none") or "none"),
        live_memory_worker_id=str(system_config.live_memory_worker_id or ""),
        live_memory_events_emitted=int(getattr(system, "live_memory_events_emitted", 0) or 0),
        live_memory_events_dropped_queue_full=int(getattr(system, "live_memory_events_dropped_queue_full", 0) or 0),
        live_memory_events_dropped_error=int(getattr(system, "live_memory_events_dropped_error", 0) or 0),
        live_memory_refresh_failed_count=int(getattr(system, "live_memory_refresh_failed_count", 0) or 0),
        live_memory_stable_contingencies_imported=int(getattr(system, "live_memory_stable_contingencies_imported", 0) or 0),
        live_memory_replay_candidates_imported=int(getattr(system, "live_memory_replay_candidates_imported", 0) or 0),
        live_memory_carrier_candidates_imported=int(getattr(system, "live_memory_carrier_candidates_imported", 0) or 0),
        live_memory_family_updates_imported=int(getattr(system, "live_memory_family_updates_imported", 0) or 0),
        live_memory_contradiction_clusters_loaded=int(getattr(system, "live_memory_contradiction_clusters_loaded", 0) or 0),
        live_memory_future_option_events_loaded=int(getattr(system, "live_memory_future_option_events_loaded", 0) or 0),
        memory_guided_action_count=int(sampler_memory_guided_summary.get("memory_guided_action_count", 0) or 0),
        memory_guided_fallback_count=int(sampler_memory_guided_summary.get("memory_guided_fallback_count", 0) or 0),
        mean_memory_action_score=float(sampler_memory_guided_summary.get("mean_memory_action_score", 0.0) or 0.0),
        selected_action_memory_score_mean=float(
            sampler_memory_guided_summary.get("selected_action_memory_score_mean", 0.0) or 0.0
        ),
        selected_action_failure_risk_mean=float(
            sampler_memory_guided_summary.get("selected_action_failure_risk_mean", 0.0) or 0.0
        ),
        selected_action_future_option_gain_mean=float(
            sampler_memory_guided_summary.get("selected_action_future_option_gain_mean", 0.0) or 0.0
        ),
        **memory_access_summary,
    )
    if contradiction_summary:
        db_path.with_name("context_contradictions.json").write_text(
            json.dumps(contradiction_summary, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
    db_path.with_name("carrier_candidates.json").write_text(
        json.dumps([item.to_dict() if hasattr(item, "to_dict") else item for item in carrier_candidates], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    db_path.with_name("memory_lifecycle_summary.json").write_text(
        json.dumps(memory_summary, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    if bool(job.get("write_debug_sidecars", False)):
        db_path.with_name("memory_replay_candidates.json").write_text(
            json.dumps(replay_candidates, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        db_path.with_name("efficiency_summary.json").write_text(
            json.dumps(efficiency_summary, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
    if not memory_access_summary and not _WORKER_MEMORY_INIT_METRICS:
        return {"legacy_future_effects_removed": True}
    return {
        "legacy_future_effects_removed": True,
        "memory_access_metrics": memory_access_summary,
        "memory_snapshot_bytes": _WORKER_MEMORY_INIT_METRICS.get("memory_snapshot_bytes"),
    }


def _evaluate_sampling_runs(
    config: InteractionSamplingConfig,
    sampling_root: Path,
    direct_fold_timings: dict[str, float] | None = None,
) -> list[dict]:
    if config.memory_output_dir and direct_streaming_manifest_exists(config.memory_output_dir):
        metrics_started_at = time.perf_counter()
        metric_rows = load_direct_streamed_job_metrics(config.memory_output_dir)
        _add_direct_fold_timing(direct_fold_timings, "metrics", time.perf_counter() - metrics_started_at)
        validation_started_at = time.perf_counter()
        validation_payloads = load_direct_streamed_validation_payloads(config.memory_output_dir)
        _add_direct_fold_timing(direct_fold_timings, "validation", time.perf_counter() - validation_started_at)
        if not metric_rows:
            raise RuntimeError("Direct streaming fold manifest missing job metrics; raw DB fallback is disabled for normal continuous sampling.")
        if not validation_payloads:
            raise RuntimeError("Direct streaming validation payloads missing; raw DB fallback is disabled.")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        grouped_payloads: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in metric_rows:
            grouped[(str(row.get("game")), str(row.get("sampler_name")))].append(dict(row))
        for payload in validation_payloads:
            grouped_payloads[(str(payload.get("game")), str(payload.get("sampler_name")))].append(dict(payload))
        rows: list[dict] = []
        for game in config.games:
            for sampler_name in config.samplers:
                metrics_started_at = time.perf_counter()
                seed_rows = grouped.get((str(game), str(sampler_name)), [])
                if not seed_rows:
                    rows.append(_failed_row(game, sampler_name, config, "missing direct-streamed job metrics"))
                    _add_direct_fold_timing(direct_fold_timings, "metrics", time.perf_counter() - metrics_started_at)
                    continue
                aggregate = _aggregate_seed_rows(seed_rows, config)
                _add_direct_fold_timing(direct_fold_timings, "metrics", time.perf_counter() - metrics_started_at)
                validation_started_at = time.perf_counter()
                payload_rows = grouped_payloads.get((str(game), str(sampler_name)), [])
                eval_row = _best_validation_row_from_streamed_payloads(game, sampler_name, config, payload_rows)
                _add_direct_fold_timing(direct_fold_timings, "validation", time.perf_counter() - validation_started_at)
                rows.append(
                    {
                        **aggregate,
                        **eval_row,
                        "run_status": "ok",
                        "failure_reason": "",
                    }
                )
        return rows
    rows: list[dict] = []
    for game in config.games:
        for sampler_name in config.samplers:
            try:
                seed_rows = []
                for seed in config.seeds:
                    path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                    if not _sampling_db_ready(path):
                        raise FileNotFoundError(path)
                    seed_rows.append(_run_metrics(path, game, sampler_name, seed, config))
                aggregate = _aggregate_seed_rows(seed_rows, config)
                try:
                    eval_row = _best_validation_row(game, sampler_name, config, sampling_root)
                except Exception as exc:
                    failed = _failed_row(game, sampler_name, config, f"{type(exc).__name__}: {exc}")
                    rows.append({**failed, **aggregate, "run_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
                    continue
                rows.append({**aggregate, **eval_row, "run_status": "ok", "failure_reason": ""})
            except Exception as exc:
                rows.append(_failed_row(game, sampler_name, config, f"{type(exc).__name__}: {exc}"))
    return rows


def _add_direct_fold_timing(timings: dict[str, float] | None, phase: str, elapsed_seconds: float) -> None:
    if timings is not None:
        timings[phase] = float(timings.get(phase, 0.0) + max(0.0, elapsed_seconds))


def _collect_temporal_milestones(config: InteractionSamplingConfig, sampling_root: Path) -> dict:
    if config.memory_output_dir and direct_streaming_manifest_exists(config.memory_output_dir):
        rows = load_direct_streamed_temporal_milestones(config.memory_output_dir)
        if not rows:
            raise RuntimeError("Direct streaming fold manifest missing; raw DB fallback is disabled for normal continuous sampling.")
        return {"by_game_sampler_seed": rows}
    rows: list[dict[str, object]] = []
    for game in config.games:
        for sampler_name in config.samplers:
            for seed in config.seeds:
                path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                if not _sampling_db_ready(path):
                    continue
                rows.append(
                    _temporal_milestones_for_db(
                        path,
                        game=game,
                        sampler_name=sampler_name,
                        seed=int(seed),
                    )
                )
    return {"by_game_sampler_seed": rows}


def _temporal_milestones_for_db(path: Path, *, game: str, sampler_name: str, seed: int) -> dict[str, object]:
    offset = 0
    default = {
        "game": game,
        "sampler": sampler_name,
        "seed": int(seed),
        "first_interaction_step": None,
        "first_contingency_candidate_step": None,
        "first_stable_contingency_step": None,
        "first_prediction_violation_step": None,
        "first_high_replay_priority_step": None,
        "first_transformation_family_step": None,
        "first_stable_transformation_family_step": None,
        "first_carrier_candidate_step": None,
        "first_emergent_carrier_step": None,
    }
    try:
        with sqlite3.connect(path) as connection:
            offset = int(_json_metadata_value(connection, "global_step_offset", fallback=0))
            default["first_interaction_step"] = _first_value(
                connection,
                "SELECT MIN(COALESCE(global_step, id + ?)) FROM interactions",
                parameters=(offset,),
            )
            default["first_contingency_candidate_step"] = _first_value(
                connection,
                """
                SELECT MIN(COALESCE(global_step, interaction_id + ?))
                FROM prediction_results
                WHERE actual_family IS NOT NULL
                """,
                parameters=(offset,),
            )
            default["first_prediction_violation_step"] = _first_value(
                connection,
                """
                SELECT MIN(COALESCE(global_step, interaction_id + ?))
                FROM prediction_results
                WHERE COALESCE(prediction_error, 0) = 1
                   OR COALESCE(context_contradiction, 0) = 1
                   OR COALESCE(isf_prediction_error, 0.0) > 0.0
                """,
                parameters=(offset,),
            )
            default["first_high_replay_priority_step"] = _first_value(
                connection,
                f"""
                SELECT MIN(COALESCE(global_step, id + ?))
                FROM interactions
                WHERE COALESCE(memory_replay_priority, 0.0) >= {DEFAULT_HIGH_REPLAY_PRIORITY_THRESHOLD}
                """,
                parameters=(offset,),
            )
            default["first_transformation_family_step"] = _first_value(
                connection,
                """
                SELECT MIN(COALESCE(global_step, interaction_id + ?))
                FROM prediction_results
                WHERE actual_family IS NOT NULL
                """,
                parameters=(offset,),
            )
            default["first_stable_transformation_family_step"] = _stable_transformation_family_step(connection)
            default["first_stable_contingency_step"] = _stable_contingency_step(connection)
            default["first_carrier_candidate_step"] = _first_value(
                connection,
                """
                SELECT MIN(COALESCE(global_step, id + ?))
                FROM interactions
                WHERE COALESCE(carrier_event_recorded, 0) = 1
                   OR carrier_signature IS NOT NULL
                """,
                parameters=(offset,),
            )
    except sqlite3.DatabaseError:
        return default
    return default


def _stable_contingency_step(connection: sqlite3.Connection) -> int | None:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if not {"prediction_results", "sampling_metadata"}.issubset(tables):
        return None
    support_threshold = int(_json_metadata_value(connection, "contingency_support_threshold", fallback=20))
    confidence_threshold = float(_json_metadata_value(connection, "contingency_confidence_threshold", fallback=0.8))
    offset = int(_json_metadata_value(connection, "global_step_offset", fallback=0))
    try:
        rows = connection.execute(
            """
            SELECT COALESCE(global_step, interaction_id + ?), context_level, context_signature, action, actual_family
            FROM prediction_results
            WHERE actual_family IS NOT NULL
            ORDER BY interaction_id ASC
            """,
            (offset,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    by_key: Counter[tuple[int, str, int, int]] = Counter()
    totals: Counter[tuple[int, str, int]] = Counter()
    for interaction_id, context_level, context_signature, action, actual_family in rows:
        level = int(context_level or 0)
        key = (level, str(context_signature), int(action), int(actual_family))
        context_key = (level, str(context_signature), int(action))
        by_key[key] += 1
        totals[context_key] += 1
        support = by_key[key]
        confidence = support / max(1, totals[context_key])
        if support >= support_threshold and confidence >= confidence_threshold:
            return int(interaction_id)
    return None


def _stable_transformation_family_step(connection: sqlite3.Connection) -> int | None:
    support_threshold = int(_json_metadata_value(connection, "transformation_family_stable_support", fallback=DEFAULT_STABLE_TRANSFORMATION_FAMILY_SUPPORT))
    offset = int(_json_metadata_value(connection, "global_step_offset", fallback=0))
    try:
        rows = connection.execute(
            """
            SELECT COALESCE(global_step, interaction_id + ?), actual_family
            FROM prediction_results
            WHERE actual_family IS NOT NULL
            ORDER BY interaction_id ASC
            """,
            (offset,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    counts: Counter[int] = Counter()
    for interaction_id, actual_family in rows:
        family = int(actual_family)
        counts[family] += 1
        if counts[family] >= support_threshold:
            return int(interaction_id)
    return None


def _first_value(connection: sqlite3.Connection, query: str, *, parameters: tuple[object, ...] = ()) -> int | None:
    try:
        row = connection.execute(query, parameters).fetchone()
    except sqlite3.DatabaseError:
        return None
    value = None if row is None else row[0]
    return None if value is None else int(value)


def _json_metadata_value(connection: sqlite3.Connection, key: str, *, fallback: int | float) -> int | float:
    try:
        row = connection.execute("SELECT value FROM sampling_metadata WHERE key = ?", (key,)).fetchone()
    except sqlite3.DatabaseError:
        return fallback
    if row is None or row[0] is None:
        return fallback
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return fallback


def _run_metrics(path: Path, game: str, sampler_name: str, seed: int, config: InteractionSamplingConfig) -> dict:
    diagnostics = compute_run_diagnostics(path, game=game, seed=seed, steps=config.steps, horizon=config.horizon)
    diagnostics.update(_read_isf_metrics(path))
    diagnostics.update(_read_context_contradiction_metrics(path))
    diagnostics.update(_read_memory_lifecycle_metrics(path))
    diagnostics.update(_read_adaptive_context_metrics(path))
    metadata = _read_sampling_metadata(path)
    diagnostics.update(metadata)
    diagnostics.update(_read_efficiency_metrics(path))
    diagnostics["sampler_name"] = sampler_name
    return diagnostics


def _best_validation_row(game: str, sampler_name: str, config: InteractionSamplingConfig, sampling_root: Path) -> dict:
    validation_context_depth = (
        int(config.max_context_depth)
        if bool(config.adaptive_context_expansion) and config.max_context_depth is not None
        else int(config.context_depth)
    )
    train = []
    for seed in config.train_seeds:
        examples = load_prefuture_examples(sampling_db_path(sampling_root, game, sampler_name, config.steps, seed))
        train.extend([item for item in examples if int(item.features["context_level"]) <= validation_context_depth])
    test = load_prefuture_examples(sampling_db_path(sampling_root, game, sampler_name, config.steps, config.test_seed))
    test = [item for item in test if int(item.features["context_level"]) <= validation_context_depth]
    if not train or not test:
        raise ValueError("insufficient train/test stable contingencies")
    candidates = []
    for feature_set in ID_FREE_FEATURE_SETS:
        for classifier in PREFUTURE_CLASSIFIERS:
            row = evaluate_id_free_config(
                game=game,
                feature_set=feature_set,
                classifier=classifier,
                train_seeds=config.train_seeds,
                test_seed=config.test_seed,
                steps=config.steps,
                horizon=config.horizon,
                train_examples=train,
                test_examples=test,
            )
            if row is not None:
                candidates.append(row)
    if not candidates:
        raise ValueError("no validation candidates")
    best = max(candidates, key=lambda row: (game_passes({"run_status": "ok", **row}), row["id_free_macro_f1"], row["id_free_accuracy"], row["non_preserve_recall_any"]))
    return {
        "feature_set": best["feature_set"],
        "classifier": best["classifier"],
        "train_seeds": list(config.train_seeds),
        "test_seed": int(config.test_seed),
        "majority_baseline_accuracy": best["majority_baseline_accuracy"],
        "majority_baseline_macro_f1": best["majority_baseline_macro_f1"],
        "id_free_accuracy": best["id_free_accuracy"],
        "id_free_macro_f1": best["id_free_macro_f1"],
        "id_free_vs_majority_delta": best["id_free_vs_majority_delta"],
        "non_preserve_recall_any": best["non_preserve_recall_any"],
        "forbidden_future_feature_check_passed": best["forbidden_future_feature_check_passed"],
        "forbidden_id_feature_check_passed": best["forbidden_id_feature_check_passed"],
        "pass_status": game_passes({"run_status": "ok", **best}),
    }


def _prefuture_examples_from_payloads(payloads: list[dict]) -> list[PrefutureExample]:
    examples: list[PrefutureExample] = []
    for payload in payloads:
        for item in payload.get("examples", []) or []:
            contingency_key = item.get("contingency_key", [])
            if isinstance(contingency_key, list):
                normalized_key = tuple(
                    tuple(part) if isinstance(part, list) else part
                    for part in contingency_key
                )
            else:
                normalized_key = tuple(contingency_key)
            examples.append(
                PrefutureExample(
                    contingency_id=int(item.get("contingency_id", 0) or 0),
                    contingency_key=normalized_key,
                    features={str(key): float(value) for key, value in dict(item.get("features", {})).items()},
                    label=str(item.get("label", "PRESERVE")),
                )
            )
    return examples


def _best_validation_row_from_streamed_payloads(
    game: str,
    sampler_name: str,
    config: InteractionSamplingConfig,
    payloads: list[dict],
) -> dict:
    validation_context_depth = (
        int(config.max_context_depth)
        if bool(config.adaptive_context_expansion) and config.max_context_depth is not None
        else int(config.context_depth)
    )
    payloads_by_seed: dict[int, list[dict]] = defaultdict(list)
    for payload in payloads:
        payloads_by_seed[int(payload.get("seed", 0) or 0)].append(dict(payload))
    train = _prefuture_examples_from_payloads([payload for seed in config.train_seeds for payload in payloads_by_seed.get(int(seed), [])])
    train = [item for item in train if int(item.features["context_level"]) <= validation_context_depth]
    test = _prefuture_examples_from_payloads(payloads_by_seed.get(int(config.test_seed), []))
    test = [item for item in test if int(item.features["context_level"]) <= validation_context_depth]
    if not train or not test:
        raise RuntimeError("Direct streaming validation payloads missing; raw DB fallback is disabled.")
    candidates = []
    for feature_set in ID_FREE_FEATURE_SETS:
        for classifier in PREFUTURE_CLASSIFIERS:
            row = evaluate_id_free_config(
                game=game,
                feature_set=feature_set,
                classifier=classifier,
                train_seeds=config.train_seeds,
                test_seed=config.test_seed,
                steps=config.steps,
                horizon=config.horizon,
                train_examples=train,
                test_examples=test,
            )
            if row is not None:
                candidates.append(row)
    if not candidates:
        raise RuntimeError("Direct streaming validation payloads missing; raw DB fallback is disabled.")
    best = max(candidates, key=lambda row: (game_passes({"run_status": "ok", **row}), row["id_free_macro_f1"], row["id_free_accuracy"], row["non_preserve_recall_any"]))
    return {
        "feature_set": best["feature_set"],
        "classifier": best["classifier"],
        "train_seeds": list(config.train_seeds),
        "test_seed": int(config.test_seed),
        "majority_baseline_accuracy": best["majority_baseline_accuracy"],
        "majority_baseline_macro_f1": best["majority_baseline_macro_f1"],
        "id_free_accuracy": best["id_free_accuracy"],
        "id_free_macro_f1": best["id_free_macro_f1"],
        "id_free_vs_majority_delta": best["id_free_vs_majority_delta"],
        "non_preserve_recall_any": best["non_preserve_recall_any"],
        "forbidden_future_feature_check_passed": best["forbidden_future_feature_check_passed"],
        "forbidden_id_feature_check_passed": best["forbidden_id_feature_check_passed"],
        "pass_status": game_passes({"run_status": "ok", **best}),
    }


def _aggregate_seed_rows(seed_rows: list[dict], config: InteractionSamplingConfig) -> dict:
    first = seed_rows[0]
    sums = Counter()
    for row in seed_rows:
        for key in (
            "total_interactions",
            "usable_interactions",
            "reset_count",
            "terminal_count",
            "stable_contingency_count",
            "transformation_family_count",
            "high_isf_interaction_count",
            "graph_edge_follows_count",
            "graph_edge_generated_count",
            "graph_edge_member_of_count",
            "graph_edge_predicts_count",
            "graph_edge_enables_count",
            "graph_edge_blocks_count",
            "graph_edge_restricts_count",
            "graph_edge_expands_count",
            "graph_edge_terminates_count",
            "graph_edge_reversible_with_count",
            "graph_edge_explains_count",
            "graph_edge_similar_role_to_count",
            "graph_edge_depends_on_count",
            "graph_edge_contradicts_count",
            "graph_edge_total_count",
            "context_contradiction_count",
            "contradicted_context_count",
            "contradicted_context_action_count",
            "repeated_contradiction_count",
            "context_expansion_suggested_count",
            "adaptive_context_expansion_count",
            "adaptive_context_active_action_count",
            "adaptive_context_depth_used_count",
            "adaptive_context_expansion_applied_count",
            "carrier_candidate_count",
            "emergent_carrier_count",
            "carrier_spatial_candidate_count",
            "carrier_object_candidate_count",
            "carrier_cell_candidate_count",
            "carrier_context_action_fallback_candidate_count",
            "emergent_spatial_carrier_count",
            "emergent_object_carrier_count",
            "emergent_cell_carrier_count",
            "emergent_context_action_fallback_count",
            "carrier_event_count",
            "carrier_max_support",
            "memory_record_count",
            "memory_active_count",
            "memory_protected_count",
            "memory_compressed_count",
            "memory_forgotten_count",
            "memory_replay_candidate_count",
            "high_priority_replay_count",
            "efficiency_event_count",
            "efficiency_no_effect_action_count",
            "efficiency_repeated_state_count",
            "efficiency_repeated_context_action_count",
            "efficiency_terminal_outcome_count",
            "efficiency_distinct_outcome_count",
            "efficiency_future_option_gain_per_cost_count",
            "posthoc_future_option_delta_count",
        ):
            sums[key] += int(row.get(key, 0) or 0)
    aggregate = {
        "game": first["game"],
        "family": family_for_game(first["game"]),
        "sampler_name": first["sampler_name"],
        "steps": first["steps"],
        "horizon": first["horizon"],
        "context_depth": int(first.get("context_depth", config.context_depth)),
        "context_length": int(first.get("context_length", config.context_depth)),
        "total_interactions": sums["total_interactions"],
        "usable_interactions": sums["usable_interactions"],
        "reset_count": sums["reset_count"],
        "terminal_count": sums["terminal_count"],
        "no_change_ratio": float(np.mean([row.get("no_change_ratio", 0.0) for row in seed_rows])),
        "unique_transformation_families": sums["transformation_family_count"],
        "stable_contingency_count": sums["stable_contingency_count"],
        "prediction_accuracy": float(np.mean([row.get("prediction_accuracy", 0.0) for row in seed_rows])),
        "mean_isf_total": float(np.mean([row.get("mean_isf_total", 0.0) or 0.0 for row in seed_rows])),
        "max_isf_total": float(max((row.get("max_isf_total", 0.0) or 0.0) for row in seed_rows)),
        "mean_isf_survival_impact": float(np.mean([row.get("mean_isf_survival_impact", 0.0) or 0.0 for row in seed_rows])),
        "mean_isf_prediction_error": float(np.mean([row.get("mean_isf_prediction_error", 0.0) or 0.0 for row in seed_rows])),
        "mean_isf_learning_value": float(np.mean([row.get("mean_isf_learning_value", 0.0) or 0.0 for row in seed_rows])),
        "mean_isf_transfer_potential": float(np.mean([row.get("mean_isf_transfer_potential", 0.0) or 0.0 for row in seed_rows])),
        "mean_isf_explanatory_potential": float(np.mean([row.get("mean_isf_explanatory_potential", 0.0) or 0.0 for row in seed_rows])),
        "high_isf_interaction_count": sums["high_isf_interaction_count"],
        "graph_edge_follows_count": sums["graph_edge_follows_count"],
        "graph_edge_generated_count": sums["graph_edge_generated_count"],
        "graph_edge_member_of_count": sums["graph_edge_member_of_count"],
        "graph_edge_predicts_count": sums["graph_edge_predicts_count"],
        "graph_edge_enables_count": sums["graph_edge_enables_count"],
        "graph_edge_blocks_count": sums["graph_edge_blocks_count"],
        "graph_edge_restricts_count": sums["graph_edge_restricts_count"],
        "graph_edge_expands_count": sums["graph_edge_expands_count"],
        "graph_edge_terminates_count": sums["graph_edge_terminates_count"],
        "graph_edge_reversible_with_count": sums["graph_edge_reversible_with_count"],
        "graph_edge_explains_count": sums["graph_edge_explains_count"],
        "graph_edge_similar_role_to_count": sums["graph_edge_similar_role_to_count"],
        "graph_edge_depends_on_count": sums["graph_edge_depends_on_count"],
        "graph_edge_contradicts_count": sums["graph_edge_contradicts_count"],
        "graph_edge_total_count": sums["graph_edge_total_count"],
        "context_contradiction_count": sums["context_contradiction_count"],
        "contradicted_context_count": sums["contradicted_context_count"],
        "contradicted_context_action_count": sums["contradicted_context_action_count"],
        "repeated_contradiction_count": sums["repeated_contradiction_count"],
        "context_expansion_suggested_count": sums["context_expansion_suggested_count"],
        "adaptive_context_expansion_enabled": bool(first.get("adaptive_context_expansion_enabled", False)),
        "base_context_depth": int(first.get("base_context_depth", config.context_depth)),
        "max_context_depth": int(first.get("max_context_depth", config.max_context_depth or config.context_depth)),
        "memory_query_enabled": bool(first.get("memory_query_enabled", False)),
        "memory_action_selection_enabled": bool(first.get("memory_action_selection_enabled", False)),
        "restore_compact_graph": bool(first.get("restore_compact_graph", False)),
        "restore_compact_substrate": bool(first.get("restore_compact_substrate", False)),
        "adaptive_context_expansion_count": sums["adaptive_context_expansion_count"],
        "adaptive_context_active_action_count": sums["adaptive_context_active_action_count"],
        "adaptive_context_max_depth_reached": max(int(row.get("adaptive_context_max_depth_reached", 0) or 0) for row in seed_rows),
        "adaptive_context_depth_used_count": sums["adaptive_context_depth_used_count"],
        "adaptive_context_expansion_applied_count": sums["adaptive_context_expansion_applied_count"],
        "mean_suggested_context_depth": float(np.mean([row.get("mean_suggested_context_depth", 0.0) or 0.0 for row in seed_rows])),
        "max_suggested_context_depth": float(max((row.get("max_suggested_context_depth", 0.0) or 0.0 for row in seed_rows))),
        "memory_guided_action_count": int(sum(int(row.get("memory_guided_action_count", 0) or 0) for row in seed_rows)),
        "memory_guided_fallback_count": int(sum(int(row.get("memory_guided_fallback_count", 0) or 0) for row in seed_rows)),
        "mean_memory_action_score": _mean_or_none(row.get("mean_memory_action_score") for row in seed_rows),
        "selected_action_memory_score_mean": _mean_or_none(
            row.get("selected_action_memory_score_mean") for row in seed_rows
        ),
        "selected_action_failure_risk_mean": _mean_or_none(
            row.get("selected_action_failure_risk_mean") for row in seed_rows
        ),
        "selected_action_future_option_gain_mean": _mean_or_none(
            row.get("selected_action_future_option_gain_mean") for row in seed_rows
        ),
        "carrier_candidate_count": sums["carrier_candidate_count"],
        "emergent_carrier_count": sums["emergent_carrier_count"],
        "carrier_spatial_candidate_count": sums["carrier_spatial_candidate_count"],
        "carrier_object_candidate_count": sums["carrier_object_candidate_count"],
        "carrier_cell_candidate_count": sums["carrier_cell_candidate_count"],
        "carrier_context_action_fallback_candidate_count": sums["carrier_context_action_fallback_candidate_count"],
        "emergent_spatial_carrier_count": sums["emergent_spatial_carrier_count"],
        "emergent_object_carrier_count": sums["emergent_object_carrier_count"],
        "emergent_cell_carrier_count": sums["emergent_cell_carrier_count"],
        "emergent_context_action_fallback_count": sums["emergent_context_action_fallback_count"],
        "carrier_event_count": sums["carrier_event_count"],
        "carrier_max_support": sums["carrier_max_support"],
        "carrier_mean_support": float(np.mean([row.get("carrier_mean_support", 0.0) or 0.0 for row in seed_rows])),
        "mean_carrier_prediction_lift": float(np.mean([row.get("mean_carrier_prediction_lift", 0.0) or 0.0 for row in seed_rows])),
        "max_carrier_prediction_lift": float(max((row.get("max_carrier_prediction_lift", 0.0) or 0.0 for row in seed_rows))),
        "mean_carrier_compression_gain": float(np.mean([row.get("mean_carrier_compression_gain", 0.0) or 0.0 for row in seed_rows])),
        "max_carrier_compression_gain": float(max((row.get("max_carrier_compression_gain", 0.0) or 0.0 for row in seed_rows))),
        "memory_record_count": sums["memory_record_count"],
        "memory_active_count": sums["memory_active_count"],
        "memory_protected_count": sums["memory_protected_count"],
        "memory_compressed_count": sums["memory_compressed_count"],
        "memory_forgotten_count": sums["memory_forgotten_count"],
        "memory_replay_candidate_count": sums["memory_replay_candidate_count"],
        "memory_max_replay_priority": float(max((row.get("memory_max_replay_priority", 0.0) or 0.0 for row in seed_rows))),
        "memory_mean_replay_priority": _mean_or_none(row.get("memory_mean_replay_priority") for row in seed_rows),
        "mean_memory_replay_priority": _mean_or_none(row.get("memory_mean_replay_priority") for row in seed_rows),
        "high_priority_replay_count": sums["high_priority_replay_count"],
        "efficiency_event_count": sums["efficiency_event_count"],
        "efficiency_total_action_cost": float(sum(float(row.get("efficiency_total_action_cost", 0.0) or 0.0) for row in seed_rows)),
        "efficiency_mean_action_cost": _mean_or_none(row.get("efficiency_mean_action_cost") for row in seed_rows),
        "efficiency_no_effect_action_count": sums["efficiency_no_effect_action_count"],
        "efficiency_no_effect_action_ratio": sums["efficiency_no_effect_action_count"] / max(1, sums["efficiency_event_count"]),
        "efficiency_repeated_state_count": sums["efficiency_repeated_state_count"],
        "efficiency_repeated_state_ratio": sums["efficiency_repeated_state_count"] / max(1, sums["efficiency_event_count"]),
        "efficiency_repeated_context_action_count": sums["efficiency_repeated_context_action_count"],
        "efficiency_repeated_context_action_ratio": sums["efficiency_repeated_context_action_count"] / max(1, sums["efficiency_event_count"]),
        "efficiency_terminal_outcome_count": sums["efficiency_terminal_outcome_count"],
        "efficiency_distinct_outcome_count": sums["efficiency_distinct_outcome_count"],
        "efficiency_future_option_gain_per_cost_count": sums["efficiency_future_option_gain_per_cost_count"],
        "posthoc_future_option_delta_count": sums["posthoc_future_option_delta_count"],
        "efficiency_mean_normalized_solve_efficiency": _mean_or_none(
            row.get("efficiency_mean_normalized_solve_efficiency") for row in seed_rows
        ),
        "efficiency_max_normalized_solve_efficiency": _max_or_none(
            row.get("efficiency_max_normalized_solve_efficiency") for row in seed_rows
        ),
        "efficiency_mean_equivalent_outcome_cost_gap": _mean_or_none(
            row.get("efficiency_mean_equivalent_outcome_cost_gap") for row in seed_rows
        ),
        "efficiency_max_equivalent_outcome_cost_gap": _max_or_none(
            row.get("efficiency_max_equivalent_outcome_cost_gap") for row in seed_rows
        ),
        "efficiency_mean_future_option_gain_per_cost": _mean_or_none(
            row.get("efficiency_mean_future_option_gain_per_cost") for row in seed_rows
        ),
        "efficiency_max_future_option_gain_per_cost": _max_or_none(
            row.get("efficiency_max_future_option_gain_per_cost") for row in seed_rows
        ),
        "efficiency_min_future_option_gain_per_cost": _min_or_none(
            row.get("efficiency_min_future_option_gain_per_cost") for row in seed_rows
        ),
        "legacy_future_effects_removed": True,
        "future_effect_metrics_interpretable": False,
        "future_effect_count": None,
        "preserve_count": None,
        "expand_count": None,
        "restrict_count": None,
        "collapse_count": None,
        "non_preserve_count": None,
        "non_preserve_ratio": None,
        "levels_successfully_completed": int(
            max(
                max(
                    int(row.get("final_levels_completed", 0) or 0),
                    int(row.get("final_win_levels", 0) or 0),
                    1 if str(row.get("final_state") or "") == "WIN" else 0,
                )
                for row in seed_rows
            )
        ),
        "game_completed": any(_metadata_indicates_game_completed(row) for row in seed_rows),
    }
    metric_keys = (
        "memory_restore_seconds", "memory_graph_restore_seconds", "memory_substrate_restore_seconds",
        "memory_query_count", "memory_query_seconds", "memory_query_mean_seconds", "memory_action_rank_count",
        "memory_action_rank_seconds", "live_memory_refresh_count", "live_memory_refresh_rows",
        "live_memory_refresh_seconds", "sqlite_busy_retry_count", "sqlite_busy_wait_seconds",
        "memory_snapshot_version", "memory_cache_hit_count", "memory_cache_miss_count",
        "sqlite_queries_during_action_selection", "live_memory_events_submitted", "live_memory_events_dropped",
        "live_memory_queue_peak_size", "live_memory_queue_block_seconds",
    )
    for key in metric_keys:
        values = [row.get(key) for row in seed_rows if row.get(key) is not None]
        if not values:
            aggregate[key] = None
        elif key.endswith("_seconds") or key.endswith("_mean_seconds"):
            aggregate[key] = float(np.mean([float(value) for value in values]))
        elif key == "memory_snapshot_version":
            aggregate[key] = values[0]
        else:
            aggregate[key] = int(sum(int(value or 0) for value in values))
    return aggregate


def sampler_comparison_rows(rows: list[dict]) -> list[dict]:
    by_game = defaultdict(dict)
    for row in rows:
        if row.get("run_status") == "ok":
            by_game[row["game"]][row["sampler_name"]] = row
    output = []
    for game, items in sorted(by_game.items()):
        baseline = items.get("random_baseline")
        if not baseline:
            continue
        for sampler_name, row in sorted(items.items()):
            baseline_non_preserve_count = baseline.get("non_preserve_count")
            candidate_non_preserve_count = row.get("non_preserve_count")
            baseline_non_preserve_ratio = baseline.get("non_preserve_ratio")
            candidate_non_preserve_ratio = row.get("non_preserve_ratio")
            output.append(
                {
                    "game": game,
                    "family": family_for_game(game),
                    "sampler_name": sampler_name,
                    "delta_non_preserve_count": (
                        None
                        if candidate_non_preserve_count is None or baseline_non_preserve_count is None
                        else int(candidate_non_preserve_count) - int(baseline_non_preserve_count)
                    ),
                    "delta_non_preserve_ratio": (
                        None
                        if candidate_non_preserve_ratio is None or baseline_non_preserve_ratio is None
                        else float(candidate_non_preserve_ratio) - float(baseline_non_preserve_ratio)
                    ),
                    "delta_unique_transformation_families": int(row["unique_transformation_families"]) - int(baseline["unique_transformation_families"]),
                    "delta_stable_contingency_count": int(row["stable_contingency_count"]) - int(baseline["stable_contingency_count"]),
                    "delta_prediction_accuracy": float(row["prediction_accuracy"]) - float(baseline["prediction_accuracy"]),
                    "delta_id_free_accuracy": float(row["id_free_accuracy"]) - float(baseline["id_free_accuracy"]),
                    "delta_id_free_macro_f1": float(row["id_free_macro_f1"]) - float(baseline["id_free_macro_f1"]),
                    "delta_non_preserve_recall_any": float(row["non_preserve_recall_any"]) - float(baseline["non_preserve_recall_any"]),
                    "became_pass_from_fail": bool(row["pass_status"]) and not bool(baseline["pass_status"]),
                }
            )
    return output


def best_by_game(rows: list[dict]) -> list[dict]:
    output = []
    for game, items in sorted(_group_ok_by_game(rows).items()):
        output.append(
            max(
                items,
                key=lambda row: (
                    bool(row["pass_status"]),
                    float(row["id_free_macro_f1"]),
                    float(row["id_free_accuracy"]),
                    -float(row["no_change_ratio"]),
                ),
            )
        )
    return output


def summary_by_family(best_rows: list[dict]) -> list[dict]:
    by_family = defaultdict(list)
    for row in best_rows:
        by_family[family_for_game(row["game"])].append(row)
    return [
        {
            "family": family,
            "games_tested": len(rows),
            "games_passed": sum(1 for row in rows if row["pass_status"]),
            "mean_non_preserve_ratio": None,
            "mean_non_preserve_count": None,
            "mean_id_free_macro_f1": float(np.mean([row["id_free_macro_f1"] for row in rows])),
            "best_sampler": Counter(row["sampler_name"] for row in rows).most_common(1)[0][0],
        }
        for family, rows in sorted(by_family.items())
    ]


def validation_summary(rows: list[dict], comparison: list[dict], best_rows: list[dict]) -> dict:
    failed_runs = [row for row in rows if row.get("run_status") == "failed"]
    failure_reason_counts = dict(Counter(str(row.get("failure_reason") or "unknown") for row in failed_runs))
    if failed_runs:
        return {
            "diagnostic_success": False,
            "failed_run_count": len(failed_runs),
            "failure_reason_counts": failure_reason_counts,
            "scientific_conclusion": None,
            "mean_isf_total": None,
            "max_isf_total": None,
            "mean_isf_survival_impact": None,
            "mean_isf_prediction_error": None,
            "mean_isf_learning_value": None,
            "mean_isf_transfer_potential": None,
            "mean_isf_explanatory_potential": None,
            "high_isf_interaction_count": 0,
            "context_contradiction_count": 0,
            "contradicted_context_count": 0,
            "contradicted_context_action_count": 0,
            "repeated_contradiction_count": 0,
            "context_expansion_suggested_count": 0,
            "adaptive_context_expansion_enabled": False,
            "base_context_depth": None,
            "max_context_depth": None,
            "adaptive_context_expansion_count": 0,
            "adaptive_context_active_action_count": 0,
            "adaptive_context_max_depth_reached": None,
            "adaptive_context_depth_used_count": 0,
            "adaptive_context_expansion_applied_count": 0,
            "mean_suggested_context_depth": None,
            "max_suggested_context_depth": None,
            "carrier_candidate_count": 0,
            "emergent_carrier_count": 0,
            "carrier_spatial_candidate_count": 0,
            "carrier_object_candidate_count": 0,
            "carrier_cell_candidate_count": 0,
            "carrier_context_action_fallback_candidate_count": 0,
            "emergent_spatial_carrier_count": 0,
            "emergent_object_carrier_count": 0,
            "emergent_cell_carrier_count": 0,
            "emergent_context_action_fallback_count": 0,
            "carrier_event_count": 0,
            "carrier_max_support": 0,
            "carrier_mean_support": None,
            "mean_carrier_prediction_lift": None,
            "max_carrier_prediction_lift": None,
            "mean_carrier_compression_gain": None,
            "max_carrier_compression_gain": None,
            "memory_record_count": 0,
            "memory_active_count": 0,
            "memory_protected_count": 0,
            "memory_compressed_count": 0,
            "memory_forgotten_count": 0,
            "memory_replay_candidate_count": 0,
            "memory_max_replay_priority": None,
            "memory_mean_replay_priority": None,
            "mean_memory_replay_priority": None,
            "high_priority_replay_count": 0,
            "efficiency_event_count": 0,
            "efficiency_total_action_cost": 0.0,
            "efficiency_mean_action_cost": None,
            "efficiency_no_effect_action_count": 0,
            "efficiency_no_effect_action_ratio": 0.0,
            "efficiency_repeated_state_count": 0,
            "efficiency_repeated_state_ratio": 0.0,
            "efficiency_repeated_context_action_count": 0,
            "efficiency_repeated_context_action_ratio": 0.0,
            "efficiency_terminal_outcome_count": 0,
            "efficiency_distinct_outcome_count": 0,
            "efficiency_future_option_gain_per_cost_count": 0,
            "posthoc_future_option_delta_count": 0,
            "efficiency_mean_normalized_solve_efficiency": None,
            "efficiency_max_normalized_solve_efficiency": None,
            "efficiency_mean_equivalent_outcome_cost_gap": None,
            "efficiency_max_equivalent_outcome_cost_gap": None,
            "efficiency_mean_future_option_gain_per_cost": None,
            "efficiency_max_future_option_gain_per_cost": None,
            "efficiency_min_future_option_gain_per_cost": None,
        }
    failed_games = set(FAILED_REPRESENTATIVES)
    weak_games = set()
    strong_games = set()
    by_game_sampler = {(row["game"], row["sampler_name"]): row for row in rows if row.get("run_status") == "ok"}
    for row in comparison:
        game = row["game"]
        if game not in failed_games or row["sampler_name"] == "random_baseline":
            continue
        baseline = by_game_sampler.get((game, "random_baseline"))
        candidate = by_game_sampler.get((game, row["sampler_name"]))
        if not baseline or not candidate:
            continue
        if (
            baseline.get("non_preserve_count") is not None
            and baseline.get("non_preserve_ratio") is not None
            and candidate.get("non_preserve_count") is not None
            and candidate.get("non_preserve_ratio") is not None
        ):
            base_count = max(1, int(baseline["non_preserve_count"]))
            base_ratio = max(1e-9, float(baseline["non_preserve_ratio"]))
            if int(candidate["non_preserve_count"]) >= 2 * base_count and float(candidate["non_preserve_ratio"]) >= 2 * base_ratio:
                weak_games.add(game)
        if bool(candidate["pass_status"]) and not bool(baseline["pass_status"]):
            strong_games.add(game)
    ok_rows = [row for row in rows if row.get("run_status") == "ok"]
    return {
        "diagnostic_success": bool(rows) and any(row["sampler_name"] == "random_baseline" for row in rows) and bool(best_rows),
        "failed_run_count": 0,
        "failure_reason_counts": {},
        "sampling_repair_weak_pass": len(weak_games) >= 2,
        "sampling_repair_strong_pass": len(strong_games) >= 2,
        "sampling_repair_very_strong_pass": len(strong_games) >= 3,
        "weak_games": sorted(weak_games),
        "strong_games": sorted(strong_games),
        "forbidden_feature_checks_pass": all(
            bool(row.get("forbidden_future_feature_check_passed")) and bool(row.get("forbidden_id_feature_check_passed"))
            for row in rows
            if row.get("run_status") == "ok"
        ),
        "scientific_conclusion": "sampling_repair" if len(strong_games) >= 2 else "diagnostic_only",
        "legacy_future_effects_removed": True,
        "mean_isf_total": float(np.mean([row.get("mean_isf_total", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "max_isf_total": float(max((row.get("max_isf_total", 0.0) or 0.0) for row in ok_rows)) if ok_rows else None,
        "mean_isf_survival_impact": float(np.mean([row.get("mean_isf_survival_impact", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "mean_isf_prediction_error": float(np.mean([row.get("mean_isf_prediction_error", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "mean_isf_learning_value": float(np.mean([row.get("mean_isf_learning_value", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "mean_isf_transfer_potential": float(np.mean([row.get("mean_isf_transfer_potential", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "mean_isf_explanatory_potential": float(np.mean([row.get("mean_isf_explanatory_potential", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "high_isf_interaction_count": int(sum(int(row.get("high_isf_interaction_count", 0) or 0) for row in ok_rows)),
        "context_contradiction_count": int(sum(int(row.get("context_contradiction_count", 0) or 0) for row in ok_rows)),
        "contradicted_context_count": int(sum(int(row.get("contradicted_context_count", 0) or 0) for row in ok_rows)),
        "contradicted_context_action_count": int(sum(int(row.get("contradicted_context_action_count", 0) or 0) for row in ok_rows)),
        "repeated_contradiction_count": int(sum(int(row.get("repeated_contradiction_count", 0) or 0) for row in ok_rows)),
        "context_expansion_suggested_count": int(sum(int(row.get("context_expansion_suggested_count", 0) or 0) for row in ok_rows)),
        "adaptive_context_expansion_enabled": any(bool(row.get("adaptive_context_expansion_enabled")) for row in ok_rows),
        "base_context_depth": _max_or_none(row.get("base_context_depth") for row in ok_rows),
        "max_context_depth": _max_or_none(row.get("max_context_depth") for row in ok_rows),
        "adaptive_context_expansion_count": int(sum(int(row.get("adaptive_context_expansion_count", 0) or 0) for row in ok_rows)),
        "adaptive_context_active_action_count": int(sum(int(row.get("adaptive_context_active_action_count", 0) or 0) for row in ok_rows)),
        "adaptive_context_max_depth_reached": _max_or_none(row.get("adaptive_context_max_depth_reached") for row in ok_rows),
        "adaptive_context_depth_used_count": int(sum(int(row.get("adaptive_context_depth_used_count", 0) or 0) for row in ok_rows)),
        "adaptive_context_expansion_applied_count": int(
            sum(int(row.get("adaptive_context_expansion_applied_count", 0) or 0) for row in ok_rows)
        ),
        "mean_suggested_context_depth": float(np.mean([row.get("mean_suggested_context_depth", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "max_suggested_context_depth": float(max((row.get("max_suggested_context_depth", 0.0) or 0.0 for row in ok_rows))) if ok_rows else None,
        "carrier_candidate_count": int(sum(int(row.get("carrier_candidate_count", 0) or 0) for row in ok_rows)),
        "emergent_carrier_count": int(sum(int(row.get("emergent_carrier_count", 0) or 0) for row in ok_rows)),
        "carrier_spatial_candidate_count": int(sum(int(row.get("carrier_spatial_candidate_count", 0) or 0) for row in ok_rows)),
        "carrier_object_candidate_count": int(sum(int(row.get("carrier_object_candidate_count", 0) or 0) for row in ok_rows)),
        "carrier_cell_candidate_count": int(sum(int(row.get("carrier_cell_candidate_count", 0) or 0) for row in ok_rows)),
        "carrier_context_action_fallback_candidate_count": int(
            sum(int(row.get("carrier_context_action_fallback_candidate_count", 0) or 0) for row in ok_rows)
        ),
        "emergent_spatial_carrier_count": int(sum(int(row.get("emergent_spatial_carrier_count", 0) or 0) for row in ok_rows)),
        "emergent_object_carrier_count": int(sum(int(row.get("emergent_object_carrier_count", 0) or 0) for row in ok_rows)),
        "emergent_cell_carrier_count": int(sum(int(row.get("emergent_cell_carrier_count", 0) or 0) for row in ok_rows)),
        "emergent_context_action_fallback_count": int(
            sum(int(row.get("emergent_context_action_fallback_count", 0) or 0) for row in ok_rows)
        ),
        "carrier_event_count": int(sum(int(row.get("carrier_event_count", 0) or 0) for row in ok_rows)),
        "carrier_max_support": int(max((row.get("carrier_max_support", 0) or 0) for row in ok_rows)) if ok_rows else 0,
        "carrier_mean_support": float(np.mean([row.get("carrier_mean_support", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "mean_carrier_prediction_lift": float(np.mean([row.get("mean_carrier_prediction_lift", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "max_carrier_prediction_lift": float(max((row.get("max_carrier_prediction_lift", 0.0) or 0.0 for row in ok_rows))) if ok_rows else None,
        "mean_carrier_compression_gain": float(np.mean([row.get("mean_carrier_compression_gain", 0.0) or 0.0 for row in ok_rows])) if ok_rows else None,
        "max_carrier_compression_gain": float(max((row.get("max_carrier_compression_gain", 0.0) or 0.0 for row in ok_rows))) if ok_rows else None,
        "memory_record_count": int(sum(int(row.get("memory_record_count", 0) or 0) for row in ok_rows)),
        "memory_active_count": int(sum(int(row.get("memory_active_count", 0) or 0) for row in ok_rows)),
        "memory_protected_count": int(sum(int(row.get("memory_protected_count", 0) or 0) for row in ok_rows)),
        "memory_compressed_count": int(sum(int(row.get("memory_compressed_count", 0) or 0) for row in ok_rows)),
        "memory_forgotten_count": int(sum(int(row.get("memory_forgotten_count", 0) or 0) for row in ok_rows)),
        "memory_replay_candidate_count": int(sum(int(row.get("memory_replay_candidate_count", 0) or 0) for row in ok_rows)),
        "memory_max_replay_priority": _max_or_none(row.get("memory_max_replay_priority") for row in ok_rows),
        "memory_mean_replay_priority": _mean_or_none(row.get("memory_mean_replay_priority") for row in ok_rows),
        "mean_memory_replay_priority": _mean_or_none(row.get("memory_mean_replay_priority") for row in ok_rows),
        "high_priority_replay_count": int(sum(int(row.get("high_priority_replay_count", 0) or 0) for row in ok_rows)),
        "efficiency_event_count": int(sum(int(row.get("efficiency_event_count", 0) or 0) for row in ok_rows)),
        "efficiency_total_action_cost": float(sum(float(row.get("efficiency_total_action_cost", 0.0) or 0.0) for row in ok_rows)),
        "efficiency_mean_action_cost": _mean_or_none(row.get("efficiency_mean_action_cost") for row in ok_rows),
        "efficiency_no_effect_action_count": int(sum(int(row.get("efficiency_no_effect_action_count", 0) or 0) for row in ok_rows)),
        "efficiency_no_effect_action_ratio": (
            float(sum(int(row.get("efficiency_no_effect_action_count", 0) or 0) for row in ok_rows))
            / max(1, int(sum(int(row.get("efficiency_event_count", 0) or 0) for row in ok_rows)))
        ),
        "efficiency_repeated_state_count": int(sum(int(row.get("efficiency_repeated_state_count", 0) or 0) for row in ok_rows)),
        "efficiency_repeated_state_ratio": (
            float(sum(int(row.get("efficiency_repeated_state_count", 0) or 0) for row in ok_rows))
            / max(1, int(sum(int(row.get("efficiency_event_count", 0) or 0) for row in ok_rows)))
        ),
        "efficiency_repeated_context_action_count": int(sum(int(row.get("efficiency_repeated_context_action_count", 0) or 0) for row in ok_rows)),
        "efficiency_repeated_context_action_ratio": (
            float(sum(int(row.get("efficiency_repeated_context_action_count", 0) or 0) for row in ok_rows))
            / max(1, int(sum(int(row.get("efficiency_event_count", 0) or 0) for row in ok_rows)))
        ),
        "efficiency_terminal_outcome_count": int(sum(int(row.get("efficiency_terminal_outcome_count", 0) or 0) for row in ok_rows)),
        "efficiency_distinct_outcome_count": int(sum(int(row.get("efficiency_distinct_outcome_count", 0) or 0) for row in ok_rows)),
        "efficiency_future_option_gain_per_cost_count": int(
            sum(int(row.get("efficiency_future_option_gain_per_cost_count", 0) or 0) for row in ok_rows)
        ),
        "posthoc_future_option_delta_count": int(
            sum(int(row.get("posthoc_future_option_delta_count", 0) or 0) for row in ok_rows)
        ),
        "efficiency_mean_normalized_solve_efficiency": _mean_or_none(
            row.get("efficiency_mean_normalized_solve_efficiency") for row in ok_rows
        ),
        "efficiency_max_normalized_solve_efficiency": _max_or_none(
            row.get("efficiency_max_normalized_solve_efficiency") for row in ok_rows
        ),
        "efficiency_mean_equivalent_outcome_cost_gap": _mean_or_none(
            row.get("efficiency_mean_equivalent_outcome_cost_gap") for row in ok_rows
        ),
        "efficiency_max_equivalent_outcome_cost_gap": _max_or_none(
            row.get("efficiency_max_equivalent_outcome_cost_gap") for row in ok_rows
        ),
        "efficiency_mean_future_option_gain_per_cost": _mean_or_none(
            row.get("efficiency_mean_future_option_gain_per_cost") for row in ok_rows
        ),
        "efficiency_max_future_option_gain_per_cost": _max_or_none(
            row.get("efficiency_max_future_option_gain_per_cost") for row in ok_rows
        ),
        "efficiency_min_future_option_gain_per_cost": _min_or_none(
            row.get("efficiency_min_future_option_gain_per_cost") for row in ok_rows
        ),
    }


def write_interaction_sampling_reports(payload: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(payload["runs"], output / "interaction_sampling_v05c_report.csv")
    _write_csv(payload["sampler_comparison"], output / "interaction_sampling_v05c_sampler_comparison.csv")
    _write_csv(payload["best_by_game"], output / "interaction_sampling_v05c_best_by_game.csv")
    _write_csv(payload["summary_by_family"], output / "interaction_sampling_v05c_summary_by_family.csv")
    (output / "interaction_sampling_v05c_recommended_next_steps.txt").write_text(_recommended_next_steps(payload), encoding="utf-8")
    (output / "interaction_sampling_v05c_report.txt").write_text(_format_text(payload), encoding="utf-8")


def sampling_db_path(root: Path, game: str, sampler_name: str, steps: int, seed: int) -> Path:
    return root / game / sampler_name / f"steps_{int(steps)}" / f"seed_{int(seed)}.sqlite"


def _write_sampling_metadata(path: Path, **values) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sampling_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT OR REPLACE INTO sampling_metadata (key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in values.items()],
        )
        connection.commit()


def _read_sampling_metadata(path: Path) -> dict:
    with sqlite3.connect(path) as connection:
        try:
            rows = connection.execute("SELECT key, value FROM sampling_metadata").fetchall()
        except sqlite3.DatabaseError:
            return {}
    return {str(key): json.loads(value) for key, value in rows}


def expected_levels_by_game() -> dict[str, int]:
    for path in GAMES_MD_CANDIDATES:
        counts = _parse_expected_levels_from_games_md(path)
        if counts:
            return counts
    return {}


def _parse_expected_levels_from_games_md(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [item.strip() for item in line.split("|")[1:-1]]
        if len(parts) < 4:
            continue
        game_id = parts[0]
        if not re.fullmatch(r"[a-z]{2}\d{2}", game_id):
            continue
        try:
            counts[game_id] = int(parts[3])
        except ValueError:
            continue
    return counts


def compute_epoch_completion_counters(records: list[dict[str, object]]) -> dict[str, object]:
    expected_counts = expected_levels_by_game()
    unique_records: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        game_id = str(record.get("game_id") or "").strip()
        level_id = str(record.get("level_id") or record.get("level_name") or "").strip()
        if not game_id or not level_id:
            continue
        completed = record.get("completed")
        if completed is None:
            completed = record.get("success")
        if not bool(completed):
            continue
        unique_records.setdefault((game_id, level_id), dict(record))
    completed_levels_by_game: dict[str, int] = {}
    for game_id, _level_id in sorted(unique_records):
        completed_levels_by_game[game_id] = int(completed_levels_by_game.get(game_id, 0) + 1)
    solved_games = sorted(
        game_id
        for game_id, completed_count in completed_levels_by_game.items()
        if game_id in expected_counts and int(completed_count) >= int(expected_counts[game_id])
    )
    return {
        "levels_successfully_completed_per_epoch": len(unique_records),
        "games_solved_per_epoch": len(solved_games),
        "solved_games": solved_games,
        "completed_levels_by_game": completed_levels_by_game,
    }


def _metadata_indicates_game_completed(metadata: dict[str, object]) -> bool:
    final_state = str(metadata.get("final_state") or "")
    final_levels_completed = int(metadata.get("final_levels_completed", 0) or 0)
    final_win_levels = int(metadata.get("final_win_levels", 0) or 0)
    return final_state == "WIN" or (final_win_levels > 0 and final_levels_completed >= final_win_levels)


def _collect_level_completion_records(config: InteractionSamplingConfig, sampling_root: Path) -> list[dict[str, object]]:
    if config.memory_output_dir and direct_streaming_manifest_exists(config.memory_output_dir):
        records: list[dict[str, object]] = []
        for metadata in load_direct_streamed_job_metrics(config.memory_output_dir):
            game = str(metadata.get("game") or "")
            sampler_name = str(metadata.get("sampler_name") or "")
            seed = int(metadata.get("seed", 0) or 0)
            completed_count = max(
                int(metadata.get("final_levels_completed", 0) or 0),
                int(metadata.get("final_win_levels", 0) or 0),
                1 if str(metadata.get("final_state") or "") == "WIN" else 0,
            )
            final_state = str(metadata.get("final_state") or "")
            for level_index in range(1, max(0, completed_count) + 1):
                level_name = f"level_{level_index:04d}"
                records.append(
                    {
                        "game_id": game,
                        "level_id": level_name,
                        "level_name": level_name,
                        "completed": True,
                        "success": True,
                        "seed": seed,
                        "sampler": sampler_name,
                        "steps_used": None,
                        "final_state": final_state,
                        "source_run_db": None,
                    }
                )
        return records
    records: list[dict[str, object]] = []
    for game in config.games:
        for sampler_name in config.samplers:
            for seed in config.seeds:
                path = sampling_db_path(sampling_root, game, sampler_name, config.steps, seed)
                if not _sampling_db_ready(path):
                    continue
                metadata = _read_sampling_metadata(path)
                completed_count = max(
                    int(metadata.get("final_levels_completed", 0) or 0),
                    int(metadata.get("final_win_levels", 0) or 0),
                    1 if str(metadata.get("final_state") or "") == "WIN" else 0,
                )
                final_state = str(metadata.get("final_state") or "")
                for level_index in range(1, max(0, completed_count) + 1):
                    level_name = f"level_{level_index:04d}"
                    records.append(
                        {
                            "game_id": game,
                            "level_id": level_name,
                            "level_name": level_name,
                            "completed": True,
                            "success": True,
                            "seed": int(seed),
                            "sampler": sampler_name,
                            "steps_used": None,
                            "final_state": final_state,
                            "source_run_db": str(path),
                        }
                    )
    return records


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


def _aggregate_memory_job_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_keys = (
        "memory_restore_seconds", "memory_graph_restore_seconds", "memory_substrate_restore_seconds",
        "memory_query_count", "memory_query_seconds", "memory_query_mean_seconds", "memory_action_rank_count",
        "memory_action_rank_seconds", "live_memory_refresh_count", "live_memory_refresh_rows",
        "live_memory_refresh_seconds", "sqlite_busy_retry_count", "sqlite_busy_wait_seconds",
        "memory_cache_hit_count", "memory_cache_miss_count", "sqlite_queries_during_action_selection",
        "live_memory_delta_rpc_calls", "live_memory_delta_lagged_beyond_retention_count",
        "role_candidates_examined", "concept_candidates_examined", "role_lookup_seconds", "concept_lookup_seconds",
        "exact_contingency_index_rows", "fallback_contingency_index_rows", "family_index_rows",
        "replay_index_rows", "carrier_index_rows", "future_option_index_rows", "role_index_rows",
        "concept_index_rows", "graph_edge_index_rows", "memory_snapshot_build_seconds",
        "memory_snapshot_serialized_bytes", "memory_snapshot_rss_increase_bytes",
        "live_memory_events_submitted", "live_memory_events_dropped", "live_memory_queue_peak_size",
        "live_memory_queue_block_seconds",
    )
    result: dict[str, object] = {key: 0 for key in metric_keys}
    result["memory_snapshot_version"] = next((row.get("memory_snapshot_version") for row in rows if row.get("memory_snapshot_version") is not None), None)
    result["memory_snapshot_bytes"] = 0
    result["memory_snapshot_estimated_ram_bytes"] = next(
        (row.get("memory_snapshot_estimated_ram_bytes") for row in rows if row.get("memory_snapshot_estimated_ram_bytes")),
        {},
    )
    result["live_memory_delta_refresh_states"] = sorted(
        {str(row.get("live_memory_delta_refresh_state")) for row in rows if row.get("live_memory_delta_refresh_state")}
    )
    for row in rows:
        result["memory_snapshot_bytes"] = max(int(result["memory_snapshot_bytes"] or 0), int(row.get("memory_snapshot_bytes", 0) or 0))
        for key in metric_keys:
            value = row.get(key)
            if value is None:
                continue
            if key in {"memory_snapshot_build_seconds", "memory_snapshot_serialized_bytes", "memory_snapshot_rss_increase_bytes"}:
                result[key] = max(int(result[key] or 0), int(value or 0))
            elif key.endswith("_seconds") or key.endswith("_mean_seconds"):
                result[key] = float(result[key] or 0.0) + float(value)
            else:
                result[key] = int(result[key] or 0) + int(value or 0)
    result["memory_job_count"] = len(rows)
    return result


def _default_worker_execution_stats(config: InteractionSamplingConfig) -> dict[str, object]:
    return {
        "requested_workers": int(config.workers),
        "initial_workers": int(config.initial_workers if config.initial_workers is not None else config.workers),
        "peak_workers": 0,
        "fold_workers": 0,
        "worker_ramp_enabled": bool(config.enable_worker_ramp),
        "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
        "initial_worker_ramp_delay_seconds": float(config.initial_worker_ramp_delay_seconds),
        "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
        "ram_used_percent_at_start": 0.0,
        "ramp_event_count": 0,
        "ramp_events": [],
        "parallel_sidecar_fold_enabled": False,
        "parallel_sidecar_fold_shard_count": 0,
        "direct_streaming_fold_enabled": bool(config.direct_streaming_fold_enabled),
        "direct_streaming_fold_writer_count": 1 if bool(config.direct_streaming_fold_enabled and config.memory_output_dir) else 0,
        "direct_streaming_fold_worker_count": int(config.direct_streaming_fold_workers if bool(config.direct_streaming_fold_enabled) else 0),
        "direct_streaming_fold_shard_count": 0,
        "direct_streaming_fold_shard_root": None,
        "direct_streaming_fold_job_count": 0,
        "direct_streaming_fold_success_count": 0,
        "direct_streaming_fold_failed_count": 0,
        "direct_streaming_fold_deleted_raw_count": 0,
        "direct_streaming_fold_manifest_path": None,
        "direct_streaming_fold_shards_deleted": False,
        "direct_streaming_fold_merge_started_at": None,
        "direct_streaming_fold_merge_finished_at": None,
        "direct_streaming_fold_merge_seconds": None,
        "direct_streaming_fold_jobs_submitted": 0,
        "direct_streaming_fold_jobs_completed": 0,
        "direct_streaming_fold_jobs_failed": 0,
        "direct_streaming_fold_raw_deleted_after_shard_fold_count": 0,
        "direct_streaming_fold_finalized_main_memory": False,
        "direct_streaming_fold_total_raw_bytes": 0,
        "direct_streaming_fold_total_shard_bytes_added": 0,
        "direct_streaming_fold_mean_job_seconds": 0.0,
        "direct_streaming_fold_mean_write_mb_per_second": 0.0,
        "direct_streaming_shard_synchronous": str(config.direct_streaming_shard_synchronous),
        "legacy_sidecar_fold_removed": True,
        "no_backpressure": True,
        "shared_live_memory_enabled": bool(str(config.shared_live_memory) != "none"),
        "shared_live_memory_mode": str(config.shared_live_memory),
        "live_memory_writer_started": False,
        "live_memory_writer_exitcode": None,
        "live_memory_writer_forced_terminated": False,
        "live_memory_queue_maxsize": int(config.live_memory_queue_maxsize),
        "live_memory_batch_size": int(config.live_memory_batch_size),
        "live_memory_flush_seconds": float(config.live_memory_flush_seconds),
        "live_memory_delta_max_events": int(config.live_memory_delta_max_events),
        "live_memory_delta_batch_limit": int(config.live_memory_delta_batch_limit),
        "live_memory_summary_path": None,
        "live_memory_event_counts": None,
        "live_memory_events_submitted": 0,
        "live_memory_events_written": 0,
        "live_memory_batches_written": 0,
        "live_memory_queue_peak_size": 0,
        "live_memory_queue_block_seconds": 0.0,
        "live_memory_events_dropped": 0,
        "memory_metrics": _aggregate_memory_job_metrics([]),
    }


def _merge_worker_execution_stats(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_events = list(left.get("ramp_events", []) or [])
    right_events = list(right.get("ramp_events", []) or [])
    left_memory = dict(left.get("memory_metrics", {}) or {})
    right_memory = dict(right.get("memory_metrics", {}) or {})
    memory_metrics = dict(left_memory)
    for key, value in right_memory.items():
        if key == "memory_snapshot_version":
            memory_metrics[key] = value or memory_metrics.get(key)
        elif key == "memory_snapshot_bytes":
            memory_metrics[key] = max(int(memory_metrics.get(key, 0) or 0), int(value or 0))
        elif key == "memory_job_count":
            memory_metrics[key] = int(memory_metrics.get(key, 0) or 0) + int(value or 0)
        elif isinstance(value, (int, float)):
            memory_metrics[key] = (memory_metrics.get(key, 0) or 0) + value
    return {
        "requested_workers": int(max(int(left.get("requested_workers", 0) or 0), int(right.get("requested_workers", 0) or 0))),
        "initial_workers": int(max(int(left.get("initial_workers", 0) or 0), int(right.get("initial_workers", 0) or 0))),
        "fold_workers": int(max(int(left.get("fold_workers", 0) or 0), int(right.get("fold_workers", 0) or 0))),
        "peak_workers": int(max(int(left.get("peak_workers", 0) or 0), int(right.get("peak_workers", 0) or 0))),
        "worker_ramp_enabled": bool(left.get("worker_ramp_enabled", False) or right.get("worker_ramp_enabled", False)),
        "ram_ramp_threshold_percent": float(max(float(left.get("ram_ramp_threshold_percent", 0.0) or 0.0), float(right.get("ram_ramp_threshold_percent", 0.0) or 0.0))),
        "initial_worker_ramp_delay_seconds": float(max(float(left.get("initial_worker_ramp_delay_seconds", 0.0) or 0.0), float(right.get("initial_worker_ramp_delay_seconds", 0.0) or 0.0))),
        "per_worker_ramp_delay_seconds": float(max(float(left.get("per_worker_ramp_delay_seconds", 0.0) or 0.0), float(right.get("per_worker_ramp_delay_seconds", 0.0) or 0.0))),
        "ram_used_percent_at_start": float(max(float(left.get("ram_used_percent_at_start", 0.0) or 0.0), float(right.get("ram_used_percent_at_start", 0.0) or 0.0))),
        "ramp_event_count": int(len(left_events) + len(right_events)),
        "ramp_events": [*left_events, *right_events],
        "parallel_sidecar_fold_enabled": bool(left.get("parallel_sidecar_fold_enabled", False) or right.get("parallel_sidecar_fold_enabled", False)),
        "parallel_sidecar_fold_shard_count": int(left.get("parallel_sidecar_fold_shard_count", 0) or 0) + int(right.get("parallel_sidecar_fold_shard_count", 0) or 0),
        "direct_streaming_fold_enabled": bool(left.get("direct_streaming_fold_enabled", False) or right.get("direct_streaming_fold_enabled", False)),
        "direct_fold_shutdown_timings": {
            key: float((left.get("direct_fold_shutdown_timings", {}) or {}).get(key, 0.0) or 0.0)
            + float((right.get("direct_fold_shutdown_timings", {}) or {}).get(key, 0.0) or 0.0)
            for key in set((left.get("direct_fold_shutdown_timings", {}) or {}))
            | set((right.get("direct_fold_shutdown_timings", {}) or {}))
        },
        "direct_fold_shutdown_started_at": min(
            (
                float(value)
                for value in (
                    left.get("direct_fold_shutdown_started_at"),
                    right.get("direct_fold_shutdown_started_at"),
                )
                if isinstance(value, (int, float))
            ),
            default=None,
        ),
        "direct_streaming_fold_writer_count": int(max(int(left.get("direct_streaming_fold_writer_count", 0) or 0), int(right.get("direct_streaming_fold_writer_count", 0) or 0))),
        "direct_streaming_fold_worker_count": int(max(int(left.get("direct_streaming_fold_worker_count", 0) or 0), int(right.get("direct_streaming_fold_worker_count", 0) or 0))),
        "direct_streaming_fold_shard_count": int(max(int(left.get("direct_streaming_fold_shard_count", 0) or 0), int(right.get("direct_streaming_fold_shard_count", 0) or 0))),
        "direct_streaming_fold_shard_root": right.get("direct_streaming_fold_shard_root") or left.get("direct_streaming_fold_shard_root"),
        "direct_streaming_fold_job_count": int(left.get("direct_streaming_fold_job_count", 0) or 0) + int(right.get("direct_streaming_fold_job_count", 0) or 0),
        "direct_streaming_fold_success_count": int(left.get("direct_streaming_fold_success_count", 0) or 0) + int(right.get("direct_streaming_fold_success_count", 0) or 0),
        "direct_streaming_fold_failed_count": int(left.get("direct_streaming_fold_failed_count", 0) or 0) + int(right.get("direct_streaming_fold_failed_count", 0) or 0),
        "direct_streaming_fold_deleted_raw_count": int(left.get("direct_streaming_fold_deleted_raw_count", 0) or 0) + int(right.get("direct_streaming_fold_deleted_raw_count", 0) or 0),
        "direct_streaming_fold_manifest_path": right.get("direct_streaming_fold_manifest_path") or left.get("direct_streaming_fold_manifest_path"),
        "direct_streaming_fold_shards_deleted": bool(left.get("direct_streaming_fold_shards_deleted", False) or right.get("direct_streaming_fold_shards_deleted", False)),
        "direct_streaming_fold_merge_started_at": right.get("direct_streaming_fold_merge_started_at") or left.get("direct_streaming_fold_merge_started_at"),
        "direct_streaming_fold_merge_finished_at": right.get("direct_streaming_fold_merge_finished_at") or left.get("direct_streaming_fold_merge_finished_at"),
        "direct_streaming_fold_merge_seconds": right.get("direct_streaming_fold_merge_seconds") or left.get("direct_streaming_fold_merge_seconds"),
        "direct_streaming_fold_jobs_submitted": int(left.get("direct_streaming_fold_jobs_submitted", 0) or 0) + int(right.get("direct_streaming_fold_jobs_submitted", 0) or 0),
        "direct_streaming_fold_jobs_completed": int(left.get("direct_streaming_fold_jobs_completed", 0) or 0) + int(right.get("direct_streaming_fold_jobs_completed", 0) or 0),
        "direct_streaming_fold_jobs_failed": int(left.get("direct_streaming_fold_jobs_failed", 0) or 0) + int(right.get("direct_streaming_fold_jobs_failed", 0) or 0),
        "direct_streaming_fold_raw_deleted_after_shard_fold_count": int(left.get("direct_streaming_fold_raw_deleted_after_shard_fold_count", 0) or 0) + int(right.get("direct_streaming_fold_raw_deleted_after_shard_fold_count", 0) or 0),
        "direct_streaming_fold_finalized_main_memory": bool(left.get("direct_streaming_fold_finalized_main_memory", False) or right.get("direct_streaming_fold_finalized_main_memory", False)),
        "direct_streaming_fold_total_raw_bytes": int(left.get("direct_streaming_fold_total_raw_bytes", 0) or 0) + int(right.get("direct_streaming_fold_total_raw_bytes", 0) or 0),
        "direct_streaming_fold_total_shard_bytes_added": int(left.get("direct_streaming_fold_total_shard_bytes_added", 0) or 0) + int(right.get("direct_streaming_fold_total_shard_bytes_added", 0) or 0),
        "direct_streaming_fold_mean_job_seconds": float(max(float(left.get("direct_streaming_fold_mean_job_seconds", 0.0) or 0.0), float(right.get("direct_streaming_fold_mean_job_seconds", 0.0) or 0.0))),
        "direct_streaming_fold_mean_write_mb_per_second": float(max(float(left.get("direct_streaming_fold_mean_write_mb_per_second", 0.0) or 0.0), float(right.get("direct_streaming_fold_mean_write_mb_per_second", 0.0) or 0.0))),
        "direct_streaming_shard_synchronous": str(right.get("direct_streaming_shard_synchronous") or left.get("direct_streaming_shard_synchronous") or "off"),
        "legacy_sidecar_fold_removed": bool(left.get("legacy_sidecar_fold_removed", False) or right.get("legacy_sidecar_fold_removed", False)),
        "no_backpressure": bool(left.get("no_backpressure", False) or right.get("no_backpressure", False)),
        "shared_live_memory_enabled": bool(left.get("shared_live_memory_enabled", False) or right.get("shared_live_memory_enabled", False)),
        "shared_live_memory_mode": str(right.get("shared_live_memory_mode") or left.get("shared_live_memory_mode") or "none"),
        "live_memory_writer_started": bool(left.get("live_memory_writer_started", False) or right.get("live_memory_writer_started", False)),
        "live_memory_writer_exitcode": right.get("live_memory_writer_exitcode") if right.get("live_memory_writer_started") else left.get("live_memory_writer_exitcode"),
        "live_memory_writer_forced_terminated": bool(left.get("live_memory_writer_forced_terminated", False) or right.get("live_memory_writer_forced_terminated", False)),
        "live_memory_queue_maxsize": int(max(int(left.get("live_memory_queue_maxsize", 0) or 0), int(right.get("live_memory_queue_maxsize", 0) or 0))),
        "live_memory_batch_size": int(max(int(left.get("live_memory_batch_size", 0) or 0), int(right.get("live_memory_batch_size", 0) or 0))),
        "live_memory_flush_seconds": float(max(float(left.get("live_memory_flush_seconds", 0.0) or 0.0), float(right.get("live_memory_flush_seconds", 0.0) or 0.0))),
        "live_memory_delta_max_events": int(max(int(left.get("live_memory_delta_max_events", 0) or 0), int(right.get("live_memory_delta_max_events", 0) or 0))),
        "live_memory_delta_batch_limit": int(max(int(left.get("live_memory_delta_batch_limit", 0) or 0), int(right.get("live_memory_delta_batch_limit", 0) or 0))),
        "live_memory_summary_path": right.get("live_memory_summary_path") or left.get("live_memory_summary_path"),
        "live_memory_event_counts": right.get("live_memory_event_counts") or left.get("live_memory_event_counts"),
        "live_memory_events_submitted": int(left.get("live_memory_events_submitted", 0) or 0) + int(right.get("live_memory_events_submitted", 0) or 0),
        "live_memory_events_written": int(left.get("live_memory_events_written", 0) or 0) + int(right.get("live_memory_events_written", 0) or 0),
        "live_memory_batches_written": int(left.get("live_memory_batches_written", 0) or 0) + int(right.get("live_memory_batches_written", 0) or 0),
        "live_memory_queue_peak_size": int(max(int(left.get("live_memory_queue_peak_size", 0) or 0), int(right.get("live_memory_queue_peak_size", 0) or 0))),
        "live_memory_queue_block_seconds": float(left.get("live_memory_queue_block_seconds", 0.0) or 0.0) + float(right.get("live_memory_queue_block_seconds", 0.0) or 0.0),
        "live_memory_events_dropped": int(left.get("live_memory_events_dropped", 0) or 0) + int(right.get("live_memory_events_dropped", 0) or 0),
        "memory_metrics": memory_metrics,
    }


def _sampling_db_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with sqlite3.connect(path) as connection:
            tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            required_tables = {"interactions", "deltas", "contingencies", "prediction_results", "sampling_metadata"}
            if not required_tables.issubset(tables):
                return False
            if "future_effects" in tables:
                return True
            row = connection.execute(
                "SELECT value FROM sampling_metadata WHERE key IN ('future_effects_postprocessing_skipped', 'fast_postprocessing_enabled')"
            ).fetchall()
            metadata = {json.loads(value) for (value,) in row}
            return True in metadata
    except sqlite3.DatabaseError:
        return False


def _read_isf_metrics(path: Path) -> dict:
    with sqlite3.connect(path) as connection:
        try:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        except sqlite3.DatabaseError:
            return _default_isf_metrics()
        if "isf_total" not in columns:
            return _default_isf_metrics()
        rows = connection.execute(
            """
            SELECT
                isf_total,
                isf_survival_impact,
                isf_prediction_error,
                isf_learning_value,
                isf_transfer_potential,
                isf_explanatory_potential
            FROM interactions
            WHERE isf_total IS NOT NULL
            """
        ).fetchall()
    if not rows:
        return _default_isf_metrics()
    totals = [float(row[0]) for row in rows if row[0] is not None]
    survival = [float(row[1]) for row in rows if row[1] is not None]
    prediction = [float(row[2]) for row in rows if row[2] is not None]
    learning = [float(row[3]) for row in rows if row[3] is not None]
    transfer = [float(row[4]) for row in rows if row[4] is not None]
    explanatory = [float(row[5]) for row in rows if row[5] is not None]
    return {
        "mean_isf_total": float(np.mean(totals)) if totals else 0.0,
        "max_isf_total": float(max(totals)) if totals else 0.0,
        "mean_isf_survival_impact": float(np.mean(survival)) if survival else 0.0,
        "mean_isf_prediction_error": float(np.mean(prediction)) if prediction else 0.0,
        "mean_isf_learning_value": float(np.mean(learning)) if learning else 0.0,
        "mean_isf_transfer_potential": float(np.mean(transfer)) if transfer else 0.0,
        "mean_isf_explanatory_potential": float(np.mean(explanatory)) if explanatory else 0.0,
        "high_isf_interaction_count": sum(1 for value in totals if value >= 0.70),
    }


def _default_isf_metrics() -> dict:
    return {
        "mean_isf_total": 0.0,
        "max_isf_total": 0.0,
        "mean_isf_survival_impact": 0.0,
        "mean_isf_prediction_error": 0.0,
        "mean_isf_learning_value": 0.0,
        "mean_isf_transfer_potential": 0.0,
        "mean_isf_explanatory_potential": 0.0,
        "high_isf_interaction_count": 0,
        "graph_edge_follows_count": 0,
        "graph_edge_generated_count": 0,
        "graph_edge_member_of_count": 0,
        "graph_edge_predicts_count": 0,
        "graph_edge_enables_count": 0,
        "graph_edge_blocks_count": 0,
        "graph_edge_restricts_count": 0,
        "graph_edge_expands_count": 0,
        "graph_edge_terminates_count": 0,
        "graph_edge_reversible_with_count": 0,
        "graph_edge_explains_count": 0,
        "graph_edge_similar_role_to_count": 0,
        "graph_edge_depends_on_count": 0,
        "graph_edge_contradicts_count": 0,
        "graph_edge_total_count": 0,
        "context_contradiction_count": 0,
        "contradicted_context_count": 0,
        "contradicted_context_action_count": 0,
        "repeated_contradiction_count": 0,
        "context_expansion_suggested_count": 0,
        "mean_suggested_context_depth": 0.0,
        "max_suggested_context_depth": 0.0,
        "carrier_candidate_count": 0,
        "emergent_carrier_count": 0,
        "carrier_spatial_candidate_count": 0,
        "carrier_object_candidate_count": 0,
        "carrier_cell_candidate_count": 0,
        "carrier_context_action_fallback_candidate_count": 0,
        "emergent_spatial_carrier_count": 0,
        "emergent_object_carrier_count": 0,
        "emergent_cell_carrier_count": 0,
        "emergent_context_action_fallback_count": 0,
        "carrier_event_count": 0,
        "carrier_max_support": 0,
        "carrier_mean_support": 0.0,
        "mean_carrier_prediction_lift": 0.0,
        "max_carrier_prediction_lift": 0.0,
        "mean_carrier_compression_gain": 0.0,
        "max_carrier_compression_gain": 0.0,
        "memory_record_count": 0,
        "memory_active_count": 0,
        "memory_protected_count": 0,
        "memory_compressed_count": 0,
        "memory_forgotten_count": 0,
        "memory_replay_candidate_count": 0,
        "memory_max_replay_priority": 0.0,
        "memory_mean_replay_priority": 0.0,
        "mean_memory_replay_priority": 0.0,
        "high_priority_replay_count": 0,
        "efficiency_event_count": 0,
        "efficiency_total_action_cost": 0.0,
        "efficiency_mean_action_cost": 0.0,
        "efficiency_no_effect_action_count": 0,
        "efficiency_no_effect_action_ratio": 0.0,
        "efficiency_repeated_state_count": 0,
        "efficiency_repeated_state_ratio": 0.0,
        "efficiency_repeated_context_action_count": 0,
        "efficiency_repeated_context_action_ratio": 0.0,
        "efficiency_terminal_outcome_count": 0,
        "efficiency_distinct_outcome_count": 0,
        "efficiency_future_option_gain_per_cost_count": 0,
        "posthoc_future_option_delta_count": 0,
        "efficiency_mean_normalized_solve_efficiency": 0.0,
        "efficiency_max_normalized_solve_efficiency": 0.0,
        "efficiency_mean_equivalent_outcome_cost_gap": 0.0,
        "efficiency_max_equivalent_outcome_cost_gap": 0.0,
        "efficiency_mean_future_option_gain_per_cost": 0.0,
        "efficiency_max_future_option_gain_per_cost": 0.0,
        "efficiency_min_future_option_gain_per_cost": 0.0,
    }


def _read_context_contradiction_metrics(path: Path) -> dict:
    default = {
        "context_contradiction_count": 0,
        "contradicted_context_count": 0,
        "contradicted_context_action_count": 0,
        "repeated_contradiction_count": 0,
        "context_expansion_suggested_count": 0,
        "mean_suggested_context_depth": 0.0,
        "max_suggested_context_depth": 0.0,
        "carrier_candidate_count": 0,
        "emergent_carrier_count": 0,
        "carrier_spatial_candidate_count": 0,
        "carrier_object_candidate_count": 0,
        "carrier_cell_candidate_count": 0,
        "carrier_context_action_fallback_candidate_count": 0,
        "emergent_spatial_carrier_count": 0,
        "emergent_object_carrier_count": 0,
        "emergent_cell_carrier_count": 0,
        "emergent_context_action_fallback_count": 0,
        "carrier_event_count": 0,
        "carrier_max_support": 0,
        "carrier_mean_support": 0.0,
        "mean_carrier_prediction_lift": 0.0,
        "max_carrier_prediction_lift": 0.0,
        "mean_carrier_compression_gain": 0.0,
        "max_carrier_compression_gain": 0.0,
        "memory_record_count": 0,
        "memory_active_count": 0,
        "memory_protected_count": 0,
        "memory_compressed_count": 0,
        "memory_forgotten_count": 0,
        "memory_replay_candidate_count": 0,
        "memory_max_replay_priority": 0.0,
        "memory_mean_replay_priority": 0.0,
        "mean_memory_replay_priority": 0.0,
        "high_priority_replay_count": 0,
        "efficiency_event_count": 0,
        "efficiency_total_action_cost": 0.0,
        "efficiency_mean_action_cost": 0.0,
        "efficiency_no_effect_action_count": 0,
        "efficiency_no_effect_action_ratio": 0.0,
        "efficiency_repeated_state_count": 0,
        "efficiency_repeated_state_ratio": 0.0,
        "efficiency_repeated_context_action_count": 0,
        "efficiency_repeated_context_action_ratio": 0.0,
        "efficiency_terminal_outcome_count": 0,
        "efficiency_distinct_outcome_count": 0,
        "efficiency_future_option_gain_per_cost_count": 0,
        "posthoc_future_option_delta_count": 0,
        "efficiency_mean_normalized_solve_efficiency": 0.0,
        "efficiency_max_normalized_solve_efficiency": 0.0,
        "efficiency_mean_equivalent_outcome_cost_gap": 0.0,
        "efficiency_max_equivalent_outcome_cost_gap": 0.0,
        "efficiency_mean_future_option_gain_per_cost": 0.0,
        "efficiency_max_future_option_gain_per_cost": 0.0,
        "efficiency_min_future_option_gain_per_cost": 0.0,
    }
    with sqlite3.connect(path) as connection:
        try:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(prediction_results)").fetchall()}
        except sqlite3.DatabaseError:
            return default
        required = {"context_contradiction", "context_expansion_suggested", "suggested_context_depth"}
        if not required.issubset(columns):
            return default
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN context_contradiction = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN context_expansion_suggested = 1 THEN 1 ELSE 0 END),
                AVG(CASE WHEN suggested_context_depth IS NOT NULL THEN suggested_context_depth END),
                MAX(suggested_context_depth)
            FROM prediction_results
            """
        ).fetchone()
    return {
        **default,
        "context_contradiction_count": int(row[0] or 0),
        "context_expansion_suggested_count": int(row[1] or 0),
        "mean_suggested_context_depth": float(row[2] or 0.0),
        "max_suggested_context_depth": float(row[3] or 0.0),
    }


def _read_memory_lifecycle_metrics(path: Path) -> dict:
    default = {
        "memory_record_count": 0,
        "memory_active_count": 0,
        "memory_protected_count": 0,
        "memory_compressed_count": 0,
        "memory_forgotten_count": 0,
        "memory_replay_candidate_count": 0,
        "memory_max_replay_priority": 0.0,
        "memory_mean_replay_priority": 0.0,
        "mean_memory_replay_priority": 0.0,
        "high_priority_replay_count": 0,
        "posthoc_future_option_delta_count": 0,
    }
    with sqlite3.connect(path) as connection:
        try:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        except sqlite3.DatabaseError:
            return default
        required = {"memory_status", "memory_replay_priority", "memory_replay_candidate"}
        if not required.issubset(columns):
            return default
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN memory_status = 'active' THEN 1 ELSE 0 END),
                SUM(CASE WHEN memory_status = 'protected' THEN 1 ELSE 0 END),
                SUM(CASE WHEN memory_status = 'compressed' THEN 1 ELSE 0 END),
                SUM(CASE WHEN memory_status = 'forgotten' THEN 1 ELSE 0 END),
                SUM(CASE WHEN memory_replay_candidate = 1 THEN 1 ELSE 0 END),
                MAX(COALESCE(memory_replay_priority, 0.0)),
                AVG(CASE WHEN memory_replay_candidate = 1 THEN COALESCE(memory_replay_priority, 0.0) END),
                SUM(CASE WHEN COALESCE(memory_replay_priority, 0.0) >= 0.70 THEN 1 ELSE 0 END)
            FROM interactions
            """
        ).fetchone()
    return {
        **default,
        "memory_record_count": int(row[0] or 0),
        "memory_active_count": int(row[1] or 0),
        "memory_protected_count": int(row[2] or 0),
        "memory_compressed_count": int(row[3] or 0),
        "memory_forgotten_count": int(row[4] or 0),
        "memory_replay_candidate_count": int(row[5] or 0),
        "memory_max_replay_priority": float(row[6] or 0.0),
        "memory_mean_replay_priority": float(row[7] or 0.0),
        "mean_memory_replay_priority": float(row[7] or 0.0),
        "high_priority_replay_count": int(row[8] or 0),
    }


def _read_adaptive_context_metrics(path: Path) -> dict:
    default = {
        "adaptive_context_depth_used_count": 0,
        "adaptive_context_expansion_applied_count": 0,
    }
    with sqlite3.connect(path) as connection:
        try:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        except sqlite3.DatabaseError:
            return default
        required = {"context_depth_used", "adaptive_context_expansion_applied", "adaptive_context_depth_after"}
        if not required.issubset(columns):
            return default
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN context_depth_used IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN adaptive_context_expansion_applied = 1 THEN 1 ELSE 0 END)
            FROM interactions
            """
        ).fetchone()
    return {
        **default,
        "adaptive_context_depth_used_count": int(row[0] or 0),
        "adaptive_context_expansion_applied_count": int(row[1] or 0),
    }


def _read_efficiency_metrics(path: Path) -> dict:
    default = {
        "efficiency_event_count": 0,
        "efficiency_total_action_cost": 0.0,
        "efficiency_mean_action_cost": 0.0,
        "efficiency_no_effect_action_count": 0,
        "efficiency_no_effect_action_ratio": 0.0,
        "efficiency_repeated_state_count": 0,
        "efficiency_repeated_state_ratio": 0.0,
        "efficiency_repeated_context_action_count": 0,
        "efficiency_repeated_context_action_ratio": 0.0,
        "efficiency_terminal_outcome_count": 0,
        "efficiency_distinct_outcome_count": 0,
        "efficiency_future_option_gain_per_cost_count": 0,
        "posthoc_future_option_delta_count": 0,
        "efficiency_mean_normalized_solve_efficiency": 0.0,
        "efficiency_max_normalized_solve_efficiency": 0.0,
        "efficiency_mean_equivalent_outcome_cost_gap": 0.0,
        "efficiency_max_equivalent_outcome_cost_gap": 0.0,
        "efficiency_mean_future_option_gain_per_cost": 0.0,
        "efficiency_max_future_option_gain_per_cost": 0.0,
        "efficiency_min_future_option_gain_per_cost": 0.0,
    }
    with sqlite3.connect(path) as connection:
        try:
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        except sqlite3.DatabaseError:
            return default
        required = {
            "efficiency_action_cost",
            "efficiency_repeated_state",
            "efficiency_repeated_context_action",
            "efficiency_no_effect_action",
            "efficiency_terminal_outcome",
            "efficiency_outcome_signature",
        }
        if not required.issubset(columns):
            return default
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(COALESCE(efficiency_action_cost, 0.0)),
                AVG(COALESCE(efficiency_action_cost, 0.0)),
                SUM(CASE WHEN efficiency_no_effect_action = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN efficiency_repeated_state = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN efficiency_repeated_context_action = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN efficiency_terminal_outcome = 1 THEN 1 ELSE 0 END),
                COUNT(DISTINCT efficiency_outcome_signature),
                AVG(efficiency_normalized_solve_efficiency),
                MAX(efficiency_normalized_solve_efficiency),
                AVG(efficiency_equivalent_outcome_cost_gap),
                MAX(efficiency_equivalent_outcome_cost_gap),
                COUNT(CASE WHEN efficiency_future_option_gain_per_cost IS NOT NULL THEN 1 END),
                COUNT(efficiency_future_option_gain_per_cost),
                AVG(efficiency_future_option_gain_per_cost),
                MAX(efficiency_future_option_gain_per_cost),
                MIN(efficiency_future_option_gain_per_cost)
            FROM interactions
            """
        ).fetchone()
    event_count = int(row[0] or 0)
    no_effect_count = int(row[3] or 0)
    repeated_state_count = int(row[4] or 0)
    repeated_context_action_count = int(row[5] or 0)
    return {
        **default,
        "efficiency_event_count": event_count,
        "efficiency_total_action_cost": float(row[1] or 0.0),
        "efficiency_mean_action_cost": float(row[2] or 0.0),
        "efficiency_no_effect_action_count": no_effect_count,
        "efficiency_no_effect_action_ratio": no_effect_count / max(1, event_count),
        "efficiency_repeated_state_count": repeated_state_count,
        "efficiency_repeated_state_ratio": repeated_state_count / max(1, event_count),
        "efficiency_repeated_context_action_count": repeated_context_action_count,
        "efficiency_repeated_context_action_ratio": repeated_context_action_count / max(1, event_count),
        "efficiency_terminal_outcome_count": int(row[6] or 0),
        "efficiency_distinct_outcome_count": int(row[7] or 0),
        "efficiency_mean_normalized_solve_efficiency": float(row[8] or 0.0),
        "efficiency_max_normalized_solve_efficiency": float(row[9] or 0.0),
        "efficiency_mean_equivalent_outcome_cost_gap": float(row[10] or 0.0),
        "efficiency_max_equivalent_outcome_cost_gap": float(row[11] or 0.0),
        "posthoc_future_option_delta_count": int(row[12] or 0),
        "efficiency_future_option_gain_per_cost_count": int(row[13] or 0),
        "efficiency_mean_future_option_gain_per_cost": float(row[14] or 0.0),
        "efficiency_max_future_option_gain_per_cost": float(row[15] or 0.0),
        "efficiency_min_future_option_gain_per_cost": float(row[16] or 0.0),
    }


def _mean_or_none(values) -> float | None:
    items = [float(value) for value in values if value is not None]
    return float(np.mean(items)) if items else None


def _max_or_none(values) -> float | None:
    items = [float(value) for value in values if value is not None]
    return float(max(items)) if items else None


def _min_or_none(values) -> float | None:
    items = [float(value) for value in values if value is not None]
    return float(min(items)) if items else None


def _future_option_deltas_by_interaction_id(db_path: Path, *, horizon: int) -> dict[str, float]:
    del db_path, horizon
    return {}


def _apply_future_option_efficiency_diagnostics(db_path: Path, deltas_by_interaction_id: dict[str, float]) -> None:
    if not deltas_by_interaction_id:
        return
    with sqlite3.connect(db_path) as connection:
        interaction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        prediction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(prediction_results)").fetchall()}
        if not {
            "id",
            "efficiency_action_cost",
            "efficiency_future_option_gain_per_cost",
        }.issubset(interaction_columns):
            return
        if not {
            "interaction_id",
            "efficiency_action_cost",
            "efficiency_future_option_gain_per_cost",
        }.issubset(prediction_columns):
            return
        rows = [(float(delta), str(interaction_id)) for interaction_id, delta in deltas_by_interaction_id.items()]
        if not rows:
            return
        connection.executemany(
            """
            UPDATE interactions
            SET efficiency_future_option_gain_per_cost = ? / efficiency_action_cost
            WHERE id = ?
              AND efficiency_action_cost IS NOT NULL
              AND efficiency_action_cost > 0
            """,
            rows,
        )
        connection.executemany(
            """
            UPDATE prediction_results
            SET efficiency_future_option_gain_per_cost = ? / efficiency_action_cost
            WHERE interaction_id = ?
              AND efficiency_action_cost IS NOT NULL
              AND efficiency_action_cost > 0
            """,
            rows,
        )
        connection.commit()


def _group_ok_by_game(rows: list[dict]) -> dict[str, list[dict]]:
    by_game = defaultdict(list)
    for row in rows:
        if row.get("run_status") == "ok":
            by_game[row["game"]].append(row)
    return by_game


def _failed_row(game: str, sampler_name: str, config: InteractionSamplingConfig, reason: str) -> dict:
    return {
        "game": game,
        "family": family_for_game(game),
        "sampler_name": sampler_name,
        "steps": config.steps,
        "horizon": config.horizon,
        "context_depth": config.context_depth,
        "context_length": config.context_depth,
        "adaptive_context_expansion_enabled": bool(config.adaptive_context_expansion),
        "base_context_depth": config.context_depth,
        "max_context_depth": config.max_context_depth if config.max_context_depth is not None else config.context_depth,
        "train_seeds": list(config.train_seeds),
        "test_seed": config.test_seed,
        "run_status": "failed",
        "failure_reason": reason,
        "pass_status": False,
        "forbidden_future_feature_check_passed": True,
        "forbidden_id_feature_check_passed": True,
        "mean_isf_total": 0.0,
        "max_isf_total": 0.0,
        "mean_isf_survival_impact": 0.0,
        "mean_isf_prediction_error": 0.0,
        "mean_isf_learning_value": 0.0,
        "mean_isf_transfer_potential": 0.0,
        "mean_isf_explanatory_potential": 0.0,
        "high_isf_interaction_count": 0,
        "graph_edge_follows_count": 0,
        "graph_edge_generated_count": 0,
        "graph_edge_member_of_count": 0,
        "graph_edge_predicts_count": 0,
        "graph_edge_enables_count": 0,
        "graph_edge_blocks_count": 0,
        "graph_edge_restricts_count": 0,
        "graph_edge_expands_count": 0,
        "graph_edge_terminates_count": 0,
        "graph_edge_reversible_with_count": 0,
        "graph_edge_explains_count": 0,
        "graph_edge_similar_role_to_count": 0,
        "graph_edge_depends_on_count": 0,
        "graph_edge_contradicts_count": 0,
        "graph_edge_total_count": 0,
        "context_contradiction_count": 0,
        "contradicted_context_count": 0,
        "contradicted_context_action_count": 0,
        "repeated_contradiction_count": 0,
        "context_expansion_suggested_count": 0,
        "adaptive_context_expansion_count": 0,
        "adaptive_context_active_action_count": 0,
        "adaptive_context_max_depth_reached": 0,
        "adaptive_context_depth_used_count": 0,
        "adaptive_context_expansion_applied_count": 0,
        "mean_suggested_context_depth": 0.0,
        "max_suggested_context_depth": 0.0,
        "carrier_candidate_count": 0,
        "emergent_carrier_count": 0,
        "carrier_spatial_candidate_count": 0,
        "carrier_object_candidate_count": 0,
        "carrier_cell_candidate_count": 0,
        "carrier_context_action_fallback_candidate_count": 0,
        "emergent_spatial_carrier_count": 0,
        "emergent_object_carrier_count": 0,
        "emergent_cell_carrier_count": 0,
        "emergent_context_action_fallback_count": 0,
        "carrier_event_count": 0,
        "carrier_max_support": 0,
        "carrier_mean_support": 0.0,
        "mean_carrier_prediction_lift": 0.0,
        "max_carrier_prediction_lift": 0.0,
        "mean_carrier_compression_gain": 0.0,
        "max_carrier_compression_gain": 0.0,
        "memory_record_count": 0,
        "memory_active_count": 0,
        "memory_protected_count": 0,
        "memory_compressed_count": 0,
        "memory_forgotten_count": 0,
        "memory_replay_candidate_count": 0,
        "memory_max_replay_priority": 0.0,
        "memory_mean_replay_priority": 0.0,
        "mean_memory_replay_priority": 0.0,
        "high_priority_replay_count": 0,
        "efficiency_event_count": 0,
        "efficiency_total_action_cost": 0.0,
        "efficiency_mean_action_cost": 0.0,
        "efficiency_no_effect_action_count": 0,
        "efficiency_no_effect_action_ratio": 0.0,
        "efficiency_repeated_state_count": 0,
        "efficiency_repeated_state_ratio": 0.0,
        "efficiency_repeated_context_action_count": 0,
        "efficiency_repeated_context_action_ratio": 0.0,
        "efficiency_terminal_outcome_count": 0,
        "efficiency_distinct_outcome_count": 0,
        "efficiency_future_option_gain_per_cost_count": 0,
        "efficiency_mean_normalized_solve_efficiency": 0.0,
        "efficiency_max_normalized_solve_efficiency": 0.0,
        "efficiency_mean_equivalent_outcome_cost_gap": 0.0,
        "efficiency_max_equivalent_outcome_cost_gap": 0.0,
        "efficiency_mean_future_option_gain_per_cost": 0.0,
        "efficiency_max_future_option_gain_per_cost": 0.0,
        "efficiency_min_future_option_gain_per_cost": 0.0,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), (dict, list, tuple)) else row.get(field) for field in fields})


def _format_text(payload: dict) -> str:
    validation = payload["validation"]
    lines = [
        "ARC-AGI3 v0.5c Interaction Sampling Repair",
        f"validation={validation}",
        f"future_option_efficiency_posthoc_only={payload.get('future_option_efficiency_posthoc_only')}",
        "",
    ]
    if not validation.get("diagnostic_success", False):
        lines.append("operational failure: no scientific pass/fail conclusion")
        return "\n".join(lines) + "\n"
    lines.append("best by game:")
    for row in payload["best_by_game"]:
        non_preserve_count = row.get("non_preserve_count")
        non_preserve_ratio = row.get("non_preserve_ratio")
        lines.append(
            f"{row['game']} sampler={row['sampler_name']} pass={row['pass_status']} "
            f"nonP={'n/a' if non_preserve_count is None else non_preserve_count} "
            f"ratio={'n/a' if non_preserve_ratio is None else f'{float(non_preserve_ratio):.3f}'} "
            f"acc={row['id_free_accuracy']:.3f} macro_f1={row['id_free_macro_f1']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _recommended_next_steps(payload: dict) -> str:
    validation = payload["validation"]
    if not validation.get("diagnostic_success", False):
        return (
            "Recommended next step: fix operational failures before drawing any scientific conclusion.\n"
            f"failed_run_count={validation.get('failed_run_count', 0)}\n"
            f"failure_reason_counts={validation.get('failure_reason_counts', {})}\n"
        )
    if validation["sampling_repair_strong_pass"]:
        recommendation = "v0.5d broad validation using the best non-planning sampler."
    elif validation["sampling_repair_weak_pass"]:
        recommendation = "v0.5d broaden sampling repair, then repair features/context if validation still fails."
    else:
        recommendation = "family-specific diagnostics; simple reactive exploration may be insufficient."
    return (
        f"Recommended next step: {recommendation}\n"
        f"weak_games={validation['weak_games']}\n"
        f"strong_games={validation['strong_games']}\n"
        "Do not move to carrier discovery from v0.5c alone.\n"
    )
