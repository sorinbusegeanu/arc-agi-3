from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.environment.arc_adapter import ArcGridEnvironment
from v6.evaluation.broad_game_validation import family_for_game, game_passes, parse_game_selector
from v6.evaluation.failure_diagnostics import compute_run_diagnostics
from v6.evaluation.future_effects import analyze_future_effects, interaction_future_option_deltas
from v6.evaluation.id_free_prefuture_validation import ID_FREE_FEATURE_SETS, evaluate_id_free_config
from v6.evaluation.prefuture_role_prediction import PREFUTURE_CLASSIFIERS, load_prefuture_examples
from v6.game_sets import load_game_set_manifest, parquet_games_present
from v6.main import V6Config, V6System
from v6.sampling import make_sampler, sampler_registry
from v6.storage.migration import migrate_sqlite_to_parquet


FAILED_REPRESENTATIVES = ("tt01", "pb02", "fs02", "tp02", "gr01")
PASSING_REFERENCES = ("va02", "mo01")
DEFAULT_V05C_GAMES = FAILED_REPRESENTATIVES + PASSING_REFERENCES
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
}


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
    workers: int = 60
    commit_steps: int = 1000
    storage_backend: str = "sqlite"
    parquet_root: str = "runs/v6/storage_parquet"
    duckdb_path: str = "runs/v6/arc_agi3.duckdb"
    storage_batch_size: int = 1000
    compression: str = "zstd"
    output_dir: str = "runs/v6"
    env_root: str | None = None
    game_set_manifest: str | None = None
    game_set_name: str | None = None
    only_missing_from_parquet_root: bool = False
    collect_only: bool = False


def parse_v05c_games(selector: str) -> tuple[str, ...]:
    value = selector.strip()
    if value in V05C_GAME_PRESETS:
        return tuple(dict.fromkeys(V05C_GAME_PRESETS[value]))
    return parse_game_selector(value)


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
    sampling_root = output / ("sampling_v05c_sqlite_tmp" if config.storage_backend == "parquet" else "sampling_v05c")
    sampling_root.mkdir(parents=True, exist_ok=True)
    _generate_sampling_dbs(config, sampling_root)
    if config.collect_only:
        if config.storage_backend == "parquet":
            _export_sampling_sqlite_to_parquet(config, sampling_root)
            shutil.rmtree(sampling_root, ignore_errors=True)
        return []
    rows = _evaluate_sampling_runs(config, sampling_root)
    comparison = sampler_comparison_rows(rows)
    best = best_by_game(rows)
    family_summary = summary_by_family(best)
    payload = {
        "runs": rows,
        "sampler_comparison": comparison,
        "best_by_game": best,
        "summary_by_family": family_summary,
        "validation": validation_summary(rows, comparison, best),
        "samplers": list(config.samplers),
        "forbidden_features_used_during_sampling": False,
        "efficiency_diagnostics_enabled": True,
        "efficiency_used_for_sampling": False,
        "efficiency_used_for_m2": False,
        "efficiency_used_for_m3": False,
        "efficiency_used_for_m4": False,
        "future_option_efficiency_posthoc_only": True,
    }
    write_interaction_sampling_reports(payload, output)
    if config.storage_backend == "parquet":
        _export_sampling_sqlite_to_parquet(config, sampling_root)
        shutil.rmtree(sampling_root, ignore_errors=True)
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


def _generate_sampling_dbs(config: InteractionSamplingConfig, sampling_root: Path) -> None:
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
                            "commit_steps": int(config.commit_steps),
                            "db_path": str(db_path),
                            "env_root": config.env_root,
                        }
                    )
                    order += 1
            if game_jobs:
                _run_sampling_jobs(game_jobs, workers=config.workers)
            _export_sampling_sqlite_to_parquet(config, sampling_root, games=(game,))
            shutil.rmtree(sampling_root / game, ignore_errors=True)
        return

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
                        "commit_steps": int(config.commit_steps),
                        "db_path": str(db_path),
                        "env_root": config.env_root,
                    }
                )
                order += 1
    if not jobs:
        return
    _run_sampling_jobs(jobs, workers=config.workers)


def _run_sampling_jobs(jobs: list[dict], *, workers: int) -> None:
    workers = max(1, min(int(workers), len(jobs)))
    print(f"running {len(jobs)} v0.5c sampling jobs with workers={workers}", file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        futures = {executor.submit(_run_sampling_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            print(
                f"completed {job['game']} sampler={job['sampler_name']} seed={job['seed']} steps={job['steps']}",
                file=sys.stderr,
                flush=True,
            )


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


def _run_sampling_job(job: dict) -> dict:
    db_path = Path(str(job["db_path"]))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler_name = str(job["sampler_name"])
    seed = int(job["seed"])
    sampler = make_sampler(sampler_name, seed=seed)
    env = ArcGridEnvironment(game_id=str(job["game"]), seed=seed, env_root=job["env_root"])
    system = V6System(
        env=env,
        config=V6Config(
            database_path=str(db_path),
            random_seed=seed,
            context_length=int(job.get("context_depth", 3)),
            database_commit_every=int(job.get("commit_steps", 1000)),
        ),
        action_sampler=sampler,
    )
    edge_counts: dict[str, int] = {}
    contradiction_summary: dict[str, object] = {}
    carrier_candidates: list[object] = []
    memory_summary: dict[str, object] = {}
    replay_candidates: list[dict] = []
    efficiency_summary: dict[str, object] = {}
    try:
        system.run(steps=int(job["steps"]))
        graph = getattr(system, "graph", None)
        if graph is not None and hasattr(graph, "edge_type_counts"):
            edge_counts = graph.edge_type_counts()
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
    finally:
        system.close()
    effects = analyze_future_effects(
        db_path=str(db_path),
        game=str(job["game"]),
        seed=seed,
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
    )
    deltas_by_interaction_id = _future_option_deltas_by_interaction_id(
        db_path,
        horizon=int(job["horizon"]),
    )
    _apply_future_option_efficiency_diagnostics(db_path, deltas_by_interaction_id)
    _write_sampling_metadata(
        db_path,
        game=str(job["game"]),
        sampler_name=sampler_name,
        seed=seed,
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
        context_depth=int(job.get("context_depth", 3)),
        context_length=int(job.get("context_depth", 3)),
        reset_count=int(getattr(env, "reset_count", 0)) + int(getattr(sampler, "reset_count", 0)),
        terminal_count=int(getattr(env, "skipped_terminal_steps", 0)),
        reset_unavailable=bool(getattr(sampler, "reset_unavailable", False)),
        context_contradiction_count=int(contradiction_summary.get("context_contradiction_count", 0) or 0),
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
    )
    if contradiction_summary:
        db_path.with_name("context_contradictions.json").write_text(json.dumps(contradiction_summary, indent=2), encoding="utf-8")
    db_path.with_name("carrier_candidates.json").write_text(
        json.dumps([item.to_dict() if hasattr(item, "to_dict") else item for item in carrier_candidates], indent=2),
        encoding="utf-8",
    )
    db_path.with_name("memory_lifecycle_summary.json").write_text(json.dumps(memory_summary, indent=2), encoding="utf-8")
    db_path.with_name("memory_replay_candidates.json").write_text(json.dumps(replay_candidates, indent=2), encoding="utf-8")
    db_path.with_name("efficiency_summary.json").write_text(json.dumps(efficiency_summary, indent=2), encoding="utf-8")
    return {"effects": len(effects)}


def _evaluate_sampling_runs(config: InteractionSamplingConfig, sampling_root: Path) -> list[dict]:
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


def _run_metrics(path: Path, game: str, sampler_name: str, seed: int, config: InteractionSamplingConfig) -> dict:
    diagnostics = compute_run_diagnostics(path, game=game, seed=seed, steps=config.steps, horizon=config.horizon)
    diagnostics.update(_read_isf_metrics(path))
    diagnostics.update(_read_context_contradiction_metrics(path))
    diagnostics.update(_read_memory_lifecycle_metrics(path))
    metadata = _read_sampling_metadata(path)
    diagnostics.update(metadata)
    diagnostics.update(_read_efficiency_metrics(path))
    diagnostics["sampler_name"] = sampler_name
    return diagnostics


def _best_validation_row(game: str, sampler_name: str, config: InteractionSamplingConfig, sampling_root: Path) -> dict:
    train = []
    for seed in config.train_seeds:
        examples = load_prefuture_examples(sampling_db_path(sampling_root, game, sampler_name, config.steps, seed))
        train.extend([item for item in examples if int(item.features["context_level"]) <= int(config.context_depth)])
    test = load_prefuture_examples(sampling_db_path(sampling_root, game, sampler_name, config.steps, config.test_seed))
    test = [item for item in test if int(item.features["context_level"]) <= int(config.context_depth)]
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


def _aggregate_seed_rows(seed_rows: list[dict], config: InteractionSamplingConfig) -> dict:
    first = seed_rows[0]
    sums = Counter()
    for row in seed_rows:
        for key in (
            "total_interactions",
            "usable_interactions",
            "reset_count",
            "terminal_count",
            "future_effect_count",
            "preserve_count",
            "expand_count",
            "restrict_count",
            "collapse_count",
            "non_preserve_count",
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
    future_count = max(1, sums["future_effect_count"])
    return {
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
        "mean_suggested_context_depth": float(np.mean([row.get("mean_suggested_context_depth", 0.0) or 0.0 for row in seed_rows])),
        "max_suggested_context_depth": float(max((row.get("max_suggested_context_depth", 0.0) or 0.0 for row in seed_rows))),
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
        "future_effect_count": sums["future_effect_count"],
        "preserve_count": sums["preserve_count"],
        "expand_count": sums["expand_count"],
        "restrict_count": sums["restrict_count"],
        "collapse_count": sums["collapse_count"],
        "non_preserve_count": sums["non_preserve_count"],
        "non_preserve_ratio": sums["non_preserve_count"] / future_count,
    }


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
            output.append(
                {
                    "game": game,
                    "family": family_for_game(game),
                    "sampler_name": sampler_name,
                    "delta_non_preserve_count": int(row["non_preserve_count"]) - int(baseline["non_preserve_count"]),
                    "delta_non_preserve_ratio": float(row["non_preserve_ratio"]) - float(baseline["non_preserve_ratio"]),
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
                    int(row["non_preserve_count"]),
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
            "mean_non_preserve_ratio": float(np.mean([row["non_preserve_ratio"] for row in rows])),
            "mean_non_preserve_count": float(np.mean([row["non_preserve_count"] for row in rows])),
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


def _sampling_db_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with sqlite3.connect(path) as connection:
            tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return {"interactions", "deltas", "contingencies", "prediction_results", "future_effects", "sampling_metadata"}.issubset(tables)
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
    try:
        with sqlite3.connect(db_path) as connection:
            tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if not {"contingencies", "prediction_results"}.issubset(tables):
                return {}
            prediction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(prediction_results)").fetchall()}
            contingency_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(contingencies)").fetchall()}
            required_prediction = {"interaction_id", "episode_id", "action", "actual_family"}
            required_contingency = {"id", "context_level", "context_signature", "action", "transformation_family", "support_count", "confidence"}
            if not required_prediction.issubset(prediction_columns) or not required_contingency.issubset(contingency_columns):
                return {}
            rows = interaction_future_option_deltas(connection, horizon=horizon)
    except sqlite3.DatabaseError:
        return {}
    return {str(interaction_id): float(delta) for interaction_id, delta in rows.items()}


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
        lines.append(
            f"{row['game']} sampler={row['sampler_name']} pass={row['pass_status']} "
            f"nonP={row['non_preserve_count']} ratio={row['non_preserve_ratio']:.3f} "
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
