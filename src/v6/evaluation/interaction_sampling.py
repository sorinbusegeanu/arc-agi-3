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
from v6.evaluation.future_effects import analyze_future_effects
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
    return InteractionSamplingConfig(
        **{
            **config.__dict__,
            "games": tuple(dict.fromkeys(selected_games)),
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
            context_length=3,
            database_commit_every=int(job.get("commit_steps", 1000)),
        ),
        action_sampler=sampler,
    )
    try:
        system.run(steps=int(job["steps"]))
    finally:
        system.close()
    effects = analyze_future_effects(
        db_path=str(db_path),
        game=str(job["game"]),
        seed=seed,
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
    )
    _write_sampling_metadata(
        db_path,
        game=str(job["game"]),
        sampler_name=sampler_name,
        seed=seed,
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
        reset_count=int(getattr(env, "reset_count", 0)) + int(getattr(sampler, "reset_count", 0)),
        terminal_count=int(getattr(env, "skipped_terminal_steps", 0)),
        reset_unavailable=bool(getattr(sampler, "reset_unavailable", False)),
    )
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
                eval_row = _best_validation_row(game, sampler_name, config, sampling_root)
                aggregate = _aggregate_seed_rows(seed_rows)
                rows.append({**aggregate, **eval_row, "run_status": "ok", "failure_reason": ""})
            except Exception as exc:
                rows.append(_failed_row(game, sampler_name, config, f"{type(exc).__name__}: {exc}"))
    return rows


def _run_metrics(path: Path, game: str, sampler_name: str, seed: int, config: InteractionSamplingConfig) -> dict:
    diagnostics = compute_run_diagnostics(path, game=game, seed=seed, steps=config.steps, horizon=config.horizon)
    metadata = _read_sampling_metadata(path)
    diagnostics.update(metadata)
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


def _aggregate_seed_rows(seed_rows: list[dict]) -> dict:
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
        ):
            sums[key] += int(row.get(key, 0) or 0)
    future_count = max(1, sums["future_effect_count"])
    return {
        "game": first["game"],
        "family": family_for_game(first["game"]),
        "sampler_name": first["sampler_name"],
        "steps": first["steps"],
        "horizon": first["horizon"],
        "context_depth": first.get("context_depth", 1),
        "total_interactions": sums["total_interactions"],
        "usable_interactions": sums["usable_interactions"],
        "reset_count": sums["reset_count"],
        "terminal_count": sums["terminal_count"],
        "no_change_ratio": float(np.mean([row.get("no_change_ratio", 0.0) for row in seed_rows])),
        "unique_transformation_families": sums["transformation_family_count"],
        "stable_contingency_count": sums["stable_contingency_count"],
        "prediction_accuracy": float(np.mean([row.get("prediction_accuracy", 0.0) for row in seed_rows])),
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
        "train_seeds": list(config.train_seeds),
        "test_seed": config.test_seed,
        "run_status": "failed",
        "failure_reason": reason,
        "pass_status": False,
        "forbidden_future_feature_check_passed": True,
        "forbidden_id_feature_check_passed": True,
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
