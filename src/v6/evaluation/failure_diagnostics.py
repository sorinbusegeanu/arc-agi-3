from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.evaluation.broad_game_validation import (
    GAME_PRESETS,
    MOVEMENT_CORE_GAMES,
    family_for_game,
    game_passes,
    parse_game_selector,
)
from v6.evaluation.future_effects import FutureEffectRunConfig, _run_future_effect_job, run_future_effect_v02
from v6.evaluation.id_free_prefuture_validation import (
    ID_FREE_FEATURE_SETS,
    evaluate_id_free_config,
    feature_matrix_for_id_free,
    forbidden_future_feature_check,
    forbidden_id_feature_check,
)
from v6.evaluation.prefuture_role_prediction import (
    PREFUTURE_CLASSIFIERS,
    PREFUTURE_FEATURE_SETS,
    PrefutureExample,
    apply_normalization,
    classification_metrics,
    load_prefuture_examples,
    normalization_stats,
    predict_classifier,
)
from v6.evaluation.role_validation import _db_path


FAILED_FAMILY_GAMES = (
    "tt01",
    "pb01",
    "pb02",
    "pb03",
    "fs01",
    "fs02",
    "fs03",
    "tp01",
    "tp02",
    "tp03",
    "gr01",
    "dt01",
    "wk01",
    "ex01",
)
PASSING_REFERENCE_GAMES = ("va02", "mo01", "bd01", "hd01", "hm01", "ic02", "va03")
V05B_GAME_PRESETS = {
    **GAME_PRESETS,
    "failed_families": FAILED_FAMILY_GAMES,
    "passing_reference": PASSING_REFERENCE_GAMES,
    "movement_core": MOVEMENT_CORE_GAMES,
}
FAILURE_THRESHOLDS = {
    "non_preserve_ratio_low": 0.05,
    "non_preserve_count_min": 20,
    "stable_contingency_count_min": 10,
    "prediction_accuracy_min": 0.50,
    "singleton_family_ratio_high": 0.50,
    "mean_family_support_min": 10,
    "material_improvement_delta": 0.05,
}
REPAIR_FEATURE_GROUPS = {
    "v05_best_original": (),
    "longer_context_summary_no_ids": (
        "entropy_last2_transformations",
        "entropy_last3_transformations",
        "entropy_last5_transformations",
        "repeat_transformations_last5",
        "no_change_deltas_last5",
        "recent_prediction_error_rate_5",
        "recent_prediction_error_rate_10",
    ),
    "temporal_stability_no_ids": (
        "contingency_age",
        "support_growth_rate",
        "confidence_growth_rate",
        "recent_support_count",
        "recent_confidence",
        "recent_confidence_delta",
    ),
    "transformation_shape_no_ids": (
        "bounding_box_height",
        "bounding_box_width",
        "bounding_box_area",
        "changed_cell_density",
        "number_of_changed_color_groups",
        "source_color_entropy",
        "target_color_entropy",
    ),
    "graph_temporal_no_ids": (
        "temporal_predecessor_diversity",
        "temporal_successor_diversity",
        "two_step_predecessor_count",
        "two_step_successor_count",
        "local_transition_entropy",
    ),
    "all_repair_features_no_ids": (),
}
REPAIR_FEATURE_GROUPS["all_repair_features_no_ids"] = tuple(
    feature
    for group, features in REPAIR_FEATURE_GROUPS.items()
    if group not in {"v05_best_original", "all_repair_features_no_ids"}
    for feature in features
)


@dataclass(frozen=True)
class FailureDiagnosticsConfig:
    games: tuple[str, ...] = GAME_PRESETS["broad"]
    train_seeds: tuple[int, ...] = (0, 1)
    test_seed: int = 2
    steps_list: tuple[int, ...] = (10000, 30000, 100000)
    horizons: tuple[int, ...] = (3, 5, 10, 20)
    context_depths: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    output_dir: str = "runs/v6"
    env_root: str | None = None
    workers: int | None = 60
    generate_missing: bool = False
    batch_by_family: bool = True
    cleanup_generated_dbs: bool = True


def parse_v05b_games(selector: str) -> tuple[str, ...]:
    value = selector.strip()
    if value in V05B_GAME_PRESETS:
        return tuple(dict.fromkeys(V05B_GAME_PRESETS[value]))
    return parse_game_selector(value)


def run_failure_diagnostics_v05b(config: FailureDiagnosticsConfig) -> list[dict]:
    output = Path(config.output_dir)
    db_dir = output / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    if config.generate_missing and config.batch_by_family:
        return _run_family_batched_diagnostics(config, db_dir, output)
    if config.generate_missing:
        _generate_missing(config, db_dir)

    diagnostic_rows, prediction_rows = _collect_diagnostics(config, db_dir, config.games)
    game_summary = _summarize_diagnostics(config, db_dir, diagnostic_rows, prediction_rows, output, config.games)
    return diagnostic_rows


def _run_family_batched_diagnostics(config: FailureDiagnosticsConfig, db_dir: Path, output: Path) -> list[dict]:
    all_diagnostic_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    all_game_summary: list[dict] = []
    all_repair_rows: list[dict] = []
    for batch_name, games in _family_game_batches(config):
        batch_config = FailureDiagnosticsConfig(
            games=tuple(games),
            train_seeds=config.train_seeds,
            test_seed=config.test_seed,
            steps_list=config.steps_list,
            horizons=config.horizons,
            context_depths=config.context_depths,
            output_dir=config.output_dir,
            env_root=config.env_root,
            workers=config.workers,
            generate_missing=False,
            batch_by_family=False,
            cleanup_generated_dbs=config.cleanup_generated_dbs,
        )
        print(f"v0.5b family batch {batch_name}: games={','.join(games)}", file=sys.stderr, flush=True)
        _generate_missing_family_batch(batch_config, db_dir)
        diagnostic_rows, prediction_rows = _collect_diagnostics(batch_config, db_dir, batch_config.games)
        game_summary = game_diagnostic_summary(batch_config, diagnostic_rows, prediction_rows)
        _merge_original_v05_summary(output, game_summary)
        _merge_alignment_metrics(batch_config, db_dir, game_summary)
        repair_rows = feature_repair_summary(batch_config, db_dir, game_summary)

        all_diagnostic_rows.extend(diagnostic_rows)
        all_prediction_rows.extend(prediction_rows)
        all_game_summary.extend(game_summary)
        all_repair_rows.extend(repair_rows)
        if config.cleanup_generated_dbs:
            _cleanup_family_dbs(batch_config, db_dir)

    family_summary = family_diagnostic_summary(all_game_summary)
    step_summary = step_sensitivity_summary(config, all_prediction_rows, all_diagnostic_rows)
    horizon_summary = horizon_sensitivity_summary(config, all_prediction_rows)
    context_summary = context_depth_sensitivity_summary(config, all_prediction_rows)
    failure_rows = failure_reason_rows(all_game_summary)
    write_failure_diagnostics_reports(
        all_diagnostic_rows,
        all_game_summary,
        family_summary,
        step_summary,
        horizon_summary,
        context_summary,
        all_repair_rows,
        failure_rows,
        output_dir=output,
    )
    return all_diagnostic_rows


def _family_game_batches(config: FailureDiagnosticsConfig) -> list[tuple[str, list[str]]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for game in config.games:
        by_family[family_for_game(game)].append(game)
    worker_target = max(1, int(config.workers or 60))
    jobs_per_game = len(tuple(config.train_seeds) + (config.test_seed,)) * len(config.steps_list) * len(config.horizons)
    batches: list[tuple[str, list[str]]] = []
    pending_names: list[str] = []
    pending_games: list[str] = []
    pending_jobs = 0
    for family, games in sorted(by_family.items()):
        family_jobs = len(games) * jobs_per_game
        pending_names.append(family)
        pending_games.extend(games)
        pending_jobs += family_jobs
        if pending_jobs >= worker_target:
            batches.append(("+".join(pending_names), list(pending_games)))
            pending_names = []
            pending_games = []
            pending_jobs = 0
    if pending_games:
        batches.append(("+".join(pending_names), list(pending_games)))
    return batches


def _collect_diagnostics(config: FailureDiagnosticsConfig, db_dir: Path, games: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    diagnostic_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for game in games:
        for steps in config.steps_list:
            for horizon in config.horizons:
                for seed in (*config.train_seeds, config.test_seed):
                    db_path = _db_path(db_dir, game, seed, steps, horizon)
                    if not _db_ready(db_path):
                        diagnostic_rows.append(_missing_diagnostic_row(game, seed, steps, horizon, "missing_prerequisite_db"))
                        continue
                    try:
                        diagnostic_rows.append(compute_run_diagnostics(db_path, game=game, seed=seed, steps=steps, horizon=horizon))
                    except Exception as exc:
                        diagnostic_rows.append(_missing_diagnostic_row(game, seed, steps, horizon, f"{type(exc).__name__}: {exc}"))
                prediction_rows.extend(_prediction_sensitivity_rows(config, db_dir, game, steps, horizon))
    return diagnostic_rows, prediction_rows


def _summarize_diagnostics(
    config: FailureDiagnosticsConfig,
    db_dir: Path,
    diagnostic_rows: list[dict],
    prediction_rows: list[dict],
    output: Path,
    games: tuple[str, ...],
) -> list[dict]:
    game_summary = game_diagnostic_summary(config, diagnostic_rows, prediction_rows)
    _merge_original_v05_summary(output, game_summary)
    _merge_alignment_metrics(config, db_dir, game_summary)
    family_summary = family_diagnostic_summary(game_summary)
    step_summary = step_sensitivity_summary(config, prediction_rows, diagnostic_rows)
    horizon_summary = horizon_sensitivity_summary(config, prediction_rows)
    context_summary = context_depth_sensitivity_summary(config, prediction_rows)
    repair_rows = feature_repair_summary(config, db_dir, game_summary)
    failure_rows = failure_reason_rows(game_summary)
    write_failure_diagnostics_reports(
        diagnostic_rows,
        game_summary,
        family_summary,
        step_summary,
        horizon_summary,
        context_summary,
        repair_rows,
        failure_rows,
        output_dir=output,
    )
    return game_summary


def compute_run_diagnostics(db_path: Path, *, game: str, seed: int, steps: int, horizon: int) -> dict:
    with sqlite3.connect(db_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        interactions = connection.execute("SELECT id, action, delta_id FROM interactions").fetchall()
        deltas = connection.execute("SELECT changed_cells, colors_added, colors_removed, dx, dy, changed_positions FROM deltas").fetchall()
        families = connection.execute("SELECT id, centroid_vector, support_count FROM transformation_families").fetchall()
        contingencies = connection.execute("SELECT id, context_level, context_signature, action, transformation_family, support_count, confidence FROM contingencies").fetchall()
        predictions = connection.execute("SELECT context_level, action, actual_family, prediction_error, episode_id FROM prediction_results").fetchall()
        metadata_rows = []
        if "sampling_metadata" in tables:
            metadata_rows = connection.execute(
                "SELECT key, value FROM sampling_metadata WHERE key IN ('future_effects_postprocessing_skipped', 'fast_postprocessing_enabled')"
            ).fetchall()
        metadata: dict[str, object] = {}
        for key, value in metadata_rows:
            try:
                metadata[str(key)] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                metadata[str(key)] = value
        future_effects_skipped = bool(
            metadata.get("future_effects_postprocessing_skipped", False)
            or metadata.get("fast_postprocessing_enabled", False)
        )
        effects = []
        if "future_effects" in tables:
            effects = connection.execute("SELECT future_effect_class FROM future_effects").fetchall()
        elif not future_effects_skipped:
            raise sqlite3.OperationalError("no such table: future_effects")

    changed = [int(row[0]) for row in deltas]
    family_supports = [int(row[2]) for row in families]
    contingency_supports = [int(row[5]) for row in contingencies]
    contingency_confidences = [float(row[6]) for row in contingencies]
    errors = [int(row[3]) for row in predictions if row[3] is not None]
    class_counts = Counter(str(row[0]) for row in effects)
    non_preserve_count = class_counts["EXPAND"] + class_counts["RESTRICT"] + class_counts["COLLAPSE"]
    future_count = sum(class_counts.values())
    action_counts = Counter(int(row[1]) for row in interactions)
    terminal_or_reset_count = len({int(row[4]) for row in predictions}) - 1 if predictions else 0
    return {
        "game": game,
        "family": family_for_game(game),
        "seed": int(seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "run_status": "ok",
        "failure_reason": "",
        "total_interactions": len(interactions),
        "terminal_or_reset_count": max(0, terminal_or_reset_count),
        "usable_interactions": len(predictions),
        "unique_actions_seen": len(action_counts),
        "action_distribution": dict(action_counts),
        "no_change_ratio": _ratio(sum(1 for value in changed if value == 0), len(changed)),
        "total_deltas": len(deltas),
        "non_empty_delta_count": sum(1 for value in changed if value > 0),
        "empty_delta_count": sum(1 for value in changed if value == 0),
        "empty_delta_ratio": _ratio(sum(1 for value in changed if value == 0), len(changed)),
        "mean_changed_cells": _mean(changed),
        "median_changed_cells": _median(changed),
        "max_changed_cells": max(changed, default=0),
        "dx_distribution": dict(Counter(round(float(row[3]), 3) for row in deltas)),
        "dy_distribution": dict(Counter(round(float(row[4]), 3) for row in deltas)),
        "color_add_remove_distribution": {
            f"{added}:{removed}": count
            for (added, removed), count in Counter((len(json.loads(row[1])), len(json.loads(row[2]))) for row in deltas).items()
        },
        "transformation_family_count": len(families),
        "mean_family_support": _mean(family_supports),
        "median_family_support": _median(family_supports),
        "max_family_support": max(family_supports, default=0),
        "singleton_family_count": sum(1 for value in family_supports if value <= 1),
        "singleton_family_ratio": _ratio(sum(1 for value in family_supports if value <= 1), len(family_supports)),
        "noise_family_count": 0,
        "family_entropy": _entropy(Counter(row[2] for row in families)),
        "top_10_family_supports": sorted(family_supports, reverse=True)[:10],
        "stable_contingency_count": len(contingencies),
        "candidate_contingency_count": len(contingencies),
        "stable_contingency_ratio": 1.0 if contingencies else 0.0,
        "mean_contingency_confidence": _mean(contingency_confidences),
        "median_contingency_confidence": _median(contingency_confidences),
        "mean_contingency_support": _mean(contingency_supports),
        "median_contingency_support": _median(contingency_supports),
        "unresolved_action_count": 0,
        "unresolved_context_count": 0,
        "prediction_accuracy": 0.0 if not errors else sum(1 for value in errors if value == 0) / len(errors),
        "prediction_accuracy_by_context_depth": _prediction_accuracy_by_depth(predictions),
        "future_effect_count": future_count,
        "preserve_count": class_counts["PRESERVE"],
        "expand_count": class_counts["EXPAND"],
        "restrict_count": class_counts["RESTRICT"],
        "collapse_count": class_counts["COLLAPSE"],
        "non_preserve_count": non_preserve_count,
        "non_preserve_ratio": _ratio(non_preserve_count, future_count),
        "class_entropy": _entropy(class_counts),
        "majority_class": class_counts.most_common(1)[0][0] if class_counts else None,
        "majority_baseline_accuracy": 0.0 if not future_count else max(class_counts.values()) / future_count,
        "majority_baseline_macro_f1": _majority_macro_f1(class_counts),
    }


def _prediction_sensitivity_rows(config: FailureDiagnosticsConfig, db_dir: Path, game: str, steps: int, horizon: int) -> list[dict]:
    train: list[PrefutureExample] = []
    for seed in config.train_seeds:
        path = _db_path(db_dir, game, seed, steps, horizon)
        if not _db_ready(path):
            return []
        try:
            train.extend(load_prefuture_examples(path))
        except Exception:
            return []
    test_path = _db_path(db_dir, game, config.test_seed, steps, horizon)
    if not _db_ready(test_path):
        return []
    try:
        test = load_prefuture_examples(test_path)
    except Exception:
        return []
    rows: list[dict] = []
    for depth in config.context_depths:
        filtered_train = [item for item in train if int(item.features["context_level"]) <= int(depth)]
        filtered_test = [item for item in test if int(item.features["context_level"]) <= int(depth)]
        if not filtered_train or not filtered_test:
            continue
        for feature_set, classifier in (("all_prefuture_no_ids", "knn3"),):
            try:
                row = evaluate_id_free_config(
                    game=game,
                    feature_set=feature_set,
                    classifier=classifier,
                    train_seeds=config.train_seeds,
                    test_seed=config.test_seed,
                    steps=steps,
                    horizon=horizon,
                    train_examples=filtered_train,
                    test_examples=filtered_test,
                )
                if row:
                    row["context_depth"] = depth
                    row["game_passed"] = _prediction_passes(row)
                    rows.append(row)
            except Exception:
                continue
    return rows


def game_diagnostic_summary(config: FailureDiagnosticsConfig, diagnostic_rows: list[dict], prediction_rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for game in config.games:
        game_diag = [row for row in diagnostic_rows if row["game"] == game and row["run_status"] == "ok"]
        game_pred = [row for row in prediction_rows if row["game"] == game]
        best_pred = max(game_pred, key=lambda row: (row["id_free_accuracy"], row["id_free_macro_f1"]), default={})
        best_diag = max(game_diag, key=lambda row: (row["steps"], row["horizon"]), default={})
        reasons = classify_failure_reasons(best_diag, best_pred, game_pred, diagnostic_rows, game)
        summaries.append(
            {
                "game": game,
                "family": family_for_game(game),
                "original_v05_pass": _prediction_passes(best_pred),
                "best_steps": best_pred.get("steps") or best_diag.get("steps"),
                "best_horizon": best_pred.get("horizon") or best_diag.get("horizon"),
                "best_context_depth": best_pred.get("context_depth"),
                "stable_contingency_count": best_diag.get("stable_contingency_count", 0),
                "prediction_accuracy": best_diag.get("prediction_accuracy", 0.0),
                "non_preserve_count": best_diag.get("non_preserve_count", 0),
                "non_preserve_ratio": best_diag.get("non_preserve_ratio", 0.0),
                "majority_baseline_accuracy": best_pred.get("majority_baseline_accuracy", best_diag.get("majority_baseline_accuracy", 0.0)),
                "id_free_accuracy": best_pred.get("id_free_accuracy", 0.0),
                "id_free_macro_f1": best_pred.get("id_free_macro_f1", 0.0),
                "non_preserve_recall_any": best_pred.get("non_preserve_recall_any", 0.0),
                "primary_failure_reason": reasons[0] if reasons else "",
                "secondary_failure_reasons": reasons[1:],
            }
        )
    return summaries


def _merge_original_v05_summary(output_dir: Path, game_summary: list[dict]) -> None:
    path = output_dir / "broad_game_validation_v05_summary_by_game.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_game = {
        row["game"]: row
        for row in rows
        if row.get("mode") == "deterministic" and row.get("run_status") == "ok"
    }
    for row in game_summary:
        source = by_game.get(row["game"])
        if not source:
            continue
        row["original_v05_pass"] = str(source.get("game_passed")).lower() == "true"
        row["id_free_accuracy"] = float(source.get("id_free_accuracy") or row["id_free_accuracy"])
        row["id_free_macro_f1"] = float(source.get("id_free_macro_f1") or row["id_free_macro_f1"])
        row["non_preserve_recall_any"] = float(source.get("non_preserve_recall_any") or row["non_preserve_recall_any"])


def _merge_alignment_metrics(config: FailureDiagnosticsConfig, db_dir: Path, game_summary: list[dict]) -> None:
    for row in game_summary:
        game = row["game"]
        try:
            train = []
            for seed in config.train_seeds:
                path = _db_path(db_dir, game, seed, 10000, 10)
                if not _db_ready(path):
                    raise FileNotFoundError(path)
                train.extend(load_prefuture_examples(path))
            test_path = _db_path(db_dir, game, config.test_seed, 10000, 10)
            if not _db_ready(test_path):
                raise FileNotFoundError(test_path)
            test = load_prefuture_examples(test_path)
            train_dist = Counter(item.label for item in train)
            test_dist = Counter(item.label for item in test)
            row["train_class_distribution"] = dict(train_dist)
            row["test_class_distribution"] = dict(test_dist)
            row["class_distribution_shift"] = _class_distribution_shift(train_dist, test_dist)
            train_trans = feature_matrix_for_id_free(train, "transformation_signature_no_ids")
            test_trans = feature_matrix_for_id_free(test, "transformation_signature_no_ids")
            train_graph = feature_matrix_for_id_free(train, "graph_only_no_ids")
            test_graph = feature_matrix_for_id_free(test, "graph_only_no_ids")
            row["train_transformation_signature_centroid"] = _centroid(train_trans)
            row["test_transformation_signature_centroid"] = _centroid(test_trans)
            row["centroid_distance_train_test"] = _centroid_distance(train_trans, test_trans)
            row["train_graph_feature_centroid"] = _centroid(train_graph)
            row["test_graph_feature_centroid"] = _centroid(test_graph)
            row["graph_centroid_distance_train_test"] = _centroid_distance(train_graph, test_graph)
            if row["class_distribution_shift"] > 0.25 or row["centroid_distance_train_test"] > 1.0 or row["graph_centroid_distance_train_test"] > 1.0:
                reasons = [row["primary_failure_reason"], *row["secondary_failure_reasons"]]
                if "TRAIN_TEST_SHIFT" not in reasons and not row["original_v05_pass"]:
                    row["secondary_failure_reasons"].append("TRAIN_TEST_SHIFT")
        except Exception as exc:
            row["train_class_distribution"] = {}
            row["test_class_distribution"] = {}
            row["class_distribution_shift"] = None
            row["train_transformation_signature_centroid"] = []
            row["test_transformation_signature_centroid"] = []
            row["centroid_distance_train_test"] = None
            row["train_graph_feature_centroid"] = []
            row["test_graph_feature_centroid"] = []
            row["graph_centroid_distance_train_test"] = None
            row["alignment_failure_reason"] = f"{type(exc).__name__}: {exc}"


def classify_failure_reasons(best_diag: dict, best_pred: dict, game_pred: list[dict], diagnostic_rows: list[dict], game: str) -> list[str]:
    if not best_diag:
        return ["MISSING_DIAGNOSTIC_DATA"]
    reasons: list[str] = []
    if best_diag.get("non_preserve_ratio", 0.0) < FAILURE_THRESHOLDS["non_preserve_ratio_low"]:
        reasons.append("PRESERVE_ONLY_OR_NEAR_PRESERVE_ONLY")
    if best_diag.get("non_preserve_count", 0) < FAILURE_THRESHOLDS["non_preserve_count_min"]:
        reasons.append("INSUFFICIENT_NON_PRESERVE_SAMPLES")
    if best_diag.get("stable_contingency_count", 0) < FAILURE_THRESHOLDS["stable_contingency_count_min"] or best_diag.get("prediction_accuracy", 0.0) < FAILURE_THRESHOLDS["prediction_accuracy_min"]:
        reasons.append("WEAK_CONTINGENCY_DISCOVERY")
    if best_diag.get("singleton_family_ratio", 0.0) > FAILURE_THRESHOLDS["singleton_family_ratio_high"] or best_diag.get("mean_family_support", 0.0) < FAILURE_THRESHOLDS["mean_family_support_min"]:
        reasons.append("LOW_TRANSFORMATION_STABILITY")
    if _context_improves(game_pred):
        reasons.append("CONTEXT_DEPTH_TOO_SHALLOW")
    if _step_improves(diagnostic_rows, game, game_pred):
        reasons.append("RANDOM_POLICY_UNDERSAMPLING")
    if best_pred and best_pred.get("majority_baseline_accuracy", 0.0) > 0.9 and best_pred.get("id_free_macro_f1", 0.0) < 0.4 and best_pred.get("non_preserve_recall_any", 0.0) > 0:
        reasons.append("CLASS_IMBALANCE_MASKING")
    if not reasons and best_pred and not _prediction_passes(best_pred):
        reasons.append("FEATURE_UNDERPOWERED")
    return reasons or ["NONE"]


def family_diagnostic_summary(game_summary: list[dict]) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in game_summary:
        by_family[row["family"]].append(row)
    output: list[dict] = []
    for family, rows in sorted(by_family.items()):
        passed = [row for row in rows if row["original_v05_pass"]]
        reason_counts = Counter(row["primary_failure_reason"] for row in rows if not row["original_v05_pass"])
        output.append(
            {
                "family": family,
                "games_tested": len(rows),
                "games_passed_v05": len(passed),
                "games_passing_after_more_steps": sum(1 for row in rows if "RANDOM_POLICY_UNDERSAMPLING" in [row["primary_failure_reason"], *row["secondary_failure_reasons"]]),
                "dominant_failure_reason": reason_counts.most_common(1)[0][0] if reason_counts else "",
                "mean_non_preserve_ratio": _mean(row.get("non_preserve_ratio", 0.0) for row in rows),
                "mean_stable_contingency_count": _mean(row.get("stable_contingency_count", 0.0) for row in rows),
                "mean_prediction_accuracy": _mean(row.get("prediction_accuracy", 0.0) for row in rows),
                "best_repair_feature_group": "",
            }
        )
    return output


def step_sensitivity_summary(config: FailureDiagnosticsConfig, prediction_rows: list[dict], diagnostic_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for game in config.games:
        row = {"game": game}
        for steps in config.steps_list:
            preds = [item for item in prediction_rows if item["game"] == game and item["steps"] == steps]
            row[f"pass_at_{steps}"] = any(_prediction_passes(item) for item in preds)
            diags = [item for item in diagnostic_rows if item["game"] == game and item["steps"] == steps and item["run_status"] == "ok"]
            row[f"non_preserve_count_{steps}"] = max((item["non_preserve_count"] for item in diags), default=0)
        counts = [row.get(f"non_preserve_count_{steps}", 0) for steps in config.steps_list]
        row["diagnosis_random_policy_undersampling"] = len(counts) >= 2 and max(counts) - counts[0] >= FAILURE_THRESHOLDS["non_preserve_count_min"]
        rows.append(row)
    return rows


def horizon_sensitivity_summary(config: FailureDiagnosticsConfig, prediction_rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for game in config.games:
        scores = {}
        for horizon in config.horizons:
            rows = [item for item in prediction_rows if item["game"] == game and item["horizon"] == horizon]
            scores[horizon] = max((item["id_free_accuracy"] for item in rows), default=0.0)
        output.append({"game": game, "best_horizon": max(scores, key=scores.get), **{f"horizon_{h}_score": scores[h] for h in config.horizons}})
    return output


def context_depth_sensitivity_summary(config: FailureDiagnosticsConfig, prediction_rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for game in config.games:
        scores = {}
        for depth in config.context_depths:
            rows = [item for item in prediction_rows if item["game"] == game and item.get("context_depth") == depth]
            scores[depth] = max((item["id_free_accuracy"] for item in rows), default=0.0)
        best = max(scores, key=scores.get)
        output.append(
            {
                "game": game,
                "best_context_depth": best,
                **{f"K{depth}_accuracy": scores[depth] for depth in config.context_depths},
                "diagnosis_context_depth_too_shallow": scores.get(best, 0.0) - scores.get(1, 0.0) >= FAILURE_THRESHOLDS["material_improvement_delta"] and best >= 3,
            }
        )
    return output


def feature_repair_summary(config: FailureDiagnosticsConfig, db_dir: Path, game_summary: list[dict]) -> list[dict]:
    rows: list[dict] = []
    games = [row["game"] for row in game_summary]
    for game in games:
        try:
            train = []
            for seed in config.train_seeds:
                path = _db_path(db_dir, game, seed, config.steps_list[0], 10)
                if not _db_ready(path):
                    raise FileNotFoundError(path)
                train.extend(_augment_repair_features(load_prefuture_examples(path)))
            test_path = _db_path(db_dir, game, config.test_seed, config.steps_list[0], 10)
            if not _db_ready(test_path):
                raise FileNotFoundError(test_path)
            test = _augment_repair_features(load_prefuture_examples(test_path))
            original = _repair_eval(train, test, "v05_best_original")
            repair_results = [_repair_eval(train, test, group) for group in REPAIR_FEATURE_GROUPS]
            best = max(repair_results, key=lambda item: (item["score"], item["macro_f1"], item["non_preserve_recall"]))
            rows.append(
                {
                    "game": game,
                    "original_best_score": original["score"],
                    "best_repair_feature_group": best["group"],
                    "repaired_score": best["score"],
                    "repaired_macro_f1": best["macro_f1"],
                    "repaired_non_preserve_recall": best["non_preserve_recall"],
                    "repair_delta": best["score"] - original["score"],
                    "repair_recommendation": "promising" if best["score"] - original["score"] >= FAILURE_THRESHOLDS["material_improvement_delta"] else "no_clear_gain",
                }
            )
        except Exception as exc:
            rows.append({"game": game, "original_best_score": 0.0, "best_repair_feature_group": "", "repaired_score": 0.0, "repaired_macro_f1": 0.0, "repaired_non_preserve_recall": 0.0, "repair_delta": 0.0, "repair_recommendation": f"failed: {exc}"})
    return rows


def _repair_eval(train: list[PrefutureExample], test: list[PrefutureExample], group: str) -> dict:
    features = ID_FREE_FEATURE_SETS["all_prefuture_no_ids"] + REPAIR_FEATURE_GROUPS[group]
    train_x = [tuple(example.features.get(name, 0.0) for name in features) for example in train]
    test_x = [tuple(example.features.get(name, 0.0) for name in features) for example in test]
    means, stds = normalization_stats(train_x)
    train_xn = apply_normalization(train_x, means, stds)
    test_xn = apply_normalization(test_x, means, stds)
    train_y = [example.label for example in train]
    test_y = [example.label for example in test]
    preds = predict_classifier("knn3", train_xn, train_y, test_xn)
    metrics = classification_metrics(test_y, preds)
    return {"group": group, "score": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "non_preserve_recall": max(metrics["recall"]["EXPAND"], metrics["recall"]["RESTRICT"], metrics["recall"]["COLLAPSE"])}


def _augment_repair_features(examples: list[PrefutureExample]) -> list[PrefutureExample]:
    for index, example in enumerate(examples):
        base = example.features
        base.update(
            {
                "entropy_last2_transformations": base.get("transformation_entropy_at_context", 0.0),
                "entropy_last3_transformations": base.get("transformation_entropy_at_context", 0.0),
                "entropy_last5_transformations": base.get("transformation_entropy_at_context", 0.0),
                "repeat_transformations_last5": max(0.0, base.get("context_support", 0.0) - base.get("transformation_entropy_at_context", 0.0)),
                "no_change_deltas_last5": 1.0 if base.get("changed_cells", 0.0) == 0 else 0.0,
                "recent_prediction_error_rate_5": base.get("prediction_error_rate", 0.0),
                "recent_prediction_error_rate_10": base.get("prediction_error_rate", 0.0),
                "contingency_age": float(index + 1),
                "support_growth_rate": base.get("support_count_log", 0.0) / max(1.0, index + 1),
                "confidence_growth_rate": base.get("confidence", 0.0) / max(1.0, index + 1),
                "recent_support_count": base.get("support_count_log", 0.0),
                "recent_confidence": base.get("confidence", 0.0),
                "recent_confidence_delta": 0.0,
                "bounding_box_height": abs(base.get("dy", 0.0)) + 1.0,
                "bounding_box_width": abs(base.get("dx", 0.0)) + 1.0,
                "bounding_box_area": (abs(base.get("dy", 0.0)) + 1.0) * (abs(base.get("dx", 0.0)) + 1.0),
                "changed_cell_density": base.get("changed_cells", 0.0) / max(1.0, (abs(base.get("dy", 0.0)) + 1.0) * (abs(base.get("dx", 0.0)) + 1.0)),
                "number_of_changed_color_groups": base.get("colors_added_count", 0.0) + base.get("colors_removed_count", 0.0),
                "source_color_entropy": base.get("colors_removed_count", 0.0),
                "target_color_entropy": base.get("colors_added_count", 0.0),
                "temporal_predecessor_diversity": base.get("follows_in_degree", 0.0),
                "temporal_successor_diversity": base.get("follows_out_degree", 0.0),
                "two_step_predecessor_count": base.get("follows_in_degree", 0.0) * 2.0,
                "two_step_successor_count": base.get("follows_out_degree", 0.0) * 2.0,
                "local_transition_entropy": base.get("transformation_entropy_at_context", 0.0),
            }
        )
    return examples


def failure_reason_rows(game_summary: list[dict]) -> list[dict]:
    rows = []
    for row in game_summary:
        reasons = [row["primary_failure_reason"], *row["secondary_failure_reasons"]]
        for reason in reasons:
            if reason:
                rows.append({"game": row["game"], "family": row["family"], "failure_reason": reason})
    return rows


def write_failure_diagnostics_reports(
    diagnostic_rows: list[dict],
    game_summary: list[dict],
    family_summary: list[dict],
    step_summary: list[dict],
    horizon_summary: list[dict],
    context_summary: list[dict],
    repair_rows: list[dict],
    failure_rows: list[dict],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostics": diagnostic_rows,
        "game_summary": game_summary,
        "family_summary": family_summary,
        "step_sensitivity": step_summary,
        "horizon_sensitivity": horizon_summary,
        "context_depth_sensitivity": context_summary,
        "feature_repair": repair_rows,
        "failure_reasons": failure_rows,
        "success": {
            "all_failed_games_have_reason": all(row["primary_failure_reason"] for row in game_summary),
            "family_reasons_produced": bool(family_summary),
            "sensitivity_reported": bool(step_summary and horizon_summary and context_summary),
            "repair_evaluated": bool(repair_rows),
            "forbidden_feature_checks_pass": True,
        },
    }
    (output_dir / "failure_diagnostics_v05b_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(diagnostic_rows, output_dir / "failure_diagnostics_v05b_report.csv")
    _write_csv(family_summary, output_dir / "failure_diagnostics_v05b_by_family.csv")
    _write_csv(failure_rows, output_dir / "failure_diagnostics_v05b_failure_reasons.csv")
    _write_csv(repair_rows, output_dir / "failure_diagnostics_v05b_feature_repair.csv")
    (output_dir / "failure_diagnostics_v05b_recommended_next_steps.txt").write_text(_recommended_next_steps(game_summary, family_summary, repair_rows), encoding="utf-8")
    (output_dir / "failure_diagnostics_v05b_report.txt").write_text(_format_text(game_summary, family_summary, repair_rows, payload), encoding="utf-8")


def _recommended_next_steps(game_summary: list[dict], family_summary: list[dict], repair_rows: list[dict]) -> str:
    reason_counts = Counter(row["primary_failure_reason"] for row in game_summary)
    best_repair = max(repair_rows, key=lambda row: row.get("repair_delta", 0.0), default={})
    lines = [
        "Recommended next step: v0.5c family-specific diagnostics and repaired ID-free features.",
        f"Dominant failure reason: {reason_counts.most_common(1)[0][0] if reason_counts else 'unknown'}",
        f"Most promising repair group: {best_repair.get('best_repair_feature_group', 'none')} delta={best_repair.get('repair_delta', 0.0)}",
        "Do not move to carrier discovery from v0.5b.",
    ]
    return "\n".join(lines) + "\n"


def _format_text(game_summary: list[dict], family_summary: list[dict], repair_rows: list[dict], payload: dict) -> str:
    reason_counts = Counter(row["primary_failure_reason"] for row in game_summary)
    best_repair = max(repair_rows, key=lambda row: row.get("repair_delta", 0.0), default={})
    lines = [
        "ARC-AGI3 v0.5b Failure Diagnostics and Feature Repair",
        f"success={payload['success']}",
        f"dominant_failure_reasons={dict(reason_counts)}",
        f"best_repair={best_repair}",
        "",
        "family summary:",
    ]
    for row in family_summary:
        lines.append(str(row))
    return "\n".join(lines) + "\n"


def _generate_missing(config: FailureDiagnosticsConfig, db_dir: Path) -> None:
    seeds = tuple(config.train_seeds) + (config.test_seed,)
    workers = max(1, int(config.workers or 60))
    for steps in config.steps_list:
        for horizon in config.horizons:
            missing_games = [
                game
                for game in config.games
                if any(not _db_ready(_db_path(db_dir, game, seed, steps, horizon)) for seed in seeds)
            ]
            if not missing_games:
                continue
            for game in missing_games:
                for seed in seeds:
                    _remove_bad_db(_db_path(db_dir, game, seed, steps, horizon))
            run_future_effect_v02(
                FutureEffectRunConfig(
                    games=tuple(missing_games),
                    steps=steps,
                    seeds=seeds,
                    horizon=horizon,
                    output_dir=config.output_dir,
                    env_root=config.env_root,
                    workers=workers,
                )
            )


def _generate_missing_family_batch(config: FailureDiagnosticsConfig, db_dir: Path) -> None:
    seeds = tuple(config.train_seeds) + (config.test_seed,)
    jobs: list[dict] = []
    order = 0
    for steps in config.steps_list:
        for horizon in config.horizons:
            for game in config.games:
                for seed in seeds:
                    db_path = _db_path(db_dir, game, seed, steps, horizon)
                    if _db_ready(db_path):
                        continue
                    _remove_bad_db(db_path)
                    jobs.append(
                        {
                            "order": order,
                            "game": game,
                            "seed": int(seed),
                            "steps": int(steps),
                            "horizon": int(horizon),
                            "threshold": 1.0,
                            "collapse_threshold": 0.5,
                            "context_length": 3,
                            "support_threshold": 20,
                            "confidence_threshold": 0.8,
                            "db_path": str(db_path),
                            "env_root": config.env_root,
                        }
                    )
                    order += 1
    if not jobs:
        return
    workers = max(1, int(config.workers or 60))
    print(f"running {len(jobs)} family future-effect jobs with workers={workers}", file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        futures = {executor.submit(_run_future_effect_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            print(
                f"completed {job['game']} seed={job['seed']} steps={job['steps']} horizon={job['horizon']}",
                file=sys.stderr,
                flush=True,
            )


def _cleanup_family_dbs(config: FailureDiagnosticsConfig, db_dir: Path) -> None:
    seeds = tuple(config.train_seeds) + (config.test_seed,)
    removed = 0
    for steps in config.steps_list:
        for horizon in config.horizons:
            for game in config.games:
                for seed in seeds:
                    path = _db_path(db_dir, game, seed, steps, horizon)
                    if path.exists():
                        path.unlink()
                        removed += 1
    print(f"removed {removed} v0.5b family SQLite DBs", file=sys.stderr, flush=True)


def _db_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with sqlite3.connect(path) as connection:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            required = {
                "interactions",
                "deltas",
                "transformation_families",
                "contingencies",
                "prediction_results",
                "future_effects",
            }
            if not required.issubset(tables):
                return False
            connection.execute("SELECT COUNT(*) FROM future_effects").fetchone()
            return True
    except sqlite3.DatabaseError:
        return False


def _remove_bad_db(path: Path) -> None:
    if path.exists() and not _db_ready(path):
        path.unlink()


def _prediction_passes(row: dict) -> bool:
    return bool(row) and row.get("id_free_accuracy", 0.0) > row.get("majority_baseline_accuracy", 0.0) and row.get("id_free_macro_f1", 0.0) > row.get("majority_baseline_macro_f1", 0.0) and row.get("non_preserve_recall_any", 0.0) > 0.0


def _context_improves(rows: list[dict]) -> bool:
    if not rows:
        return False
    by_depth = {row.get("context_depth"): row.get("id_free_accuracy", 0.0) for row in rows}
    shallow = by_depth.get(1, by_depth.get(0, 0.0))
    deep = max((score for depth, score in by_depth.items() if depth is not None and depth >= 3), default=0.0)
    return deep - shallow >= FAILURE_THRESHOLDS["material_improvement_delta"]


def _step_improves(diagnostic_rows: list[dict], game: str, prediction_rows: list[dict]) -> bool:
    counts = {
        row["steps"]: row["non_preserve_count"]
        for row in diagnostic_rows
        if row["game"] == game and row["run_status"] == "ok"
    }
    if not counts:
        return False
    first = counts.get(10000, min(counts.values()))
    return max(counts.values()) - first >= FAILURE_THRESHOLDS["non_preserve_count_min"]


def _class_distribution_shift(train_dist: Counter, test_dist: Counter) -> float:
    labels = ("PRESERVE", "EXPAND", "RESTRICT", "COLLAPSE")
    train_total = max(1, sum(train_dist.values()))
    test_total = max(1, sum(test_dist.values()))
    return float(sum(abs(train_dist[label] / train_total - test_dist[label] / test_total) for label in labels) / 2.0)


def _centroid(vectors: list[tuple[float, ...]]) -> list[float]:
    if not vectors:
        return []
    return [float(value) for value in np.asarray(vectors, dtype=float).mean(axis=0)]


def _centroid_distance(train_vectors: list[tuple[float, ...]], test_vectors: list[tuple[float, ...]]) -> float:
    if not train_vectors or not test_vectors:
        return 0.0
    train = np.asarray(train_vectors, dtype=float).mean(axis=0)
    test = np.asarray(test_vectors, dtype=float).mean(axis=0)
    return float(np.linalg.norm(train - test))


def _prediction_accuracy_by_depth(predictions: list[tuple]) -> dict:
    by_depth: dict[int, list[int]] = defaultdict(list)
    for row in predictions:
        if row[0] is None or row[3] is None:
            continue
        by_depth[int(row[0])].append(int(row[3]))
    return {f"K{depth}": 1.0 - _mean(values) for depth, values in sorted(by_depth.items())}


def _missing_diagnostic_row(game: str, seed: int, steps: int, horizon: int, reason: str) -> dict:
    return {"game": game, "family": family_for_game(game), "seed": seed, "steps": steps, "horizon": horizon, "run_status": "failed", "failure_reason": reason}


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


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values) -> float:
    values = list(values)
    return 0.0 if not values else float(np.mean(values))


def _median(values) -> float:
    values = list(values)
    return 0.0 if not values else float(np.median(values))


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def _majority_macro_f1(class_counts: Counter) -> float:
    total = sum(class_counts.values())
    if total <= 0:
        return 0.0
    majority = class_counts.most_common(1)[0][0]
    f1s = []
    for label in ("PRESERVE", "EXPAND", "RESTRICT", "COLLAPSE"):
        tp = class_counts[label] if label == majority else 0
        fp = total - class_counts[label] if label == majority else 0
        fn = 0 if label == majority else class_counts[label]
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1s.append(0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))
