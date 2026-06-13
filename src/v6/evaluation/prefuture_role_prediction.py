from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.evaluation.future_effects import FutureEffectRunConfig, load_future_effects, run_future_effect_v02
from v6.evaluation.role_candidates import FUTURE_EFFECT_CLASSES, ROLE_DISCOVERY_GAMES
from v6.evaluation.role_validation import _db_path


FORBIDDEN_FEATURE_NAMES = {
    "future_effect_class_id",
    "mean_fo_before",
    "mean_fo_after",
    "mean_delta_fo",
    "median_delta_fo",
    "std_delta_fo",
    "positive_delta_ratio",
    "negative_delta_ratio",
    "zero_delta_ratio",
    "collapse_ratio",
    "FO_after",
    "FO_before",
    "delta_fo",
}
PREFUTURE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "contingency_only": (
        "context_level",
        "confidence",
        "support_count_log",
        "prediction_error_rate",
        "context_support",
        "action_entropy_at_context",
        "transformation_entropy_at_context",
    ),
    "transformation_signature": (
        "confidence",
        "support_count_log",
        "transformation_family_support_log",
        "changed_cells",
        "dx",
        "dy",
        "colors_added_count",
        "colors_removed_count",
    ),
    "graph_only": (
        "contingency_in_degree",
        "contingency_out_degree",
        "follows_in_degree",
        "follows_out_degree",
        "cooccurrence_degree",
        "clustering_coefficient",
        "degree_centrality",
    ),
    "contingency_plus_graph": (),
    "contingency_plus_transformation": (),
    "all_prefuture_no_ids": (),
    "all_prefuture_with_ids": (),
}
PREFUTURE_FEATURE_SETS["contingency_plus_graph"] = PREFUTURE_FEATURE_SETS["contingency_only"] + PREFUTURE_FEATURE_SETS["graph_only"]
PREFUTURE_FEATURE_SETS["contingency_plus_transformation"] = PREFUTURE_FEATURE_SETS["contingency_only"] + PREFUTURE_FEATURE_SETS["transformation_signature"]
PREFUTURE_FEATURE_SETS["all_prefuture_no_ids"] = (
    PREFUTURE_FEATURE_SETS["contingency_only"]
    + PREFUTURE_FEATURE_SETS["transformation_signature"]
    + PREFUTURE_FEATURE_SETS["graph_only"]
)
PREFUTURE_FEATURE_SETS["all_prefuture_with_ids"] = PREFUTURE_FEATURE_SETS["all_prefuture_no_ids"] + (
    "action_id",
    "transformation_family_id",
)
PREFUTURE_CLASSIFIERS = ("nearest_centroid", "knn3", "logistic_regression")
PRIMARY_GAMES = {"va02", "mo01", "ic01"}


@dataclass(frozen=True)
class PrefutureConfig:
    games: tuple[str, ...] = ROLE_DISCOVERY_GAMES
    train_seeds: tuple[int, ...] = (0, 1)
    test_seed: int = 2
    steps: int = 10000
    horizon: int = 10
    threshold: float = 1.0
    collapse_threshold: float = 0.5
    context_length: int = 3
    support_threshold: int = 20
    confidence_threshold: float = 0.8
    output_dir: str = "runs/v6"
    env_root: str | None = None
    workers: int | None = None
    reuse_v02: bool = True


@dataclass(frozen=True)
class PrefutureExample:
    contingency_id: int
    contingency_key: tuple
    features: dict[str, float]
    label: str


def run_prefuture_role_prediction_v04c(config: PrefutureConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = tuple(config.train_seeds) + (int(config.test_seed),)
    expected = [_db_path(db_dir, game, seed, config.steps, config.horizon) for game in config.games for seed in all_seeds]
    if not config.reuse_v02 or any(not path.exists() for path in expected):
        print("running prerequisite v0.2 future-effect jobs", file=sys.stderr, flush=True)
        run_future_effect_v02(
            FutureEffectRunConfig(
                games=config.games,
                steps=config.steps,
                seeds=all_seeds,
                horizon=config.horizon,
                threshold=config.threshold,
                collapse_threshold=config.collapse_threshold,
                context_length=config.context_length,
                support_threshold=config.support_threshold,
                confidence_threshold=config.confidence_threshold,
                output_dir=config.output_dir,
                env_root=config.env_root,
                workers=config.workers,
            )
        )

    rows: list[dict] = []
    for game in config.games:
        train_examples: list[PrefutureExample] = []
        for seed in config.train_seeds:
            train_examples.extend(load_prefuture_examples(_db_path(db_dir, game, seed, config.steps, config.horizon)))
        test_examples = load_prefuture_examples(_db_path(db_dir, game, config.test_seed, config.steps, config.horizon))
        for feature_set in PREFUTURE_FEATURE_SETS:
            for classifier in PREFUTURE_CLASSIFIERS:
                row = evaluate_prefuture_classifier(
                    game=game,
                    feature_set=feature_set,
                    classifier=classifier,
                    train_seeds=config.train_seeds,
                    test_seed=config.test_seed,
                    steps=config.steps,
                    horizon=config.horizon,
                    train_examples=train_examples,
                    test_examples=test_examples,
                )
                if row is not None:
                    rows.append(row)
    write_prefuture_reports(rows, output_dir=output_dir)
    return rows


def evaluate_prefuture_classifier(
    *,
    game: str,
    feature_set: str,
    classifier: str,
    train_seeds: tuple[int, ...],
    test_seed: int,
    steps: int,
    horizon: int,
    train_examples: list[PrefutureExample],
    test_examples: list[PrefutureExample],
) -> dict | None:
    forbidden_ok = forbidden_feature_check(feature_set)
    train_x = feature_matrix(train_examples, feature_set)
    test_x = feature_matrix(test_examples, feature_set)
    means, stds = normalization_stats(train_x)
    train_xn = apply_normalization(train_x, means, stds)
    test_xn = apply_normalization(test_x, means, stds)
    train_y = [example.label for example in train_examples]
    test_y = [example.label for example in test_examples]
    if classifier == "logistic_regression" and len(set(train_y)) < 2:
        return None

    predictions = predict_classifier(classifier, train_xn, train_y, test_xn)
    majority = majority_label(train_y)
    baseline_predictions = [majority for _ in test_y]
    contingency_predictions = contingency_baseline_predictions(train_examples, test_examples, majority)
    stratified_predictions = stratified_predictions_from_train(train_y, len(test_y))
    metrics = classification_metrics(test_y, predictions)
    majority_metrics = classification_metrics(test_y, baseline_predictions)
    baseline_accuracy = accuracy(test_y, baseline_predictions)
    contingency_accuracy = accuracy(test_y, contingency_predictions)
    prefuture_accuracy = accuracy(test_y, predictions)
    return {
        "game": game,
        "mode": "prefuture",
        "feature_set": feature_set,
        "classifier": classifier,
        "train_seeds": list(train_seeds),
        "test_seed": int(test_seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "train_sample_count": len(train_examples),
        "test_sample_count": len(test_examples),
        "class_distribution_train": dict(Counter(train_y)),
        "class_distribution_test": dict(Counter(test_y)),
        "baseline_accuracy": baseline_accuracy,
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "contingency_baseline_accuracy": contingency_accuracy,
        "random_stratified_baseline_accuracy": accuracy(test_y, stratified_predictions),
        "prefuture_role_accuracy": prefuture_accuracy,
        "prefuture_role_macro_f1": metrics["macro_f1"],
        "prefuture_role_vs_baseline_delta": prefuture_accuracy - baseline_accuracy,
        "prefuture_role_vs_contingency_delta": prefuture_accuracy - contingency_accuracy,
        "preserve_precision": metrics["precision"]["PRESERVE"],
        "preserve_recall": metrics["recall"]["PRESERVE"],
        "expand_precision": metrics["precision"]["EXPAND"],
        "expand_recall": metrics["recall"]["EXPAND"],
        "restrict_precision": metrics["precision"]["RESTRICT"],
        "restrict_recall": metrics["recall"]["RESTRICT"],
        "collapse_precision": metrics["precision"]["COLLAPSE"],
        "collapse_recall": metrics["recall"]["COLLAPSE"],
        "confusion_matrix_json": metrics["confusion_matrix"],
        "forbidden_feature_check_passed": forbidden_ok,
    }


def load_prefuture_examples(db_path: Path) -> list[PrefutureExample]:
    with sqlite3.connect(db_path) as connection:
        effects = load_future_effects(connection)
        contingencies = _load_contingencies(connection)
        families = _load_families(connection)
        prediction_rows = _load_prediction_rows(connection)
    prediction_stats = _prediction_stats(prediction_rows)
    graph_stats = _graph_stats(contingencies, prediction_rows)
    examples: list[PrefutureExample] = []
    for effect in effects:
        contingency = contingencies.get(int(effect.contingency_id))
        if contingency is None:
            continue
        family = families.get(int(contingency["transformation_family"]), {})
        context_key = tuple(contingency["context_signature"])
        contingency_key = (
            int(contingency["context_level"]),
            context_key,
            int(contingency["action"]),
            int(contingency["transformation_family"]),
        )
        context_stats = prediction_stats["contexts"].get(context_key, {})
        cont_stats = prediction_stats["contingencies"].get(contingency_key, {})
        graph = graph_stats.get(int(effect.contingency_id), {})
        features = {
            "context_level": float(contingency["context_level"]),
            "action_id": float(contingency["action"]),
            "transformation_family_id": float(contingency["transformation_family"]),
            "confidence": float(contingency["confidence"]),
            "support_count_log": math.log1p(float(contingency["support_count"])),
            "prediction_error_rate": float(cont_stats.get("prediction_error_rate", 0.0)),
            "context_support": float(context_stats.get("support", 0.0)),
            "action_entropy_at_context": float(context_stats.get("action_entropy", 0.0)),
            "transformation_entropy_at_context": float(context_stats.get("family_entropy", 0.0)),
            "transformation_family_support_log": math.log1p(float(family.get("support_count", 0.0))),
            "changed_cells": float(family.get("changed_cells", 0.0)),
            "dx": float(family.get("dx", 0.0)),
            "dy": float(family.get("dy", 0.0)),
            "colors_added_count": float(family.get("colors_added_count", 0.0)),
            "colors_removed_count": float(family.get("colors_removed_count", 0.0)),
            "contingency_in_degree": float(graph.get("contingency_in_degree", 0.0)),
            "contingency_out_degree": float(graph.get("contingency_out_degree", 0.0)),
            "follows_in_degree": float(graph.get("follows_in_degree", 0.0)),
            "follows_out_degree": float(graph.get("follows_out_degree", 0.0)),
            "cooccurrence_degree": float(graph.get("cooccurrence_degree", 0.0)),
            "clustering_coefficient": float(graph.get("clustering_coefficient", 0.0)),
            "degree_centrality": float(graph.get("degree_centrality", 0.0)),
            "pagerank": float(graph.get("pagerank", graph.get("degree_centrality", 0.0))),
        }
        examples.append(
            PrefutureExample(
                contingency_id=int(effect.contingency_id),
                contingency_key=contingency_key,
                features=features,
                label=str(effect.future_effect_class),
            )
        )
    return examples


def feature_matrix(examples: list[PrefutureExample], feature_set: str) -> list[tuple[float, ...]]:
    if feature_set not in PREFUTURE_FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    return [
        tuple(
            float(example.features.get("pagerank", example.features.get("degree_centrality", 0.0)))
            if name == "pagerank"
            else float(example.features[name])
            for name in PREFUTURE_FEATURE_SETS[feature_set]
        )
        for example in examples
    ]


def forbidden_feature_check(feature_set: str) -> bool:
    names = set(PREFUTURE_FEATURE_SETS[feature_set])
    return not bool(names & FORBIDDEN_FEATURE_NAMES)


def normalization_stats(vectors: list[tuple[float, ...]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not vectors:
        return (), ()
    matrix = np.asarray(vectors, dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0.0] = 1.0
    return tuple(float(value) for value in means), tuple(float(value) for value in stds)


def apply_normalization(vectors: list[tuple[float, ...]], means: tuple[float, ...], stds: tuple[float, ...]) -> np.ndarray:
    if not vectors:
        return np.empty((0, len(means)), dtype=float)
    matrix = np.asarray(vectors, dtype=float)
    return (matrix - np.asarray(means, dtype=float)) / np.asarray(stds, dtype=float)


def predict_classifier(classifier: str, train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    if len(test_x) == 0:
        return []
    if len(train_x) == 0:
        return ["PRESERVE" for _ in range(len(test_x))]
    if classifier == "nearest_centroid":
        return nearest_centroid_predictions(train_x, train_y, test_x)
    if classifier == "knn3":
        return knn_predictions(train_x, train_y, test_x, k=3)
    if classifier == "logistic_regression":
        return logistic_predictions(train_x, train_y, test_x)
    raise ValueError(f"unknown classifier: {classifier}")


def nearest_centroid_predictions(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    centroids = {
        label: train_x[[index for index, item in enumerate(train_y) if item == label]].mean(axis=0)
        for label in sorted(set(train_y))
    }
    return [
        min(centroids, key=lambda label: float(np.linalg.norm(row - centroids[label])))
        for row in test_x
    ]


def knn_predictions(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray, *, k: int) -> list[str]:
    predictions: list[str] = []
    for row in test_x:
        distances = np.linalg.norm(train_x - row, axis=1)
        indices = np.argsort(distances)[: max(1, min(int(k), len(train_y)))]
        counter = Counter(train_y[int(index)] for index in indices)
        predictions.append(majority_from_counter(counter))
    return predictions


def logistic_predictions(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced")
        model.fit(train_x, train_y)
        return [str(item) for item in model.predict(test_x)]
    except Exception:
        return nearest_centroid_predictions(train_x, train_y, test_x)


def classification_metrics(true_labels: list[str], predictions: list[str]) -> dict:
    confusion = {label: {predicted: 0 for predicted in FUTURE_EFFECT_CLASSES} for label in FUTURE_EFFECT_CLASSES}
    for actual, predicted in zip(true_labels, predictions, strict=True):
        confusion[actual][predicted] += 1
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1s: list[float] = []
    for label in FUTURE_EFFECT_CLASSES:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in FUTURE_EFFECT_CLASSES if actual != label)
        fn = sum(count for predicted, count in confusion[label].items() if predicted != label)
        precision[label] = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall[label] = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1s.append(0.0 if precision[label] + recall[label] == 0.0 else 2 * precision[label] * recall[label] / (precision[label] + recall[label]))
    return {
        "accuracy": accuracy(true_labels, predictions),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion,
    }


def write_prefuture_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = best_configs(rows)
    payload = {
        "runs": rows,
        "best_configs": best,
        "validation": validation_summary(rows),
        "forbidden_features": sorted(FORBIDDEN_FEATURE_NAMES),
        "feature_sets": {key: list(value) for key, value in PREFUTURE_FEATURE_SETS.items()},
    }
    (output / "prefuture_role_prediction_v04c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, output / "prefuture_role_prediction_v04c_report.csv")
    write_csv(best, output / "prefuture_role_prediction_v04c_best.csv")
    (output / "prefuture_role_prediction_v04c_report.txt").write_text(format_text_report(rows, best, payload), encoding="utf-8")


def best_configs(rows: list[dict]) -> list[dict]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[str(row["game"])].append(row)
    return [
        max(items, key=lambda row: (row["prefuture_role_accuracy"], row["prefuture_role_macro_f1"], row["prefuture_role_vs_baseline_delta"]))
        for _game, items in sorted(by_key.items())
    ]


def validation_summary(rows: list[dict]) -> dict:
    primary = [row for row in rows if row["game"] in PRIMARY_GAMES]
    weak = [
        row
        for row in primary
        if row["prefuture_role_accuracy"] > row["baseline_accuracy"]
        and row["prefuture_role_macro_f1"] > row["majority_baseline_macro_f1"]
        and has_non_preserve_recall(row)
    ]
    strong_games = {
        row["game"]
        for row in primary
        if row["prefuture_role_accuracy"] > row["baseline_accuracy"]
        and has_non_preserve_recall(row)
        and row["forbidden_feature_check_passed"]
    }
    return {
        "weak_pass": bool(weak),
        "weak_passing_configs": [
            f"{row['game']}/{row['feature_set']}/{row['classifier']}" for row in weak
        ],
        "strong_pass": len(strong_games) >= 2,
        "strong_passing_games": sorted(strong_games),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "mode",
        "feature_set",
        "classifier",
        "train_seeds",
        "test_seed",
        "steps",
        "horizon",
        "train_sample_count",
        "test_sample_count",
        "class_distribution_train",
        "class_distribution_test",
        "baseline_accuracy",
        "majority_baseline_macro_f1",
        "contingency_baseline_accuracy",
        "random_stratified_baseline_accuracy",
        "prefuture_role_accuracy",
        "prefuture_role_macro_f1",
        "prefuture_role_vs_baseline_delta",
        "prefuture_role_vs_contingency_delta",
        "preserve_precision",
        "preserve_recall",
        "expand_precision",
        "expand_recall",
        "restrict_precision",
        "restrict_recall",
        "collapse_precision",
        "collapse_recall",
        "confusion_matrix_json",
        "forbidden_feature_check_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field])
                    if field in {"train_seeds", "class_distribution_train", "class_distribution_test", "confusion_matrix_json"}
                    else row.get(field)
                    for field in fieldnames
                }
            )


def format_text_report(rows: list[dict], best: list[dict], payload: dict) -> str:
    lines = [
        "ARC-AGI3 v0.4c Pre-Future Role Prediction Report",
        "offline validation only; no future-effect-derived input features",
        "",
        "best configs:",
    ]
    for row in best:
        lines.append(
            f"{row['game']} feature_set={row['feature_set']} classifier={row['classifier']} "
            f"acc={row['prefuture_role_accuracy']:.3f} macro_f1={row['prefuture_role_macro_f1']:.3f} "
            f"baseline={row['baseline_accuracy']:.3f} contingency={row['contingency_baseline_accuracy']:.3f} "
            f"recalls=P:{row['preserve_recall']:.3f} E:{row['expand_recall']:.3f} "
            f"R:{row['restrict_recall']:.3f} C:{row['collapse_recall']:.3f}"
        )
    lines.append("")
    validation = payload["validation"]
    lines.append(f"weak_pass={validation['weak_pass']} strong_pass={validation['strong_pass']}")
    lines.append(f"strong_passing_games={validation['strong_passing_games']}")
    lines.append("")
    lines.append("top primary configs:")
    primary = [row for row in rows if row["game"] in PRIMARY_GAMES]
    for row in sorted(primary, key=lambda item: (item["prefuture_role_accuracy"], item["prefuture_role_macro_f1"]), reverse=True)[:12]:
        lines.append(
            f"{row['game']} {row['feature_set']} {row['classifier']} "
            f"acc={row['prefuture_role_accuracy']:.3f} macro={row['prefuture_role_macro_f1']:.3f} "
            f"base={row['baseline_accuracy']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _load_contingencies(connection: sqlite3.Connection) -> dict[int, dict]:
    rows = connection.execute(
        """
        SELECT id, context_level, context_signature, action, transformation_family, support_count, confidence
        FROM contingencies
        """
    ).fetchall()
    return {
        int(row[0]): {
            "context_level": int(row[1]),
            "context_signature": tuple(json.loads(row[2])),
            "action": int(row[3]),
            "transformation_family": int(row[4]),
            "support_count": int(row[5]),
            "confidence": float(row[6]),
        }
        for row in rows
    }


def _load_families(connection: sqlite3.Connection) -> dict[int, dict]:
    rows = connection.execute("SELECT id, centroid_vector, support_count FROM transformation_families").fetchall()
    families: dict[int, dict] = {}
    for family_id, centroid_json, support_count in rows:
        vector = list(json.loads(centroid_json))
        padded = vector + [0.0] * (5 - len(vector))
        families[int(family_id)] = {
            "support_count": int(support_count),
            "changed_cells": float(padded[0]),
            "dx": float(padded[1]),
            "dy": float(padded[2]),
            "colors_added_count": float(padded[3]),
            "colors_removed_count": float(padded[4]),
        }
    return families


def _load_prediction_rows(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        """
        SELECT interaction_id, episode_id, context_level, context_signature, action, predicted_family, actual_family, prediction_error
        FROM prediction_results
        ORDER BY interaction_id
        """
    ).fetchall()
    return [
        {
            "interaction_id": int(row[0]),
            "episode_id": int(row[1]),
            "context_level": None if row[2] is None else int(row[2]),
            "context_signature": tuple(json.loads(row[3])),
            "action": int(row[4]),
            "predicted_family": None if row[5] is None else int(row[5]),
            "actual_family": None if row[6] is None else int(row[6]),
            "prediction_error": None if row[7] is None else int(row[7]),
        }
        for row in rows
    ]


def _prediction_stats(rows: list[dict]) -> dict:
    contexts: dict[tuple, dict] = {}
    contingencies: dict[tuple, dict] = {}
    by_context: dict[tuple, list[dict]] = defaultdict(list)
    by_contingency: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        context = tuple(row["context_signature"])
        by_context[context].append(row)
        if row["actual_family"] is not None:
            key = (row["context_level"] or 0, context, row["action"], row["actual_family"])
            by_contingency[key].append(row)
    for context, items in by_context.items():
        contexts[context] = {
            "support": len(items),
            "action_entropy": entropy(Counter(item["action"] for item in items)),
            "family_entropy": entropy(Counter(item["actual_family"] for item in items if item["actual_family"] is not None)),
        }
    for key, items in by_contingency.items():
        errors = [item["prediction_error"] for item in items if item["prediction_error"] is not None]
        contingencies[key] = {"prediction_error_rate": 0.0 if not errors else sum(errors) / len(errors)}
    return {"contexts": contexts, "contingencies": contingencies}


def _graph_stats(contingencies: dict[int, dict], rows: list[dict]) -> dict[int, dict]:
    by_context: dict[tuple, set[int]] = defaultdict(set)
    by_family: dict[int, set[int]] = defaultdict(set)
    for contingency_id, item in contingencies.items():
        by_context[tuple(item["context_signature"])].add(contingency_id)
        by_family[int(item["transformation_family"])].add(contingency_id)

    follows_out: dict[int, set[int]] = defaultdict(set)
    follows_in: dict[int, set[int]] = defaultdict(set)
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["actual_family"] is not None:
            by_episode[int(row["episode_id"])].append(row)
    for items in by_episode.values():
        for first, second in zip(items, items[1:]):
            a = int(first["actual_family"])
            b = int(second["actual_family"])
            follows_out[a].add(b)
            follows_in[b].add(a)

    total = max(1, len(contingencies) - 1)
    stats: dict[int, dict] = {}
    for contingency_id, item in contingencies.items():
        context_neighbors = set(by_context[tuple(item["context_signature"])]) - {contingency_id}
        family_neighbors = set(by_family[int(item["transformation_family"])]) - {contingency_id}
        neighbors = context_neighbors | family_neighbors
        stats[contingency_id] = {
            "contingency_in_degree": len(context_neighbors),
            "contingency_out_degree": len(family_neighbors),
            "follows_in_degree": len(follows_in[int(item["transformation_family"])]),
            "follows_out_degree": len(follows_out[int(item["transformation_family"])]),
            "cooccurrence_degree": len(neighbors),
            "clustering_coefficient": clustering_coefficient(neighbors, contingencies),
            "degree_centrality": len(neighbors) / total,
            "pagerank": len(neighbors) / total,
        }
    return stats


def clustering_coefficient(neighbors: set[int], contingencies: dict[int, dict]) -> float:
    if len(neighbors) < 2:
        return 0.0
    neighbor_list = list(neighbors)
    possible = len(neighbor_list) * (len(neighbor_list) - 1) / 2
    edges = 0
    for index, left in enumerate(neighbor_list):
        for right in neighbor_list[index + 1 :]:
            if (
                tuple(contingencies[left]["context_signature"]) == tuple(contingencies[right]["context_signature"])
                or int(contingencies[left]["transformation_family"]) == int(contingencies[right]["transformation_family"])
            ):
                edges += 1
    return edges / possible


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def contingency_baseline_predictions(train: list[PrefutureExample], test: list[PrefutureExample], fallback: str) -> list[str]:
    counts: dict[tuple, Counter[str]] = defaultdict(Counter)
    for example in train:
        counts[example.contingency_key][example.label] += 1
    return [majority_from_counter(counts[example.contingency_key]) if example.contingency_key in counts else fallback for example in test]


def stratified_predictions_from_train(train_y: list[str], count: int) -> list[str]:
    if not train_y:
        return ["PRESERVE"] * count
    labels = sorted(Counter(train_y).items(), key=lambda item: (-item[1], item[0]))
    sequence = [label for label, label_count in labels for _ in range(label_count)]
    return [sequence[index % len(sequence)] for index in range(count)]


def accuracy(true_labels: list[str], predictions: list[str]) -> float:
    if not true_labels:
        return 0.0
    return sum(1 for actual, predicted in zip(true_labels, predictions, strict=True) if actual == predicted) / len(true_labels)


def majority_label(labels: list[str]) -> str:
    return majority_from_counter(Counter(labels))


def majority_from_counter(counter: Counter[str]) -> str:
    if not counter:
        return "PRESERVE"
    return max(FUTURE_EFFECT_CLASSES, key=lambda label: (counter[label], -FUTURE_EFFECT_CLASSES.index(label)))


def has_non_preserve_recall(row: dict) -> bool:
    return row["expand_recall"] > 0.0 or row["restrict_recall"] > 0.0 or row["collapse_recall"] > 0.0
