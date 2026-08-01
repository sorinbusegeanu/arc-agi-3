from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from v6.evaluation.future_effects import FutureEffectRunConfig, run_future_effect_v02
from v6.evaluation.prefuture_role_prediction import (
    FORBIDDEN_FEATURE_NAMES,
    PREFUTURE_CLASSIFIERS,
    PRIMARY_GAMES,
    PrefutureExample,
    accuracy,
    apply_normalization,
    classification_metrics,
    contingency_baseline_predictions,
    feature_matrix,
    load_prefuture_examples,
    majority_label,
    normalization_stats,
    predict_classifier,
    stratified_predictions_from_train,
)
from v6.evaluation.role_candidates import ROLE_DISCOVERY_GAMES
from v6.evaluation.role_validation import _db_path


FORBIDDEN_ID_FEATURE_NAMES = {"action_id", "transformation_family_id"}
ID_FREE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "contingency_only_no_ids": (
        "context_level",
        "confidence",
        "support_count_log",
        "prediction_error_rate",
        "context_support",
        "action_entropy_at_context",
        "transformation_entropy_at_context",
    ),
    "transformation_signature_no_ids": (
        "confidence",
        "support_count_log",
        "transformation_family_support_log",
        "changed_cells",
        "dx",
        "dy",
        "colors_added_count",
        "colors_removed_count",
    ),
    "graph_only_no_ids": (
        "contingency_in_degree",
        "contingency_out_degree",
        "follows_in_degree",
        "follows_out_degree",
        "cooccurrence_degree",
        "clustering_coefficient",
        "degree_centrality",
        "pagerank",
    ),
}
ID_FREE_FEATURE_SETS["contingency_plus_graph_no_ids"] = (
    ID_FREE_FEATURE_SETS["contingency_only_no_ids"] + ID_FREE_FEATURE_SETS["graph_only_no_ids"]
)
ID_FREE_FEATURE_SETS["contingency_plus_transformation_signature_no_ids"] = (
    ID_FREE_FEATURE_SETS["contingency_only_no_ids"] + ID_FREE_FEATURE_SETS["transformation_signature_no_ids"]
)
ID_FREE_FEATURE_SETS["all_prefuture_no_ids"] = (
    ID_FREE_FEATURE_SETS["contingency_only_no_ids"]
    + ID_FREE_FEATURE_SETS["transformation_signature_no_ids"]
    + ID_FREE_FEATURE_SETS["graph_only_no_ids"]
)


@dataclass(frozen=True)
class IdFreePrefutureConfig:
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
class PreparedIdFreeValidationGroup:
    """Inputs shared by every feature/classifier configuration for one split."""

    train_examples: list[PrefutureExample]
    test_examples: list[PrefutureExample]
    train_y: list[str]
    test_y: list[str]
    majority_accuracy: float
    majority_macro_f1: float
    contingency_accuracy: float
    random_stratified_accuracy: float


@dataclass(frozen=True)
class PreparedIdFreeFeatureSet:
    feature_set: str
    train_xn: object
    test_xn: object


def run_id_free_prefuture_validation_v04d(config: IdFreePrefutureConfig) -> list[dict]:
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
        train: list[PrefutureExample] = []
        for seed in config.train_seeds:
            train.extend(_load_id_free_examples(_db_path(db_dir, game, seed, config.steps, config.horizon)))
        test = _load_id_free_examples(_db_path(db_dir, game, config.test_seed, config.steps, config.horizon))
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
                    rows.append(row)
    write_id_free_reports(rows, output_dir=output_dir)
    return rows


def evaluate_id_free_config(
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
    prepared_group = prepare_id_free_validation_group(train_examples, test_examples)
    prepared_features = prepare_id_free_feature_set(prepared_group, feature_set)
    return evaluate_prepared_id_free_config(
        game=game,
        feature_set=feature_set,
        classifier=classifier,
        train_seeds=train_seeds,
        test_seed=test_seed,
        steps=steps,
        horizon=horizon,
        prepared_group=prepared_group,
        prepared_features=prepared_features,
    )


def prepare_id_free_validation_group(
    train_examples: list[PrefutureExample],
    test_examples: list[PrefutureExample],
) -> PreparedIdFreeValidationGroup:
    """Prepare labels and baselines once for a validation task."""
    train_y = [item.label for item in train_examples]
    test_y = [item.label for item in test_examples]
    majority = majority_label(train_y)
    majority_predictions = [majority for _ in test_y]
    contingency_predictions = contingency_baseline_predictions(train_examples, test_examples, majority)
    stratified_predictions = stratified_predictions_from_train(train_y, len(test_y))
    majority_metrics = classification_metrics(test_y, majority_predictions)
    return PreparedIdFreeValidationGroup(
        train_examples=train_examples,
        test_examples=test_examples,
        train_y=train_y,
        test_y=test_y,
        majority_accuracy=accuracy(test_y, majority_predictions),
        majority_macro_f1=float(majority_metrics["macro_f1"]),
        contingency_accuracy=accuracy(test_y, contingency_predictions),
        random_stratified_accuracy=accuracy(test_y, stratified_predictions),
    )


def prepare_id_free_feature_set(
    prepared_group: PreparedIdFreeValidationGroup,
    feature_set: str,
) -> PreparedIdFreeFeatureSet:
    """Build and normalize a feature set once, then reuse it for classifiers."""
    train_x = feature_matrix_for_id_free(prepared_group.train_examples, feature_set)
    test_x = feature_matrix_for_id_free(prepared_group.test_examples, feature_set)
    means, stds = normalization_stats(train_x)
    return PreparedIdFreeFeatureSet(
        feature_set=feature_set,
        train_xn=apply_normalization(train_x, means, stds),
        test_xn=apply_normalization(test_x, means, stds),
    )


def evaluate_prepared_id_free_config(
    *,
    game: str,
    feature_set: str,
    classifier: str,
    train_seeds: tuple[int, ...],
    test_seed: int,
    steps: int,
    horizon: int,
    prepared_group: PreparedIdFreeValidationGroup,
    prepared_features: PreparedIdFreeFeatureSet,
) -> dict | None:
    """Evaluate one classifier without rebuilding features, labels, or baselines."""
    train_y = prepared_group.train_y
    test_y = prepared_group.test_y
    if classifier == "logistic_regression" and len(set(train_y)) < 2:
        return None
    predictions = predict_classifier(classifier, prepared_features.train_xn, train_y, prepared_features.test_xn)
    metrics = classification_metrics(test_y, predictions)
    id_free_accuracy = accuracy(test_y, predictions)
    return {
        "game": game,
        "feature_set": feature_set,
        "classifier": classifier,
        "train_seeds": list(train_seeds),
        "test_seed": int(test_seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "train_sample_count": len(prepared_group.train_examples),
        "test_sample_count": len(prepared_group.test_examples),
        "class_distribution_train": dict(__import__("collections").Counter(train_y)),
        "class_distribution_test": dict(__import__("collections").Counter(test_y)),
        "majority_baseline_accuracy": prepared_group.majority_accuracy,
        "majority_baseline_macro_f1": prepared_group.majority_macro_f1,
        "contingency_baseline_accuracy": prepared_group.contingency_accuracy,
        "random_stratified_accuracy": prepared_group.random_stratified_accuracy,
        "id_free_accuracy": id_free_accuracy,
        "id_free_macro_f1": metrics["macro_f1"],
        "id_free_vs_majority_delta": id_free_accuracy - prepared_group.majority_accuracy,
        "id_free_vs_contingency_delta": id_free_accuracy - prepared_group.contingency_accuracy,
        "preserve_precision": metrics["precision"]["PRESERVE"],
        "preserve_recall": metrics["recall"]["PRESERVE"],
        "expand_precision": metrics["precision"]["EXPAND"],
        "expand_recall": metrics["recall"]["EXPAND"],
        "restrict_precision": metrics["precision"]["RESTRICT"],
        "restrict_recall": metrics["recall"]["RESTRICT"],
        "collapse_precision": metrics["precision"]["COLLAPSE"],
        "collapse_recall": metrics["recall"]["COLLAPSE"],
        "non_preserve_recall_any": max(metrics["recall"]["EXPAND"], metrics["recall"]["RESTRICT"], metrics["recall"]["COLLAPSE"]),
        "confusion_matrix_json": metrics["confusion_matrix"],
        "forbidden_future_feature_check_passed": forbidden_future_feature_check(feature_set),
        "forbidden_id_feature_check_passed": forbidden_id_feature_check(feature_set),
    }


def feature_matrix_for_id_free(examples: list[PrefutureExample], feature_set: str) -> list[tuple[float, ...]]:
    _install_id_free_sets_temporarily()
    return feature_matrix(examples, feature_set)


def forbidden_future_feature_check(feature_set: str) -> bool:
    return not bool(set(ID_FREE_FEATURE_SETS[feature_set]) & FORBIDDEN_FEATURE_NAMES)


def forbidden_id_feature_check(feature_set: str) -> bool:
    return not bool(set(ID_FREE_FEATURE_SETS[feature_set]) & FORBIDDEN_ID_FEATURE_NAMES)


def write_id_free_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = best_configs(rows)
    payload = {
        "runs": rows,
        "best_configs": best,
        "validation": validation_summary(rows),
        "feature_sets": {key: list(value) for key, value in ID_FREE_FEATURE_SETS.items()},
        "forbidden_future_features": sorted(FORBIDDEN_FEATURE_NAMES),
        "forbidden_id_features": sorted(FORBIDDEN_ID_FEATURE_NAMES),
    }
    (output / "id_free_prefuture_validation_v04d_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, output / "id_free_prefuture_validation_v04d_report.csv")
    write_csv(best, output / "id_free_prefuture_validation_v04d_best.csv")
    (output / "id_free_prefuture_validation_v04d_report.txt").write_text(format_text(rows, best, payload), encoding="utf-8")


def best_configs(rows: list[dict]) -> list[dict]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game"])].append(row)
    return [
        max(items, key=lambda row: (row["id_free_macro_f1"], row["id_free_accuracy"], row["non_preserve_recall_any"]))
        for _game, items in sorted(by_game.items())
    ]


def validation_summary(rows: list[dict]) -> dict:
    primary = [row for row in rows if row["game"] in PRIMARY_GAMES]
    passing = [
        row
        for row in primary
        if row["id_free_accuracy"] > row["majority_baseline_accuracy"]
        and row["id_free_macro_f1"] > row["majority_baseline_macro_f1"]
        and row["non_preserve_recall_any"] > 0.0
        and row["forbidden_future_feature_check_passed"]
        and row["forbidden_id_feature_check_passed"]
    ]
    strong_games = sorted(
        {
            row["game"]
            for row in primary
            if row["id_free_accuracy"] > row["majority_baseline_accuracy"]
            and row["non_preserve_recall_any"] > 0.0
            and row["forbidden_future_feature_check_passed"]
            and row["forbidden_id_feature_check_passed"]
        }
    )
    return {
        "weak_pass": bool(passing),
        "weak_passing_configs": [f"{row['game']}/{row['feature_set']}/{row['classifier']}" for row in passing],
        "strong_pass": len(strong_games) >= 2,
        "strong_passing_games": strong_games,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
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
        "majority_baseline_accuracy",
        "majority_baseline_macro_f1",
        "contingency_baseline_accuracy",
        "random_stratified_accuracy",
        "id_free_accuracy",
        "id_free_macro_f1",
        "id_free_vs_majority_delta",
        "id_free_vs_contingency_delta",
        "preserve_precision",
        "preserve_recall",
        "expand_precision",
        "expand_recall",
        "restrict_precision",
        "restrict_recall",
        "collapse_precision",
        "collapse_recall",
        "non_preserve_recall_any",
        "confusion_matrix_json",
        "forbidden_future_feature_check_passed",
        "forbidden_id_feature_check_passed",
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


def format_text(rows: list[dict], best: list[dict], payload: dict) -> str:
    lines = ["ARC-AGI3 v0.4d ID-Free Pre-Future Role Validation", "offline validation only; no future-effect or local ID input features", "", "best configs:"]
    for row in best:
        lines.append(
            f"{row['game']} feature_set={row['feature_set']} classifier={row['classifier']} "
            f"acc={row['id_free_accuracy']:.3f} macro_f1={row['id_free_macro_f1']:.3f} "
            f"majority={row['majority_baseline_accuracy']:.3f} contingency={row['contingency_baseline_accuracy']:.3f} "
            f"non_preserve={row['non_preserve_recall_any']:.3f}"
        )
    validation = payload["validation"]
    lines.append("")
    lines.append(f"weak_pass={validation['weak_pass']} strong_pass={validation['strong_pass']}")
    lines.append(f"strong_passing_games={validation['strong_passing_games']}")
    return "\n".join(lines) + "\n"


def _load_id_free_examples(path: Path) -> list[PrefutureExample]:
    examples = load_prefuture_examples(path)
    for example in examples:
        if "pagerank" not in example.features:
            example.features["pagerank"] = example.features.get("degree_centrality", 0.0)
    return examples


def _install_id_free_sets_temporarily() -> None:
    from v6.evaluation import prefuture_role_prediction

    prefuture_role_prediction.PREFUTURE_FEATURE_SETS.update(ID_FREE_FEATURE_SETS)
