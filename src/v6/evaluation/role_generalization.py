from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.evaluation.future_effects import FutureEffectRunConfig, run_future_effect_v02
from v6.evaluation.role_candidates import FUTURE_EFFECT_CLASSES, ROLE_DISCOVERY_GAMES, ROLE_MODES
from v6.evaluation.role_validation import (
    ALLOWED_ASSIGNMENT_FEATURE_NAMES,
    TrainRole,
    ValidationExample,
    _apply_normalization,
    _classification_metrics,
    _contingency_history,
    _cosine,
    _db_path,
    _load_examples,
    _majority_label,
    _normalization_stats,
)


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full_allowed": ALLOWED_ASSIGNMENT_FEATURE_NAMES,
    "no_action": tuple(name for name in ALLOWED_ASSIGNMENT_FEATURE_NAMES if name != "action_id"),
    "no_family": tuple(name for name in ALLOWED_ASSIGNMENT_FEATURE_NAMES if name != "transformation_family_id"),
    "no_action_no_family": tuple(
        name
        for name in ALLOWED_ASSIGNMENT_FEATURE_NAMES
        if name not in {"action_id", "transformation_family_id"}
    ),
    "future_effect_only_no_label": (
        "mean_fo_before",
        "mean_fo_after",
        "mean_delta_fo",
        "std_delta_fo",
        "positive_delta_ratio",
        "negative_delta_ratio",
        "zero_delta_ratio",
        "collapse_ratio",
    ),
    "structural_no_ids": (
        "context_level",
        "confidence",
        "support_count_log",
        "mean_fo_before",
        "mean_fo_after",
        "mean_delta_fo",
        "std_delta_fo",
        "positive_delta_ratio",
        "negative_delta_ratio",
        "zero_delta_ratio",
        "collapse_ratio",
    ),
}
SIMILARITY_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85)
PRIMARY_GAMES = {"va02", "mo01", "ic01"}


@dataclass(frozen=True)
class RoleGeneralizationConfig:
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


def run_role_generalization_v04b(config: RoleGeneralizationConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = tuple(config.train_seeds) + (int(config.test_seed),)
    expected = [
        _db_path(db_dir, game, seed, config.steps, config.horizon)
        for game in config.games
        for seed in all_seeds
    ]
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
        train_examples: list[ValidationExample] = []
        for seed in config.train_seeds:
            train_examples.extend(_load_examples(_db_path(db_dir, game, seed, config.steps, config.horizon)))
        test_examples = _load_examples(_db_path(db_dir, game, config.test_seed, config.steps, config.horizon))
        for mode in ROLE_MODES:
            for feature_set in FEATURE_SETS:
                for threshold in SIMILARITY_THRESHOLDS:
                    rows.append(
                        validate_generalization_config(
                            game=game,
                            mode=mode,
                            feature_set=feature_set,
                            similarity_threshold=threshold,
                            train_seeds=config.train_seeds,
                            test_seed=config.test_seed,
                            steps=config.steps,
                            horizon=config.horizon,
                            train_examples=train_examples,
                            test_examples=test_examples,
                        )
                    )
    write_role_generalization_reports(rows, output_dir=output_dir)
    return rows


def validate_generalization_config(
    *,
    game: str,
    mode: str,
    feature_set: str,
    similarity_threshold: float,
    train_seeds: tuple[int, ...],
    test_seed: int,
    steps: int,
    horizon: int,
    train_examples: list[ValidationExample],
    test_examples: list[ValidationExample],
) -> dict:
    train_vectors = project_feature_vectors(train_examples, feature_set)
    test_vectors = project_feature_vectors(test_examples, feature_set)
    means, stds = _normalization_stats(train_vectors)
    normalized_train = [_apply_normalization(vector, means, stds) for vector in train_vectors]
    normalized_test = [_apply_normalization(vector, means, stds) for vector in test_vectors]

    train_roles = train_roles_for_feature_set(mode, feature_set, train_examples, normalized_train)
    train_majority = _majority_label([example.label for example in train_examples])
    contingency_history = _contingency_history(train_examples)
    true_labels = [example.label for example in test_examples]
    baseline_predictions = [train_majority for _example in test_examples]
    contingency_predictions = [
        contingency_history.get(example.contingency_key, train_majority)
        for example in test_examples
    ]
    role_predictions, assigned_count = role_predictions_at_threshold(
        train_roles,
        normalized_test,
        similarity_threshold=float(similarity_threshold),
    )

    role_metrics = _classification_metrics(true_labels, role_predictions)
    baseline_accuracy = _accuracy(true_labels, baseline_predictions)
    contingency_accuracy = _accuracy(true_labels, contingency_predictions)
    test_count = len(test_examples)
    strict_accuracy = role_metrics["strict_accuracy"]
    return {
        "game": game,
        "mode": mode,
        "feature_set": feature_set,
        "similarity_threshold": float(similarity_threshold),
        "train_seeds": list(train_seeds),
        "test_seed": int(test_seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "train_role_count": len(train_roles),
        "test_contingency_count": test_count,
        "assignment_coverage": 0.0 if test_count == 0 else assigned_count / test_count,
        "strict_role_accuracy": strict_accuracy,
        "assigned_only_role_accuracy": role_metrics["assigned_only_accuracy"],
        "strict_macro_f1": role_metrics["macro_f1"],
        "assigned_only_macro_f1": role_metrics["assigned_only_macro_f1"],
        "baseline_accuracy": baseline_accuracy,
        "contingency_accuracy": contingency_accuracy,
        "role_vs_baseline_delta": strict_accuracy - baseline_accuracy,
        "role_vs_contingency_delta": strict_accuracy - contingency_accuracy,
        "per_class_precision": role_metrics["per_class_precision"],
        "per_class_recall": role_metrics["per_class_recall"],
        "confusion_matrix": role_metrics["confusion_matrix"],
        "preserve_recall": role_metrics["per_class_recall"]["PRESERVE"],
        "expand_recall": role_metrics["per_class_recall"]["EXPAND"],
        "restrict_recall": role_metrics["per_class_recall"]["RESTRICT"],
        "collapse_recall": role_metrics["per_class_recall"]["COLLAPSE"],
        "unassigned_test_contingencies": test_count - assigned_count,
        "assigned_test_contingencies": assigned_count,
        "label_leakage_prevented": True,
    }


def project_feature_vectors(examples: list[ValidationExample], feature_set: str) -> list[tuple[float, ...]]:
    indices = feature_indices(feature_set)
    return [tuple(example.raw_vector[index] for index in indices) for example in examples]


def feature_indices(feature_set: str) -> tuple[int, ...]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    index_by_name = {name: index for index, name in enumerate(ALLOWED_ASSIGNMENT_FEATURE_NAMES)}
    return tuple(index_by_name[name] for name in FEATURE_SETS[feature_set])


def train_roles_for_feature_set(
    mode: str,
    feature_set: str,
    examples: list[ValidationExample],
    normalized_vectors: list[tuple[float, ...]],
) -> list[TrainRole]:
    groups = _deterministic_groups_for_feature_set(feature_set, examples) if mode == "deterministic" else _vector_groups(normalized_vectors)
    roles: list[TrainRole] = []
    for index, member_indices in enumerate(sorted(groups, key=lambda group: (-len(group), min(group))), start=1):
        vectors = np.array([normalized_vectors[member_index] for member_index in member_indices], dtype=float)
        labels = [examples[member_index].label for member_index in member_indices]
        roles.append(
            TrainRole(
                role_id=f"R{index}",
                mode=mode,
                member_count=len(member_indices),
                prototype_vector=tuple(float(value) for value in np.mean(vectors, axis=0)),
                dominant_future_effect_class=_majority_label(labels),
            )
        )
    return roles


def role_predictions_at_threshold(
    train_roles: list[TrainRole],
    normalized_test: list[tuple[float, ...]],
    *,
    similarity_threshold: float,
) -> tuple[list[str | None], int]:
    predictions: list[str | None] = []
    assigned = 0
    for vector in normalized_test:
        if not train_roles:
            predictions.append(None)
            continue
        best = max(train_roles, key=lambda role: _cosine(vector, role.prototype_vector))
        if _cosine(vector, best.prototype_vector) < float(similarity_threshold):
            predictions.append(None)
            continue
        predictions.append(best.dominant_future_effect_class)
        assigned += 1
    return predictions, assigned


def write_role_generalization_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_rows = best_configs(rows)
    payload = {
        "runs": rows,
        "best_configs": best_rows,
        "validation": _validation_summary(rows),
        "label_leakage_prevented": True,
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "similarity_thresholds": list(SIMILARITY_THRESHOLDS),
        "diagnostic_only": True,
    }
    (output / "role_generalization_v04b_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, output / "role_generalization_v04b_report.csv")
    _write_csv(best_rows, output / "role_generalization_v04b_best.csv")
    (output / "role_generalization_v04b_report.txt").write_text(_format_text_report(rows, best_rows, payload), encoding="utf-8")


def best_configs(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["game"]), str(row["mode"]))].append(row)
    return [
        max(
            items,
            key=lambda row: (
                row["strict_role_accuracy"],
                row["assignment_coverage"],
                row["assigned_only_role_accuracy"],
                -row["similarity_threshold"],
            ),
        )
        for _key, items in sorted(by_key.items())
    ]


def _deterministic_groups_for_feature_set(feature_set: str, examples: list[ValidationExample]) -> list[list[int]]:
    feature_names = set(FEATURE_SETS[feature_set])
    key_indices = [
        index
        for name, index in (
            ("context_level", 0),
            ("action_id", 1),
            ("transformation_family_id", 2),
        )
        if name in feature_names
    ]
    if not key_indices:
        key_indices = [5, 6, 7, 12]
    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[tuple(round(float(example.raw_vector[item]), 6) for item in key_indices)].append(index)
    return [groups[key] for key in sorted(groups)]


def _vector_groups(normalized_vectors: list[tuple[float, ...]]) -> list[list[int]]:
    if len(normalized_vectors) < 3:
        return [[index] for index in range(len(normalized_vectors))]
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean").fit_predict(
            np.array(normalized_vectors, dtype=float)
        )
    except Exception:
        return [[index] for index in range(len(normalized_vectors))]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        if int(label) >= 0:
            groups[int(label)].append(index)
    if len(groups) < 2:
        return [[index] for index in range(len(normalized_vectors))]
    return [groups[label] for label in sorted(groups)]


def _validation_summary(rows: list[dict]) -> dict:
    primary = [row for row in rows if row["game"] in PRIMARY_GAMES]
    passing = [
        row
        for row in primary
        if row["strict_role_accuracy"] > row["baseline_accuracy"]
        and row["assignment_coverage"] >= 0.5
        and _has_non_preserve_recall(row)
    ]
    return {
        "passes": bool(passing),
        "passing_configs": [
            f"{row['game']}/{row['mode']}/{row['feature_set']}/{row['similarity_threshold']:.2f}"
            for row in passing
        ],
        "strong_pass": {
            "mo01_collapse_strict_improved": _class_strict_improved(primary, "mo01", "COLLAPSE"),
            "va02_expand_or_collapse_strict_improved": (
                _class_strict_improved(primary, "va02", "EXPAND")
                or _class_strict_improved(primary, "va02", "COLLAPSE")
            ),
            "ic01_restrict_or_expand_strict_improved": (
                _class_strict_improved(primary, "ic01", "RESTRICT")
                or _class_strict_improved(primary, "ic01", "EXPAND")
            ),
        },
    }


def _class_strict_improved(rows: list[dict], game: str, label: str) -> bool:
    recall_key = f"{label.lower()}_recall"
    return any(
        row["game"] == game
        and row["strict_role_accuracy"] > row["baseline_accuracy"]
        and row[recall_key] > 0.0
        for row in rows
    )


def _has_non_preserve_recall(row: dict) -> bool:
    return row["expand_recall"] > 0.0 or row["restrict_recall"] > 0.0 or row["collapse_recall"] > 0.0


def _accuracy(true_labels: list[str], predictions: list[str | None]) -> float:
    if not true_labels:
        return 0.0
    return sum(1 for actual, predicted in zip(true_labels, predictions, strict=True) if actual == predicted) / len(true_labels)


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "mode",
        "feature_set",
        "similarity_threshold",
        "train_seeds",
        "test_seed",
        "steps",
        "horizon",
        "train_role_count",
        "test_contingency_count",
        "assignment_coverage",
        "strict_role_accuracy",
        "assigned_only_role_accuracy",
        "strict_macro_f1",
        "assigned_only_macro_f1",
        "baseline_accuracy",
        "contingency_accuracy",
        "role_vs_baseline_delta",
        "role_vs_contingency_delta",
        "preserve_recall",
        "expand_recall",
        "restrict_recall",
        "collapse_recall",
        "unassigned_test_contingencies",
        "assigned_test_contingencies",
        "label_leakage_prevented",
        "confusion_matrix",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field])
                    if field in {"train_seeds", "confusion_matrix"}
                    else row.get(field)
                    for field in fieldnames
                }
            )


def _format_text_report(rows: list[dict], best_rows: list[dict], payload: dict) -> str:
    lines = [
        "ARC-AGI3 v0.4b Role Generalization Report",
        "diagnostic sweep only; strict_role_accuracy counts unassigned as incorrect; label_leakage_prevented=true",
        "",
        "best configs:",
    ]
    for row in best_rows:
        lines.append(
            f"{row['game']} mode={row['mode']} feature_set={row['feature_set']} threshold={row['similarity_threshold']:.2f} "
            f"coverage={row['assignment_coverage']:.3f} strict={row['strict_role_accuracy']:.3f} "
            f"assigned={row['assigned_only_role_accuracy']:.3f} baseline={row['baseline_accuracy']:.3f} "
            f"d_base={row['role_vs_baseline_delta']:.3f} recalls="
            f"P:{row['preserve_recall']:.3f} E:{row['expand_recall']:.3f} "
            f"R:{row['restrict_recall']:.3f} C:{row['collapse_recall']:.3f}"
        )
    lines.append("")
    validation = payload["validation"]
    lines.append(f"validation_pass={validation['passes']} passing_configs={validation['passing_configs']}")
    lines.append(f"strong_pass={validation['strong_pass']}")
    lines.append("")
    lines.append("top strict primary configs:")
    primary = [row for row in rows if row["game"] in PRIMARY_GAMES]
    for row in sorted(primary, key=lambda item: (item["strict_role_accuracy"], item["assignment_coverage"]), reverse=True)[:12]:
        lines.append(
            f"{row['game']} {row['mode']} {row['feature_set']} t={row['similarity_threshold']:.2f} "
            f"strict={row['strict_role_accuracy']:.3f} coverage={row['assignment_coverage']:.3f} "
            f"assigned={row['assigned_only_role_accuracy']:.3f} baseline={row['baseline_accuracy']:.3f}"
        )
    return "\n".join(lines) + "\n"
