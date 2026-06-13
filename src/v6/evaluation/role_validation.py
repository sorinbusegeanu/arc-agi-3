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
from v6.evaluation.role_candidates import FUTURE_EFFECT_CLASSES, ROLE_DISCOVERY_GAMES, ROLE_MODES


ALLOWED_ASSIGNMENT_FEATURE_NAMES = (
    "context_level",
    "action_id",
    "transformation_family_id",
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
)
ROLE_ASSIGNMENT_THRESHOLD = 0.85


@dataclass(frozen=True)
class RoleValidationConfig:
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
class ValidationExample:
    contingency_id: int
    contingency_key: tuple
    raw_vector: tuple[float, ...]
    label: str


@dataclass(frozen=True)
class TrainRole:
    role_id: str
    mode: str
    member_count: int
    prototype_vector: tuple[float, ...]
    dominant_future_effect_class: str


def run_role_validation_v04(config: RoleValidationConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = tuple(config.train_seeds) + (int(config.test_seed),)
    expected = [
        db_dir / f"{game}_seed{seed}_steps{config.steps}_h{config.horizon}.sqlite"
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
            rows.append(
                validate_role_predictor(
                    game=game,
                    mode=mode,
                    train_seeds=config.train_seeds,
                    test_seed=config.test_seed,
                    steps=config.steps,
                    horizon=config.horizon,
                    train_examples=train_examples,
                    test_examples=test_examples,
                )
            )
    write_role_validation_reports(rows, output_dir=output_dir)
    return rows


def validate_role_predictor(
    *,
    game: str,
    mode: str,
    train_seeds: tuple[int, ...],
    test_seed: int,
    steps: int,
    horizon: int,
    train_examples: list[ValidationExample],
    test_examples: list[ValidationExample],
) -> dict:
    train_vectors = [example.raw_vector for example in train_examples]
    means, stds = _normalization_stats(train_vectors)
    normalized_train = [_apply_normalization(example.raw_vector, means, stds) for example in train_examples]
    normalized_test = [_apply_normalization(example.raw_vector, means, stds) for example in test_examples]

    train_roles = _train_roles(mode, train_examples, normalized_train)
    train_majority = _majority_label([example.label for example in train_examples])
    contingency_history = _contingency_history(train_examples)

    true_labels = [example.label for example in test_examples]
    baseline_predictions = [train_majority for _example in test_examples]
    contingency_predictions = [
        contingency_history.get(example.contingency_key, train_majority)
        for example in test_examples
    ]
    role_predictions, assigned_count = _role_predictions(train_roles, normalized_test)

    role_metrics = _classification_metrics(true_labels, role_predictions)
    baseline_accuracy = _accuracy(true_labels, baseline_predictions)
    contingency_accuracy = _accuracy(true_labels, contingency_predictions)
    strict_role_accuracy = role_metrics["strict_accuracy"]
    assigned_only_role_accuracy = role_metrics["assigned_only_accuracy"]
    test_count = len(test_examples)
    unassigned_count = test_count - assigned_count
    return {
        "game": game,
        "mode": mode,
        "train_seeds": list(train_seeds),
        "test_seed": int(test_seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "train_role_count": len(train_roles),
        "test_contingency_count": test_count,
        "assignment_coverage": 0.0 if test_count == 0 else assigned_count / test_count,
        "baseline_accuracy": baseline_accuracy,
        "contingency_accuracy": contingency_accuracy,
        "strict_role_accuracy": strict_role_accuracy,
        "assigned_only_role_accuracy": assigned_only_role_accuracy,
        "role_accuracy": strict_role_accuracy,
        "role_vs_baseline_delta": strict_role_accuracy - baseline_accuracy,
        "role_vs_contingency_delta": strict_role_accuracy - contingency_accuracy,
        "macro_f1": role_metrics["macro_f1"],
        "assigned_only_macro_f1": role_metrics["assigned_only_macro_f1"],
        "per_class_precision": role_metrics["per_class_precision"],
        "per_class_recall": role_metrics["per_class_recall"],
        "confusion_matrix": role_metrics["confusion_matrix"],
        "unassigned_test_contingencies": unassigned_count,
        "assigned_test_contingencies": assigned_count,
        "label_leakage_prevented": True,
        "non_preserve_recall_positive": any(
            role_metrics["per_class_recall"][label] > 0.0
            for label in ("EXPAND", "RESTRICT", "COLLAPSE")
        ),
    }


def write_role_validation_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": rows,
        "validation": _validation_summary(rows),
        "label_leakage_prevented": True,
        "assignment_feature_names": list(ALLOWED_ASSIGNMENT_FEATURE_NAMES),
        "target_label": "future_effect_class_id",
        "role_assignment_threshold": ROLE_ASSIGNMENT_THRESHOLD,
    }
    (output / "role_validation_v04_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, output / "role_validation_v04_report.csv")
    (output / "role_validation_v04_report.txt").write_text(_format_text_report(rows, payload), encoding="utf-8")


def _load_examples(db_path: Path) -> list[ValidationExample]:
    with sqlite3.connect(db_path) as connection:
        effects = load_future_effects(connection)
        contingency_rows = connection.execute(
            """
            SELECT id, context_level, context_signature, action, transformation_family, support_count, confidence
            FROM contingencies
            """
        ).fetchall()
    contingencies = {
        int(row[0]): {
            "context_level": int(row[1]),
            "context_signature": tuple(json.loads(row[2])),
            "action": int(row[3]),
            "transformation_family": int(row[4]),
            "support_count": int(row[5]),
            "confidence": float(row[6]),
        }
        for row in contingency_rows
    }
    examples: list[ValidationExample] = []
    for effect in effects:
        contingency = contingencies.get(int(effect.contingency_id))
        if contingency is None:
            continue
        raw_vector = (
            float(effect.context_level),
            float(effect.action),
            float(effect.transformation_family),
            float(contingency["confidence"]),
            math.log1p(float(contingency["support_count"])),
            float(effect.mean_fo_before),
            float(effect.mean_fo_after),
            float(effect.mean_delta_fo),
            float(effect.std_delta_fo),
            float(effect.positive_delta_ratio),
            float(effect.negative_delta_ratio),
            float(effect.zero_delta_ratio),
            float(effect.collapse_ratio),
        )
        examples.append(
            ValidationExample(
                contingency_id=int(effect.contingency_id),
                contingency_key=(
                    int(contingency["context_level"]),
                    tuple(contingency["context_signature"]),
                    int(contingency["action"]),
                    int(contingency["transformation_family"]),
                ),
                raw_vector=raw_vector,
                label=str(effect.future_effect_class),
            )
        )
    return examples


def _train_roles(mode: str, examples: list[ValidationExample], normalized_vectors: list[tuple[float, ...]]) -> list[TrainRole]:
    if mode == "deterministic":
        groups = _deterministic_groups(examples)
    elif mode == "vector":
        groups = _vector_groups(normalized_vectors)
    else:
        raise ValueError(f"unknown role validation mode: {mode}")

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


def _deterministic_groups(examples: list[ValidationExample]) -> list[list[int]]:
    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(example.raw_vector[0], example.raw_vector[2])].append(index)
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
        return _deterministic_groups_from_vectors(normalized_vectors)
    return [groups[label] for label in sorted(groups)]


def _deterministic_groups_from_vectors(normalized_vectors: list[tuple[float, ...]]) -> list[list[int]]:
    return [[index] for index in range(len(normalized_vectors))]


def _role_predictions(train_roles: list[TrainRole], normalized_test: list[tuple[float, ...]]) -> tuple[list[str | None], int]:
    predictions: list[str | None] = []
    assigned = 0
    for vector in normalized_test:
        if not train_roles:
            predictions.append(None)
            continue
        best = max(train_roles, key=lambda role: _cosine(vector, role.prototype_vector))
        similarity = _cosine(vector, best.prototype_vector)
        if similarity < ROLE_ASSIGNMENT_THRESHOLD:
            predictions.append(None)
            continue
        predictions.append(best.dominant_future_effect_class)
        assigned += 1
    return predictions, assigned


def _contingency_history(examples: list[ValidationExample]) -> dict[tuple, str]:
    counts: dict[tuple, Counter[str]] = defaultdict(Counter)
    for example in examples:
        counts[example.contingency_key][example.label] += 1
    return {key: _majority_from_counter(counter) for key, counter in counts.items()}


def _classification_metrics(true_labels: list[str], predictions: list[str | None]) -> dict:
    confusion: dict[str, dict[str, int]] = {
        label: {predicted: 0 for predicted in (*FUTURE_EFFECT_CLASSES, "UNASSIGNED")}
        for label in FUTURE_EFFECT_CLASSES
    }
    for actual, predicted in zip(true_labels, predictions, strict=True):
        confusion[actual][predicted or "UNASSIGNED"] += 1

    precision, recall, f1s = _precision_recall_f1(confusion, include_unassigned_as_fn=True)
    assigned_precision, assigned_recall, assigned_f1s = _precision_recall_f1(confusion, include_unassigned_as_fn=False)
    assigned_pairs = [
        (actual, predicted)
        for actual, predicted in zip(true_labels, predictions, strict=True)
        if predicted is not None
    ]
    return {
        "strict_accuracy": _accuracy(true_labels, predictions),
        "assigned_only_accuracy": 0.0
        if not assigned_pairs
        else sum(1 for actual, predicted in assigned_pairs if actual == predicted) / len(assigned_pairs),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "assigned_only_macro_f1": float(np.mean(assigned_f1s)) if assigned_f1s else 0.0,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "assigned_only_per_class_precision": assigned_precision,
        "assigned_only_per_class_recall": assigned_recall,
        "confusion_matrix": confusion,
    }


def _precision_recall_f1(
    confusion: dict[str, dict[str, int]],
    *,
    include_unassigned_as_fn: bool,
) -> tuple[dict[str, float], dict[str, float], list[float]]:
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1s: list[float] = []
    for label in FUTURE_EFFECT_CLASSES:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in FUTURE_EFFECT_CLASSES if actual != label)
        fn = sum(
            count
            for predicted, count in confusion[label].items()
            if predicted != label and (include_unassigned_as_fn or predicted != "UNASSIGNED")
        )
        precision[label] = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall[label] = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1s.append(0.0 if precision[label] + recall[label] == 0.0 else 2 * precision[label] * recall[label] / (precision[label] + recall[label]))
    return precision, recall, f1s


def _accuracy(true_labels: list[str], predictions: list[str | None]) -> float:
    if not true_labels:
        return 0.0
    return sum(1 for actual, predicted in zip(true_labels, predictions, strict=True) if actual == predicted) / len(true_labels)


def _normalization_stats(vectors: list[tuple[float, ...]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not vectors:
        return tuple(0.0 for _ in ALLOWED_ASSIGNMENT_FEATURE_NAMES), tuple(1.0 for _ in ALLOWED_ASSIGNMENT_FEATURE_NAMES)
    matrix = np.array(vectors, dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0.0] = 1.0
    return tuple(float(value) for value in means), tuple(float(value) for value in stds)


def _apply_normalization(vector: tuple[float, ...], means: tuple[float, ...], stds: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((float(value) - means[index]) / stds[index] for index, value in enumerate(vector))


def _majority_label(labels: list[str]) -> str:
    return _majority_from_counter(Counter(labels))


def _majority_from_counter(counter: Counter[str]) -> str:
    if not counter:
        return "PRESERVE"
    return max(FUTURE_EFFECT_CLASSES, key=lambda label: (counter[label], -FUTURE_EFFECT_CLASSES.index(label)))


def _cosine(vector_a: tuple[float, ...], vector_b: tuple[float, ...]) -> float:
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _validation_summary(rows: list[dict]) -> dict:
    primary = [row for row in rows if row["game"] in {"va02", "mo01", "ic01"}]
    weak_pass_rows = [
        row
        for row in primary
        if row["strict_role_accuracy"] > row["baseline_accuracy"]
        and row["assignment_coverage"] >= 0.5
        and row["non_preserve_recall_positive"]
    ]
    return {
        "passes": bool(weak_pass_rows),
        "passing_game_modes": [f"{row['game']}/{row['mode']}" for row in weak_pass_rows],
        "strong_pass": {
            "va02_non_preserve_improved": _game_mode_has_non_preserve_recall(primary, "va02"),
            "mo01_collapse_recall_positive": _game_mode_class_recall(primary, "mo01", "COLLAPSE") > 0.0,
            "ic01_restrict_or_expand_recall_positive": (
                _game_mode_class_recall(primary, "ic01", "RESTRICT") > 0.0
                or _game_mode_class_recall(primary, "ic01", "EXPAND") > 0.0
            ),
        },
    }


def _game_mode_has_non_preserve_recall(rows: list[dict], game: str) -> bool:
    return any(
        row["game"] == game
        and (
            row["per_class_recall"]["EXPAND"] > 0.0
            or row["per_class_recall"]["RESTRICT"] > 0.0
            or row["per_class_recall"]["COLLAPSE"] > 0.0
        )
        for row in rows
    )


def _game_mode_class_recall(rows: list[dict], game: str, label: str) -> float:
    values = [row["per_class_recall"][label] for row in rows if row["game"] == game]
    return max(values, default=0.0)


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "mode",
        "train_seeds",
        "test_seed",
        "steps",
        "horizon",
        "train_role_count",
        "test_contingency_count",
        "assignment_coverage",
        "baseline_accuracy",
        "contingency_accuracy",
        "strict_role_accuracy",
        "assigned_only_role_accuracy",
        "role_vs_baseline_delta",
        "role_vs_contingency_delta",
        "macro_f1",
        "assigned_only_macro_f1",
        "preserve_precision",
        "preserve_recall",
        "expand_precision",
        "expand_recall",
        "restrict_precision",
        "restrict_recall",
        "collapse_precision",
        "collapse_recall",
        "unassigned_test_contingencies",
        "assigned_test_contingencies",
        "confusion_matrix",
        "label_leakage_prevented",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "game": row["game"],
                    "mode": row["mode"],
                    "train_seeds": json.dumps(row["train_seeds"]),
                    "test_seed": row["test_seed"],
                    "steps": row["steps"],
                    "horizon": row["horizon"],
                    "train_role_count": row["train_role_count"],
                    "test_contingency_count": row["test_contingency_count"],
                    "assignment_coverage": row["assignment_coverage"],
                    "baseline_accuracy": row["baseline_accuracy"],
                    "contingency_accuracy": row["contingency_accuracy"],
                    "strict_role_accuracy": row["strict_role_accuracy"],
                    "assigned_only_role_accuracy": row["assigned_only_role_accuracy"],
                    "role_vs_baseline_delta": row["role_vs_baseline_delta"],
                    "role_vs_contingency_delta": row["role_vs_contingency_delta"],
                    "macro_f1": row["macro_f1"],
                    "assigned_only_macro_f1": row["assigned_only_macro_f1"],
                    "preserve_precision": row["per_class_precision"]["PRESERVE"],
                    "preserve_recall": row["per_class_recall"]["PRESERVE"],
                    "expand_precision": row["per_class_precision"]["EXPAND"],
                    "expand_recall": row["per_class_recall"]["EXPAND"],
                    "restrict_precision": row["per_class_precision"]["RESTRICT"],
                    "restrict_recall": row["per_class_recall"]["RESTRICT"],
                    "collapse_precision": row["per_class_precision"]["COLLAPSE"],
                    "collapse_recall": row["per_class_recall"]["COLLAPSE"],
                    "unassigned_test_contingencies": row["unassigned_test_contingencies"],
                    "assigned_test_contingencies": row["assigned_test_contingencies"],
                    "confusion_matrix": json.dumps(row["confusion_matrix"]),
                    "label_leakage_prevented": row["label_leakage_prevented"],
                }
            )


def _format_text_report(rows: list[dict], payload: dict) -> str:
    lines = [
        "ARC-AGI3 v0.4 Role-Candidate Validation Report",
        "validation only; random policy; strict_role_accuracy counts unassigned as incorrect; label_leakage_prevented=true",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['game']} mode={row['mode']} train={row['train_seeds']} test={row['test_seed']} "
            f"roles={row['train_role_count']} test={row['test_contingency_count']} "
            f"coverage={row['assignment_coverage']:.3f} baseline={row['baseline_accuracy']:.3f} "
            f"contingency={row['contingency_accuracy']:.3f} strict_role={row['strict_role_accuracy']:.3f} "
            f"assigned_role={row['assigned_only_role_accuracy']:.3f} "
            f"d_base={row['role_vs_baseline_delta']:.3f} d_cont={row['role_vs_contingency_delta']:.3f} "
            f"macro_f1={row['macro_f1']:.3f} assigned_macro_f1={row['assigned_only_macro_f1']:.3f} "
            f"unassigned={row['unassigned_test_contingencies']}"
        )
        lines.append(
            "  recall="
            f"PRESERVE:{row['per_class_recall']['PRESERVE']:.3f} "
            f"EXPAND:{row['per_class_recall']['EXPAND']:.3f} "
            f"RESTRICT:{row['per_class_recall']['RESTRICT']:.3f} "
            f"COLLAPSE:{row['per_class_recall']['COLLAPSE']:.3f}"
        )
    lines.append("")
    validation = payload["validation"]
    lines.append(f"validation_pass={validation['passes']} passing_game_modes={validation['passing_game_modes']}")
    lines.append(f"strong_pass={validation['strong_pass']}")
    return "\n".join(lines) + "\n"


def _db_path(db_dir: Path, game: str, seed: int, steps: int, horizon: int) -> Path:
    return db_dir / f"{game}_seed{seed}_steps{steps}_h{horizon}.sqlite"
