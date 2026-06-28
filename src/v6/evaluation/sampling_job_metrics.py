from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _connect_sampling_read_db(db_path: Path, *, busy_timeout_ms: int = 60000) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=max(1.0, float(busy_timeout_ms) / 1000.0))
    connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    return connection


def compute_sampling_job_metrics(
    db_path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
    config: Any,
) -> dict:
    from v6.evaluation import interaction_sampling as mod

    with _patched_sqlite_busy_timeout(60000):
        return mod._run_metrics(Path(db_path), game, sampler_name, int(seed), config)


def compute_sampling_job_temporal_milestones(
    db_path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
) -> dict:
    from v6.evaluation import interaction_sampling as mod

    with _patched_sqlite_busy_timeout(60000):
        return mod._temporal_milestones_for_db(
            Path(db_path),
            game=game,
            sampler_name=sampler_name,
            seed=int(seed),
        )


def compute_sampling_job_validation_payload(
    db_path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
    config: Any,
) -> dict:
    examples = _load_validation_examples(Path(db_path))
    return {
        "game": str(game),
        "sampler_name": str(sampler_name),
        "seed": int(seed),
        "steps": int(getattr(config, "steps", 0) or 0),
        "horizon": int(getattr(config, "horizon", 0) or 0),
        "examples": [
            {
                "contingency_id": int(item.contingency_id),
                "contingency_key": list(item.contingency_key),
                "features": {str(key): float(value) for key, value in item.features.items()},
                "label": str(item.label),
            }
            for item in examples
        ],
    }


def _load_validation_examples(db_path: Path) -> list[Any]:
    from v6.evaluation.prefuture_role_prediction import load_prefuture_examples

    with _connect_sampling_read_db(db_path) as connection:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "future_effects" in tables:
        return load_prefuture_examples(db_path)
    return _load_validation_examples_from_current_schema(db_path)


def _load_validation_examples_from_current_schema(db_path: Path) -> list[Any]:
    from v6.evaluation.prefuture_role_prediction import PrefutureExample

    with _connect_sampling_read_db(db_path) as connection:
        connection.row_factory = sqlite3.Row
        prediction_rows = connection.execute(
            """
            SELECT interaction_id, context_level, context_signature, action,
                   predicted_family, actual_family, prediction_error,
                   context_contradiction, efficiency_no_effect_action,
                   efficiency_future_option_gain_per_cost, outcome_state, level_completed_event
            FROM prediction_results
            ORDER BY interaction_id ASC
            """
        ).fetchall()
        delta_rows = {
            int(row["id"]): row
            for row in connection.execute(
                """
                SELECT d.id, d.changed_cells, d.dx, d.dy, d.colors_added, d.colors_removed
                FROM deltas AS d
                """
            ).fetchall()
        }
        interaction_to_delta = {
            int(row["id"]): int(row["delta_id"])
            for row in connection.execute("SELECT id, delta_id FROM interactions WHERE delta_id IS NOT NULL").fetchall()
        }
    if not prediction_rows:
        return []
    context_action_counts: Counter[tuple[int, str, int, str]] = Counter()
    context_counts: Counter[str] = Counter()
    context_action_distribution: dict[str, Counter[int]] = defaultdict(Counter)
    context_family_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    for row in prediction_rows:
        context_signature = str(row["context_signature"] or "[]")
        actual_family = str(row["actual_family"] if row["actual_family"] is not None else "__none__")
        key = (
            int(row["context_level"] or 0),
            context_signature,
            int(row["action"] or 0),
            actual_family,
        )
        context_action_counts[key] += 1
        context_counts[context_signature] += 1
        context_action_distribution[context_signature][int(row["action"] or 0)] += 1
        context_family_distribution[context_signature][actual_family] += 1
        family_counts[actual_family] += 1

    examples: list[PrefutureExample] = []
    for row in prediction_rows:
        context_level = int(row["context_level"] or 0)
        context_signature = str(row["context_signature"] or "[]")
        action = int(row["action"] or 0)
        actual_family = str(row["actual_family"] if row["actual_family"] is not None else "__none__")
        key = (context_level, context_signature, action, actual_family)
        support_count = context_action_counts[key]
        prediction_error_rate = float(row["prediction_error"] or 0.0)
        delta = delta_rows.get(interaction_to_delta.get(int(row["interaction_id"] or 0), -1))
        examples.append(
            PrefutureExample(
                contingency_id=int(row["interaction_id"] or 0),
                contingency_key=(context_level, (context_signature,), action, actual_family),
                features={
                    "context_level": float(context_level),
                    "confidence": 0.0,
                    "support_count_log": math.log1p(float(support_count)),
                    "prediction_error_rate": prediction_error_rate,
                    "context_support": float(context_counts[context_signature]),
                    "action_entropy_at_context": _entropy(context_action_distribution[context_signature]),
                    "transformation_entropy_at_context": _entropy(context_family_distribution[context_signature]),
                    "transformation_family_support_log": math.log1p(float(family_counts[actual_family])),
                    "changed_cells": float(delta["changed_cells"] or 0.0) if delta is not None else 0.0,
                    "dx": float(delta["dx"] or 0.0) if delta is not None else 0.0,
                    "dy": float(delta["dy"] or 0.0) if delta is not None else 0.0,
                    "colors_added_count": float(len(_json_list(delta["colors_added"])) if delta is not None else 0),
                    "colors_removed_count": float(len(_json_list(delta["colors_removed"])) if delta is not None else 0),
                    "contingency_in_degree": 0.0,
                    "contingency_out_degree": 0.0,
                    "follows_in_degree": 0.0,
                    "follows_out_degree": 0.0,
                    "cooccurrence_degree": 0.0,
                    "clustering_coefficient": 0.0,
                    "degree_centrality": 0.0,
                    "pagerank": 0.0,
                },
                label=_derived_future_label(row),
            )
        )
    return examples


@contextmanager
def _patched_sqlite_busy_timeout(busy_timeout_ms: int):
    original_connect = sqlite3.connect

    def _connect_with_busy_timeout(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("timeout", max(1.0, float(busy_timeout_ms) / 1000.0))
        connection = original_connect(*args, **kwargs)
        connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        return connection

    sqlite3.connect = _connect_with_busy_timeout
    try:
        yield
    finally:
        sqlite3.connect = original_connect


def _derived_future_label(row: sqlite3.Row) -> str:
    outcome_state = str(row["outcome_state"] or "")
    if outcome_state == "GAME_OVER":
        return "COLLAPSE"
    if bool(row["level_completed_event"]) or outcome_state == "WIN":
        return "EXPAND"
    if bool(row["context_contradiction"]):
        return "RESTRICT"
    if float(row["efficiency_future_option_gain_per_cost"] or 0.0) > 0.0:
        return "EXPAND"
    if bool(row["efficiency_no_effect_action"]):
        return "PRESERVE"
    if float(row["prediction_error"] or 0.0) > 0.0:
        return "RESTRICT"
    return "PRESERVE"


def _entropy(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    value = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = float(count) / total
        value -= p * math.log(p)
    return float(value)


def _json_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []
