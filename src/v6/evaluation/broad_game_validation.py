from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.evaluation.future_effects import _run_future_effect_job
from v6.evaluation.id_free_prefuture_validation import (
    ID_FREE_FEATURE_SETS,
    evaluate_id_free_config,
    forbidden_future_feature_check,
    forbidden_id_feature_check,
)
from v6.evaluation.prefuture_role_prediction import PREFUTURE_CLASSIFIERS
from v6.evaluation.role_validation import _db_path
from v6.evaluation.id_free_prefuture_validation import _load_id_free_examples


CONTROL_GAMES = ("ez01", "ez02", "ez03", "ez04")
ORIGINAL_PRIMARY_GAMES = ("va02", "mo01", "ic01")
MOVEMENT_CORE_GAMES = (
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
    "ic02",
    "ic03",
    "va01",
    "va03",
    "nw01",
    "bd01",
    "gr01",
    "dt01",
    "wk01",
    "rf01",
    "zq01",
    "hm01",
    "ex01",
    "dl01",
    "hd01",
)
OPTIONAL_EXPLORATORY_GAMES = ("ul01", "sv01", "sk01")
GAME_PRESETS = {
    "control": CONTROL_GAMES,
    "original_primary": ORIGINAL_PRIMARY_GAMES,
    "movement_core": MOVEMENT_CORE_GAMES,
    "broad": CONTROL_GAMES + ORIGINAL_PRIMARY_GAMES + MOVEMENT_CORE_GAMES,
    "all": CONTROL_GAMES + ORIGINAL_PRIMARY_GAMES + MOVEMENT_CORE_GAMES + OPTIONAL_EXPLORATORY_GAMES,
}
GAME_FAMILIES = {
    "movement_baseline": CONTROL_GAMES,
    "collection_survival": ("tt01",),
    "push_crate": ("pb01", "pb02", "pb03"),
    "switches": ("fs01", "fs02", "fs03"),
    "teleport": ("tp01", "tp02", "tp03"),
    "slide": ("ic01", "ic02", "ic03"),
    "coverage_path": ("va01", "va02", "va03", "bd01", "hm01"),
    "forced_or_modified_movement": ("nw01", "rf01", "mo01", "dl01"),
    "timing_or_hazard": ("zq01", "hd01"),
    "topology_or_environment": ("gr01", "dt01", "wk01", "ex01"),
    "optional_exploratory": OPTIONAL_EXPLORATORY_GAMES,
}
SIMPLEST_FEATURE_ORDER = {
    "contingency_only_no_ids": 0,
    "transformation_signature_no_ids": 1,
    "graph_only_no_ids": 2,
    "contingency_plus_graph_no_ids": 3,
    "contingency_plus_transformation_signature_no_ids": 4,
    "all_prefuture_no_ids": 5,
}
V05_MODES = ("deterministic", "vector")


@dataclass(frozen=True)
class BroadValidationConfig:
    games: tuple[str, ...] = GAME_PRESETS["broad"]
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
    workers: int | None = 60
    reuse_v02: bool = True


def parse_game_selector(selector: str) -> tuple[str, ...]:
    value = selector.strip()
    if value in GAME_PRESETS:
        return tuple(dict.fromkeys(GAME_PRESETS[value]))
    games = tuple(item.strip() for item in value.split(",") if item.strip())
    if not games:
        raise ValueError("expected game preset or comma-separated game list")
    return games


def family_for_game(game: str) -> str:
    for family, games in GAME_FAMILIES.items():
        if game in games:
            return family
    return "unknown"


def run_broad_validation_v05(config: BroadValidationConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    _ensure_broad_dbs(config, db_dir)
    rows: list[dict] = []
    for game in config.games:
        try:
            train = []
            for seed in config.train_seeds:
                train.extend(_load_id_free_examples(_db_path(db_dir, game, seed, config.steps, config.horizon)))
            test = _load_id_free_examples(_db_path(db_dir, game, config.test_seed, config.steps, config.horizon))
            if not train or not test:
                raise ValueError("insufficient train/test stable contingencies")
            for mode in V05_MODES:
                if mode == "vector":
                    rows.append(_failed_row(game, mode, config, "vector_prefuture_mode_not_supported"))
                    continue
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
                        if row is None:
                            continue
                        rows.append(_v05_row(row, mode=mode, run_status="ok", failure_reason=""))
        except Exception as exc:
            rows.append(_failed_row(game, "deterministic", config, f"{type(exc).__name__}: {exc}"))
            rows.append(_failed_row(game, "vector", config, "skipped_after_game_failure"))
    write_broad_reports(rows, output_dir=output_dir)
    return rows


def write_broad_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = best_configs(rows)
    summary_by_game = summary_by_game_rows(rows, best)
    summary_by_family = summary_by_family_rows(summary_by_game)
    payload = {
        "runs": rows,
        "best_configs": best,
        "summary_by_game": summary_by_game,
        "summary_by_family": summary_by_family,
        "validation": validation_summary(summary_by_game),
        "analysis": analysis_answers(summary_by_game, summary_by_family, best),
    }
    (output / "broad_game_validation_v05_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, output / "broad_game_validation_v05_report.csv")
    _write_csv(best, output / "broad_game_validation_v05_best.csv")
    _write_csv(summary_by_game, output / "broad_game_validation_v05_summary_by_game.csv")
    _write_csv(summary_by_family, output / "broad_game_validation_v05_summary_by_family.csv")
    (output / "broad_game_validation_v05_report.txt").write_text(_format_text(payload), encoding="utf-8")


def best_configs(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("run_status") == "ok":
            by_key[(row["game"], row["mode"])].append(row)
    best = [
        max(
            items,
            key=lambda row: (
                _num(row["id_free_macro_f1"]),
                _num(row["id_free_accuracy"]),
                _num(row["non_preserve_recall_any"]),
                -SIMPLEST_FEATURE_ORDER.get(str(row["feature_set"]), 99),
            ),
        )
        for _key, items in sorted(by_key.items())
    ]
    failed_keys = {(row["game"], row["mode"]) for row in rows if row.get("run_status") != "ok"}
    for game, mode in sorted(failed_keys - set(by_key)):
        candidates = [row for row in rows if row["game"] == game and row["mode"] == mode]
        if candidates:
            best.append(candidates[0])
    return sorted(best, key=lambda row: (row["game"], row["mode"]))


def summary_by_game_rows(rows: list[dict], best: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in best:
        passed = game_passes(row)
        output.append(
            {
                "game": row["game"],
                "family": family_for_game(row["game"]),
                "mode": row["mode"],
                "run_status": row["run_status"],
                "game_passed": passed,
                "best_feature_set": row.get("feature_set"),
                "best_classifier": row.get("classifier"),
                "id_free_accuracy": row.get("id_free_accuracy"),
                "majority_baseline_accuracy": row.get("majority_baseline_accuracy"),
                "id_free_macro_f1": row.get("id_free_macro_f1"),
                "majority_baseline_macro_f1": row.get("majority_baseline_macro_f1"),
                "non_preserve_recall_any": row.get("non_preserve_recall_any"),
                "failure_reason": row.get("failure_reason") or diagnose_failure(row),
            }
        )
    return output


def summary_by_family_rows(game_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in game_rows:
        if row["mode"] == "deterministic":
            by_family[row["family"]].append(row)
    for family, items in sorted(by_family.items()):
        passed = [row for row in items if row["game_passed"]]
        rows.append(
            {
                "family": family,
                "games_tested": len(items),
                "games_passed": len(passed),
                "pass_rate": 0.0 if not items else len(passed) / len(items),
                "mean_id_free_accuracy": _mean(row.get("id_free_accuracy") for row in items),
                "mean_majority_baseline_accuracy": _mean(row.get("majority_baseline_accuracy") for row in items),
                "mean_id_free_macro_f1": _mean(row.get("id_free_macro_f1") for row in items),
                "mean_non_preserve_recall_any": _mean(row.get("non_preserve_recall_any") for row in items),
                "strongest_game": _extreme_game(items, largest=True),
                "weakest_game": _extreme_game(items, largest=False),
            }
        )
    return rows


def validation_summary(game_rows: list[dict]) -> dict:
    deterministic = [
        row
        for row in game_rows
        if row["mode"] == "deterministic"
        and row["family"] != "movement_baseline"
        and row["game"] not in CONTROL_GAMES
        and row["run_status"] == "ok"
    ]
    tested = len(deterministic)
    passed = sum(1 for row in deterministic if row["game_passed"])
    families_passed = sorted({row["family"] for row in deterministic if row["game_passed"]})
    pass_rate = 0.0 if tested == 0 else passed / tested
    return {
        "non_control_broad_games_tested": tested,
        "non_control_broad_games_passed": passed,
        "pass_rate": pass_rate,
        "weak_pass": pass_rate >= 0.30,
        "strong_pass": pass_rate >= 0.50,
        "very_strong_pass": pass_rate >= 0.50 and len(families_passed) >= 3,
        "families_passed": families_passed,
    }


def analysis_answers(game_rows: list[dict], family_rows: list[dict], best: list[dict]) -> dict:
    validation = validation_summary(game_rows)
    deterministic_ok = [row for row in best if row.get("mode") == "deterministic" and row.get("run_status") == "ok"]
    feature_counts = Counter(row["feature_set"] for row in deterministic_ok)
    classifier_counts = Counter(row["classifier"] for row in deterministic_ok)
    failed_families = [row["family"] for row in family_rows if int(row["games_passed"]) == 0]
    return {
        "broad_validation_games_passed": validation["non_control_broad_games_passed"],
        "families_passed": validation["families_passed"],
        "families_failed": failed_families,
        "best_feature_set_overall": feature_counts.most_common(1)[0][0] if feature_counts else None,
        "best_classifier_overall": classifier_counts.most_common(1)[0][0] if classifier_counts else None,
        "id_free_beat_majority_beyond_va02_mo01": any(
            row["game"] not in {"va02", "mo01"}
            and row["mode"] == "deterministic"
            and row["run_status"] == "ok"
            and _num(row["id_free_accuracy"]) > _num(row["majority_baseline_accuracy"])
            for row in best
        ),
        "non_preserve_recovered_across_families": sorted(
            {
                family_for_game(row["game"])
                for row in best
                if row["mode"] == "deterministic"
                and row["run_status"] == "ok"
                and _num(row["non_preserve_recall_any"]) > 0.0
            }
        ),
    }


def game_passes(row: dict) -> bool:
    return (
        row.get("run_status") == "ok"
        and bool(row.get("forbidden_future_feature_check_passed"))
        and bool(row.get("forbidden_id_feature_check_passed"))
        and _num(row.get("id_free_accuracy")) > _num(row.get("majority_baseline_accuracy"))
        and _num(row.get("id_free_macro_f1")) > _num(row.get("majority_baseline_macro_f1"))
        and _num(row.get("non_preserve_recall_any")) > 0.0
    )


def diagnose_failure(row: dict) -> str:
    if row.get("run_status") != "ok":
        return str(row.get("failure_reason") or "run failed")
    train_dist = _json_or_dict(row.get("class_distribution_train"))
    test_dist = _json_or_dict(row.get("class_distribution_test"))
    if sum(count for label, count in test_dist.items() if label != "PRESERVE") == 0:
        return "PRESERVE-only distribution"
    if sum(count for label, count in train_dist.items() if label != "PRESERVE") < 3:
        return "insufficient non-PRESERVE samples"
    if _num(row.get("contingency_baseline_accuracy")) <= _num(row.get("majority_baseline_accuracy")):
        return "weak contingency discovery"
    if _num(row.get("non_preserve_recall_any")) <= 0.0:
        return "poor train/test role alignment"
    return "other"


def _ensure_broad_dbs(config: BroadValidationConfig, db_dir: Path) -> None:
    all_seeds = tuple(config.train_seeds) + (int(config.test_seed),)
    jobs: list[dict] = []
    order = 0
    for game in config.games:
        for seed in all_seeds:
            db_path = _db_path(db_dir, game, seed, config.steps, config.horizon)
            if config.reuse_v02 and _db_ready(db_path):
                continue
            if db_path.exists():
                db_path.unlink()
            jobs.append(
                {
                    "order": order,
                    "game": game,
                    "seed": int(seed),
                    "steps": int(config.steps),
                    "horizon": int(config.horizon),
                    "threshold": float(config.threshold),
                    "collapse_threshold": float(config.collapse_threshold),
                    "context_length": int(config.context_length),
                    "support_threshold": int(config.support_threshold),
                    "confidence_threshold": float(config.confidence_threshold),
                    "db_path": str(db_path),
                    "env_root": config.env_root,
                }
            )
            order += 1
    if not jobs:
        return
    workers = max(1, int(config.workers or 60))
    print(f"running {len(jobs)} broad future-effect jobs with workers={workers}", file=sys.stderr, flush=True)
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


def _v05_row(row: dict, *, mode: str, run_status: str, failure_reason: str) -> dict:
    output = {
        "game": row["game"],
        "mode": mode,
        "feature_set": row["feature_set"],
        "classifier": row["classifier"],
        "train_seeds": row["train_seeds"],
        "test_seed": row["test_seed"],
        "steps": row["steps"],
        "horizon": row["horizon"],
        "train_sample_count": row["train_sample_count"],
        "test_sample_count": row["test_sample_count"],
        "class_distribution_train": row["class_distribution_train"],
        "class_distribution_test": row["class_distribution_test"],
        "majority_baseline_accuracy": row["majority_baseline_accuracy"],
        "majority_baseline_macro_f1": row["majority_baseline_macro_f1"],
        "contingency_baseline_accuracy": row["contingency_baseline_accuracy"],
        "random_stratified_accuracy": row["random_stratified_accuracy"],
        "id_free_accuracy": row["id_free_accuracy"],
        "id_free_macro_f1": row["id_free_macro_f1"],
        "id_free_vs_majority_delta": row["id_free_vs_majority_delta"],
        "id_free_vs_contingency_delta": row["id_free_vs_contingency_delta"],
        "preserve_precision": row["preserve_precision"],
        "preserve_recall": row["preserve_recall"],
        "expand_precision": row["expand_precision"],
        "expand_recall": row["expand_recall"],
        "restrict_precision": row["restrict_precision"],
        "restrict_recall": row["restrict_recall"],
        "collapse_precision": row["collapse_precision"],
        "collapse_recall": row["collapse_recall"],
        "non_preserve_recall_any": row["non_preserve_recall_any"],
        "non_preserve_class_count_train": _non_preserve_count(row["class_distribution_train"]),
        "non_preserve_class_count_test": _non_preserve_count(row["class_distribution_test"]),
        "confusion_matrix_json": row["confusion_matrix_json"],
        "forbidden_future_feature_check_passed": row["forbidden_future_feature_check_passed"],
        "forbidden_id_feature_check_passed": row["forbidden_id_feature_check_passed"],
        "run_status": run_status,
        "failure_reason": failure_reason,
    }
    return output


def _failed_row(game: str, mode: str, config: BroadValidationConfig, reason: str) -> dict:
    return {
        "game": game,
        "mode": mode,
        "feature_set": "",
        "classifier": "",
        "train_seeds": list(config.train_seeds),
        "test_seed": int(config.test_seed),
        "steps": int(config.steps),
        "horizon": int(config.horizon),
        "train_sample_count": 0,
        "test_sample_count": 0,
        "class_distribution_train": {},
        "class_distribution_test": {},
        "majority_baseline_accuracy": None,
        "majority_baseline_macro_f1": None,
        "contingency_baseline_accuracy": None,
        "random_stratified_accuracy": None,
        "id_free_accuracy": None,
        "id_free_macro_f1": None,
        "id_free_vs_majority_delta": None,
        "id_free_vs_contingency_delta": None,
        "preserve_precision": None,
        "preserve_recall": None,
        "expand_precision": None,
        "expand_recall": None,
        "restrict_precision": None,
        "restrict_recall": None,
        "collapse_precision": None,
        "collapse_recall": None,
        "non_preserve_recall_any": None,
        "non_preserve_class_count_train": 0,
        "non_preserve_class_count_test": 0,
        "confusion_matrix_json": {},
        "forbidden_future_feature_check_passed": True,
        "forbidden_id_feature_check_passed": True,
        "run_status": "failed",
        "failure_reason": reason,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field])
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fieldnames
                }
            )


def _format_text(payload: dict) -> str:
    validation = payload["validation"]
    analysis = payload["analysis"]
    lines = [
        "ARC-AGI3 v0.5 Broad Game Validation Report",
        "offline validation only; ID-free pre-future default; vector mode recorded as unsupported where applicable",
        "",
        f"non_control_passed={validation['non_control_broad_games_passed']}/{validation['non_control_broad_games_tested']} "
        f"pass_rate={validation['pass_rate']:.3f}",
        f"weak_pass={validation['weak_pass']} strong_pass={validation['strong_pass']} very_strong_pass={validation['very_strong_pass']}",
        f"families_passed={validation['families_passed']}",
        "",
        "best deterministic configs:",
    ]
    for row in payload["summary_by_game"]:
        if row["mode"] != "deterministic":
            continue
        lines.append(
            f"{row['game']} family={row['family']} status={row['run_status']} passed={row['game_passed']} "
            f"feature={row['best_feature_set']} classifier={row['best_classifier']} "
            f"acc={_fmt(row['id_free_accuracy'])} baseline={_fmt(row['majority_baseline_accuracy'])} "
            f"macro={_fmt(row['id_free_macro_f1'])} nonP={_fmt(row['non_preserve_recall_any'])} "
            f"failure={row['failure_reason']}"
        )
    lines.append("")
    lines.append(f"analysis={analysis}")
    return "\n".join(lines) + "\n"


def _non_preserve_count(distribution: dict) -> int:
    return int(sum(value for label, value in distribution.items() if label != "PRESERVE"))


def _json_or_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return {}


def _num(value) -> float:
    return float(value) if value is not None and value != "" else 0.0


def _mean(values) -> float:
    nums = [_num(value) for value in values if value is not None and value != ""]
    return 0.0 if not nums else float(np.mean(nums))


def _extreme_game(rows: list[dict], *, largest: bool) -> str | None:
    ok = [row for row in rows if row["run_status"] == "ok"]
    if not ok:
        return None
    return max(ok, key=lambda row: _num(row.get("id_free_macro_f1")))["game"] if largest else min(ok, key=lambda row: _num(row.get("id_free_macro_f1")))["game"]


def _fmt(value) -> str:
    return "NA" if value is None or value == "" else f"{float(value):.3f}"
