from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


H01_JSON_NAME = "h01_contingency_emergence_report.json"
H01_TXT_NAME = "h01_contingency_emergence_report.txt"
H01_MD_NAME = "h01_contingency_emergence.md"
H01_READY_JSON_NAME = "h01_ready_runs.json"
H01_READY_TXT_NAME = "h01_ready_runs.txt"
INPUT_REPORT_JSON_NAME = "interaction_sampling_v05c_report.json"
INPUT_REPORT_TXT_NAME = "interaction_sampling_v05c_report.txt"

H01_DEFAULTS: dict[str, Any] = {
    "hypothesis_id": "H01",
    "hypothesis_name": "Contingency emergence from interaction history",
    "hypothesis_statement": "Contingencies emerge from accumulated interaction history before higher-level structures are required.",
    "decision": "INCONCLUSIVE",
    "scientific_conclusion": "",
    "source_run_dir": "",
    "input_report_found": False,
    "input_report_txt_found": False,
    "db_found": False,
    "db_paths_total": 0,
    "db_paths_inspected": 0,
    "db_scan_truncated": False,
    "selected_db_paths": [],
    "tables_seen": [],
    "total_interaction_count": None,
    "memory_record_count": None,
    "contingency_candidate_count": None,
    "discovered_contingency_count": None,
    "stable_contingency_count": None,
    "prediction_accuracy": None,
    "mean_prediction_accuracy": None,
    "context_lift": None,
    "mean_context_lift": None,
    "contradiction_count": None,
    "repeated_contradiction_count": None,
    "context_expansion_suggested_count": None,
    "per_game_contingency_counts": {},
    "per_sampler_contingency_counts": {},
    "cross_game_contingency_presence": None,
    "percentage_games_with_stable_contingency": None,
    "percentage_samplers_with_stable_contingency": None,
    "stability_approximated": False,
    "evidence_for": [],
    "evidence_against": [],
    "missing_evidence": [],
    "acceptance_checks": {
        "interactions_present": None,
        "contingencies_present": None,
        "stable_contingencies_present": None,
        "multi_game_support": None,
        "multi_sampler_support": None,
        "prediction_or_context_signal_present": None,
    },
}


def evaluate_h01_contingency_emergence(run_dir: Path, output_dir: Path, *, memory_dir: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / INPUT_REPORT_JSON_NAME
    report_txt_path = run_dir / INPUT_REPORT_TXT_NAME
    report = _load_json(report_path)
    sqlite_paths = _find_sqlite_paths(run_dir)

    result = dict(H01_DEFAULTS)
    result["source_run_dir"] = str(run_dir)
    result["input_report_found"] = report is not None
    result["input_report_txt_found"] = report_txt_path.exists()
    result["db_found"] = bool(sqlite_paths)
    result["db_paths_total"] = len(sqlite_paths)
    result["evidence_source"] = "raw_epoch_db"

    if report is None and memory_dir is not None:
        compact_metrics = _extract_compact_memory_metrics(Path(memory_dir))
        result.update(compact_metrics)
        result["evidence_source"] = "compact_memory"
        result["decision"] = "PARTIALLY_VALID" if _gt(compact_metrics.get("stable_contingency_count"), 0) else "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H01 evaluated from compact memory after raw cleanup."
            if _gt(compact_metrics.get("stable_contingency_count"), 0)
            else "H01 remains inconclusive because compact memory lacks stable contingency evidence."
        )
        _finalize_h01_result(result, output_dir)
        return result
    if report is None:
        result["missing_evidence"].append(f"Required input report missing: {INPUT_REPORT_JSON_NAME}")
        result["scientific_conclusion"] = "H01 cannot be evaluated because the required interaction-sampling report is missing."
        _finalize_h01_result(result, output_dir)
        return result

    report_metrics = _extract_report_metrics(report)
    report_has_runs = bool(report.get("runs"))
    needs_db_fallback = (
        not report_has_runs
        or report_metrics.get("total_interaction_count") is None
        or report_metrics.get("stable_contingency_count") is None
    )
    db_metrics = _extract_db_metrics(sqlite_paths) if sqlite_paths and needs_db_fallback else {}
    result.update(report_metrics)
    if memory_dir is not None and not sqlite_paths:
        compact_metrics = _extract_compact_memory_metrics(Path(memory_dir))
        for key, value in compact_metrics.items():
            if result.get(key) is None or result.get(key) == {}:
                result[key] = value
        result["evidence_source"] = "mixed" if report is not None else "compact_memory"

    for key in (
        "total_interaction_count",
        "memory_record_count",
        "contingency_candidate_count",
        "discovered_contingency_count",
        "stable_contingency_count",
        "prediction_accuracy",
        "mean_prediction_accuracy",
        "context_lift",
        "mean_context_lift",
    ):
        if result.get(key) is None:
            result[key] = db_metrics.get(key)

    if not result["per_game_contingency_counts"]:
        result["per_game_contingency_counts"] = db_metrics.get("per_game_contingency_counts", {})
    if not result["per_sampler_contingency_counts"]:
        result["per_sampler_contingency_counts"] = db_metrics.get("per_sampler_contingency_counts", {})

    result["cross_game_contingency_presence"] = db_metrics.get("cross_game_contingency_presence")
    result["db_paths_inspected"] = db_metrics.get("db_paths_inspected", 0)
    result["selected_db_paths"] = db_metrics.get("selected_db_paths", [])
    result["tables_seen"] = db_metrics.get("tables_seen", [])
    result["stability_approximated"] = bool(db_metrics.get("stability_approximated", False)) and not bool(
        report_metrics.get("stable_contingency_count")
    )

    if (
        result.get("contingency_candidate_count") is None
        and result.get("stable_contingency_count") is not None
    ):
        result["contingency_candidate_count"] = result["stable_contingency_count"]
        result["stability_approximated"] = not report_has_runs
    if (
        result.get("discovered_contingency_count") is None
        and result.get("stable_contingency_count") is not None
    ):
        result["discovered_contingency_count"] = result["stable_contingency_count"]
        result["stability_approximated"] = not report_has_runs

    game_counts = result.get("per_game_contingency_counts", {})
    sampler_counts = result.get("per_sampler_contingency_counts", {})
    game_positive = sum(1 for value in game_counts.values() if int(value or 0) > 0)
    sampler_positive = sum(1 for value in sampler_counts.values() if int(value or 0) > 0)
    result["percentage_games_with_stable_contingency"] = (
        100.0 * game_positive / max(1, len(game_counts)) if game_counts else None
    )
    result["percentage_samplers_with_stable_contingency"] = (
        100.0 * sampler_positive / max(1, len(sampler_counts)) if sampler_counts else None
    )

    interactions_present = _gt(result.get("total_interaction_count"), 0)
    contingencies_present = _gt(result.get("discovered_contingency_count"), 0) or _gt(result.get("stable_contingency_count"), 0)
    stable_present = _gt(result.get("stable_contingency_count"), 0)
    multi_game_support = game_positive >= 2
    multi_sampler_support = sampler_positive >= 2
    signal_present = (
        _gt(result.get("prediction_accuracy"), 0.0)
        or _gt(result.get("mean_prediction_accuracy"), 0.0)
        or _gt(result.get("context_lift"), 0.0)
        or _gt(result.get("mean_context_lift"), 0.0)
    )

    result["acceptance_checks"] = {
        "interactions_present": interactions_present,
        "contingencies_present": contingencies_present,
        "stable_contingencies_present": stable_present,
        "multi_game_support": multi_game_support,
        "multi_sampler_support": multi_sampler_support,
        "prediction_or_context_signal_present": signal_present,
    }

    if not result["input_report_txt_found"]:
        result["missing_evidence"].append(f"Optional text report missing: {INPUT_REPORT_TXT_NAME}")
    if result.get("cross_game_contingency_presence") is None:
        result["missing_evidence"].append("Cross-game contingency identity is not derivable from current run artifacts.")

    if interactions_present and contingencies_present:
        result["evidence_for"].append(
            f"Contingencies emerge from interaction history in this run ({int(result.get('stable_contingency_count') or 0)} stable contingencies)."
        )
    if multi_game_support:
        result["evidence_for"].append(f"Stable contingencies appear across multiple games ({game_positive}).")
    if multi_sampler_support:
        result["evidence_for"].append(f"Stable contingencies appear across multiple samplers ({sampler_positive}).")
    if signal_present:
        result["evidence_for"].append("Prediction or context signal is non-zero in the sampled runs.")

    if interactions_present and not contingencies_present:
        result["evidence_against"].append("Interactions are present but no contingencies were detected.")
    elif stable_present and not (multi_game_support or multi_sampler_support):
        result["evidence_against"].append("Contingencies are present but remain sparse across games and samplers.")
    if not signal_present and contingencies_present:
        result["evidence_against"].append("Contingencies are present but prediction/context signal is weak or unavailable.")

    if not interactions_present and not result["db_found"] and not report.get("runs"):
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H01 remains inconclusive because the current run does not expose enough interaction or contingency evidence."
        )
    elif interactions_present and not contingencies_present:
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H01 is not supported in this run because interactions are present but no contingencies emerge."
        )
    elif stable_present and signal_present and (multi_game_support or multi_sampler_support):
        result["decision"] = "VALID"
        result["scientific_conclusion"] = (
            "H01 is supported in this run. Stable contingencies emerge from interaction history across multiple games or samplers with non-zero prediction/context signal."
        )
    elif contingencies_present:
        result["decision"] = "PARTIALLY_VALID"
        if result["stability_approximated"]:
            result["scientific_conclusion"] = (
                "H01 is partially supported in this run. Contingencies emerge from interaction history, but stability evidence is approximate."
            )
        else:
            result["scientific_conclusion"] = (
                "H01 is partially supported in this run. Contingencies emerge from interaction history, but they remain sparse across games or samplers."
            )
    else:
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H01 remains inconclusive because usable contingency evidence could not be established from the current artifacts."
        )

    _finalize_h01_result(result, output_dir)
    return result


def _extract_compact_memory_metrics(memory_dir: Path) -> dict[str, Any]:
    current_state = memory_dir / "current_state.sqlite"
    if not current_state.exists():
        return {}
    with sqlite3.connect(current_state) as connection:
        per_game = dict(connection.execute("SELECT COALESCE(game, 'unknown'), COUNT(*) FROM stable_contingencies GROUP BY COALESCE(game, 'unknown')").fetchall())
        per_sampler = dict(connection.execute("SELECT COALESCE(sampler, 'unknown'), COUNT(*) FROM stable_contingencies GROUP BY COALESCE(sampler, 'unknown')").fetchall())
        stable_count = int(connection.execute("SELECT COUNT(*) FROM stable_contingencies WHERE support_count >= 20").fetchone()[0])
        summary_row = connection.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'total_interactions_seen'"
        ).fetchone()
        interaction_count = 0
        if summary_row is not None and summary_row[0] is not None:
            try:
                interaction_count = int(json.loads(summary_row[0]))
            except Exception:
                try:
                    interaction_count = int(summary_row[0])
                except Exception:
                    interaction_count = 0
        return {
            "total_interaction_count": interaction_count if interaction_count > 0 else None,
            "stable_contingency_count": stable_count,
            "contingency_candidate_count": int(connection.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
            "discovered_contingency_count": int(connection.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
            "per_game_contingency_counts": {str(key): int(value) for key, value in per_game.items()},
            "per_sampler_contingency_counts": {str(key): int(value) for key, value in per_sampler.items()},
            "mean_prediction_accuracy": None,
        }


def find_h01_ready_runs(runs_root: Path, output_dir: Path) -> dict:
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for report_path in sorted(runs_root.rglob(INPUT_REPORT_JSON_NAME)):
        run_dir = report_path.parent
        report = _load_json(report_path)
        sqlite_paths = _find_sqlite_paths(run_dir)
        metrics = _extract_report_metrics(report)
        has_contingency_evidence = any(
            _gt(metrics.get(field), 0)
            for field in ("contingency_candidate_count", "discovered_contingency_count", "stable_contingency_count")
        ) or bool(report and report.get("runs"))
        h01_ready = bool(report) and (
            _gt(metrics.get("total_interaction_count"), 0) or has_contingency_evidence or bool(sqlite_paths)
        )
        runs.append(
            {
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "has_sqlite_db": bool(sqlite_paths),
                "sqlite_db_count": len(sqlite_paths),
                "memory_record_count": metrics.get("memory_record_count"),
                "total_interaction_count": metrics.get("total_interaction_count"),
                "discovered_contingency_count": metrics.get("discovered_contingency_count"),
                "stable_contingency_count": metrics.get("stable_contingency_count"),
                "prediction_accuracy": metrics.get("prediction_accuracy"),
                "context_lift": metrics.get("context_lift"),
                "h01_ready": h01_ready,
                "recommended_output_dir": str(output_dir),
            }
        )

    runs.sort(
        key=lambda row: (
            not bool(row["h01_ready"]),
            -(int(row.get("stable_contingency_count") or 0)),
            -(int(row.get("discovered_contingency_count") or 0)),
            -(int(row.get("total_interaction_count") or 0)),
            row["run_dir"],
        )
    )
    ready_runs = [row for row in runs if row["h01_ready"]]
    recommended_run = None
    if ready_runs:
        recommended_run = {
            "run_dir": ready_runs[0]["run_dir"],
            "recommended_output_dir": ready_runs[0]["recommended_output_dir"],
        }

    result = {
        "runs_root": str(runs_root),
        "candidate_count": len(runs),
        "ready_count": len(ready_runs),
        "runs": runs,
        "recommended_run": recommended_run,
    }
    (output_dir / H01_READY_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / H01_READY_TXT_NAME).write_text(_format_h01_ready_text(result), encoding="utf-8")
    return result


def _extract_report_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    validation = report.get("validation", {}) if isinstance(report, dict) else {}
    runs = [row for row in report.get("runs", []) if not isinstance(row, dict) or row.get("run_status", "ok") == "ok"]
    if not runs and validation:
        return {
            "memory_record_count": _coerce_int(validation.get("memory_record_count")),
            "prediction_accuracy": _coerce_float(validation.get("prediction_accuracy")),
            "mean_prediction_accuracy": _coerce_float(validation.get("mean_prediction_accuracy")),
            "context_lift": _coerce_float(validation.get("context_lift")),
            "mean_context_lift": _coerce_float(validation.get("mean_context_lift")),
            "contradiction_count": _coerce_int(validation.get("context_contradiction_count")),
            "repeated_contradiction_count": _coerce_int(validation.get("repeated_contradiction_count")),
            "context_expansion_suggested_count": _coerce_int(validation.get("context_expansion_suggested_count")),
        }

    per_game: dict[str, int] = {}
    per_sampler: dict[str, int] = {}
    total_interaction_count = 0
    memory_record_count = 0
    stable_contingency_count = 0
    discovered_contingency_count = 0
    prediction_values: list[float] = []
    context_lift_values: list[float] = []

    for row in runs:
        if not isinstance(row, dict):
            continue
        total_interaction_count += int(row.get("total_interactions", 0) or 0)
        memory_record_count += int(row.get("memory_record_count", 0) or 0)
        stable = int(row.get("stable_contingency_count", 0) or 0)
        discovered = int(row.get("discovered_contingency_count", stable) or 0)
        stable_contingency_count += stable
        discovered_contingency_count += discovered
        if row.get("prediction_accuracy") is not None:
            prediction_values.append(float(row["prediction_accuracy"]))
        if row.get("context_lift") is not None:
            context_lift_values.append(float(row["context_lift"]))
        game = str(row.get("game", "unknown"))
        sampler = str(row.get("sampler_name", "unknown"))
        per_game[game] = per_game.get(game, 0) + stable
        per_sampler[sampler] = per_sampler.get(sampler, 0) + stable

    return {
        "total_interaction_count": total_interaction_count if runs else _coerce_int(validation.get("total_interaction_count")),
        "memory_record_count": memory_record_count if runs else _coerce_int(validation.get("memory_record_count")),
        "contingency_candidate_count": _coerce_int(validation.get("contingency_candidate_count")),
        "discovered_contingency_count": (
            discovered_contingency_count if runs else _coerce_int(validation.get("discovered_contingency_count"))
        ),
        "stable_contingency_count": (
            stable_contingency_count if runs else _coerce_int(validation.get("stable_contingency_count"))
        ),
        "prediction_accuracy": _coerce_float(validation.get("prediction_accuracy")) or _mean_or_none(prediction_values),
        "mean_prediction_accuracy": _coerce_float(validation.get("mean_prediction_accuracy")) or _mean_or_none(prediction_values),
        "context_lift": _coerce_float(validation.get("context_lift")) or _mean_or_none(context_lift_values),
        "mean_context_lift": _coerce_float(validation.get("mean_context_lift")) or _mean_or_none(context_lift_values),
        "contradiction_count": _coerce_int(validation.get("context_contradiction_count")),
        "repeated_contradiction_count": _coerce_int(validation.get("repeated_contradiction_count")),
        "context_expansion_suggested_count": _coerce_int(validation.get("context_expansion_suggested_count")),
        "per_game_contingency_counts": per_game,
        "per_sampler_contingency_counts": per_sampler,
    }


def _extract_db_metrics(sqlite_paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "db_paths_inspected": 0,
        "selected_db_paths": [],
        "tables_seen": [],
        "per_game_contingency_counts": {},
        "per_sampler_contingency_counts": {},
        "stability_approximated": False,
    }
    total_interactions = 0
    contingency_rows = 0
    tables_seen: set[str] = set()
    per_game: dict[str, int] = {}
    per_sampler: dict[str, int] = {}

    for path in sqlite_paths:
        try:
            with sqlite3.connect(path) as connection:
                tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                tables_seen.update(tables)
                interactions = 0
                contingencies = 0
                if "interactions" in tables:
                    interactions = int(connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
                if "contingencies" in tables:
                    contingencies = int(connection.execute("SELECT COUNT(*) FROM contingencies").fetchone()[0])
                total_interactions += interactions
                contingency_rows += contingencies
        except sqlite3.DatabaseError:
            continue

        result["db_paths_inspected"] += 1
        result["selected_db_paths"].append(str(path))
        game, sampler = _infer_game_sampler_from_path(path)
        if contingencies:
            if game:
                per_game[game] = per_game.get(game, 0) + contingencies
            if sampler:
                per_sampler[sampler] = per_sampler.get(sampler, 0) + contingencies

    result["tables_seen"] = sorted(tables_seen)
    result["total_interaction_count"] = total_interactions if result["db_paths_inspected"] else None
    result["contingency_candidate_count"] = contingency_rows if result["db_paths_inspected"] else None
    result["discovered_contingency_count"] = contingency_rows if result["db_paths_inspected"] else None
    result["stable_contingency_count"] = contingency_rows if result["db_paths_inspected"] else None
    result["per_game_contingency_counts"] = per_game
    result["per_sampler_contingency_counts"] = per_sampler
    result["stability_approximated"] = bool(result["db_paths_inspected"])
    result["cross_game_contingency_presence"] = None
    return result


def _find_sqlite_paths(run_dir: Path) -> list[Path]:
    paths = []
    for pattern in ("*.sqlite", "*.db", "*.sqlite3"):
        paths.extend(run_dir.rglob(pattern))
    return sorted(set(paths))


def _infer_game_sampler_from_path(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    game = None
    sampler = None
    if "sampling_v05c" in parts:
        idx = parts.index("sampling_v05c")
        if len(parts) > idx + 2:
            game = parts[idx + 1]
            sampler = parts[idx + 2]
    elif len(parts) >= 4:
        game = parts[-4]
        sampler = parts[-3]
    return game, sampler


def _finalize_h01_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / H01_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = _format_h01_text(result)
    (output_dir / H01_TXT_NAME).write_text(text, encoding="utf-8")
    (output_dir / H01_MD_NAME).write_text(text, encoding="utf-8")


def _format_h01_text(result: dict[str, Any]) -> str:
    lines = [
        "# H01 Hypothesis Report",
        "",
        "Hypothesis statement:",
        result["hypothesis_statement"],
        "",
        "Decision:",
        str(result["decision"]),
        "",
        "Core metrics:",
        f"- total interaction count: {_fmt(result.get('total_interaction_count'))}",
        f"- memory_record_count: {_fmt(result.get('memory_record_count'))}",
        f"- contingency_candidate_count: {_fmt(result.get('contingency_candidate_count'))}",
        f"- discovered_contingency_count: {_fmt(result.get('discovered_contingency_count'))}",
        f"- stable_contingency_count: {_fmt(result.get('stable_contingency_count'))}",
        f"- mean_prediction_accuracy: {_fmt(result.get('mean_prediction_accuracy'))}",
        f"- mean_context_lift: {_fmt(result.get('mean_context_lift'))}",
        f"- contradiction_count: {_fmt(result.get('contradiction_count'))}",
        f"- repeated_contradiction_count: {_fmt(result.get('repeated_contradiction_count'))}",
        f"- context_expansion_suggested_count: {_fmt(result.get('context_expansion_suggested_count'))}",
        "",
        "Coverage:",
        f"- per-game contingency counts: {json.dumps(result.get('per_game_contingency_counts', {}), sort_keys=True)}",
        f"- per-sampler contingency counts: {json.dumps(result.get('per_sampler_contingency_counts', {}), sort_keys=True)}",
        f"- cross-game contingency presence: {_fmt(result.get('cross_game_contingency_presence'))}",
        f"- percentage of games with at least one stable contingency: {_fmt(result.get('percentage_games_with_stable_contingency'))}",
        f"- percentage of samplers with at least one stable contingency: {_fmt(result.get('percentage_samplers_with_stable_contingency'))}",
        "",
        "Evidence for:",
    ]
    lines.extend([f"- {item}" for item in result.get("evidence_for", [])] or ["- none"])
    lines.extend(["", "Evidence against:"])
    lines.extend([f"- {item}" for item in result.get("evidence_against", [])] or ["- none"])
    lines.extend(["", "Missing evidence:"])
    lines.extend([f"- {item}" for item in result.get("missing_evidence", [])] or ["- none"])
    lines.extend(["", "Acceptance checklist:"])
    lines.extend([f"- {key}: {value}" for key, value in result.get("acceptance_checks", {}).items()])
    lines.extend(["", "Final scientific conclusion:", result.get("scientific_conclusion", "")])
    return "\n".join(lines) + "\n"


def _format_h01_ready_text(result: dict[str, Any]) -> str:
    lines = [
        f"Total reports found: {result['candidate_count']}",
        f"Ready count: {result['ready_count']}",
        "",
    ]
    if result.get("recommended_run") is None:
        lines.append("No H01-ready existing v05c run found.")
    else:
        lines.extend(
            [
                "Top recommended run:",
                result["recommended_run"]["run_dir"],
                "",
                "Run:",
                "PYTHONPATH=src python -m v6.cli hypothesis-h01-report \\",
                f"  --run-dir {result['recommended_run']['run_dir']} \\",
                f"  --output-dir {result['recommended_run']['recommended_output_dir']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gt(value: Any, threshold: float) -> bool:
    try:
        return value is not None and float(value) > float(threshold)
    except (TypeError, ValueError):
        return False


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "null" if value is None else str(value)


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
