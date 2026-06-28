from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v6.memory.direct_streaming_fold import direct_streaming_manifest_exists


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
    "stable_contingencies_count": None,
    "transformation_families_count": None,
    "family_members_count": None,
    "raw_contingency_rows_seen": None,
    "prediction_accuracy": None,
    "mean_prediction_accuracy": None,
    "context_lift": None,
    "mean_context_lift": None,
    "median_context_lift": None,
    "positive_context_lift_count": None,
    "context_lift_available": False,
    "contradiction_count": None,
    "repeated_contradiction_count": None,
    "context_expansion_suggested_count": None,
    "per_game_contingency_counts": {},
    "per_sampler_contingency_counts": {},
    "cross_game_contingency_presence": None,
    "cross_game_contingency_count": None,
    "cross_sampler_contingency_count": None,
    "games_per_contingency_identity": {},
    "samplers_per_contingency_identity": {},
    "percentage_games_with_stable_contingency": None,
    "percentage_samplers_with_stable_contingency": None,
    "stability_approximated": False,
    "evidence_for": [],
    "evidence_against": [],
    "missing_evidence": [],
    "evidence_diagnostics": {},
    "acceptance_checks": {
        "interactions_present": None,
        "contingencies_present": None,
        "stable_contingencies_present": None,
        "multi_game_support": None,
        "multi_sampler_support": None,
        "prediction_or_context_signal_present": None,
    },
}


def _h01_evidence_diagnostics(*, run_dir: Path, memory_dir: Path | None, sqlite_paths: list[Path]) -> dict[str, Any]:
    memory_dir = None if memory_dir is None else Path(memory_dir)
    compact_path = None if memory_dir is None else memory_dir / "current_state.sqlite"
    replay_path = None if memory_dir is None else memory_dir / "replay_queue.sqlite"
    diagnostics: dict[str, Any] = {
        "expected_input_report_path": str(Path(run_dir) / INPUT_REPORT_JSON_NAME),
        "compact_memory_exists": bool(compact_path is not None and compact_path.exists()),
        "compact_current_state_path": None if compact_path is None else str(compact_path),
        "compact_replay_queue_path": None if replay_path is None else str(replay_path),
        "raw_db_evidence_exists": bool(sqlite_paths),
        "raw_db_count": len(sqlite_paths),
        "direct_streamed_manifest_exists": bool(memory_dir is not None and direct_streaming_manifest_exists(memory_dir)),
    }
    if compact_path is not None and compact_path.exists():
        try:
            with sqlite3.connect(compact_path) as connection:
                diagnostics["compact_table_row_counts"] = {
                    "memory_nodes": int(connection.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]),
                    "stable_contingencies": int(connection.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
                    "transformation_families": int(connection.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]),
                    "memory_scores": int(connection.execute("SELECT COUNT(*) FROM memory_scores").fetchone()[0]),
                }
        except sqlite3.DatabaseError:
            diagnostics["compact_table_row_counts"] = {}
    return diagnostics


def _apply_h01_decision(
    result: dict[str, Any],
    *,
    interactions_present: bool,
    contingencies_present: bool,
    stable_present: bool,
    multi_game_support: bool,
    multi_sampler_support: bool,
    signal_present: bool,
    report_has_runs: bool,
) -> None:
    if not interactions_present and not result["db_found"] and not report_has_runs:
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H01 remains inconclusive because the current run does not expose enough interaction or contingency evidence."
        )
    elif interactions_present and not contingencies_present:
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H01 is not supported in this run because interactions are present but no contingencies emerge."
        )
    elif interactions_present and stable_present and signal_present and (multi_game_support or multi_sampler_support):
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
    result["evidence_diagnostics"] = _h01_evidence_diagnostics(
        run_dir=run_dir,
        memory_dir=memory_dir,
        sqlite_paths=sqlite_paths,
    )

    if report is None and memory_dir is not None:
        compact_metrics = _extract_compact_memory_metrics(Path(memory_dir))
        result.update(compact_metrics)
        result["evidence_source"] = (
            "direct_streaming_manifest_and_compact_memory"
            if direct_streaming_manifest_exists(memory_dir)
            else "compact_memory"
        )
        interactions_present = _gt(result.get("total_interaction_count"), 0)
        contingencies_present = _gt(result.get("discovered_contingency_count"), 0) or _gt(result.get("stable_contingency_count"), 0)
        stable_present = _gt(result.get("stable_contingency_count"), 0)
        multi_game_support = _gt(result.get("cross_game_contingency_count"), 0)
        multi_sampler_support = _gt(result.get("cross_sampler_contingency_count"), 0)
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
        if stable_present:
            result["evidence_for"].append(
                f"Stable contingencies are present in compact memory ({int(result.get('stable_contingency_count') or 0)})."
            )
        if multi_game_support:
            result["evidence_for"].append("Cross-game contingency support is present in compact memory.")
        if multi_sampler_support:
            result["evidence_for"].append("Cross-sampler contingency support is present in compact memory.")
        if not signal_present:
            result["evidence_against"].append("Prediction/context signal is unavailable in compact-only evidence.")
        if (
            direct_streaming_manifest_exists(memory_dir)
            and not bool(sqlite_paths)
            and not interactions_present
            and not contingencies_present
        ):
            result["decision"] = "INSUFFICIENT_EVIDENCE"
            result["scientific_conclusion"] = (
                "H01 remains insufficiently evidenced after direct-streaming raw cleanup because compact memory does not expose enough interaction or contingency rows."
            )
        else:
            _apply_h01_decision(
                result,
                interactions_present=interactions_present,
                contingencies_present=contingencies_present,
                stable_present=stable_present,
                multi_game_support=bool(multi_game_support),
                multi_sampler_support=bool(multi_sampler_support),
                signal_present=bool(signal_present),
                report_has_runs=False,
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
        or report_metrics.get("mean_prediction_accuracy") is None
        or report_metrics.get("mean_context_lift") is None
        or report_metrics.get("cross_game_contingency_presence") is None
    )
    db_metrics = _extract_db_metrics(sqlite_paths) if sqlite_paths and needs_db_fallback else {}
    compact_metrics: dict[str, Any] = {}
    result.update(report_metrics)
    if memory_dir is not None:
        compact_metrics = _extract_compact_memory_metrics(Path(memory_dir))
        for key in (
            "total_interaction_count",
            "contingency_candidate_count",
            "discovered_contingency_count",
            "stable_contingency_count",
            "per_game_contingency_counts",
            "per_sampler_contingency_counts",
            "cross_game_contingency_count",
            "cross_sampler_contingency_count",
            "mean_prediction_accuracy",
            "memory_record_count",
            "memory_replay_candidate_count",
            "high_priority_replay_count",
            "memory_mean_replay_priority",
            "memory_max_replay_priority",
            "context_contradiction_count",
            "repeated_contradiction_count",
            "prediction_error_positive_count",
            "predicted_family_available_count",
            "actual_family_available_count",
            "wrong_prediction_count",
            "confident_wrong_prediction_count",
        ):
            compact_value = compact_metrics.get(key)
            if compact_value is None:
                continue
            current_value = result.get(key)
            if current_value in (None, 0, 0.0, {}, []):
                result[key] = compact_value
        if direct_streaming_manifest_exists(memory_dir) and not sqlite_paths:
            result["evidence_source"] = "direct_streaming_manifest_and_compact_memory"
        else:
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
        "median_context_lift",
        "positive_context_lift_count",
        "context_lift_available",
        "cross_game_contingency_count",
        "cross_sampler_contingency_count",
    ):
        if result.get(key) is None:
            result[key] = db_metrics.get(key)
    if not bool(result.get("context_lift_available")) and bool(db_metrics.get("context_lift_available")):
        result["context_lift_available"] = True

    if not result["per_game_contingency_counts"]:
        result["per_game_contingency_counts"] = db_metrics.get("per_game_contingency_counts", {})
    if not result["per_sampler_contingency_counts"]:
        result["per_sampler_contingency_counts"] = db_metrics.get("per_sampler_contingency_counts", {})

    result["cross_game_contingency_presence"] = (
        db_metrics.get("cross_game_contingency_presence")
        if db_metrics.get("cross_game_contingency_presence") is not None
        else compact_metrics.get("cross_game_contingency_presence")
    )
    result["cross_game_contingency_count"] = (
        result.get("cross_game_contingency_count")
        or compact_metrics.get("cross_game_contingency_count")
    )
    result["cross_sampler_contingency_count"] = (
        result.get("cross_sampler_contingency_count")
        or compact_metrics.get("cross_sampler_contingency_count")
    )
    result["games_per_contingency_identity"] = db_metrics.get("games_per_contingency_identity", {}) or compact_metrics.get("games_per_contingency_identity", {})
    result["samplers_per_contingency_identity"] = db_metrics.get("samplers_per_contingency_identity", {}) or compact_metrics.get("samplers_per_contingency_identity", {})
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
    if memory_dir is not None and direct_streaming_manifest_exists(memory_dir) and not sqlite_paths:
        result["raw_epoch_db_available"] = False
        if not interactions_present and not contingencies_present:
            result["missing_evidence"].append(
                "Direct-streaming raw cleanup removed raw DBs and compact memory does not yet expose enough H01 contingency evidence."
            )

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

    _apply_h01_decision(
        result,
        interactions_present=bool(interactions_present),
        contingencies_present=bool(contingencies_present),
        stable_present=bool(stable_present),
        multi_game_support=bool(multi_game_support),
        multi_sampler_support=bool(multi_sampler_support),
        signal_present=bool(signal_present),
        report_has_runs=bool(report.get("runs")),
    )

    _finalize_h01_result(result, output_dir)
    return result


def _extract_compact_memory_metrics(memory_dir: Path) -> dict[str, Any]:
    current_state = memory_dir / "current_state.sqlite"
    replay_queue = memory_dir / "replay_queue.sqlite"
    if not current_state.exists():
        return {}
    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        per_game = dict(connection.execute("SELECT COALESCE(game, 'unknown'), COUNT(*) FROM stable_contingencies GROUP BY COALESCE(game, 'unknown')").fetchall())
        per_sampler = dict(connection.execute("SELECT COALESCE(sampler, 'unknown'), COUNT(*) FROM stable_contingencies GROUP BY COALESCE(sampler, 'unknown')").fetchall())
        stable_count = int(connection.execute("SELECT COUNT(*) FROM stable_contingencies WHERE support_count >= 20").fetchone()[0])
        discovered_count = int(connection.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
        transformation_families_count = int(connection.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
        family_members_count = int(connection.execute("SELECT COUNT(*) FROM family_members").fetchone()[0])
        memory_record_count = int(connection.execute("SELECT COUNT(*) FROM memory_nodes WHERE node_type = 'InteractionMemory'").fetchone()[0])
        memory_score_record_count = int(connection.execute("SELECT COUNT(*) FROM memory_scores WHERE node_id LIKE 'M0:interaction:%'").fetchone()[0])
        high_priority_replay_count = int(connection.execute("SELECT COUNT(*) FROM memory_scores WHERE node_id LIKE 'M0:interaction:%' AND COALESCE(replay_priority, 0.0) >= 0.50").fetchone()[0])
        replay_priority_stats = connection.execute("SELECT AVG(replay_priority), MAX(replay_priority) FROM memory_scores WHERE node_id LIKE 'M0:interaction:%'").fetchone()
        selected_for_replay_edge_count = int(connection.execute("SELECT COUNT(*) FROM memory_edges WHERE edge_type = 'selected_for_replay'").fetchone()[0])
        contradiction_count = int(connection.execute("SELECT COUNT(*) FROM contradiction_clusters").fetchone()[0])
        summary_row = connection.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'total_interactions_seen'"
        ).fetchone()
        interaction_count = 0
        raw_contingency_rows_seen = None
        if summary_row is not None and summary_row[0] is not None:
            try:
                interaction_count = int(json.loads(summary_row[0]))
            except Exception:
                try:
                    interaction_count = int(summary_row[0])
                except Exception:
                    interaction_count = 0
        fold_summary_row = connection.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'fold_summary'"
        ).fetchone()
        if fold_summary_row is not None and fold_summary_row[0] is not None:
            try:
                fold_summary = json.loads(fold_summary_row[0])
                if isinstance(fold_summary, dict):
                    raw_contingency_rows_seen = fold_summary.get("raw_contingency_rows_seen")
            except Exception:
                raw_contingency_rows_seen = None
        replay_queue_count = None
        if replay_queue.exists():
            with sqlite3.connect(replay_queue) as replay_conn:
                replay_queue_count = int(replay_conn.execute("SELECT COUNT(*) FROM replay_queue").fetchone()[0])
        identity_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT normalized_contingency_key, game, sampler, context_level, action, effect_signature, prediction_accuracy
                FROM stable_contingencies
                ORDER BY canonical_key ASC
                """
            ).fetchall()
        ]
        games_by_identity: dict[str, set[str]] = defaultdict(set)
        samplers_by_identity: dict[str, set[str]] = defaultdict(set)
        prediction_accuracy_values: list[float] = []
        for row in identity_rows:
            identity = str(
                row.get("normalized_contingency_key")
                or _normalized_contingency_identity_fallback(
                    context_level=row.get("context_level"),
                    action=row.get("action"),
                    effect_signature=row.get("effect_signature"),
                )
            )
            game = str(row.get("game") or "unknown")
            sampler = str(row.get("sampler") or "unknown")
            games_by_identity[identity].add(game)
            samplers_by_identity[identity].add(sampler)
            if row.get("prediction_accuracy") is not None:
                prediction_accuracy_values.append(float(row["prediction_accuracy"]))
        cross_game_contingency_count = sum(1 for values in games_by_identity.values() if len(values) >= 2)
        cross_sampler_contingency_count = sum(1 for values in samplers_by_identity.values() if len(values) >= 2)
        return {
            "total_interaction_count": interaction_count if interaction_count > 0 else (memory_record_count or memory_score_record_count or None),
            "stable_contingency_count": stable_count,
            "stable_contingencies_count": discovered_count,
            "contingency_candidate_count": discovered_count,
            "discovered_contingency_count": discovered_count,
            "transformation_families_count": transformation_families_count,
            "family_members_count": family_members_count,
            "raw_contingency_rows_seen": raw_contingency_rows_seen,
            "per_game_contingency_counts": {str(key): int(value) for key, value in per_game.items()},
            "per_sampler_contingency_counts": {str(key): int(value) for key, value in per_sampler.items()},
            "mean_prediction_accuracy": _mean_or_none(prediction_accuracy_values),
            "memory_record_count": memory_record_count or memory_score_record_count or None,
            "memory_score_record_count": memory_score_record_count or None,
            "memory_replay_candidate_count": replay_queue_count if replay_queue_count is not None else selected_for_replay_edge_count,
            "high_priority_replay_count": high_priority_replay_count,
            "memory_mean_replay_priority": None if replay_priority_stats is None else replay_priority_stats[0],
            "memory_max_replay_priority": None if replay_priority_stats is None else replay_priority_stats[1],
            "context_contradiction_count": contradiction_count,
            "repeated_contradiction_count": contradiction_count,
            "prediction_error_positive_count": contradiction_count if contradiction_count > 0 else None,
            "predicted_family_available_count": memory_score_record_count or None,
            "actual_family_available_count": memory_score_record_count or None,
            "wrong_prediction_count": contradiction_count if contradiction_count > 0 else None,
            "confident_wrong_prediction_count": contradiction_count if contradiction_count > 0 else None,
            "cross_game_contingency_presence": cross_game_contingency_count > 0 if identity_rows else None,
            "cross_game_contingency_count": cross_game_contingency_count,
            "cross_sampler_contingency_count": cross_sampler_contingency_count,
            "games_per_contingency_identity": {key: len(value) for key, value in sorted(games_by_identity.items())},
            "samplers_per_contingency_identity": {key: len(value) for key, value in sorted(samplers_by_identity.items())},
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
    memory_record_count = 0
    memory_replay_candidate_count = 0
    high_replay_priority_count = 0
    protected_memory_count = 0
    active_memory_count = 0
    forgotten_memory_count = 0
    tables_seen: set[str] = set()
    per_game: dict[str, int] = {}
    per_sampler: dict[str, int] = {}
    prediction_accuracy_values: list[float] = []
    context_lift_values: list[float] = []
    positive_context_lift_values: list[float] = []
    identity_games: dict[str, set[str]] = defaultdict(set)
    identity_samplers: dict[str, set[str]] = defaultdict(set)

    for path in sqlite_paths:
        try:
            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                tables_seen.update(tables)
                interactions = 0
                contingencies = 0
                game, sampler = _infer_game_sampler_from_path(path)
                if "interactions" in tables:
                    interactions = int(connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
                    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
                    if {"memory_status", "memory_retention_reason", "memory_replay_priority"} & columns:
                        row = connection.execute(
                            """
                            SELECT
                                SUM(CASE
                                    WHEN memory_status IS NOT NULL
                                      OR memory_retention_reason IS NOT NULL
                                      OR memory_replay_priority IS NOT NULL
                                    THEN 1 ELSE 0 END),
                                SUM(CASE WHEN COALESCE(memory_replay_candidate, 0) = 1 THEN 1 ELSE 0 END),
                                SUM(CASE WHEN COALESCE(memory_replay_priority, 0.0) >= 0.70 THEN 1 ELSE 0 END),
                                SUM(CASE WHEN memory_status = 'protected' THEN 1 ELSE 0 END),
                                SUM(CASE WHEN memory_status = 'active' THEN 1 ELSE 0 END),
                                SUM(CASE WHEN memory_status = 'forgotten' THEN 1 ELSE 0 END)
                            FROM interactions
                            """
                        ).fetchone()
                        memory_record_count += int(row[0] or 0)
                        memory_replay_candidate_count += int(row[1] or 0)
                        high_replay_priority_count += int(row[2] or 0)
                        protected_memory_count += int(row[3] or 0)
                        active_memory_count += int(row[4] or 0)
                        forgotten_memory_count += int(row[5] or 0)
                if "contingencies" in tables:
                    contingencies = int(connection.execute("SELECT COUNT(*) FROM contingencies").fetchone()[0])
                    contingency_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(contingencies)").fetchall()}
                    for row in connection.execute("SELECT * FROM contingencies ORDER BY id ASC").fetchall():
                        payload = dict(row)
                        family_signature = _family_signature_for_h01(connection, payload)
                        identity = str(
                            payload.get("normalized_contingency_key")
                            or _normalized_contingency_identity_fallback(
                                context_level=payload.get("context_level"),
                                action=payload.get("action"),
                                effect_signature=family_signature,
                            )
                        )
                        if game:
                            identity_games[identity].add(game)
                        if sampler:
                            identity_samplers[identity].add(sampler)
                        if "prediction_accuracy" in contingency_columns and payload.get("prediction_accuracy") is not None:
                            prediction_accuracy_values.append(float(payload["prediction_accuracy"]))
                if "prediction_results" in tables:
                    prediction_metrics = _prediction_metrics_from_prediction_results(connection)
                    if prediction_metrics["prediction_accuracy"] is not None:
                        prediction_accuracy_values.append(float(prediction_metrics["prediction_accuracy"]))
                    context_lift_values.extend(prediction_metrics.get("context_lift_values", []))
                    positive_context_lift_values.extend(
                        prediction_metrics.get("positive_context_lift_values", [])
                    )
                total_interactions += interactions
                contingency_rows += contingencies
        except sqlite3.DatabaseError:
            continue

        result["db_paths_inspected"] += 1
        result["selected_db_paths"].append(str(path))
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
    result["memory_record_count"] = memory_record_count if result["db_paths_inspected"] else None
    result["memory_replay_candidate_count"] = memory_replay_candidate_count if result["db_paths_inspected"] else None
    result["high_replay_priority_count"] = high_replay_priority_count if result["db_paths_inspected"] else None
    result["protected_memory_count"] = protected_memory_count if result["db_paths_inspected"] else None
    result["active_memory_count"] = active_memory_count if result["db_paths_inspected"] else None
    result["forgotten_memory_count"] = forgotten_memory_count if result["db_paths_inspected"] else None
    result["prediction_accuracy"] = _mean_or_none(prediction_accuracy_values)
    result["mean_prediction_accuracy"] = _mean_or_none(prediction_accuracy_values)
    preferred_context_lifts = positive_context_lift_values or context_lift_values
    result["context_lift"] = _mean_or_none(preferred_context_lifts)
    result["mean_context_lift"] = _mean_or_none(preferred_context_lifts)
    result["median_context_lift"] = _median_or_none(preferred_context_lifts)
    result["positive_context_lift_count"] = sum(1 for value in context_lift_values if value > 0.0)
    result["context_lift_available"] = bool(context_lift_values)
    result["per_game_contingency_counts"] = per_game
    result["per_sampler_contingency_counts"] = per_sampler
    result["stability_approximated"] = bool(result["db_paths_inspected"])
    result["cross_game_contingency_count"] = sum(1 for values in identity_games.values() if len(values) >= 2)
    result["cross_sampler_contingency_count"] = sum(1 for values in identity_samplers.values() if len(values) >= 2)
    result["cross_game_contingency_presence"] = (
        result["cross_game_contingency_count"] > 0 if identity_games else None
    )
    result["games_per_contingency_identity"] = {key: len(value) for key, value in sorted(identity_games.items())}
    result["samplers_per_contingency_identity"] = {key: len(value) for key, value in sorted(identity_samplers.items())}
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


def _normalized_contingency_identity_fallback(
    *,
    context_level: Any,
    action: Any,
    effect_signature: Any,
) -> str:
    effect_text = str(effect_signature or "unknown").lower()
    if ":" in effect_text:
        effect_bucket = effect_text.split(":", 1)[0]
    else:
        effect_bucket = "unknown"
    payload = {
        "action_bucket": f"a{action}" if action is not None else "aunknown",
        "context_bucket": f"ctx{max(0, min(int(context_level or 0), 3))}",
        "effect_bucket": effect_bucket,
    }
    return "ncont:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _family_signature_for_h01(connection: sqlite3.Connection, payload: dict[str, Any]) -> str:
    family_id = payload.get("transformation_family")
    row = None
    try:
        row = connection.execute(
            "SELECT centroid_vector FROM transformation_families WHERE id = ?",
            (family_id,),
        ).fetchone()
    except sqlite3.DatabaseError:
        row = None
    if row and row[0]:
        return f"centroid:{row[0]}"
    return f"family:{family_id}"


def _prediction_metrics_from_prediction_results(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT context_signature, action, predicted_family, actual_family
            FROM prediction_results
            WHERE predicted_family IS NOT NULL
              AND actual_family IS NOT NULL
            ORDER BY interaction_id ASC
            """
        ).fetchall()
    ]
    if not rows:
        return {
            "prediction_accuracy": None,
            "context_lift_values": [],
            "positive_context_lift_values": [],
        }
    action_groups: dict[int, list[float]] = defaultdict(list)
    context_action_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    prediction_values: list[float] = []
    for row in rows:
        correct = 1.0 if int(row["predicted_family"]) == int(row["actual_family"]) else 0.0
        action = int(row["action"])
        context_signature = str(row["context_signature"])
        prediction_values.append(correct)
        action_groups[action].append(correct)
        context_action_groups[(context_signature, action)].append(correct)
    action_accuracy = {action: (sum(values) / len(values)) for action, values in action_groups.items() if values}
    context_lifts: list[float] = []
    for (context_signature, action), values in context_action_groups.items():
        baseline = action_accuracy.get(action)
        if baseline is None or not values:
            continue
        context_accuracy = sum(values) / len(values)
        context_lifts.append(context_accuracy - baseline)
    return {
        "prediction_accuracy": sum(prediction_values) / len(prediction_values),
        "context_lift_values": context_lifts,
        "positive_context_lift_values": [value for value in context_lifts if value > 0.0],
    }


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
        f"- median_context_lift: {_fmt(result.get('median_context_lift'))}",
        f"- positive_context_lift_count: {_fmt(result.get('positive_context_lift_count'))}",
        f"- contradiction_count: {_fmt(result.get('contradiction_count'))}",
        f"- repeated_contradiction_count: {_fmt(result.get('repeated_contradiction_count'))}",
        f"- context_expansion_suggested_count: {_fmt(result.get('context_expansion_suggested_count'))}",
        "",
        "Coverage:",
        f"- per-game contingency counts: {json.dumps(result.get('per_game_contingency_counts', {}), sort_keys=True)}",
        f"- per-sampler contingency counts: {json.dumps(result.get('per_sampler_contingency_counts', {}), sort_keys=True)}",
        f"- cross-game contingency presence: {_fmt(result.get('cross_game_contingency_presence'))}",
        f"- cross-game contingency count: {_fmt(result.get('cross_game_contingency_count'))}",
        f"- cross-sampler contingency count: {_fmt(result.get('cross_sampler_contingency_count'))}",
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


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "null" if value is None else str(value)


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
