from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


H03_JSON_NAME = "h03_transformation_family_report.json"
H03_TXT_NAME = "h03_transformation_family_report.txt"
H03_MD_NAME = "h03_transformation_family.md"
H03_READY_JSON_NAME = "h03_ready_runs.json"
H03_READY_TXT_NAME = "h03_ready_runs.txt"
INPUT_REPORT_NAME = "interaction_sampling_v05c_report.json"
DEFAULT_MAX_DB_FILES = 1000
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MIN_FAMILY_SUPPORT = 2

H03_REQUIRED_CARRIER_FIELDS = (
    "emergent_object_carrier_count",
    "emergent_context_action_fallback_count",
    "carrier_object_candidate_count",
    "carrier_context_action_fallback_candidate_count",
)
H03_PRE_OBJECT_FIELDS = (
    "memory_record_count",
    "carrier_candidate_count",
    "emergent_carrier_count",
    "carrier_spatial_candidate_count",
    "carrier_object_candidate_count",
    "emergent_object_carrier_count",
    "carrier_context_action_fallback_candidate_count",
    "emergent_context_action_fallback_count",
)

ARTIFACT_KEYWORDS = ("family", "families", "transformation_family", "m2", "contingency", "compression")
FAMILY_TABLE_NAMES = (
    "transformation_families",
    "families",
    "m2_families",
    "contingency_families",
)
CANDIDATE_TABLE_NAMES = FAMILY_TABLE_NAMES + (
    "contingencies",
    "interactions",
    "prediction_results",
    "carrier_candidates",
    "carrier_events",
)
FAMILY_ID_COLUMNS = (
    "family_id",
    "transformation_family_id",
    "m2_family_id",
    "cluster_id",
    "signature_family",
    "compressed_signature",
    "effect_signature",
    "delta_signature",
    "transformation_family",
)
CONTINGENCY_ID_COLUMNS = (
    "contingency_id",
    "interaction_signature",
    "context_action",
    "action",
    "effect_signature",
    "delta_signature",
    "prediction_key",
    "id",
)
SUPPORT_COLUMNS = ("support", "support_count", "member_count", "occurrence_count", "count", "frequency", "total_count")
COMPRESSION_COLUMNS = ("compression_gain", "family_compression_gain", "compression_ratio", "compressed_count", "original_count")
PREDICTION_LIFT_COLUMNS = ("prediction_lift", "family_prediction_lift", "accuracy_lift", "context_lift", "confidence")
STABILITY_COLUMNS = ("stable", "is_stable", "stability_score")
SIGNATURE_COLUMNS = ("effect_signature", "delta_signature", "changed_cells_signature", "outcome_signature")
CONTEXT_COLUMNS = ("context_action", "context_signature", "interaction_signature", "prediction_key")
GAME_COLUMNS = ("game_id", "game", "game_name")
SAMPLER_COLUMNS = ("sampler", "sampler_name", "sampler_scope")
PARQUET_EXTENSIONS = {".parquet"}
SQLITE_EXTENSIONS = {".sqlite", ".db", ".sqlite3"}
DB_SAMPLER_PRIORITY = (
    "mixed",
    "low_confidence",
    "novelty_delta",
    "no_change_avoidance",
    "action_balance",
    "random_baseline",
)
DIRECT_FAMILY_UNAVAILABLE_MESSAGE = "Direct transformation-family evidence unavailable in current run artifacts."
PRE_OBJECT_UNAVAILABLE_MESSAGE = "Pre-object timing unavailable because carrier-source aggregate fields are missing."
STABILITY_APPROX_MESSAGE = "Stability approximated by support threshold."

H03_DEFAULTS: dict[str, Any] = {
    "hypothesis_id": "H03",
    "hypothesis_name": "Transformation-family formation before object carriers",
    "hypothesis_statement": "Repeated action-conditioned contingencies compress into transformation families before carrier/object emergence.",
    "decision": "INCONCLUSIVE",
    "scientific_conclusion": "",
    "source_run_dir": "",
    "input_report_found": False,
    "db_found": False,
    "db_paths_total": 0,
    "db_paths_inspected": 0,
    "db_scan_truncated": False,
    "selected_db_paths": [],
    "tables_seen": [],
    "candidate_tables_used": [],
    "artifact_paths_used": [],
    "memory_record_count": None,
    "interaction_count": None,
    "contingency_candidate_count": None,
    "discovered_contingency_count": None,
    "stable_contingency_count": None,
    "transformation_family_candidate_count": None,
    "transformation_family_count": None,
    "stable_transformation_family_count": None,
    "family_member_count_total": None,
    "family_mean_member_count": None,
    "family_median_member_count": None,
    "family_max_member_count": None,
    "singleton_family_count": None,
    "singleton_family_ratio": None,
    "compression_ratio": None,
    "compression_gain": None,
    "mean_family_compression_gain": None,
    "max_family_compression_gain": None,
    "family_prediction_lift_mean": None,
    "family_prediction_lift_median": None,
    "family_prediction_lift_max": None,
    "family_cross_context_count": None,
    "family_cross_game_count": None,
    "family_cross_sampler_count": None,
    "carrier_candidate_count": None,
    "emergent_carrier_count": None,
    "carrier_spatial_candidate_count": None,
    "carrier_object_candidate_count": None,
    "emergent_object_carrier_count": None,
    "carrier_context_action_fallback_candidate_count": None,
    "emergent_context_action_fallback_count": None,
    "family_signal_active": False,
    "compression_signal_active": False,
    "families_nontrivial": False,
    "families_stable": None,
    "pre_object_condition_satisfied": None,
    "evidence_for": [],
    "evidence_against": [],
    "missing_evidence": [],
    "acceptance_checks": {
        "contingencies_present": None,
        "transformation_families_present": None,
        "non_singleton_families_present": None,
        "singleton_family_ratio_acceptable": None,
        "compression_ratio_gt_1": None,
        "compression_gain_positive": None,
        "family_prediction_lift_non_negative": None,
        "object_carriers_absent": None,
        "context_action_fallback_absent": None,
    },
}


def evaluate_h03_transformation_family_formation(
    run_dir: Path,
    output_dir: Path,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    max_rows: int = DEFAULT_MAX_ROWS,
    scan_all_dbs: bool = True,
    prefer_db: Path | None = None,
    min_family_support: int = DEFAULT_MIN_FAMILY_SUPPORT,
) -> dict:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / INPUT_REPORT_NAME
    input_report = _load_json(report_path)
    sqlite_paths = _find_sqlite_paths(run_dir)

    result = dict(H03_DEFAULTS)
    result["source_run_dir"] = str(run_dir)
    result["input_report_found"] = input_report is not None
    result["db_found"] = bool(sqlite_paths)
    result["db_paths_total"] = len(sqlite_paths)

    report_metrics = _extract_report_metrics(input_report)
    for field, value in report_metrics.items():
        result[field] = value

    if input_report is None:
        result["missing_evidence"].append(f"Required input report missing: {INPUT_REPORT_NAME}")
        result["scientific_conclusion"] = "H03 cannot be evaluated because the required interaction-sampling report is missing."
        _finalize_h03_result(result, output_dir)
        return result

    direct_metrics = compute_h03_family_metrics_from_existing_artifacts(
        run_dir,
        max_db_files=max_db_files,
        max_rows=max_rows,
        scan_all_dbs=scan_all_dbs,
        prefer_db=prefer_db,
        min_family_support=min_family_support,
    )
    result.update(direct_metrics)
    if result.get("memory_record_count") is None:
        result["memory_record_count"] = report_metrics.get("memory_record_count")

    checks = {
        "contingencies_present": _gt(result.get("discovered_contingency_count"), 0),
        "transformation_families_present": _gt(result.get("transformation_family_count"), 0),
        "non_singleton_families_present": _gt(result.get("family_member_count_total"), result.get("transformation_family_count") or 0),
        "singleton_family_ratio_acceptable": _lte(result.get("singleton_family_ratio"), 0.50),
        "compression_ratio_gt_1": _gt(result.get("compression_ratio"), 1.0),
        "compression_gain_positive": _gt(result.get("compression_gain"), 0.0),
        "family_prediction_lift_non_negative": _gte(result.get("family_prediction_lift_mean"), 0.0),
        "object_carriers_absent": _eq(result.get("emergent_object_carrier_count"), 0),
        "context_action_fallback_absent": _eq(result.get("emergent_context_action_fallback_count"), 0),
    }
    result["acceptance_checks"] = checks
    result["family_signal_active"] = checks["transformation_families_present"] is True
    result["compression_signal_active"] = (
        checks["compression_ratio_gt_1"] is True or checks["compression_gain_positive"] is True
    )
    result["families_nontrivial"] = checks["non_singleton_families_present"] is True
    result["families_stable"] = _gt(result.get("stable_transformation_family_count"), 0)

    if result.get("emergent_object_carrier_count") is None or result.get("emergent_context_action_fallback_count") is None:
        result["pre_object_condition_satisfied"] = None
        if PRE_OBJECT_UNAVAILABLE_MESSAGE not in result["missing_evidence"]:
            result["missing_evidence"].append(PRE_OBJECT_UNAVAILABLE_MESSAGE)
    else:
        result["pre_object_condition_satisfied"] = (
            checks["object_carriers_absent"] is True and checks["context_action_fallback_absent"] is True
        )

    contingencies_present = checks["contingencies_present"] is True
    families_present = checks["transformation_families_present"] is True
    non_singleton_present = checks["non_singleton_families_present"] is True
    compression_good = checks["compression_ratio_gt_1"] is True and checks["compression_gain_positive"] is True
    compression_partial = checks["compression_ratio_gt_1"] is True or checks["compression_gain_positive"] is True
    pre_object = result["pre_object_condition_satisfied"]
    ratio_missing = result.get("singleton_family_ratio") is None
    stability_approx = bool(result.get("stability_approximated", False))

    if not direct_metrics.get("usable_direct_family_evidence", False):
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H03 remains inconclusive because the current run artifacts do not expose enough direct family or contingency evidence."
        )
    elif contingencies_present and result.get("transformation_family_count") == 0:
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H03 is not supported in this run because contingencies are present but no transformation families were detected."
        )
    elif families_present and result.get("singleton_family_count") == result.get("transformation_family_count"):
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H03 is not supported in this run because all detected families are singleton structures without compression."
        )
    elif (
        _lte(result.get("compression_ratio"), 1.0) is True
        and _lte(result.get("compression_gain"), 0.0) is True
    ):
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H03 is not supported in this run because transformation-family compression does not exceed the underlying contingency count."
        )
    elif _gt(result.get("emergent_object_carrier_count"), 0) is True and not families_present:
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H03 is not supported in this run because object carriers are already emergent before any transformation families are detected."
        )
    elif pre_object is False:
        result["decision"] = "INVALID"
        result["scientific_conclusion"] = (
            "H03 is not supported in this run because object-carrier emergence is already present and pre-object family timing is not established."
        )
    elif (
        contingencies_present
        and families_present
        and non_singleton_present
        and compression_good
        and checks["singleton_family_ratio_acceptable"] is True
        and pre_object is True
    ):
        result["decision"] = "VALID"
        result["scientific_conclusion"] = (
            "H03 is supported in this run. Repeated contingencies compress into non-singleton transformation families with positive compression gain, and this family structure is present before object carriers emerge."
        )
    elif contingencies_present and families_present and non_singleton_present and compression_partial:
        result["decision"] = "PARTIALLY_VALID"
        if pre_object is None:
            result["scientific_conclusion"] = (
                "H03 is partially supported in this run. Repeated contingencies compress into non-singleton transformation families, but pre-object timing cannot be fully validated because carrier-source aggregate fields are unavailable."
            )
        elif ratio_missing or stability_approx:
            result["scientific_conclusion"] = (
                "H03 is partially supported in this run. Transformation-family compression is present, but singleton-ratio or stability evidence remains incomplete or approximate."
            )
        elif pre_object is False:
            result["decision"] = "INVALID"
            result["scientific_conclusion"] = (
                "H03 is not supported in this run because object-carrier emergence is already present and pre-object family timing is not established."
            )
        else:
            result["scientific_conclusion"] = (
                "H03 is partially supported in this run. Repeated contingencies compress into transformation families, but the available evidence is not complete enough for robust validation."
            )
    else:
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H03 remains inconclusive because the run artifacts do not cleanly distinguish contingency structure from family-level compression."
        )

    _populate_h03_evidence_lists(result)
    _finalize_h03_result(result, output_dir)
    return result


def compute_h03_family_metrics_from_existing_artifacts(
    run_dir: Path,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    max_rows: int = DEFAULT_MAX_ROWS,
    scan_all_dbs: bool = True,
    prefer_db: Path | None = None,
    min_family_support: int = DEFAULT_MIN_FAMILY_SUPPORT,
) -> dict:
    run_dir = Path(run_dir)
    sqlite_paths = _find_sqlite_paths(run_dir)
    ranked_paths = _rank_sqlite_paths(run_dir, sqlite_paths, prefer_db=prefer_db)
    limited_paths = ranked_paths[: max(1, int(max_db_files))]
    artifact_paths = _find_h03_artifact_paths(run_dir)

    result = {
        "db_found": bool(sqlite_paths),
        "db_paths_total": len(sqlite_paths),
        "db_paths_inspected": 0,
        "db_scan_truncated": len(limited_paths) < len(ranked_paths),
        "selected_db_paths": [],
        "tables_seen": [],
        "candidate_tables_used": [],
        "artifact_paths_used": [],
        "interaction_count": None,
        "contingency_candidate_count": None,
        "discovered_contingency_count": None,
        "stable_contingency_count": None,
        "transformation_family_candidate_count": None,
        "transformation_family_count": None,
        "stable_transformation_family_count": None,
        "family_member_count_total": None,
        "family_mean_member_count": None,
        "family_median_member_count": None,
        "family_max_member_count": None,
        "singleton_family_count": None,
        "singleton_family_ratio": None,
        "compression_ratio": None,
        "compression_gain": None,
        "mean_family_compression_gain": None,
        "max_family_compression_gain": None,
        "family_prediction_lift_mean": None,
        "family_prediction_lift_median": None,
        "family_prediction_lift_max": None,
        "family_cross_context_count": None,
        "family_cross_game_count": None,
        "family_cross_sampler_count": None,
        "artifact_paths_seen": [str(path) for path in artifact_paths],
        "usable_direct_family_evidence": False,
        "stability_approximated": False,
        "missing_evidence": [],
    }

    artifact_candidate = _read_h03_artifact_candidate(artifact_paths, min_family_support=min_family_support)
    if artifact_candidate is not None:
        _merge_h03_candidate(result, artifact_candidate)
        result["artifact_paths_used"] = artifact_candidate.get("artifact_paths_used", [])

    db_candidates: list[dict[str, Any]] = []
    for sqlite_path in limited_paths:
        result["db_paths_inspected"] += 1
        try:
            with sqlite3.connect(sqlite_path) as connection:
                table_map = _load_table_map(connection)
                tables_seen = sorted(table_map)
                result["tables_seen"] = sorted(set(result["tables_seen"]) | set(tables_seen))
                candidate = _compute_h03_db_candidate(
                    connection,
                    sqlite_path=sqlite_path,
                    run_dir=run_dir,
                    table_map=table_map,
                    max_rows=max_rows,
                    min_family_support=min_family_support,
                )
                if candidate is None:
                    continue
                db_candidates.append(candidate)
                if not scan_all_dbs:
                    break
        except sqlite3.DatabaseError:
            continue

    if db_candidates:
        aggregate_candidate = _aggregate_h03_db_candidates(db_candidates, min_family_support=min_family_support)
        _merge_h03_candidate(result, aggregate_candidate)

    if not result["usable_direct_family_evidence"]:
        result["missing_evidence"].append(DIRECT_FAMILY_UNAVAILABLE_MESSAGE)
    if any(path.suffix.lower() in PARQUET_EXTENSIONS for path in artifact_paths):
        result["missing_evidence"].append("Parquet family artifacts were detected but are not inspected by this report.")
    return result


def find_h03_ready_runs(
    runs_root: Path,
    output_dir: Path,
    *,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    prefer_db: Path | None = None,
) -> dict:
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for report_path in sorted(runs_root.rglob(INPUT_REPORT_NAME)):
        run_dir = report_path.parent
        payload = _load_json(report_path)
        source = _report_source(payload)
        sqlite_paths = _rank_sqlite_paths(run_dir, _find_sqlite_paths(run_dir), prefer_db=prefer_db)
        artifact_paths = _find_h03_artifact_paths(run_dir)
        limited_paths = sqlite_paths[: max(1, int(max_db_files))]
        schema_info = _inspect_h03_schema_flags(limited_paths)
        missing_required = [field for field in H03_REQUIRED_CARRIER_FIELDS if source.get(field) is None]
        has_required = not missing_required
        entry = {
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "has_sqlite_db": bool(sqlite_paths),
            "sqlite_db_count": len(sqlite_paths),
            "has_family_artifacts": bool(artifact_paths),
            "family_artifact_paths": [str(path) for path in artifact_paths],
            "has_family_like_db_schema": schema_info["has_family_like_db_schema"],
            "has_derivable_family_schema": schema_info["has_derivable_family_schema"],
            "has_required_carrier_fields": has_required,
            "missing_required_carrier_fields": missing_required,
            "memory_record_count": _maybe_int(source.get("memory_record_count")),
            "carrier_object_candidate_count": _maybe_int(source.get("carrier_object_candidate_count")),
            "emergent_object_carrier_count": _maybe_int(source.get("emergent_object_carrier_count")),
            "carrier_context_action_fallback_candidate_count": _maybe_int(
                source.get("carrier_context_action_fallback_candidate_count")
            ),
            "emergent_context_action_fallback_count": _maybe_int(source.get("emergent_context_action_fallback_count")),
            "h03_ready": False,
            "recommended_output_dir": str(output_dir),
        }
        entry["h03_ready"] = bool(
            entry["has_sqlite_db"]
            and has_required
            and (
                entry["has_family_artifacts"]
                or entry["has_family_like_db_schema"]
                or entry["has_derivable_family_schema"]
            )
        )
        runs.append(entry)

    runs.sort(key=_h03_ready_sort_key)
    recommended = next((entry for entry in runs if entry["h03_ready"] is True), None)
    result = {
        "runs_root": str(runs_root),
        "candidate_count": len(runs),
        "ready_count": sum(1 for entry in runs if entry["h03_ready"] is True),
        "runs": runs,
        "run_best_parameters": None,
        "recommended_run": None if recommended is None else {
            "run_dir": recommended["run_dir"],
            "recommended_output_dir": recommended["recommended_output_dir"],
        },
    }
    _write_h03_ready_inventory(result, output_dir)
    return result


def run_h03_on_best_ready_run(
    runs_root: Path,
    output_dir: Path,
    *,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    max_rows: int = DEFAULT_MAX_ROWS,
    scan_all_dbs: bool = True,
    prefer_db: Path | None = None,
    min_family_support: int = DEFAULT_MIN_FAMILY_SUPPORT,
) -> dict:
    result = find_h03_ready_runs(runs_root, output_dir, max_db_files=max_db_files, prefer_db=prefer_db)
    result["run_best_parameters"] = {
        "max_db_files": int(max_db_files),
        "max_rows": int(max_rows),
        "scan_all_dbs": bool(scan_all_dbs),
        "prefer_db": None if prefer_db is None else str(prefer_db),
        "min_family_support": int(min_family_support),
    }
    _write_h03_ready_inventory(result, Path(output_dir))
    recommended = result.get("recommended_run")
    if recommended is None:
        return result
    evaluate_h03_transformation_family_formation(
        run_dir=Path(recommended["run_dir"]),
        output_dir=Path(recommended["recommended_output_dir"]),
        max_db_files=max_db_files,
        max_rows=max_rows,
        scan_all_dbs=scan_all_dbs,
        prefer_db=prefer_db,
        min_family_support=min_family_support,
    )
    return result


def _compute_h03_db_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_map: dict[str, set[str]],
    max_rows: int,
    min_family_support: int,
) -> dict[str, Any] | None:
    del max_rows
    table_order = sorted(table_map, key=lambda name: (0 if name in CANDIDATE_TABLE_NAMES else 1, name))
    interactions_count = _count_rows(connection, "interactions") if "interactions" in table_map else None
    for table_name in table_order:
        columns = table_map[table_name]
        if table_name in FAMILY_TABLE_NAMES or _first_matching(columns, FAMILY_ID_COLUMNS) is not None:
            candidate = _compute_family_table_candidate(
                connection,
                sqlite_path=sqlite_path,
                run_dir=run_dir,
                table_name=table_name,
                columns=columns,
                min_family_support=min_family_support,
                interaction_count=interactions_count,
            )
            if candidate is not None:
                return candidate
        candidate = _compute_derivable_family_candidate(
            connection,
            sqlite_path=sqlite_path,
            run_dir=run_dir,
            table_name=table_name,
            columns=columns,
            min_family_support=min_family_support,
            interaction_count=interactions_count,
        )
        if candidate is not None:
            return candidate
    return None


def _compute_family_table_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_name: str,
    columns: set[str],
    min_family_support: int,
    interaction_count: int | None,
) -> dict[str, Any] | None:
    family_column = _first_matching(columns, FAMILY_ID_COLUMNS)
    if family_column is None:
        return None
    contingency_column = _first_matching(columns, CONTINGENCY_ID_COLUMNS)
    support_column = _first_matching(columns, SUPPORT_COLUMNS)
    compression_column = _first_matching(columns, ("compression_gain", "family_compression_gain"))
    ratio_column = _first_matching(columns, ("compression_ratio",))
    original_count_column = _first_matching(columns, ("original_count",))
    compressed_count_column = _first_matching(columns, ("compressed_count",))
    prediction_lift_column = _first_matching(columns, PREDICTION_LIFT_COLUMNS)
    stability_column = _first_matching(columns, STABILITY_COLUMNS)
    context_column = _first_matching(columns, CONTEXT_COLUMNS)
    game_column = _first_matching(columns, GAME_COLUMNS)
    sampler_column = _first_matching(columns, SAMPLER_COLUMNS)

    table_ref = _quote_ident(table_name)
    family_ref = _quote_ident(family_column)
    contingency_expr = _quote_ident(contingency_column) if contingency_column is not None else "rowid"
    support_expr = _quote_ident(support_column) if support_column is not None else "1"
    context_expr = _quote_ident(context_column) if context_column is not None else "NULL"
    game_expr = _quote_ident(game_column) if game_column is not None else "NULL"
    sampler_expr = _quote_ident(sampler_column) if sampler_column is not None else "NULL"
    compression_expr = _quote_ident(compression_column) if compression_column is not None else None
    ratio_expr = _quote_ident(ratio_column) if ratio_column is not None else None
    prediction_lift_expr = _quote_ident(prediction_lift_column) if prediction_lift_column is not None else None

    if stability_column is not None:
        stable_expr = _stable_expr(stability_column)
        stability_approximated = False
    else:
        stable_expr = f"CASE WHEN CAST({support_expr} AS REAL) >= {int(min_family_support)} THEN 1 ELSE 0 END"
        stability_approximated = True

    row = connection.execute(
        f"""
        WITH grouped AS (
            SELECT
                CAST({family_ref} AS TEXT) AS family_id,
                COUNT(DISTINCT CAST({contingency_expr} AS TEXT)) AS member_count,
                MAX(CAST({support_expr} AS REAL)) AS support_value,
                SUM(CASE WHEN {context_expr} IS NOT NULL THEN 1 ELSE 0 END) AS context_rows,
                COUNT(DISTINCT CASE WHEN {context_expr} IS NOT NULL THEN CAST({context_expr} AS TEXT) END) AS distinct_contexts,
                COUNT(DISTINCT CASE WHEN {game_expr} IS NOT NULL THEN CAST({game_expr} AS TEXT) END) AS distinct_games,
                COUNT(DISTINCT CASE WHEN {sampler_expr} IS NOT NULL THEN CAST({sampler_expr} AS TEXT) END) AS distinct_samplers,
                MAX({stable_expr}) AS stable_flag
                {', AVG(CAST(' + compression_expr + ' AS REAL)) AS compression_gain_value' if compression_expr else ''}
                {', AVG(CAST(' + ratio_expr + ' AS REAL)) AS compression_ratio_value' if ratio_expr else ''}
                {', AVG(CAST(' + prediction_lift_expr + ' AS REAL)) AS prediction_lift_value' if prediction_lift_expr else ''}
            FROM {table_ref}
            WHERE {family_ref} IS NOT NULL
            GROUP BY CAST({family_ref} AS TEXT)
        )
        SELECT
            COUNT(*),
            SUM(member_count),
            SUM(CASE WHEN stable_flag = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN member_count <= 1 THEN 1 ELSE 0 END),
            AVG(member_count),
            MAX(member_count),
            AVG(CASE WHEN member_count >= 2 THEN 1.0 - (1.0 / member_count) ELSE 0.0 END),
            MAX(CASE WHEN member_count >= 2 THEN 1.0 - (1.0 / member_count) ELSE 0.0 END),
            SUM(CASE WHEN distinct_contexts > 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN distinct_games > 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN distinct_samplers > 1 THEN 1 ELSE 0 END)
            {', AVG(compression_gain_value)' if compression_expr else ''}
            {', AVG(compression_ratio_value)' if ratio_expr else ''}
            {', AVG(prediction_lift_value), MAX(prediction_lift_value)' if prediction_lift_expr else ''}
        FROM grouped
        """
    ).fetchone()
    family_count = int(row[0] or 0)
    if family_count <= 0:
        return None

    discovered = int(row[1] or 0)
    stable_families = int(row[2] or 0)
    singleton_count = int(row[3] or 0)
    family_mean_member_count = float(row[4] or 0.0)
    family_max_member_count = int(row[5] or 0)
    mean_family_compression_gain = float(row[6] or 0.0)
    max_family_compression_gain = float(row[7] or 0.0)
    cross_context_count = int(row[8] or 0)
    cross_game_count = int(row[9] or 0)
    cross_sampler_count = int(row[10] or 0)

    offset = 11
    direct_compression_gain = None
    direct_compression_ratio = None
    family_prediction_lift_mean = None
    family_prediction_lift_max = None
    if compression_expr:
        direct_compression_gain = float(row[offset] or 0.0)
        offset += 1
    if ratio_expr:
        direct_compression_ratio = float(row[offset] or 0.0)
        offset += 1
    if prediction_lift_expr:
        family_prediction_lift_mean = float(row[offset] or 0.0)
        family_prediction_lift_max = float(row[offset + 1] or 0.0)

    compression_ratio = direct_compression_ratio
    if compression_ratio is None:
        if original_count_column and compressed_count_column:
            compression_ratio = _family_ratio_from_counts(connection, table_name, original_count_column, compressed_count_column)
        elif family_count > 0:
            compression_ratio = discovered / family_count

    compression_gain = direct_compression_gain
    if compression_gain is None and family_count > 0 and discovered > 0:
        compression_gain = 1.0 - (family_count / discovered)

    member_counts = _family_member_counts(connection, table_name, family_column, contingency_column)
    candidate = {
        "selected_db_paths": [str(sqlite_path)],
        "tables_seen": [table_name],
        "candidate_tables_used": [table_name],
        "interaction_count": interaction_count,
        "contingency_candidate_count": discovered,
        "discovered_contingency_count": discovered,
        "stable_contingency_count": stable_families,
        "transformation_family_candidate_count": family_count,
        "transformation_family_count": family_count,
        "stable_transformation_family_count": stable_families,
        "family_member_count_total": discovered,
        "family_mean_member_count": family_mean_member_count,
        "family_median_member_count": float(median(member_counts)) if member_counts else None,
        "family_max_member_count": family_max_member_count,
        "singleton_family_count": singleton_count,
        "singleton_family_ratio": (singleton_count / family_count) if family_count > 0 else None,
        "compression_ratio": compression_ratio,
        "compression_gain": compression_gain,
        "mean_family_compression_gain": direct_compression_gain if direct_compression_gain is not None else mean_family_compression_gain,
        "max_family_compression_gain": direct_compression_gain if direct_compression_gain is not None else max_family_compression_gain,
        "family_prediction_lift_mean": family_prediction_lift_mean,
        "family_prediction_lift_median": family_prediction_lift_mean,
        "family_prediction_lift_max": family_prediction_lift_max,
        "family_cross_context_count": cross_context_count if context_column is not None else None,
        "family_cross_game_count": cross_game_count if game_column is not None else None,
        "family_cross_sampler_count": cross_sampler_count if sampler_column is not None else None,
        "usable_direct_family_evidence": True,
        "stability_approximated": stability_approximated,
    }
    return candidate


def _compute_derivable_family_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_name: str,
    columns: set[str],
    min_family_support: int,
    interaction_count: int | None,
) -> dict[str, Any] | None:
    del run_dir
    signature_column = _first_matching(columns, SIGNATURE_COLUMNS)
    if signature_column is None:
        return None
    context_column = _first_matching(columns, CONTEXT_COLUMNS)
    action_column = _first_matching(columns, ("action",))
    if context_column is None and action_column is None:
        return None
    support_column = _first_matching(columns, SUPPORT_COLUMNS)
    table_ref = _quote_ident(table_name)
    family_ref = _quote_ident(signature_column)
    context_expr = _quote_ident(context_column) if context_column is not None else "''"
    action_expr = f"CAST({_quote_ident(action_column)} AS TEXT)" if action_column is not None else "''"
    support_expr = _quote_ident(support_column) if support_column is not None else "1"

    row = connection.execute(
        f"""
        WITH base AS (
            SELECT
                CAST({family_ref} AS TEXT) AS family_key,
                CAST({context_expr} AS TEXT) || '|' || CAST({action_expr} AS TEXT) AS contingency_key,
                CAST({support_expr} AS REAL) AS support_value
            FROM {table_ref}
            WHERE {family_ref} IS NOT NULL
        ),
        grouped AS (
            SELECT
                family_key,
                COUNT(DISTINCT contingency_key) AS member_count,
                MAX(support_value) AS support_value
            FROM base
            GROUP BY family_key
        )
        SELECT
            COUNT(*),
            SUM(member_count),
            SUM(CASE WHEN support_value >= {int(min_family_support)} THEN 1 ELSE 0 END),
            SUM(CASE WHEN member_count <= 1 THEN 1 ELSE 0 END),
            AVG(member_count),
            MAX(member_count)
        FROM grouped
        """
    ).fetchone()
    family_count = int(row[0] or 0)
    if family_count <= 0:
        return None
    discovered = int(row[1] or 0)
    stable_families = int(row[2] or 0)
    singleton_count = int(row[3] or 0)
    member_mean = float(row[4] or 0.0)
    member_max = int(row[5] or 0)
    member_counts = _derived_family_member_counts(connection, table_name, signature_column, context_column, action_column)
    family_presence = _derived_family_presence(connection, sqlite_path, table_name, signature_column, run_dir)
    return {
        "selected_db_paths": [str(sqlite_path)],
        "tables_seen": [table_name],
        "candidate_tables_used": [table_name],
        "interaction_count": interaction_count,
        "contingency_candidate_count": discovered,
        "discovered_contingency_count": discovered,
        "stable_contingency_count": stable_families,
        "transformation_family_candidate_count": family_count,
        "transformation_family_count": family_count,
        "stable_transformation_family_count": stable_families,
        "family_member_count_total": discovered,
        "family_mean_member_count": member_mean,
        "family_median_member_count": float(median(member_counts)) if member_counts else None,
        "family_max_member_count": member_max,
        "singleton_family_count": singleton_count,
        "singleton_family_ratio": singleton_count / family_count if family_count > 0 else None,
        "compression_ratio": discovered / family_count if family_count > 0 else None,
        "compression_gain": 1.0 - (family_count / discovered) if discovered > 0 else None,
        "mean_family_compression_gain": float(mean([max(0.0, 1.0 - (1.0 / count)) for count in member_counts])) if member_counts else None,
        "max_family_compression_gain": max((max(0.0, 1.0 - (1.0 / count)) for count in member_counts), default=None),
        "family_prediction_lift_mean": None,
        "family_prediction_lift_median": None,
        "family_prediction_lift_max": None,
        "family_cross_context_count": sum(1 for item in family_presence.values() if len(item["contexts"]) > 1),
        "family_cross_game_count": sum(1 for item in family_presence.values() if len(item["games"]) > 1),
        "family_cross_sampler_count": sum(1 for item in family_presence.values() if len(item["samplers"]) > 1),
        "usable_direct_family_evidence": True,
        "stability_approximated": True,
    }


def _aggregate_h03_db_candidates(candidates: list[dict[str, Any]], *, min_family_support: int) -> dict[str, Any]:
    del min_family_support
    out = {
        "selected_db_paths": sorted({path for candidate in candidates for path in candidate.get("selected_db_paths", [])}),
        "tables_seen": sorted({table for candidate in candidates for table in candidate.get("tables_seen", [])}),
        "candidate_tables_used": sorted({table for candidate in candidates for table in candidate.get("candidate_tables_used", [])}),
        "interaction_count": _sum_or_none(candidate.get("interaction_count") for candidate in candidates),
        "contingency_candidate_count": _sum_or_none(candidate.get("contingency_candidate_count") for candidate in candidates),
        "discovered_contingency_count": _sum_or_none(candidate.get("discovered_contingency_count") for candidate in candidates),
        "stable_contingency_count": _sum_or_none(candidate.get("stable_contingency_count") for candidate in candidates),
        "transformation_family_candidate_count": _sum_or_none(candidate.get("transformation_family_candidate_count") for candidate in candidates),
        "transformation_family_count": _sum_or_none(candidate.get("transformation_family_count") for candidate in candidates),
        "stable_transformation_family_count": _sum_or_none(candidate.get("stable_transformation_family_count") for candidate in candidates),
        "family_member_count_total": _sum_or_none(candidate.get("family_member_count_total") for candidate in candidates),
        "family_mean_member_count": None,
        "family_median_member_count": None,
        "family_max_member_count": _max_or_none(candidate.get("family_max_member_count") for candidate in candidates),
        "singleton_family_count": _sum_or_none(candidate.get("singleton_family_count") for candidate in candidates),
        "singleton_family_ratio": None,
        "compression_ratio": None,
        "compression_gain": None,
        "mean_family_compression_gain": _mean_or_none(candidate.get("mean_family_compression_gain") for candidate in candidates),
        "max_family_compression_gain": _max_or_none(candidate.get("max_family_compression_gain") for candidate in candidates),
        "family_prediction_lift_mean": _mean_or_none(candidate.get("family_prediction_lift_mean") for candidate in candidates),
        "family_prediction_lift_median": _mean_or_none(candidate.get("family_prediction_lift_median") for candidate in candidates),
        "family_prediction_lift_max": _max_or_none(candidate.get("family_prediction_lift_max") for candidate in candidates),
        "family_cross_context_count": _sum_or_none(candidate.get("family_cross_context_count") for candidate in candidates),
        "family_cross_game_count": _sum_or_none(candidate.get("family_cross_game_count") for candidate in candidates),
        "family_cross_sampler_count": _sum_or_none(candidate.get("family_cross_sampler_count") for candidate in candidates),
        "usable_direct_family_evidence": any(candidate.get("usable_direct_family_evidence", False) for candidate in candidates),
        "stability_approximated": any(candidate.get("stability_approximated", False) for candidate in candidates),
    }
    family_total = out["transformation_family_count"]
    contingency_total = out["discovered_contingency_count"]
    if family_total and contingency_total:
        out["compression_ratio"] = contingency_total / family_total
        out["compression_gain"] = 1.0 - (family_total / contingency_total)
    if family_total and out["singleton_family_count"] is not None:
        out["singleton_family_ratio"] = out["singleton_family_count"] / family_total
    if family_total and out["family_member_count_total"] is not None:
        out["family_mean_member_count"] = out["family_member_count_total"] / family_total
        out["family_median_member_count"] = out["family_mean_member_count"]
    return out


def _read_h03_artifact_candidate(artifact_paths: list[Path], *, min_family_support: int) -> dict[str, Any] | None:
    for artifact_path in artifact_paths:
        rows = _load_artifact_rows(artifact_path)
        if not rows:
            continue
        candidate = _compute_h03_artifact_rows_candidate(rows, artifact_path, min_family_support=min_family_support)
        if candidate is not None:
            return candidate
    return None


def _compute_h03_artifact_rows_candidate(
    rows: list[dict[str, Any]],
    artifact_path: Path,
    *,
    min_family_support: int,
) -> dict[str, Any] | None:
    family_column = next((column for column in FAMILY_ID_COLUMNS if any(column in row for row in rows)), None)
    if family_column is None:
        return None
    member_counts: list[int] = []
    support_values: list[float] = []
    compression_values: list[float] = []
    ratio_values: list[float] = []
    lift_values: list[float] = []
    family_ids: set[str] = set()
    discovered = 0
    stable_count = 0
    for row in rows:
        family_id = row.get(family_column)
        if family_id is None:
            continue
        family_ids.add(str(family_id))
        member_count = _maybe_int(row.get("member_count"))
        if member_count is None:
            members = row.get("members")
            if isinstance(members, list):
                member_count = len(members)
        member_count = 1 if member_count is None else member_count
        member_counts.append(member_count)
        discovered += member_count
        support_value = _maybe_float(row.get(_first_existing_key(row, SUPPORT_COLUMNS)))
        if support_value is None:
            support_value = float(member_count)
        support_values.append(support_value)
        if support_value >= float(min_family_support):
            stable_count += 1
        compression_value = _maybe_float(row.get(_first_existing_key(row, ("compression_gain", "family_compression_gain"))))
        if compression_value is not None:
            compression_values.append(compression_value)
        ratio_value = _maybe_float(row.get(_first_existing_key(row, ("compression_ratio",))))
        if ratio_value is not None:
            ratio_values.append(ratio_value)
        lift_value = _maybe_float(row.get(_first_existing_key(row, PREDICTION_LIFT_COLUMNS)))
        if lift_value is not None:
            lift_values.append(lift_value)
    if not family_ids:
        return None
    family_count = len(family_ids)
    singleton_count = sum(1 for count in member_counts if count <= 1)
    return {
        "artifact_paths_used": [str(artifact_path)],
        "selected_db_paths": [],
        "tables_seen": [],
        "candidate_tables_used": [],
        "interaction_count": None,
        "contingency_candidate_count": discovered,
        "discovered_contingency_count": discovered,
        "stable_contingency_count": stable_count,
        "transformation_family_candidate_count": family_count,
        "transformation_family_count": family_count,
        "stable_transformation_family_count": stable_count,
        "family_member_count_total": discovered,
        "family_mean_member_count": float(mean(member_counts)) if member_counts else None,
        "family_median_member_count": float(median(member_counts)) if member_counts else None,
        "family_max_member_count": max(member_counts) if member_counts else None,
        "singleton_family_count": singleton_count,
        "singleton_family_ratio": singleton_count / family_count if family_count > 0 else None,
        "compression_ratio": float(mean(ratio_values)) if ratio_values else (discovered / family_count if family_count > 0 else None),
        "compression_gain": float(mean(compression_values)) if compression_values else (1.0 - (family_count / discovered) if discovered > 0 else None),
        "mean_family_compression_gain": float(mean(compression_values)) if compression_values else None,
        "max_family_compression_gain": max(compression_values) if compression_values else None,
        "family_prediction_lift_mean": float(mean(lift_values)) if lift_values else None,
        "family_prediction_lift_median": float(median(lift_values)) if lift_values else None,
        "family_prediction_lift_max": max(lift_values) if lift_values else None,
        "family_cross_context_count": None,
        "family_cross_game_count": None,
        "family_cross_sampler_count": None,
        "usable_direct_family_evidence": True,
        "stability_approximated": _first_existing_key(rows[0], STABILITY_COLUMNS) is None,
    }


def _inspect_h03_schema_flags(sqlite_paths: list[Path]) -> dict[str, bool]:
    has_family_like = False
    has_derivable = False
    for sqlite_path in sqlite_paths:
        try:
            with sqlite3.connect(sqlite_path) as connection:
                table_map = _load_table_map(connection)
        except sqlite3.DatabaseError:
            continue
        for table_name, columns in table_map.items():
            if table_name in FAMILY_TABLE_NAMES or _first_matching(columns, FAMILY_ID_COLUMNS) is not None:
                has_family_like = True
            if _first_matching(columns, SIGNATURE_COLUMNS) is not None and (
                _first_matching(columns, CONTEXT_COLUMNS) is not None or "action" in columns
            ):
                has_derivable = True
        if has_family_like and has_derivable:
            break
    return {
        "has_family_like_db_schema": has_family_like,
        "has_derivable_family_schema": has_derivable,
    }


def _find_h03_artifact_paths(run_dir: Path) -> list[Path]:
    output: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if path.name == INPUT_REPORT_NAME:
            continue
        if not any(keyword in lowered for keyword in ARTIFACT_KEYWORDS):
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt", ".parquet"}:
            continue
        output.append(path)
    return output


def _load_artifact_rows(path: Path) -> list[dict[str, Any]] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, dict):
            for key in ("families", "rows", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return None
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows or None
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _report_source(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    validation = payload.get("validation")
    if isinstance(validation, dict):
        return validation
    return payload


def _extract_report_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = _report_source(payload)
    return {field: source.get(field) for field in H03_PRE_OBJECT_FIELDS if field in source}


def _find_sqlite_paths(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SQLITE_EXTENSIONS
    ]


def _rank_sqlite_paths(run_dir: Path, sqlite_paths: list[Path], *, prefer_db: Path | None) -> list[Path]:
    preferred = _resolve_preferred_db(run_dir, prefer_db)
    return sorted(sqlite_paths, key=lambda path: _sqlite_path_sort_key(run_dir, path, preferred))


def _resolve_preferred_db(run_dir: Path, prefer_db: Path | None) -> Path | None:
    if prefer_db is None:
        return None
    candidate = Path(prefer_db)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate.resolve()


def _sqlite_path_sort_key(run_dir: Path, path: Path, preferred: Path | None) -> tuple[int, int, int, int, str]:
    sampler_rank = _sampler_rank(path)
    try:
        stat = path.stat()
        size_key = -int(stat.st_size)
        mtime_key = -int(stat.st_mtime_ns)
    except OSError:
        size_key = 0
        mtime_key = 0
    try:
        rel = str(path.relative_to(run_dir))
    except ValueError:
        rel = str(path)
    return (
        0 if preferred is not None and path.resolve() == preferred else 1,
        sampler_rank,
        size_key,
        mtime_key,
        rel,
    )


def _sampler_rank(path: Path) -> int:
    lowered = str(path).lower()
    for index, sampler_name in enumerate(DB_SAMPLER_PRIORITY):
        if sampler_name in lowered:
            return index
    return len(DB_SAMPLER_PRIORITY)


def _load_table_map(connection: sqlite3.Connection) -> dict[str, set[str]]:
    table_map: dict[str, set[str]] = {}
    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        table_name = str(row[0])
        table_map[table_name] = _table_columns(connection, table_name)
    return table_map


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({_quote_ident(table_name)})").fetchall()}
    except sqlite3.DatabaseError:
        return set()


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _first_matching(columns: set[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _first_existing_key(row: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row:
            return name
    return None


def _stable_expr(column: str) -> str:
    quoted = _quote_ident(column)
    lowered = f"LOWER(TRIM(CAST({quoted} AS TEXT)))"
    return (
        f"CASE WHEN {quoted} IS NULL THEN 0 "
        f"WHEN {lowered} IN ('1', 'true', 'yes') THEN 1 "
        f"WHEN CAST({quoted} AS REAL) >= 1 THEN 1 "
        f"ELSE 0 END"
    )


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int | None:
    try:
        row = connection.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()
    except sqlite3.DatabaseError:
        return None
    return None if row is None or row[0] is None else int(row[0])


def _family_ratio_from_counts(
    connection: sqlite3.Connection,
    table_name: str,
    original_count_column: str,
    compressed_count_column: str,
) -> float | None:
    row = connection.execute(
        f"""
        SELECT AVG(
            CASE
                WHEN CAST({_quote_ident(compressed_count_column)} AS REAL) > 0
                THEN CAST({_quote_ident(original_count_column)} AS REAL) / CAST({_quote_ident(compressed_count_column)} AS REAL)
                ELSE NULL
            END
        )
        FROM {_quote_ident(table_name)}
        """
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def _family_member_counts(
    connection: sqlite3.Connection,
    table_name: str,
    family_column: str,
    contingency_column: str | None,
) -> list[int]:
    contingency_expr = _quote_ident(contingency_column) if contingency_column is not None else "rowid"
    rows = connection.execute(
        f"""
        SELECT COUNT(DISTINCT CAST({contingency_expr} AS TEXT))
        FROM {_quote_ident(table_name)}
        WHERE {_quote_ident(family_column)} IS NOT NULL
        GROUP BY CAST({_quote_ident(family_column)} AS TEXT)
        """
    ).fetchall()
    return [int(row[0]) for row in rows if row and row[0] is not None]


def _derived_family_member_counts(
    connection: sqlite3.Connection,
    table_name: str,
    signature_column: str,
    context_column: str | None,
    action_column: str | None,
) -> list[int]:
    context_expr = _quote_ident(context_column) if context_column is not None else "''"
    action_expr = f"CAST({_quote_ident(action_column)} AS TEXT)" if action_column is not None else "''"
    rows = connection.execute(
        f"""
        WITH base AS (
            SELECT
                CAST({_quote_ident(signature_column)} AS TEXT) AS family_key,
                CAST({context_expr} AS TEXT) || '|' || CAST({action_expr} AS TEXT) AS contingency_key
            FROM {_quote_ident(table_name)}
            WHERE {_quote_ident(signature_column)} IS NOT NULL
        )
        SELECT COUNT(DISTINCT contingency_key)
        FROM base
        GROUP BY family_key
        """
    ).fetchall()
    return [int(row[0]) for row in rows if row and row[0] is not None]


def _derived_family_presence(
    connection: sqlite3.Connection,
    sqlite_path: Path,
    table_name: str,
    signature_column: str,
    run_dir: Path,
) -> dict[str, dict[str, set[str]]]:
    game_name, sampler_name = _infer_game_sampler_from_path(run_dir, sqlite_path)
    rows = connection.execute(
        f"SELECT DISTINCT CAST({_quote_ident(signature_column)} AS TEXT) FROM {_quote_ident(table_name)} WHERE {_quote_ident(signature_column)} IS NOT NULL"
    ).fetchall()
    output: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        family_key = str(row[0])
        output[family_key] = {
            "contexts": set(),
            "games": {game_name} if game_name is not None else set(),
            "samplers": {sampler_name} if sampler_name is not None else set(),
        }
    return output


def _infer_game_sampler_from_path(run_dir: Path, sqlite_path: Path) -> tuple[str | None, str | None]:
    try:
        parts = list(sqlite_path.relative_to(run_dir).parts)
    except ValueError:
        return None, None
    if "sampling_v05c" not in parts:
        return None, None
    index = parts.index("sampling_v05c")
    if len(parts) > index + 2:
        return parts[index + 1], parts[index + 2]
    return None, None


def _merge_h03_candidate(result: dict[str, Any], candidate: dict[str, Any]) -> None:
    for key, value in candidate.items():
        if key in {"selected_db_paths", "tables_seen", "candidate_tables_used", "artifact_paths_used"}:
            result[key] = sorted(set(result.get(key, [])) | set(value))
        elif key == "usable_direct_family_evidence":
            result[key] = bool(result.get(key, False) or value)
        elif key == "stability_approximated":
            result[key] = bool(result.get(key, False) or value)
        elif value is not None:
            result[key] = value


def _sum_or_none(values: Any) -> int | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return int(sum(int(value) for value in items))


def _mean_or_none(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(mean(items))


def _max_or_none(values: Any) -> int | float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return max(items)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gt(value: Any, threshold: float) -> bool | None:
    if value is None:
        return None
    return float(value) > float(threshold)


def _gte(value: Any, threshold: float) -> bool | None:
    if value is None:
        return None
    return float(value) >= float(threshold)


def _lte(value: Any, threshold: float) -> bool | None:
    if value is None:
        return None
    return float(value) <= float(threshold)


def _eq(value: Any, target: float) -> bool | None:
    if value is None:
        return None
    return float(value) == float(target)


def _h03_ready_sort_key(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        0 if entry["has_required_carrier_fields"] is True else 1,
        0 if entry["has_sqlite_db"] is True else 1,
        0
        if (
            entry["has_family_artifacts"] is True
            or entry["has_family_like_db_schema"] is True
            or entry["has_derivable_family_schema"] is True
        )
        else 1,
        -int(entry.get("memory_record_count") or 0),
    )


def _isoformat_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _write_h03_ready_inventory(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / H03_READY_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / H03_READY_TXT_NAME).write_text(_format_h03_ready_text(result), encoding="utf-8")


def _format_h03_ready_text(result: dict[str, Any]) -> str:
    lines = [
        f"Runs root: {result['runs_root']}",
        f"Total reports found: {result['candidate_count']}",
        f"Ready count: {result['ready_count']}",
        "",
    ]
    recommended = result.get("recommended_run")
    if recommended is None:
        lines.append("No H03-ready existing v05c run found. Generate a new v05c run with current code, then rerun find-h03-ready-runs.")
        return "\n".join(lines)
    lines.extend(
        [
            "Top recommended run:",
            recommended["run_dir"],
            "",
            "Run:",
            "PYTHONPATH=src python -m v6.cli hypothesis-h03-report \\",
            f"  --run-dir {recommended['run_dir']} \\",
            f"  --output-dir {recommended['recommended_output_dir']} \\",
            f"  --scan-all-dbs \\",
            f"  --max-db-files {DEFAULT_MAX_DB_FILES}",
            "",
            "Candidates:",
        ]
    )
    for entry in result.get("runs", []):
        lines.extend(
            [
                f"- run_dir: {entry['run_dir']}",
                f"  h03_ready: {entry['h03_ready']}",
                f"  has_family_artifacts: {entry['has_family_artifacts']}",
                f"  has_family_like_db_schema: {entry['has_family_like_db_schema']}",
                f"  has_derivable_family_schema: {entry['has_derivable_family_schema']}",
                f"  has_required_carrier_fields: {entry['has_required_carrier_fields']}",
            ]
        )
    return "\n".join(lines)


def _populate_h03_evidence_lists(result: dict[str, Any]) -> None:
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    missing_evidence: list[str] = list(result.get("missing_evidence", []))
    if _gt(result.get("discovered_contingency_count"), 0) is True:
        evidence_for.append(f"Contingencies are present ({int(result['discovered_contingency_count'])}).")
    else:
        evidence_against.append("No usable contingency evidence was detected.")
    if _gt(result.get("transformation_family_count"), 0) is True:
        evidence_for.append(f"Transformation families are present ({int(result['transformation_family_count'])}).")
    else:
        evidence_against.append("No transformation families were detected.")
    if _gt(result.get("family_max_member_count"), 1) is True:
        evidence_for.append(f"Non-singleton family structure is present (max family size {int(result['family_max_member_count'])}).")
    elif result.get("family_max_member_count") is not None:
        evidence_against.append("All detected families are singleton structures.")
    if _gt(result.get("compression_ratio"), 1.0) is True:
        evidence_for.append(f"Compression ratio exceeds 1 ({float(result['compression_ratio']):.3f}).")
    elif result.get("compression_ratio") is not None:
        evidence_against.append(f"Compression ratio is weak ({float(result['compression_ratio']):.3f}).")
    if _gt(result.get("compression_gain"), 0.0) is True:
        evidence_for.append(f"Compression gain is positive ({float(result['compression_gain']):.3f}).")
    elif result.get("compression_gain") is not None:
        evidence_against.append(f"Compression gain is non-positive ({float(result['compression_gain']):.3f}).")
    if result.get("pre_object_condition_satisfied") is True:
        evidence_for.append("Object carriers and context-action fallback carriers are absent in the aggregate report.")
    elif result.get("pre_object_condition_satisfied") is False:
        evidence_against.append("Object-carrier emergence is already present in the aggregate report.")
    result["evidence_for"] = evidence_for
    result["evidence_against"] = evidence_against
    result["missing_evidence"] = missing_evidence


def _finalize_h03_result(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / H03_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / H03_TXT_NAME).write_text(_format_h03_text_report(result), encoding="utf-8")
    (output_dir / H03_MD_NAME).write_text(_format_h03_markdown_report(result), encoding="utf-8")


def _format_h03_text_report(result: dict[str, Any]) -> str:
    lines = [
        "# H03 Hypothesis Report",
        "",
        "Hypothesis statement:",
        result["hypothesis_statement"],
        "",
        "Decision:",
        result["decision"],
        "",
        "Direct family evidence:",
        f"- artifacts used: {', '.join(result.get('artifact_paths_used', [])) or 'none'}",
        f"- DBs used: {', '.join(result.get('selected_db_paths', [])) or 'none'}",
        f"- tables used: {', '.join(result.get('candidate_tables_used', [])) or 'none'}",
        f"- contingency count: {_fmt_value(result.get('discovered_contingency_count'))}",
        f"- family count: {_fmt_value(result.get('transformation_family_count'))}",
        f"- stable family count: {_fmt_value(result.get('stable_transformation_family_count'))}",
        f"- total family members: {_fmt_value(result.get('family_member_count_total'))}",
        f"- mean/median/max family size: {_fmt_value(result.get('family_mean_member_count'))} / {_fmt_value(result.get('family_median_member_count'))} / {_fmt_value(result.get('family_max_member_count'))}",
        f"- singleton family ratio: {_fmt_value(result.get('singleton_family_ratio'))}",
        f"- compression ratio: {_fmt_value(result.get('compression_ratio'))}",
        f"- compression gain: {_fmt_value(result.get('compression_gain'))}",
        f"- family prediction lift: {_fmt_value(result.get('family_prediction_lift_mean'))}",
        "",
        "Pre-object evidence:",
        f"- emergent_object_carrier_count: {_fmt_value(result.get('emergent_object_carrier_count'))}",
        f"- emergent_context_action_fallback_count: {_fmt_value(result.get('emergent_context_action_fallback_count'))}",
        f"- object carrier absence conclusion: {result.get('pre_object_condition_satisfied')}",
        "",
        "Evidence for:",
        *_format_bullets(result["evidence_for"]),
        "",
        "Evidence against:",
        *_format_bullets(result["evidence_against"]),
        "",
        "Missing evidence:",
        *_format_bullets(result["missing_evidence"]),
        "",
        "Acceptance checklist:",
        *_format_acceptance_checks(result["acceptance_checks"]),
        "",
        "Final scientific conclusion:",
        result["scientific_conclusion"],
        "",
    ]
    return "\n".join(lines)


def _format_h03_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# H03 Hypothesis Report",
        "",
        "## Hypothesis statement",
        "",
        result["hypothesis_statement"],
        "",
        "## Decision",
        "",
        f"`{result['decision']}`",
        "",
        "## Direct family evidence",
        f"- artifacts used: `{', '.join(result.get('artifact_paths_used', [])) or 'none'}`",
        f"- DBs used: `{', '.join(result.get('selected_db_paths', [])) or 'none'}`",
        f"- tables used: `{', '.join(result.get('candidate_tables_used', [])) or 'none'}`",
        f"- contingency count: `{_fmt_value(result.get('discovered_contingency_count'))}`",
        f"- family count: `{_fmt_value(result.get('transformation_family_count'))}`",
        f"- stable family count: `{_fmt_value(result.get('stable_transformation_family_count'))}`",
        f"- total family members: `{_fmt_value(result.get('family_member_count_total'))}`",
        f"- mean/median/max family size: `{_fmt_value(result.get('family_mean_member_count'))} / {_fmt_value(result.get('family_median_member_count'))} / {_fmt_value(result.get('family_max_member_count'))}`",
        f"- singleton family ratio: `{_fmt_value(result.get('singleton_family_ratio'))}`",
        f"- compression ratio: `{_fmt_value(result.get('compression_ratio'))}`",
        f"- compression gain: `{_fmt_value(result.get('compression_gain'))}`",
        f"- family prediction lift: `{_fmt_value(result.get('family_prediction_lift_mean'))}`",
        "",
        "## Pre-object evidence",
        f"- emergent_object_carrier_count: `{_fmt_value(result.get('emergent_object_carrier_count'))}`",
        f"- emergent_context_action_fallback_count: `{_fmt_value(result.get('emergent_context_action_fallback_count'))}`",
        f"- object carrier absence conclusion: `{result.get('pre_object_condition_satisfied')}`",
        "",
        "## Evidence For",
        *_format_bullets(result["evidence_for"]),
        "",
        "## Evidence Against",
        *_format_bullets(result["evidence_against"]),
        "",
        "## Missing Evidence",
        *_format_bullets(result["missing_evidence"]),
        "",
        "## Acceptance Checklist",
        *_format_acceptance_checks(result["acceptance_checks"]),
        "",
        "## Final scientific conclusion",
        "",
        result["scientific_conclusion"],
        "",
    ]
    return "\n".join(lines)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_bullets(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _format_acceptance_checks(checks: dict[str, Any]) -> list[str]:
    return [f"- {name}: {value}" for name, value in checks.items()]
