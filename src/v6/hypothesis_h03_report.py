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
SIGNATURE_COLUMNS = ("effect_signature", "delta_signature", "changed_cells_signature", "outcome_signature", "centroid_vector")
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
    "row_count_available": None,
    "row_count_used": 0,
    "max_rows_applied": False,
    "selected_db_paths": [],
    "tables_seen": [],
    "candidate_tables_used": [],
    "artifact_paths_used": [],
    "global_family_merge_enabled": True,
    "global_family_count_before_merge": 0,
    "global_family_count_after_merge": 0,
    "families_merged_across_shards": 0,
    "families_merged_across_games": 0,
    "families_merged_across_samplers": 0,
    "top_global_families": [],
    "top_singleton_family_signatures": [],
    "singleton_families_by_game": {},
    "singleton_families_by_sampler": {},
    "singleton_families_by_action": {},
    "singleton_families_by_effect_type": {},
    "singleton_family_diagnostics": {
        "genuine_rare_count": 0,
        "over_specific_count": 0,
        "uncertain_count": 0,
    },
    "relaxed_canonicalization_diagnostics": {},
    "singleton_ratio_strict": None,
    "singleton_ratio_relaxed": None,
    "singleton_family_count_relaxed": None,
    "transformation_family_count_relaxed": None,
    "relaxed_family_count": None,
    "relaxed_singleton_family_count": None,
    "relaxed_singleton_family_ratio": None,
    "over_specific_singleton_count": None,
    "over_specific_singleton_ratio": None,
    "merge_safety_passed": None,
    "unsafe_relaxed_merge_count": None,
    "relaxed_decision_candidate": None,
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
    memory_dir: Path | None = None,
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
    result["evidence_source"] = "raw_epoch_db"

    report_metrics = _extract_report_metrics(input_report)
    for field, value in report_metrics.items():
        result[field] = value

    if input_report is None and memory_dir is not None:
        compact = _extract_h03_compact_metrics(Path(memory_dir))
        result.update(compact)
        result["evidence_source"] = "compact_memory"
        result["decision"] = "PARTIALLY_VALID" if _gt(result.get("transformation_family_count"), 0) else "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H03 evaluated from compact memory after raw cleanup."
            if _gt(result.get("transformation_family_count"), 0)
            else "H03 remains inconclusive because compact memory lacks family evidence."
        )
        _finalize_h03_result(result, output_dir)
        return result
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
    if memory_dir is not None and not sqlite_paths:
        compact = _extract_h03_compact_metrics(Path(memory_dir))
        for key, value in compact.items():
            if result.get(key) is None:
                result[key] = value
        result["evidence_source"] = "mixed"
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
            "H03 is supported in this run. Repeated contingencies compress into non-singleton transformation families with positive compression gain while no object-carrier emergence is detected."
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
        elif result.get("family_cross_game_count") is None or result.get("family_cross_sampler_count") is None:
            result["scientific_conclusion"] = (
                "H03 is partially supported in this run. Transformation-family compression is present, but cross-game or cross-sampler family identity is not fully exposed by the current artifacts."
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
        if result.get("relaxed_decision_candidate") == "VALID":
            result["scientific_conclusion"] += (
                " H03 remains PARTIALLY_VALID under strict canonicalization. Relaxed canonicalization suggests H03 would become VALID if over-specific centroid signatures are safely collapsed; this requires merge-safety validation."
            )
    else:
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H03 remains inconclusive because the run artifacts do not cleanly distinguish contingency structure from family-level compression."
        )

    _populate_h03_evidence_lists(result)
    _finalize_h03_result(result, output_dir)
    return result


def _extract_h03_compact_metrics(memory_dir: Path) -> dict[str, Any]:
    current_state = memory_dir / "current_state.sqlite"
    graph_db = memory_dir / "graph.sqlite"
    if not current_state.exists():
        return {}
    with sqlite3.connect(current_state) as state_conn:
        family_count = int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
        singleton_count = int(state_conn.execute("SELECT COUNT(*) FROM transformation_families WHERE member_count <= 1").fetchone()[0])
        member_total = int(state_conn.execute("SELECT COALESCE(SUM(member_count), 0) FROM transformation_families").fetchone()[0])
    graph_metrics = {"family_cross_context_count": None, "family_cross_game_count": None, "family_cross_sampler_count": None}
    if graph_db.exists():
        with sqlite3.connect(graph_db) as graph_conn:
            graph_metrics = {
                "family_cross_context_count": int(
                    graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'explains_effect'").fetchone()[0]
                ),
                "family_cross_game_count": int(
                    graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'observed_in'").fetchone()[0]
                ),
                "family_cross_sampler_count": int(
                    graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'sampled_by'").fetchone()[0]
                ),
            }
    return {
        "transformation_family_count": family_count,
        "stable_transformation_family_count": family_count,
        "family_member_count_total": member_total,
        "singleton_family_count": singleton_count,
        "singleton_family_ratio": (singleton_count / family_count) if family_count else None,
        "compression_ratio": (member_total / family_count) if family_count else None,
        "compression_gain": ((member_total / family_count) - 1.0) if family_count else None,
        **graph_metrics,
    }


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
    if int(max_db_files) <= 0:
        limited_paths = list(ranked_paths)
    else:
        limited_paths = ranked_paths[: max(1, int(max_db_files))]
    artifact_paths = _find_h03_artifact_paths(run_dir)

    result = {
        "db_found": bool(sqlite_paths),
        "db_paths_total": len(sqlite_paths),
        "db_paths_inspected": 0,
        "db_scan_truncated": len(limited_paths) < len(ranked_paths),
        "row_count_available": None,
        "row_count_used": 0,
        "max_rows_applied": False,
        "selected_db_paths": [],
        "tables_seen": [],
        "candidate_tables_used": [],
        "artifact_paths_used": [],
        "global_family_merge_enabled": True,
        "global_family_count_before_merge": 0,
        "global_family_count_after_merge": 0,
        "families_merged_across_shards": 0,
        "families_merged_across_games": 0,
        "families_merged_across_samplers": 0,
        "top_global_families": [],
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
        aggregate_candidate = _aggregate_h03_db_candidates(
            db_candidates,
            min_family_support=min_family_support,
            max_rows=max_rows,
        )
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
        if int(max_db_files) <= 0:
            limited_paths = list(sqlite_paths)
        else:
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
    table_order = sorted(table_map, key=lambda name: (0 if name in CANDIDATE_TABLE_NAMES else 1, name))
    interactions_count = _count_rows(connection, "interactions") if "interactions" in table_map else None
    for table_name in table_order:
        columns = table_map[table_name]
        if table_name == "contingencies" and "transformation_family" in columns:
            candidate = _compute_contingency_join_family_candidate(
                connection,
                sqlite_path=sqlite_path,
                run_dir=run_dir,
                table_map=table_map,
                table_name=table_name,
                columns=columns,
                max_rows=max_rows,
                min_family_support=min_family_support,
                interaction_count=interactions_count,
            )
            if candidate is not None:
                return candidate
        if table_name in FAMILY_TABLE_NAMES or _first_matching(columns, FAMILY_ID_COLUMNS) is not None:
            candidate = _compute_family_table_candidate(
                connection,
                sqlite_path=sqlite_path,
                run_dir=run_dir,
                table_name=table_name,
                columns=columns,
                max_rows=max_rows,
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
            max_rows=max_rows,
            min_family_support=min_family_support,
            interaction_count=interactions_count,
        )
        if candidate is not None:
            return candidate
    return None


def _compute_contingency_join_family_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_map: dict[str, set[str]],
    table_name: str,
    columns: set[str],
    max_rows: int,
    min_family_support: int,
    interaction_count: int | None,
) -> dict[str, Any] | None:
    family_columns = table_map.get("transformation_families")
    if not family_columns or "id" not in family_columns or "centroid_vector" not in family_columns:
        return None
    selected_sql = """
        SELECT
            c.id AS contingency_id,
            c.context_signature,
            c.action,
            c.support_count,
            tf.centroid_vector AS centroid_vector
        FROM contingencies AS c
        LEFT JOIN transformation_families AS tf
          ON c.transformation_family = tf.id
        WHERE tf.centroid_vector IS NOT NULL
        LIMIT ?
    """
    row_count_available = _count_rows(connection, table_name)
    try:
        fetched = connection.execute(selected_sql, (max(1, int(max_rows)),)).fetchall()
    except sqlite3.DatabaseError:
        return None
    rows = [
        {
            "contingency_id": item[0],
            "context_signature": item[1],
            "action": item[2],
            "support_count": item[3],
            "centroid_vector": item[4],
        }
        for item in fetched
    ]
    observations = _family_observations_from_rows(
        rows,
        sqlite_path=sqlite_path,
        run_dir=run_dir,
        family_column=None,
        semantic_column="centroid_vector",
        contingency_column="contingency_id",
        support_column="support_count",
        compression_column=None,
        ratio_column=None,
        original_count_column=None,
        compressed_count_column=None,
        prediction_lift_column=None,
        stability_column=None,
        context_column="context_signature",
        game_column=None,
        sampler_column=None,
        action_column="action",
        min_family_support=min_family_support,
    )
    if not observations:
        return None
    return {
        "selected_db_paths": [str(sqlite_path)],
        "tables_seen": [table_name, "transformation_families"],
        "candidate_tables_used": [table_name, "transformation_families"],
        "interaction_count": interaction_count,
        "row_count_available": row_count_available,
        "row_count_used": len(rows),
        "max_rows_applied": row_count_available is not None and len(rows) < row_count_available,
        "family_observations": observations,
        "global_family_count_before_merge": len({item["canonical_signature"] for item in observations}),
        "usable_direct_family_evidence": True,
        "stability_approximated": True,
    }


def _compute_family_table_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_name: str,
    columns: set[str],
    max_rows: int,
    min_family_support: int,
    interaction_count: int | None,
) -> dict[str, Any] | None:
    family_column = _first_matching(columns, FAMILY_ID_COLUMNS)
    semantic_column = _first_matching(columns, SIGNATURE_COLUMNS)
    if family_column is None and semantic_column is None:
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
    selected_columns = [name for name in {
        family_column,
        semantic_column,
        contingency_column,
        support_column,
        compression_column,
        ratio_column,
        original_count_column,
        compressed_count_column,
        prediction_lift_column,
        stability_column,
        context_column,
        game_column,
        sampler_column,
        "action" if "action" in columns else None,
    } if name is not None]
    rows, row_count_available, row_count_used = _select_rows(connection, table_name, selected_columns, max_rows=max_rows)
    if not rows:
        return None
    observations = _family_observations_from_rows(
        rows,
        sqlite_path=sqlite_path,
        run_dir=run_dir,
        family_column=family_column,
        semantic_column=semantic_column,
        contingency_column=contingency_column,
        support_column=support_column,
        compression_column=compression_column,
        ratio_column=ratio_column,
        original_count_column=original_count_column,
        compressed_count_column=compressed_count_column,
        prediction_lift_column=prediction_lift_column,
        stability_column=stability_column,
        context_column=context_column,
        game_column=game_column,
        sampler_column=sampler_column,
        action_column="action" if "action" in columns else None,
        min_family_support=min_family_support,
    )
    if not observations:
        return None
    candidate = {
        "selected_db_paths": [str(sqlite_path)],
        "tables_seen": [table_name],
        "candidate_tables_used": [table_name],
        "interaction_count": interaction_count,
        "row_count_available": row_count_available,
        "row_count_used": row_count_used,
        "max_rows_applied": row_count_available is not None and row_count_used < row_count_available,
        "family_observations": observations,
        "global_family_count_before_merge": len({item["canonical_signature"] for item in observations}),
        "usable_direct_family_evidence": True,
        "stability_approximated": stability_column is None,
    }
    return candidate


def _compute_derivable_family_candidate(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    run_dir: Path,
    table_name: str,
    columns: set[str],
    max_rows: int,
    min_family_support: int,
    interaction_count: int | None,
) -> dict[str, Any] | None:
    signature_column = _first_matching(columns, SIGNATURE_COLUMNS)
    if signature_column is None:
        return None
    context_column = _first_matching(columns, CONTEXT_COLUMNS)
    action_column = _first_matching(columns, ("action",))
    if context_column is None and action_column is None:
        return None
    support_column = _first_matching(columns, SUPPORT_COLUMNS)
    game_column = _first_matching(columns, GAME_COLUMNS)
    sampler_column = _first_matching(columns, SAMPLER_COLUMNS)
    selected_columns = [name for name in {
        signature_column,
        context_column,
        action_column,
        support_column,
        game_column,
        sampler_column,
    } if name is not None]
    rows, row_count_available, row_count_used = _select_rows(connection, table_name, selected_columns, max_rows=max_rows)
    if not rows:
        return None
    observations = _family_observations_from_rows(
        rows,
        sqlite_path=sqlite_path,
        run_dir=run_dir,
        family_column=None,
        semantic_column=signature_column,
        contingency_column=None,
        support_column=support_column,
        compression_column=None,
        ratio_column=None,
        original_count_column=None,
        compressed_count_column=None,
        prediction_lift_column=None,
        stability_column=None,
        context_column=context_column,
        game_column=game_column,
        sampler_column=sampler_column,
        action_column=action_column,
        min_family_support=min_family_support,
    )
    if not observations:
        return None
    return {
        "selected_db_paths": [str(sqlite_path)],
        "tables_seen": [table_name],
        "candidate_tables_used": [table_name],
        "interaction_count": interaction_count,
        "row_count_available": row_count_available,
        "row_count_used": row_count_used,
        "max_rows_applied": row_count_available is not None and row_count_used < row_count_available,
        "family_observations": observations,
        "global_family_count_before_merge": len({item["canonical_signature"] for item in observations}),
        "usable_direct_family_evidence": True,
        "stability_approximated": True,
    }


def _aggregate_h03_db_candidates(candidates: list[dict[str, Any]], *, min_family_support: int, max_rows: int) -> dict[str, Any]:
    global_families: dict[str, dict[str, Any]] = {}
    relaxed_families: dict[str, set[str]] = {}
    raw_family_signatures = 0
    row_count_available = 0
    row_count_used = 0
    max_rows_applied = False
    explicit_game_sampler_coverage = False
    for candidate in candidates:
        row_count_available += int(candidate.get("row_count_available") or 0)
        row_count_used += int(candidate.get("row_count_used") or 0)
        max_rows_applied = max_rows_applied or bool(candidate.get("max_rows_applied", False))
        observations = candidate.get("family_observations", [])
        raw_family_signatures += len({item["canonical_signature"] for item in observations})
        for item in observations:
            relaxed_families.setdefault(item["relaxed_canonical_signature"], set()).add(item["strict_canonical_signature"])
            family = global_families.setdefault(
                item["canonical_signature"],
                {
                    "member_keys": set(),
                    "games": set(),
                    "samplers": set(),
                    "contexts": set(),
                    "support_total": 0.0,
                    "source_db_paths": set(),
                    "compression_values": [],
                    "ratio_values": [],
                    "prediction_lift_values": [],
                    "stable_flags": [],
                    "actions": set(),
                    "effect_types": set(),
                },
            )
            family["member_keys"].add(item["member_key"])
            family["games"].update(item["games"])
            family["samplers"].update(item["samplers"])
            family["contexts"].update(item["contexts"])
            family["support_total"] += float(item["support_value"] or 0.0)
            family["source_db_paths"].add(item["source_db_path"])
            if item.get("compression_gain") is not None:
                family["compression_values"].append(float(item["compression_gain"]))
            if item.get("compression_ratio") is not None:
                family["ratio_values"].append(float(item["compression_ratio"]))
            if item.get("prediction_lift") is not None:
                family["prediction_lift_values"].append(float(item["prediction_lift"]))
            if item.get("stable_flag") is not None:
                family["stable_flags"].append(bool(item["stable_flag"]))
            if item.get("action_value") is not None:
                family["actions"].add(item["action_value"])
            if item.get("effect_type") is not None:
                family["effect_types"].add(item["effect_type"])
            if item["games"] or item["samplers"]:
                explicit_game_sampler_coverage = True

    family_count = len(global_families)
    member_counts = [len(item["member_keys"]) for item in global_families.values()]
    family_member_count_total = sum(member_counts)
    singleton_family_count = sum(1 for count in member_counts if count <= 1)
    stable_family_count = sum(
        1
        for item in global_families.values()
        if item["support_total"] >= float(min_family_support) or any(item["stable_flags"])
    )
    top_global_families = [
        {
            "canonical_signature": signature,
            "member_count": len(item["member_keys"]),
            "games": sorted(item["games"]),
            "samplers": sorted(item["samplers"]),
            "contexts": len(item["contexts"]),
            "support_total": item["support_total"],
            "source_db_count": len(item["source_db_paths"]),
        }
        for signature, item in sorted(
            global_families.items(),
            key=lambda pair: (-len(pair[1]["member_keys"]), -len(pair[1]["games"]), -len(pair[1]["samplers"]), pair[0]),
        )[:10]
    ]
    relaxed_diagnostics = _relaxed_canonicalization_diagnostics(global_families, relaxed_families)
    relaxed_family_count = relaxed_diagnostics["relaxed_family_count"]
    relaxed_singleton_count = relaxed_diagnostics["relaxed_singleton_family_count"]
    singleton_diagnostics = _singleton_diagnostics(global_families, relaxed_families)
    strict_singleton_ratio = (singleton_family_count / family_count) if family_count > 0 else None
    relaxed_singleton_ratio = (relaxed_singleton_count / relaxed_family_count) if relaxed_family_count > 0 else None
    out = {
        "selected_db_paths": sorted({path for candidate in candidates for path in candidate.get("selected_db_paths", [])}),
        "tables_seen": sorted({table for candidate in candidates for table in candidate.get("tables_seen", [])}),
        "candidate_tables_used": sorted({table for candidate in candidates for table in candidate.get("candidate_tables_used", [])}),
        "interaction_count": _sum_or_none(candidate.get("interaction_count") for candidate in candidates),
        "contingency_candidate_count": family_member_count_total if global_families else None,
        "discovered_contingency_count": family_member_count_total if global_families else None,
        "stable_contingency_count": family_member_count_total if global_families else None,
        "transformation_family_candidate_count": raw_family_signatures if global_families else None,
        "transformation_family_count": family_count if global_families else None,
        "stable_transformation_family_count": stable_family_count if global_families else None,
        "family_member_count_total": family_member_count_total if global_families else None,
        "family_mean_member_count": float(mean(member_counts)) if member_counts else None,
        "family_median_member_count": float(median(member_counts)) if member_counts else None,
        "family_max_member_count": max(member_counts) if member_counts else None,
        "singleton_family_count": singleton_family_count if global_families else None,
        "singleton_family_ratio": (singleton_family_count / family_count) if family_count > 0 else None,
        "compression_ratio": (family_member_count_total / family_count) if family_count > 0 else None,
        "compression_gain": (1.0 - (family_count / family_member_count_total)) if family_member_count_total > 0 else None,
        "mean_family_compression_gain": _mean_or_none(
            value for item in global_families.values() for value in item["compression_values"]
        ),
        "max_family_compression_gain": _max_or_none(
            value for item in global_families.values() for value in item["compression_values"]
        ),
        "family_prediction_lift_mean": _mean_or_none(
            value for item in global_families.values() for value in item["prediction_lift_values"]
        ),
        "family_prediction_lift_median": _median_or_none(
            value for item in global_families.values() for value in item["prediction_lift_values"]
        ),
        "family_prediction_lift_max": _max_or_none(
            value for item in global_families.values() for value in item["prediction_lift_values"]
        ),
        "family_cross_context_count": sum(1 for item in global_families.values() if len(item["contexts"]) > 1),
        "family_cross_game_count": (
            sum(1 for item in global_families.values() if len(item["games"]) > 1)
            if explicit_game_sampler_coverage
            else None
        ),
        "family_cross_sampler_count": (
            sum(1 for item in global_families.values() if len(item["samplers"]) > 1)
            if explicit_game_sampler_coverage
            else None
        ),
        "row_count_available": row_count_available or None,
        "row_count_used": row_count_used,
        "max_rows_applied": max_rows_applied,
        "global_family_count_before_merge": raw_family_signatures,
        "global_family_count_after_merge": family_count,
        "families_merged_across_shards": max(0, raw_family_signatures - family_count),
        "families_merged_across_games": sum(1 for item in global_families.values() if len(item["games"]) > 1),
        "families_merged_across_samplers": sum(1 for item in global_families.values() if len(item["samplers"]) > 1),
        "top_global_families": top_global_families,
        "top_singleton_family_signatures": singleton_diagnostics["top_singletons"],
        "singleton_families_by_game": singleton_diagnostics["by_game"],
        "singleton_families_by_sampler": singleton_diagnostics["by_sampler"],
        "singleton_families_by_action": singleton_diagnostics["by_action"],
        "singleton_families_by_effect_type": singleton_diagnostics["by_effect_type"],
        "singleton_family_diagnostics": singleton_diagnostics["diagnostics"],
        "relaxed_canonicalization_diagnostics": relaxed_diagnostics,
        "safe_merge_proposal_table": relaxed_diagnostics["safe_merge_proposal_table"],
        "singleton_ratio_strict": strict_singleton_ratio,
        "singleton_ratio_relaxed": relaxed_singleton_ratio,
        "singleton_family_count_relaxed": relaxed_singleton_count if relaxed_families else None,
        "transformation_family_count_relaxed": relaxed_family_count if relaxed_families else None,
        "relaxed_family_count": relaxed_family_count if relaxed_families else None,
        "relaxed_singleton_family_count": relaxed_singleton_count if relaxed_families else None,
        "relaxed_singleton_family_ratio": relaxed_singleton_ratio,
        "over_specific_singleton_count": (
            singleton_diagnostics["diagnostics"]["over_specific_context_count"]
            + singleton_diagnostics["diagnostics"]["over_specific_action_count"]
            + singleton_diagnostics["diagnostics"]["over_specific_delta_signature_count"]
        ),
        "over_specific_singleton_ratio": (
            (
                singleton_diagnostics["diagnostics"]["over_specific_context_count"]
                + singleton_diagnostics["diagnostics"]["over_specific_action_count"]
                + singleton_diagnostics["diagnostics"]["over_specific_delta_signature_count"]
            ) / singleton_family_count
            if singleton_family_count > 0
            else None
        ),
        "family_prediction_lift_before_merge": _mean_or_none(
            entry.get("family_prediction_lift_before_merge")
            for entry in relaxed_diagnostics["safe_merge_proposal_table"]
            if entry.get("family_prediction_lift_before_merge") is not None
        ),
        "family_prediction_lift_after_safe_merge": _mean_or_none(
            entry.get("family_prediction_lift_after_safe_merge")
            for entry in relaxed_diagnostics["safe_merge_proposal_table"]
            if entry.get("family_prediction_lift_after_safe_merge") is not None
        ),
        "singleton_ratio_after_safe_merge": relaxed_singleton_ratio,
        "merge_safety_passed": relaxed_diagnostics["merge_safety_passed"],
        "unsafe_relaxed_merge_count": relaxed_diagnostics["unsafe_relaxed_merge_count"],
        "relaxed_decision_candidate": (
            "VALID"
            if (
                relaxed_singleton_ratio is not None
                and relaxed_singleton_ratio <= 0.50
                and family_member_count_total > family_count
                and (1.0 - (family_count / family_member_count_total)) > 0.0
                and relaxed_diagnostics["merge_safety_passed"] is True
                and relaxed_diagnostics["unsafe_relaxed_merge_count"] == 0
            )
            else None
        ),
        "usable_direct_family_evidence": any(candidate.get("usable_direct_family_evidence", False) for candidate in candidates),
        "stability_approximated": any(candidate.get("stability_approximated", False) for candidate in candidates),
    }
    del max_rows
    return out


def _singleton_diagnostics(
    global_families: dict[str, dict[str, Any]],
    relaxed_families: dict[str, set[str]],
) -> dict[str, Any]:
    singletons = [
        (signature, item)
        for signature, item in global_families.items()
        if len(item["member_keys"]) <= 1
    ]
    by_game: dict[str, int] = {}
    by_sampler: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_effect_type: dict[str, int] = {}
    diagnostics = {
        "genuine_rare_count": 0,
        "over_specific_context_count": 0,
        "over_specific_action_count": 0,
        "over_specific_delta_signature_count": 0,
        "reset_terminal_artifact_count": 0,
        "movement_alias_count": 0,
        "unknown_count": 0,
    }
    ranked: list[dict[str, Any]] = []
    for signature, item in sorted(singletons, key=lambda pair: (-pair[1]["support_total"], pair[0])):
        for name in item["games"] or {"unknown"}:
            by_game[name] = by_game.get(name, 0) + 1
        for name in item["samplers"] or {"unknown"}:
            by_sampler[name] = by_sampler.get(name, 0) + 1
        for name in item["actions"] or {"unknown"}:
            by_action[name] = by_action.get(name, 0) + 1
        for name in item["effect_types"] or {"unknown"}:
            by_effect_type[name] = by_effect_type.get(name, 0) + 1
        relaxed_signature = _relaxed_family_signature(signature)
        relaxed_group_size = len(relaxed_families.get(relaxed_signature, set()))
        effect_text = " ".join(sorted(item.get("effect_types", set()))).lower()
        action_count = len(item.get("actions", set()))
        context_count = len(item.get("contexts", set()))
        if "terminal" in effect_text or "reset" in effect_text:
            diagnosis = "reset_terminal_artifact"
            diagnostics["reset_terminal_artifact_count"] += 1
        elif any(token in effect_text for token in ("move", "shift", "translate", "position")) and relaxed_group_size > 1:
            diagnosis = "movement_alias"
            diagnostics["movement_alias_count"] += 1
        elif relaxed_group_size > 1 and context_count <= 1:
            diagnosis = "over_specific_context"
            diagnostics["over_specific_context_count"] += 1
        elif relaxed_group_size > 1 and action_count <= 1:
            diagnosis = "over_specific_action"
            diagnostics["over_specific_action_count"] += 1
        elif relaxed_group_size > 1:
            diagnosis = "over_specific_delta_signature"
            diagnostics["over_specific_delta_signature_count"] += 1
        elif item["support_total"] >= 2.0 and len(item["contexts"]) <= 1:
            diagnosis = "genuine_rare"
            diagnostics["genuine_rare_count"] += 1
        else:
            diagnosis = "unknown"
            diagnostics["unknown_count"] += 1
        ranked.append(
            {
                "canonical_signature": signature,
                "support_total": item["support_total"],
                "games": sorted(item["games"]),
                "samplers": sorted(item["samplers"]),
                "actions": sorted(item["actions"]),
                "effect_types": sorted(item["effect_types"]),
                "source_db_count": len(item["source_db_paths"]),
                "relaxed_group_size": relaxed_group_size,
                "diagnosis": diagnosis,
            }
        )
    return {
        "top_singletons": ranked[:10],
        "by_game": dict(sorted(by_game.items())),
        "by_sampler": dict(sorted(by_sampler.items())),
        "by_action": dict(sorted(by_action.items())),
        "by_effect_type": dict(sorted(by_effect_type.items())),
        "diagnostics": diagnostics,
    }


def _relaxed_canonicalization_diagnostics(
    global_families: dict[str, dict[str, Any]],
    relaxed_families: dict[str, set[str]],
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    unsafe_count = 0
    relaxed_singletons = 0
    proposals: list[dict[str, Any]] = []
    for relaxed_signature, strict_signatures in sorted(relaxed_families.items()):
        members = [global_families[signature] for signature in strict_signatures if signature in global_families]
        effect_types = sorted({effect for item in members for effect in item.get("effect_types", set())})
        action_groups = sorted({ _action_group(action) for item in members for action in item.get("actions", set()) })
        polarities = sorted({ _effect_polarity(effect) for effect in effect_types })
        safe = (
            len(effect_types) <= 1
            and len(action_groups) <= 1
            and len(polarities) <= 1
            and not ("mixed" in polarities and len(polarities) > 1)
        )
        if not safe:
            unsafe_count += 1
        if len(strict_signatures) <= 1:
            relaxed_singletons += 1
        support_count = sum(float(item.get("support_total", 0.0) or 0.0) for item in members)
        prediction_values = [value for item in members for value in item.get("prediction_lift_values", [])]
        before_merge = _mean_or_none(prediction_values)
        after_merge = before_merge if safe else None
        proposals.append(
            {
                "strict_family_id": sorted(strict_signatures),
                "relaxed_signature": relaxed_signature,
                "action_group": action_groups[0] if len(action_groups) == 1 else "mixed",
                "effect_type": effect_types[0] if len(effect_types) == 1 else "mixed",
                "polarity": polarities[0] if len(polarities) == 1 else "mixed",
                "source_game_count": len({game for item in members for game in item.get("games", set())}),
                "source_sampler_count": len({sampler for item in members for sampler in item.get("samplers", set())}),
                "support_count": support_count,
                "merge_safety_status": "safe" if safe else "unsafe",
                "unsafe_reason": None if safe else "signature_conflict",
                "family_prediction_lift_before_merge": before_merge,
                "family_prediction_lift_after_safe_merge": after_merge,
            }
        )
        groups.append(
            {
                "relaxed_signature": relaxed_signature,
                "strict_signature_count": len(strict_signatures),
                "effect_types": effect_types,
                "action_groups": action_groups,
                "polarities": polarities,
                "safe": safe,
            }
        )
    return {
        "groups": groups[:20],
        "relaxed_family_count": len(relaxed_families),
        "relaxed_singleton_family_count": relaxed_singletons,
        "merge_safety_passed": unsafe_count == 0,
        "unsafe_relaxed_merge_count": unsafe_count,
        "safe_merge_proposal_table": proposals[:50],
    }


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


def _select_rows(
    connection: sqlite3.Connection,
    table_name: str,
    selected_columns: list[str],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], int | None, int]:
    table_ref = _quote_ident(table_name)
    row_count_available = _count_rows(connection, table_name)
    columns_sql = ", ".join(_quote_ident(name) for name in selected_columns) if selected_columns else "rowid"
    limit_value = max(1, int(max_rows))
    try:
        fetched = connection.execute(
            f"SELECT {columns_sql} FROM {table_ref} LIMIT {limit_value}"
        ).fetchall()
    except sqlite3.DatabaseError:
        return [], row_count_available, 0
    rows: list[dict[str, Any]] = []
    for item in fetched:
        row = {name: item[index] for index, name in enumerate(selected_columns)}
        rows.append(row)
    return rows, row_count_available, len(rows)


def _family_observations_from_rows(
    rows: list[dict[str, Any]],
    *,
    sqlite_path: Path,
    run_dir: Path,
    family_column: str | None,
    semantic_column: str | None,
    contingency_column: str | None,
    support_column: str | None,
    compression_column: str | None,
    ratio_column: str | None,
    original_count_column: str | None,
    compressed_count_column: str | None,
    prediction_lift_column: str | None,
    stability_column: str | None,
    context_column: str | None,
    game_column: str | None,
    sampler_column: str | None,
    action_column: str | None,
    min_family_support: int,
) -> list[dict[str, Any]]:
    path_game, path_sampler = _infer_game_sampler_from_path(run_dir, sqlite_path)
    observations: list[dict[str, Any]] = []
    for row in rows:
        canonical_signature = _canonical_family_signature(
            row,
            family_column=family_column,
            semantic_column=semantic_column,
        )
        if canonical_signature is None:
            continue
        support_value = _maybe_float(row.get(support_column)) if support_column is not None else None
        if support_value is None:
            support_value = 1.0
        ratio_value = _maybe_float(row.get(ratio_column)) if ratio_column is not None else None
        if ratio_value is None and original_count_column and compressed_count_column:
            original_count = _maybe_float(row.get(original_count_column))
            compressed_count = _maybe_float(row.get(compressed_count_column))
            if original_count is not None and compressed_count not in (None, 0.0):
                ratio_value = original_count / compressed_count
        compression_gain = _maybe_float(row.get(compression_column)) if compression_column is not None else None
        if compression_gain is None and ratio_value is not None and ratio_value > 0:
            compression_gain = 1.0 - (1.0 / ratio_value)
        game_value = _normalize_scalar(row.get(game_column)) if game_column is not None else None
        sampler_value = _normalize_scalar(row.get(sampler_column)) if sampler_column is not None else None
        context_value = _context_key(row, context_column=context_column, action_column=action_column)
        observations.append(
            {
                "canonical_signature": canonical_signature,
                "strict_canonical_signature": canonical_signature,
                "relaxed_canonical_signature": _relaxed_family_signature(canonical_signature),
                "member_key": _member_key(
                    row,
                    sqlite_path=sqlite_path,
                    contingency_column=contingency_column,
                    context_column=context_column,
                    action_column=action_column,
                    semantic_column=semantic_column,
                ),
                "games": {game_value} if game_value is not None else ({path_game} if path_game is not None else set()),
                "samplers": {sampler_value} if sampler_value is not None else ({path_sampler} if path_sampler is not None else set()),
                "contexts": {context_value} if context_value is not None else set(),
                "support_value": support_value,
                "source_db_path": str(sqlite_path),
                "compression_gain": compression_gain,
                "compression_ratio": ratio_value,
                "prediction_lift": _maybe_float(row.get(prediction_lift_column)) if prediction_lift_column is not None else None,
                "stable_flag": _stable_value(row.get(stability_column)) if stability_column is not None else (support_value >= float(min_family_support)),
                "action_value": _normalize_scalar(row.get(action_column)) if action_column is not None else None,
                "effect_type": _infer_effect_type(canonical_signature),
            }
        )
    return observations


def _canonical_family_signature(
    row: dict[str, Any],
    *,
    family_column: str | None,
    semantic_column: str | None,
) -> str | None:
    semantic_value = _normalized_signature_value(row.get(semantic_column)) if semantic_column is not None else None
    family_value = _normalized_signature_value(row.get(family_column)) if family_column is not None else None
    if family_value is not None and not _looks_local_integer_id(family_value):
        return family_value
    if semantic_value is not None:
        return semantic_value
    return None


def _normalized_signature_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return _normalize_json_like(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and isinstance(parsed, (dict, list)):
        return _normalize_json_like(parsed)
    lowered = text.lower()
    return lowered


def _normalize_json_like(value: Any) -> str:
    cleaned = _remove_volatile_fields(value)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _remove_volatile_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _remove_volatile_fields(item)
            for key, item in sorted(value.items())
            if str(key) not in {"interaction_id", "rowid", "db_path", "seed", "timestamp"}
        }
    if isinstance(value, list):
        return [_remove_volatile_fields(item) for item in value]
    return value


def _looks_local_integer_id(value: str) -> bool:
    stripped = value.strip().lower()
    if stripped.isdigit():
        return True
    if stripped.startswith("-") and stripped[1:].isdigit():
        return True
    return False


def _relaxed_family_signature(signature: str) -> str:
    try:
        parsed = json.loads(signature)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        relaxed = [_bucket_numeric_value_abs(item) if isinstance(item, (int, float)) else item for item in parsed]
        return json.dumps(relaxed, separators=(",", ":"))
    if isinstance(parsed, dict):
        relaxed = {
            key: _bucket_numeric_value_abs(value) if isinstance(value, (int, float)) else value
            for key, value in parsed.items()
        }
        return json.dumps(relaxed, sort_keys=True, separators=(",", ":"))
    return _relax_text_signature(signature)


def _bucket_numeric_value(value: float | int) -> int:
    number = float(value)
    if number == 0:
        return 0
    magnitude = abs(number)
    if magnitude <= 1:
        bucket = 1
    elif magnitude <= 4:
        bucket = 4
    elif magnitude <= 16:
        bucket = 16
    elif magnitude <= 64:
        bucket = 64
    else:
        bucket = 128
    return int(bucket if number > 0 else -bucket)


def _bucket_numeric_value_abs(value: float | int) -> int:
    return abs(_bucket_numeric_value(value))


def _relax_text_signature(signature: str) -> str:
    lowered = signature.lower().strip()
    pieces = []
    current_digits = []
    for char in lowered:
        if char.isdigit() or char in ".-":
            current_digits.append(char)
            continue
        if current_digits:
            pieces.append(str(_bucket_numeric_value_abs(_safe_float("".join(current_digits)))))
            current_digits = []
        pieces.append(char)
    if current_digits:
        pieces.append(str(_bucket_numeric_value_abs(_safe_float("".join(current_digits)))))
    return "".join(pieces)


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _infer_effect_type(signature: str) -> str:
    lowered = signature.lower()
    if lowered in {"[0,0,0,0,0]", "[0.0,0.0,0.0,0.0,0.0]", "preserve", "no_change"}:
        return "preserve_like"
    try:
        parsed = json.loads(signature)
    except Exception:
        parsed = None
    if isinstance(parsed, list) and parsed:
        non_zero = [float(item) for item in parsed if isinstance(item, (int, float)) and float(item) != 0.0]
        if not non_zero:
            return "preserve_like"
        positives = sum(1 for item in non_zero if item > 0)
        negatives = sum(1 for item in non_zero if item < 0)
        if positives and negatives:
            return "mixed_change"
        if positives:
            return "positive_change"
        if negatives:
            return "negative_change"
    if "expand" in lowered:
        return "expand_like"
    if "restrict" in lowered or "collapse" in lowered:
        return "restrict_like"
    return "signature_like"


def _effect_polarity(effect_type: str) -> str:
    if effect_type in {"positive_change", "expand_like"}:
        return "positive"
    if effect_type in {"negative_change", "restrict_like"}:
        return "negative"
    if effect_type == "mixed_change":
        return "mixed"
    return effect_type


def _action_group(action_value: str) -> str:
    try:
        action = int(action_value)
    except (TypeError, ValueError):
        return "unknown"
    if action in {1, 2, 3, 4}:
        return "movement"
    if action in {5, 6}:
        return "interaction"
    if action in {0, 7, 8, 9}:
        return f"reset_terminal_{action}"
    return f"other_{action}"


def _member_key(
    row: dict[str, Any],
    *,
    sqlite_path: Path,
    contingency_column: str | None,
    context_column: str | None,
    action_column: str | None,
    semantic_column: str | None,
) -> str:
    context_value = _normalize_scalar(row.get(context_column)) if context_column is not None else None
    action_value = _normalize_scalar(row.get(action_column)) if action_column is not None else None
    semantic_value = _normalized_signature_value(row.get(semantic_column)) if semantic_column is not None else None
    if context_value is not None and semantic_value is not None:
        return f"{context_value}|{action_value or ''}|{semantic_value}"
    if context_value is not None:
        return f"{context_value}|{action_value or ''}"
    local_id = _normalize_scalar(row.get(contingency_column)) if contingency_column is not None else None
    if local_id is None:
        local_id = _normalize_json_like(row)
    return f"{sqlite_path}:{local_id}"


def _context_key(
    row: dict[str, Any],
    *,
    context_column: str | None,
    action_column: str | None,
) -> str | None:
    context_value = _normalize_scalar(row.get(context_column)) if context_column is not None else None
    action_value = _normalize_scalar(row.get(action_column)) if action_column is not None else None
    if context_value is None and action_value is None:
        return None
    return f"{context_value or ''}|{action_value or ''}"


def _normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _stable_value(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return None


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


def _median_or_none(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(median(items))


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
        evidence_for.append("No object-carrier emergence is detected in the aggregate report.")
    elif result.get("pre_object_condition_satisfied") is False:
        evidence_against.append("Object-carrier emergence is already present in the aggregate report.")
    if _gt(result.get("families_merged_across_shards"), 0) is True:
        evidence_for.append(
            f"Global family merging reduced shard-local family inflation ({int(result['global_family_count_before_merge'])} -> {int(result['global_family_count_after_merge'])})."
        )
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
        f"- DB scan: total={_fmt_value(result.get('db_paths_total'))} inspected={_fmt_value(result.get('db_paths_inspected'))} truncated={_fmt_value(result.get('db_scan_truncated'))}",
        f"- rows: available={_fmt_value(result.get('row_count_available'))} used={_fmt_value(result.get('row_count_used'))} max_rows_applied={_fmt_value(result.get('max_rows_applied'))}",
        f"- tables used: {', '.join(result.get('candidate_tables_used', [])) or 'none'}",
        f"- global merge: enabled={_fmt_value(result.get('global_family_merge_enabled'))} before={_fmt_value(result.get('global_family_count_before_merge'))} after={_fmt_value(result.get('global_family_count_after_merge'))}",
        f"- merged across shards/games/samplers: {_fmt_value(result.get('families_merged_across_shards'))} / {_fmt_value(result.get('families_merged_across_games'))} / {_fmt_value(result.get('families_merged_across_samplers'))}",
        f"- contingency count: {_fmt_value(result.get('discovered_contingency_count'))}",
        f"- family count: {_fmt_value(result.get('transformation_family_count'))}",
        f"- stable family count: {_fmt_value(result.get('stable_transformation_family_count'))}",
        f"- total family members: {_fmt_value(result.get('family_member_count_total'))}",
        f"- mean/median/max family size: {_fmt_value(result.get('family_mean_member_count'))} / {_fmt_value(result.get('family_median_member_count'))} / {_fmt_value(result.get('family_max_member_count'))}",
        f"- singleton family ratio: {_fmt_value(result.get('singleton_family_ratio'))}",
        f"- singleton ratio strict/relaxed: {_fmt_value(result.get('singleton_ratio_strict'))} / {_fmt_value(result.get('singleton_ratio_relaxed'))}",
        f"- relaxed family count / relaxed singleton count: {_fmt_value(result.get('transformation_family_count_relaxed'))} / {_fmt_value(result.get('singleton_family_count_relaxed'))}",
        f"- relaxed merge safety passed: {_fmt_value(result.get('merge_safety_passed'))}",
        f"- unsafe relaxed merge count: {_fmt_value(result.get('unsafe_relaxed_merge_count'))}",
        f"- relaxed decision candidate: {_fmt_value(result.get('relaxed_decision_candidate'))}",
        f"- compression ratio: {_fmt_value(result.get('compression_ratio'))}",
        f"- compression gain: {_fmt_value(result.get('compression_gain'))}",
        f"- family prediction lift: {_fmt_value(result.get('family_prediction_lift_mean'))}",
        f"- cross-context/game/sampler families: {_fmt_value(result.get('family_cross_context_count'))} / {_fmt_value(result.get('family_cross_game_count'))} / {_fmt_value(result.get('family_cross_sampler_count'))}",
        f"- top global families: {json.dumps(result.get('top_global_families', []), sort_keys=True)}",
        f"- top singleton family signatures: {json.dumps(result.get('top_singleton_family_signatures', []), sort_keys=True)}",
        f"- singleton families by game: {json.dumps(result.get('singleton_families_by_game', {}), sort_keys=True)}",
        f"- singleton families by sampler: {json.dumps(result.get('singleton_families_by_sampler', {}), sort_keys=True)}",
        f"- singleton families by action: {json.dumps(result.get('singleton_families_by_action', {}), sort_keys=True)}",
        f"- singleton families by effect type: {json.dumps(result.get('singleton_families_by_effect_type', {}), sort_keys=True)}",
        f"- singleton diagnostics: {json.dumps(result.get('singleton_family_diagnostics', {}), sort_keys=True)}",
        f"- relaxed canonicalization diagnostics: {json.dumps(result.get('relaxed_canonicalization_diagnostics', {}), sort_keys=True)}",
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
        f"- DB scan: `total={_fmt_value(result.get('db_paths_total'))} inspected={_fmt_value(result.get('db_paths_inspected'))} truncated={_fmt_value(result.get('db_scan_truncated'))}`",
        f"- rows: `available={_fmt_value(result.get('row_count_available'))} used={_fmt_value(result.get('row_count_used'))} max_rows_applied={_fmt_value(result.get('max_rows_applied'))}`",
        f"- tables used: `{', '.join(result.get('candidate_tables_used', [])) or 'none'}`",
        f"- global merge: `enabled={_fmt_value(result.get('global_family_merge_enabled'))} before={_fmt_value(result.get('global_family_count_before_merge'))} after={_fmt_value(result.get('global_family_count_after_merge'))}`",
        f"- merged across shards/games/samplers: `{_fmt_value(result.get('families_merged_across_shards'))} / {_fmt_value(result.get('families_merged_across_games'))} / {_fmt_value(result.get('families_merged_across_samplers'))}`",
        f"- contingency count: `{_fmt_value(result.get('discovered_contingency_count'))}`",
        f"- family count: `{_fmt_value(result.get('transformation_family_count'))}`",
        f"- stable family count: `{_fmt_value(result.get('stable_transformation_family_count'))}`",
        f"- total family members: `{_fmt_value(result.get('family_member_count_total'))}`",
        f"- mean/median/max family size: `{_fmt_value(result.get('family_mean_member_count'))} / {_fmt_value(result.get('family_median_member_count'))} / {_fmt_value(result.get('family_max_member_count'))}`",
        f"- singleton family ratio: `{_fmt_value(result.get('singleton_family_ratio'))}`",
        f"- singleton ratio strict/relaxed: `{_fmt_value(result.get('singleton_ratio_strict'))} / {_fmt_value(result.get('singleton_ratio_relaxed'))}`",
        f"- relaxed family count / relaxed singleton count: `{_fmt_value(result.get('transformation_family_count_relaxed'))} / {_fmt_value(result.get('singleton_family_count_relaxed'))}`",
        f"- relaxed merge safety passed: `{_fmt_value(result.get('merge_safety_passed'))}`",
        f"- unsafe relaxed merge count: `{_fmt_value(result.get('unsafe_relaxed_merge_count'))}`",
        f"- relaxed decision candidate: `{_fmt_value(result.get('relaxed_decision_candidate'))}`",
        f"- compression ratio: `{_fmt_value(result.get('compression_ratio'))}`",
        f"- compression gain: `{_fmt_value(result.get('compression_gain'))}`",
        f"- family prediction lift: `{_fmt_value(result.get('family_prediction_lift_mean'))}`",
        f"- cross-context/game/sampler families: `{_fmt_value(result.get('family_cross_context_count'))} / {_fmt_value(result.get('family_cross_game_count'))} / {_fmt_value(result.get('family_cross_sampler_count'))}`",
        f"- top global families: `{json.dumps(result.get('top_global_families', []), sort_keys=True)}`",
        f"- top singleton family signatures: `{json.dumps(result.get('top_singleton_family_signatures', []), sort_keys=True)}`",
        f"- singleton families by game: `{json.dumps(result.get('singleton_families_by_game', {}), sort_keys=True)}`",
        f"- singleton families by sampler: `{json.dumps(result.get('singleton_families_by_sampler', {}), sort_keys=True)}`",
        f"- singleton families by action: `{json.dumps(result.get('singleton_families_by_action', {}), sort_keys=True)}`",
        f"- singleton families by effect type: `{json.dumps(result.get('singleton_families_by_effect_type', {}), sort_keys=True)}`",
        f"- singleton diagnostics: `{json.dumps(result.get('singleton_family_diagnostics', {}), sort_keys=True)}`",
        f"- relaxed canonicalization diagnostics: `{json.dumps(result.get('relaxed_canonicalization_diagnostics', {}), sort_keys=True)}`",
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
