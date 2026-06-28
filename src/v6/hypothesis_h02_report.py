from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from v6.memory.direct_streaming_fold import direct_streaming_manifest_exists

H02_JSON_NAME = "h02_prediction_violation_attention_report.json"
H02_TXT_NAME = "h02_prediction_violation_attention_report.txt"
H02_MD_NAME = "h02_prediction_violation_attention.md"
H02_READY_JSON_NAME = "h02_ready_runs.json"
H02_READY_TXT_NAME = "h02_ready_runs.txt"
INPUT_REPORT_NAME = "interaction_sampling_v05c_report.json"
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_DB_FILES = 20
DIRECT_LINKAGE_UNAVAILABLE_MESSAGE = (
    "Direct per-interaction prediction-error to replay-priority linkage unavailable in current DB schema. "
    "Existing run can only support PARTIALLY_VALID H02."
)
DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE = "Direct replay-lift unavailable within inspected SQLite shard limit."
H02_READY_REQUIRED_CARRIER_FIELDS = (
    "emergent_object_carrier_count",
    "emergent_context_action_fallback_count",
    "carrier_object_candidate_count",
    "carrier_context_action_fallback_candidate_count",
)
H02_READY_AGGREGATE_FIELDS = (
    "memory_record_count",
    "mean_isf_prediction_error",
    "context_contradiction_count",
    "repeated_contradiction_count",
    "context_expansion_suggested_count",
    "memory_replay_candidate_count",
    "high_priority_replay_count",
    "carrier_object_candidate_count",
    "emergent_object_carrier_count",
    "carrier_context_action_fallback_candidate_count",
    "emergent_context_action_fallback_count",
)

REPLAY_PRIORITY_COLUMNS = (
    "replay_priority",
    "memory_replay_priority",
    "priority",
    "replay_score",
    "memory_priority",
    "isf_total",
)
PREDICTION_NUMERIC_COLUMNS = (
    "prediction_error",
    "prediction_error_score",
    "isf_prediction_error",
    "pe",
    "surprise",
    "surprise_score",
)
PREDICTION_BOOLEAN_COLUMNS = (
    "contradicted_context",
    "context_contradiction",
    "contradiction",
    "prediction_violation",
    "is_prediction_violation",
)
PREDICTION_COMPARISON_PAIRS = (
    ("expected_delta", "observed_delta"),
    ("predicted_delta", "observed_delta"),
    ("predicted_outcome", "observed_outcome"),
)
HIGH_PRIORITY_FLAG_COLUMNS = (
    "high_priority_replay",
    "is_high_priority_replay",
    "high_priority",
    "replay_high_priority",
    "memory_high_priority",
    "memory_high_priority_replay",
)
DB_SAMPLER_PRIORITY = (
    "mixed",
    "low_confidence",
    "novelty_delta",
    "no_change_avoidance",
    "action_balance",
    "random_baseline",
)

H02_DEFAULTS: dict[str, Any] = {
    "hypothesis_id": "H02",
    "hypothesis_name": "Prediction violations drive attention and memory before object concepts",
    "decision": "INCONCLUSIVE",
    "h02a_replay_attention_decision": "INCONCLUSIVE",
    "h02b_pre_carrier_timing_decision": "INCONCLUSIVE",
    "h02a_replay_attention_conclusion": "",
    "h02b_pre_carrier_timing_conclusion": "",
    "h02_final_decision_basis": "",
    "scientific_conclusion": "",
    "source_run_dir": "",
    "input_report_found": False,
    "db_found": False,
    "db_path": None,
    "schema_inspected": False,
    "tables_seen": [],
    "candidate_tables_used": [],
    "prediction_violation_metric_source": None,
    "replay_priority_metric_source": None,
    "row_count_available": None,
    "row_count_used": 0,
    "prediction_violation_row_count": None,
    "non_prediction_violation_row_count": None,
    "prediction_violation_base_ratio": None,
    "mean_replay_priority_for_prediction_violating_interactions": None,
    "mean_replay_priority_for_non_prediction_violating_interactions": None,
    "prediction_violation_replay_lift": None,
    "high_priority_replay_threshold": None,
    "high_priority_replay_prediction_violation_ratio": None,
    "high_priority_replay_non_prediction_violation_ratio": None,
    "direct_replay_lift_available": False,
    "high_priority_threshold_method": None,
    "sqlite_db_count_total": 0,
    "sqlite_db_count_inspected": 0,
    "sqlite_db_inspection_truncated": False,
    "sqlite_db_skipped_count": 0,
    "total_jobs_expected": None,
    "jobs_represented_in_compact_or_manifest_evidence": None,
    "jobs_represented_in_raw_scan": None,
    "evidence_coverage_ratio": None,
    "inspected_db_paths": [],
    "selected_db_path": None,
    "mean_isf_total": None,
    "max_isf_total": None,
    "mean_isf_prediction_error": None,
    "mean_isf_learning_value": None,
    "mean_isf_transfer_potential": None,
    "mean_isf_explanatory_potential": None,
    "high_isf_interaction_count": None,
    "context_contradiction_count": None,
    "prediction_error_positive_count": None,
    "predicted_family_available_count": None,
    "actual_family_available_count": None,
    "wrong_prediction_count": None,
    "confident_wrong_prediction_count": None,
    "contradiction_event_count": None,
    "contradiction_suppressed_low_confidence_count": None,
    "contradiction_suppressed_missing_prediction_count": None,
    "contradicted_context_count": None,
    "contradicted_context_action_count": None,
    "repeated_contradiction_count": None,
    "context_expansion_suggested_count": None,
    "memory_record_count": None,
    "memory_replay_candidate_count": None,
    "memory_mean_replay_priority": None,
    "memory_max_replay_priority": None,
    "high_priority_replay_count": None,
    "carrier_candidate_count": None,
    "emergent_carrier_count": None,
    "emergent_object_carrier_count": None,
    "emergent_context_action_fallback_count": None,
    "prediction_violation_signal_active": False,
    "contradiction_signal_active": False,
    "context_expansion_pressure_active": False,
    "replay_priority_active": False,
    "object_carrier_absent_or_negligible": False,
    "carrier_timing_note": "",
    "raw_h02_evidence_incomplete": False,
    "compact_h02_fallback_used": False,
    "compact_counter_fallback_used": False,
    "direct_replay_lift_pass": False,
    "direct_replay_lift_invalid": False,
    "aggregate_invalid_core": False,
    "h02a_decision_source": "insufficient_evidence",
    "evidence_for": [],
    "evidence_against": [],
    "missing_evidence": [],
    "acceptance_checks": {
        "prediction_error_positive": None,
        "contradictions_present": None,
        "repeated_contradictions_present": None,
        "context_expansion_suggested": None,
        "replay_candidates_present": None,
        "high_priority_replay_present": None,
        "prediction_violation_replay_lift_gt_1_25": None,
        "high_priority_ratio_above_base_ratio": None,
        "object_carriers_absent": None,
        "context_action_fallback_absent": None,
    },
}

_REPORT_FIELD_NAMES = (
    "mean_isf_total",
    "max_isf_total",
    "mean_isf_prediction_error",
    "mean_isf_learning_value",
    "mean_isf_transfer_potential",
    "mean_isf_explanatory_potential",
    "high_isf_interaction_count",
    "context_contradiction_count",
    "prediction_error_positive_count",
    "predicted_family_available_count",
    "actual_family_available_count",
    "wrong_prediction_count",
    "confident_wrong_prediction_count",
    "contradiction_event_count",
    "contradiction_suppressed_low_confidence_count",
    "contradiction_suppressed_missing_prediction_count",
    "contradicted_context_count",
    "contradicted_context_action_count",
    "repeated_contradiction_count",
    "context_expansion_suggested_count",
    "memory_record_count",
    "memory_replay_candidate_count",
    "memory_mean_replay_priority",
    "memory_max_replay_priority",
    "high_priority_replay_count",
    "carrier_candidate_count",
    "emergent_carrier_count",
    "emergent_object_carrier_count",
    "emergent_context_action_fallback_count",
)

_SUM_FIELDS = {
    "high_isf_interaction_count",
    "context_contradiction_count",
    "prediction_error_positive_count",
    "predicted_family_available_count",
    "actual_family_available_count",
    "wrong_prediction_count",
    "confident_wrong_prediction_count",
    "contradiction_event_count",
    "contradiction_suppressed_low_confidence_count",
    "contradiction_suppressed_missing_prediction_count",
    "contradicted_context_count",
    "contradicted_context_action_count",
    "repeated_contradiction_count",
    "context_expansion_suggested_count",
    "memory_record_count",
    "memory_replay_candidate_count",
    "high_priority_replay_count",
    "carrier_candidate_count",
    "emergent_carrier_count",
    "emergent_object_carrier_count",
    "emergent_context_action_fallback_count",
}

_MAX_FIELDS = {"max_isf_total", "memory_max_replay_priority"}


def evaluate_h02_prediction_violation_attention(
    run_dir: Path,
    output_dir: Path,
    *,
    memory_dir: Path | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    prefer_db: str | None = None,
    scan_all_dbs: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / INPUT_REPORT_NAME
    input_report = _load_json(report_path)
    input_report_found = input_report is not None

    sqlite_paths = _find_sqlite_paths(run_dir)
    db_found = bool(sqlite_paths)

    result = dict(H02_DEFAULTS)
    result["source_run_dir"] = str(run_dir)
    result["input_report_found"] = input_report_found
    result["db_found"] = db_found
    result["sqlite_db_count_total"] = len(sqlite_paths)
    result["evidence_source"] = "raw_epoch_db"
    streamed_compact_only = bool(memory_dir is not None and direct_streaming_manifest_exists(memory_dir) and not sqlite_paths)
    if streamed_compact_only:
        result["evidence_source"] = "direct_streaming_manifest_and_compact_memory"
    if input_report_found:
        runs = [row for row in (input_report.get("runs") or []) if isinstance(row, dict)]
        result["total_jobs_expected"] = len(runs) or None
    if memory_dir is not None and direct_streaming_manifest_exists(memory_dir):
        from v6.memory.direct_streaming_fold import load_direct_streamed_job_metrics
        try:
            result["jobs_represented_in_compact_or_manifest_evidence"] = len(load_direct_streamed_job_metrics(memory_dir))
        except Exception:
            result["jobs_represented_in_compact_or_manifest_evidence"] = None

    report_metrics = _extract_report_metrics(input_report)
    db_metrics = {} if _report_has_all_fields(report_metrics) else _aggregate_db_metrics(sqlite_paths)
    direct_metrics = _compute_prediction_violation_replay_lift_from_existing_db(
        run_dir,
        max_rows=max_rows,
        max_db_files=max_db_files,
        prefer_db=prefer_db,
        scan_all_dbs=scan_all_dbs,
    )
    temporal_rows = _load_h02_temporal_rows(input_report, memory_dir)

    for field in _REPORT_FIELD_NAMES:
        result[field] = report_metrics.get(field)
        if result[field] is None:
            result[field] = db_metrics.get(field)
    compact_metrics: dict[str, Any] = {}
    raw_h02_incomplete = False
    if memory_dir is not None:
        compact_metrics = _extract_h02_compact_metrics(Path(memory_dir))
        raw_h02_incomplete = _raw_h02_evidence_is_empty_or_incomplete(
            input_report=input_report,
            sqlite_paths=sqlite_paths,
            report_metrics=report_metrics,
            db_metrics=db_metrics,
            direct_metrics=direct_metrics,
        )
        result["raw_h02_evidence_incomplete"] = raw_h02_incomplete
        if raw_h02_incomplete:
            for key, value in compact_metrics.items():
                if value is not None:
                    current_value = result.get(key)
                    if current_value in (None, 0, 0.0, False, ""):
                        result[key] = value
                        result["compact_counter_fallback_used"] = True
            result["compact_h02_fallback_used"] = any(value is not None for value in compact_metrics.values())
            result["evidence_source"] = (
                "direct_streaming_manifest_and_compact_memory"
                if streamed_compact_only
                else ("compact_memory" if not input_report_found else "mixed_compact_memory_fallback")
            )
        else:
            for key, value in compact_metrics.items():
                if value is None:
                    continue
                if key in {
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
                    "contradiction_event_count",
                }:
                    current_value = result.get(key)
                    if current_value in (None, 0, 0.0, False, ""):
                        result[key] = value
                        result["compact_counter_fallback_used"] = True
                elif result.get(key) is None:
                    result[key] = value

    result.update(direct_metrics)
    result["jobs_represented_in_raw_scan"] = int(result.get("sqlite_db_count_inspected") or 0)
    jobs_expected = int(result.get("total_jobs_expected") or 0)
    represented = max(
        int(result.get("jobs_represented_in_compact_or_manifest_evidence") or 0),
        int(result.get("jobs_represented_in_raw_scan") or 0),
    )
    result["evidence_coverage_ratio"] = (represented / jobs_expected) if jobs_expected > 0 else None

    if not input_report_found and memory_dir is not None:
        result["h02a_replay_attention_decision"] = "PARTIALLY_VALID" if _gt(result.get("memory_replay_candidate_count"), 0) else "INCONCLUSIVE"
        result["h02b_pre_carrier_timing_decision"] = "INCONCLUSIVE"
        result["h02a_replay_attention_conclusion"] = (
            "H02 evaluated from compact memory after raw cleanup."
            if _gt(result.get("memory_replay_candidate_count"), 0)
            else "H02 remains inconclusive because compact memory lacks replay evidence."
        )
        result["h02b_pre_carrier_timing_conclusion"] = "Temporal pre-carrier timing cannot be evaluated from the available evidence."
        result["h02_final_decision_basis"] = "Final H02 decision follows H02A replay/attention evidence for compatibility."
        result["decision"] = result["h02a_replay_attention_decision"]
        result["scientific_conclusion"] = result["h02a_replay_attention_conclusion"]
        _finalize_h02_result(result, output_dir)
        return result
    if streamed_compact_only and result.get("direct_replay_lift_available") is not True:
        message = "raw per-interaction replay linkage unavailable after direct streaming raw cleanup"
        if message not in result["missing_evidence"]:
            result["missing_evidence"].append(message)
    if not input_report_found:
        result["missing_evidence"].append(f"Required input report missing: {INPUT_REPORT_NAME}")
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = "H02 cannot be evaluated because the required interaction-sampling report is missing."
        _finalize_h02_result(result, output_dir)
        return result

    if not any(result.get(field) is not None for field in _REPORT_FIELD_NAMES):
        result["missing_evidence"].append("No usable aggregate report fields are available in the current run artifacts.")
        result["decision"] = "INCONCLUSIVE"
        result["scientific_conclusion"] = (
            "H02 remains inconclusive because the current report and DB artifacts do not expose enough aggregate evidence."
        )
        _finalize_h02_result(result, output_dir)
        return result
    low_coverage = result.get("evidence_coverage_ratio") is not None and float(result["evidence_coverage_ratio"]) < 0.50

    checks = {
        "prediction_error_positive": _gt(result.get("mean_isf_prediction_error"), 0.0),
        "contradictions_present": _gt(result.get("context_contradiction_count"), 0),
        "repeated_contradictions_present": _gt(result.get("repeated_contradiction_count"), 0),
        "context_expansion_suggested": _gt(result.get("context_expansion_suggested_count"), 0),
        "replay_candidates_present": _gt(result.get("memory_replay_candidate_count"), 0),
        "high_priority_replay_present": _gt(result.get("high_priority_replay_count"), 0),
        "prediction_violation_replay_lift_gt_1_25": _gt(result.get("prediction_violation_replay_lift"), 1.25),
        "high_priority_ratio_above_base_ratio": _compare_gt(
            result.get("high_priority_replay_prediction_violation_ratio"),
            result.get("prediction_violation_base_ratio"),
        ),
        "object_carriers_absent": _eq(result.get("emergent_object_carrier_count"), 0),
        "context_action_fallback_absent": _eq(result.get("emergent_context_action_fallback_count"), 0),
    }
    result["acceptance_checks"] = checks

    result["prediction_violation_signal_active"] = checks["prediction_error_positive"] is True
    result["contradiction_signal_active"] = checks["contradictions_present"] is True
    result["context_expansion_pressure_active"] = checks["context_expansion_suggested"] is True
    result["replay_priority_active"] = (
        checks["replay_candidates_present"] is True and checks["high_priority_replay_present"] is True
    )
    result["object_carrier_absent_or_negligible"] = (
        checks["object_carriers_absent"] is True and checks["context_action_fallback_absent"] is True
    )

    aggregate_signals_pass = all(
        (
            checks["prediction_error_positive"] is True,
            checks["contradictions_present"] is True,
            checks["repeated_contradictions_present"] is True,
            checks["context_expansion_suggested"] is True,
            checks["replay_candidates_present"] is True,
        )
    )
    direct_replay_lift_pass = all(
        (
            result.get("direct_replay_lift_available") is True,
            checks["prediction_violation_replay_lift_gt_1_25"] is True,
            checks["high_priority_ratio_above_base_ratio"] is True,
        )
    )
    invalid_core = any(
        (
            checks["prediction_error_positive"] is not True,
            checks["contradictions_present"] is not True,
            checks["replay_candidates_present"] is not True,
        )
    )
    direct_replay_lift_invalid = (
        aggregate_signals_pass
        and result.get("direct_replay_lift_available") is True
        and _gt(result.get("prediction_violation_replay_lift"), 1.0) is False
    )
    result["direct_replay_lift_pass"] = bool(direct_replay_lift_pass)
    result["direct_replay_lift_invalid"] = bool(direct_replay_lift_invalid)
    result["aggregate_invalid_core"] = bool(invalid_core)

    compact_has_prediction_error = _gt(result.get("mean_isf_prediction_error"), 0.0) is True
    compact_has_contradiction = _gt(result.get("context_contradiction_count"), 0) is True
    compact_has_replay = _gt(result.get("memory_replay_candidate_count"), 0) is True

    _decide_h02a_from_checks(
        result=result,
        checks=checks,
        raw_h02_incomplete=raw_h02_incomplete,
        compact_has_prediction_error=compact_has_prediction_error,
        compact_has_contradiction=compact_has_contradiction,
        compact_has_replay=compact_has_replay,
        aggregate_signals_pass=aggregate_signals_pass,
        direct_replay_lift_pass=direct_replay_lift_pass,
        direct_replay_lift_invalid=direct_replay_lift_invalid,
        invalid_core=invalid_core,
    )
    if low_coverage:
        if result["h02a_replay_attention_decision"] == "VALID":
            result["h02a_replay_attention_decision"] = "PARTIALLY_VALID_WITH_LOW_COVERAGE"
        elif result["h02a_replay_attention_decision"] in {"PARTIALLY_VALID", "INVALID"}:
            result["h02a_replay_attention_decision"] = "PARTIALLY_VALID_WITH_LOW_COVERAGE"
        else:
            result["h02a_replay_attention_decision"] = "INSUFFICIENT_EVIDENCE"
        _append_unique(
            result.setdefault("missing_evidence", []),
            f"H02 evidence coverage is low ({float(result['evidence_coverage_ratio']):.3f}); compact/manifest evidence does not yet represent enough jobs.",
        )

    timing = _evaluate_h02b_pre_carrier_timing(result, temporal_rows)
    result["h02b_pre_carrier_timing_decision"] = timing["decision"]
    result["h02b_pre_carrier_timing_conclusion"] = timing["conclusion"]
    result["carrier_timing_note"] = timing["note"]
    result["decision"] = result["h02a_replay_attention_decision"]
    result["h02_final_decision_basis"] = "Final H02 decision follows H02A replay/attention evidence; H02B is reported as a separate timing qualifier."
    result["scientific_conclusion"] = f"{result['h02a_replay_attention_conclusion']} {result['h02b_pre_carrier_timing_conclusion']}".strip()

    _populate_evidence_lists(result)
    _finalize_h02_result(result, output_dir)
    return result


def _raw_h02_evidence_is_empty_or_incomplete(
    *,
    input_report: dict[str, Any] | None,
    sqlite_paths: list[Path],
    report_metrics: dict[str, Any],
    db_metrics: dict[str, Any],
    direct_metrics: dict[str, Any],
) -> bool:
    if input_report is None:
        return True
    runs = input_report.get("runs")
    if not isinstance(runs, list) or not runs:
        return True
    candidates = [
        report_metrics.get("mean_isf_prediction_error"),
        report_metrics.get("context_contradiction_count"),
        report_metrics.get("repeated_contradiction_count"),
        report_metrics.get("memory_replay_candidate_count"),
        report_metrics.get("high_priority_replay_count"),
        direct_metrics.get("prediction_violation_replay_lift"),
        db_metrics.get("mean_isf_prediction_error"),
        db_metrics.get("context_contradiction_count"),
        db_metrics.get("repeated_contradiction_count"),
        db_metrics.get("memory_replay_candidate_count"),
        db_metrics.get("high_priority_replay_count"),
    ]
    positive_present = any(value not in (None, 0, 0.0, False, "") for value in candidates)
    if not positive_present:
        return True
    if direct_metrics.get("direct_replay_lift_available") is not True:
        aggregate_fields = (
            report_metrics.get("mean_isf_prediction_error"),
            report_metrics.get("context_contradiction_count"),
            report_metrics.get("repeated_contradiction_count"),
            report_metrics.get("memory_replay_candidate_count"),
            report_metrics.get("high_priority_replay_count"),
            db_metrics.get("mean_isf_prediction_error"),
            db_metrics.get("context_contradiction_count"),
            db_metrics.get("repeated_contradiction_count"),
            db_metrics.get("memory_replay_candidate_count"),
            db_metrics.get("high_priority_replay_count"),
        )
        if all(value in (None, 0, 0.0, False, "") for value in aggregate_fields):
            return True
    return False


def _decide_h02a_from_checks(
    *,
    result: dict[str, Any],
    checks: dict[str, Any],
    raw_h02_incomplete: bool,
    compact_has_prediction_error: bool,
    compact_has_contradiction: bool,
    compact_has_replay: bool,
    aggregate_signals_pass: bool,
    direct_replay_lift_pass: bool,
    direct_replay_lift_invalid: bool,
    invalid_core: bool,
) -> dict[str, Any]:
    result["direct_replay_lift_pass"] = bool(direct_replay_lift_pass)
    result["direct_replay_lift_invalid"] = bool(direct_replay_lift_invalid)
    result["aggregate_invalid_core"] = bool(invalid_core)
    if raw_h02_incomplete and not compact_has_prediction_error and not compact_has_contradiction and not compact_has_replay:
        result["h02a_replay_attention_decision"] = "INCONCLUSIVE"
        result["h02a_replay_attention_conclusion"] = (
            "H02A remains inconclusive because raw aggregate evidence is incomplete and compact memory does not provide enough replay/contradiction evidence."
        )
        result["h02a_decision_source"] = "insufficient_evidence"
        _append_unique(
            result.setdefault("missing_evidence", []),
            "H02 raw aggregate evidence incomplete and compact replay/contradiction evidence insufficient.",
        )
    elif raw_h02_incomplete and compact_has_replay and not compact_has_prediction_error:
        result["h02a_replay_attention_decision"] = "PARTIALLY_VALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02A is partially supported from compact replay evidence, but direct prediction-violation linkage is unavailable."
        )
        result["h02a_decision_source"] = "compact_fallback"
        _append_unique(result.setdefault("missing_evidence", []), "Direct prediction-violation to replay-priority linkage unavailable.")
    elif direct_replay_lift_pass:
        result["h02a_replay_attention_decision"] = "VALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02A is supported in this run. Direct replay-lift evidence shows prediction-violating interactions receive substantially higher replay priority than non-violating interactions."
        )
        result["h02a_decision_source"] = "direct_replay_lift"
        if invalid_core:
            _append_unique(
                result.setdefault("missing_evidence", []),
                "Aggregate replay/contradiction counters are unavailable or zero despite direct replay-lift evidence.",
            )
    elif direct_replay_lift_invalid:
        result["h02a_replay_attention_decision"] = "INVALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02 is not supported in this run because direct replay-lift evidence is available but does not show higher replay priority for prediction-violating interactions."
        )
        result["h02a_decision_source"] = "direct_replay_lift_invalid"
    elif aggregate_signals_pass and checks["replay_candidates_present"] is True and checks["high_priority_replay_present"] is True:
        result["h02a_replay_attention_decision"] = "PARTIALLY_VALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02A is partially supported in this run. Aggregate prediction-violation, contradiction, and replay-candidate signals are present, but direct replay-lift evidence is unavailable or not strong enough for full validation."
        )
        result["h02a_decision_source"] = "aggregate_signals"
    elif invalid_core and result.get("direct_replay_lift_available") is not True:
        result["h02a_replay_attention_decision"] = "INVALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02 is not supported in this run because prediction-violation signals or replayable memory pressure are absent."
        )
        result["h02a_decision_source"] = "aggregate_invalid_core"
    elif invalid_core and result.get("direct_replay_lift_available") is True:
        result["h02a_replay_attention_decision"] = "PARTIALLY_VALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02A has positive direct replay-lift evidence, but aggregate contradiction/replay counters are incomplete or inconsistent."
        )
        result["h02a_decision_source"] = "aggregate_signals"
        _append_unique(
            result.setdefault("missing_evidence", []),
            "Aggregate replay/contradiction counters are unavailable or zero despite direct replay-lift evidence.",
        )
    elif not aggregate_signals_pass and result.get("direct_replay_lift_available") is not True:
        result["h02a_replay_attention_decision"] = "INCONCLUSIVE"
        result["h02a_replay_attention_conclusion"] = (
            "H02A remains inconclusive because both the aggregate signals and the direct replay-lift evidence are unavailable."
        )
        result["h02a_decision_source"] = "insufficient_evidence"
    else:
        result["h02a_replay_attention_decision"] = "PARTIALLY_VALID"
        result["h02a_replay_attention_conclusion"] = (
            "H02A has some replay/attention evidence, but the aggregate contradiction or replay-priority signals do not fully satisfy the stronger validation conditions."
        )
        result["h02a_decision_source"] = "aggregate_signals"
    return result


def _extract_h02_compact_metrics(memory_dir: Path) -> dict[str, Any]:
    current_state = memory_dir / "current_state.sqlite"
    replay_db = memory_dir / "replay_queue.sqlite"
    if not current_state.exists():
        return {}
    with sqlite3.connect(current_state) as state_conn:
        replay_rows = []
        if replay_db.exists():
            with sqlite3.connect(replay_db) as replay_conn:
                replay_rows = replay_conn.execute("SELECT priority_score FROM replay_queue").fetchall()
        priorities = [float(row[0] or 0.0) for row in replay_rows]
        contradiction_count = int(state_conn.execute("SELECT COUNT(*) FROM contradiction_clusters").fetchone()[0])
        memory_record_count = int(state_conn.execute("SELECT COUNT(*) FROM memory_nodes WHERE node_type = 'InteractionMemory'").fetchone()[0])
        score_record_count = int(state_conn.execute("SELECT COUNT(*) FROM memory_scores WHERE node_id LIKE 'M0:interaction:%'").fetchone()[0])
        selected_for_replay_edge_count = int(state_conn.execute("SELECT COUNT(*) FROM memory_edges WHERE edge_type = 'selected_for_replay'").fetchone()[0])
        score_stats = state_conn.execute(
            """
            SELECT
                COUNT(*) AS score_record_count_with_priority,
                SUM(CASE WHEN replay_priority >= 0.8 THEN 1 ELSE 0 END) AS high_priority_score_count,
                AVG(replay_priority),
                MAX(replay_priority)
            FROM memory_scores
            WHERE node_id LIKE 'M0:interaction:%'
              AND replay_priority IS NOT NULL
            """
        ).fetchone()
        high_priority = [value for value in priorities if value >= 0.8]
        return {
            "memory_record_count": memory_record_count or score_record_count or None,
            "memory_replay_candidate_count": len(replay_rows) if replay_rows else selected_for_replay_edge_count,
            "memory_mean_replay_priority": (sum(priorities) / len(priorities)) if priorities else (score_stats[2] if score_stats else None),
            "memory_max_replay_priority": max(priorities, default=0.0) if priorities else (score_stats[3] if score_stats else None),
            "high_priority_replay_count": len(high_priority) if replay_rows else int((score_stats[1] if score_stats else 0) or 0),
            "context_contradiction_count": contradiction_count,
            "repeated_contradiction_count": contradiction_count,
            "contradiction_event_count": contradiction_count,
            "prediction_error_positive_count": contradiction_count if contradiction_count > 0 else None,
            "predicted_family_available_count": (score_stats[0] if score_stats and score_stats[0] else score_record_count) or None,
            "actual_family_available_count": (score_stats[0] if score_stats and score_stats[0] else score_record_count) or None,
            "wrong_prediction_count": contradiction_count if contradiction_count > 0 else None,
            "confident_wrong_prediction_count": contradiction_count if contradiction_count > 0 else None,
            "prediction_violation_base_ratio": (contradiction_count / len(replay_rows)) if replay_rows else None,
            "high_priority_replay_prediction_violation_ratio": (len(high_priority) / len(replay_rows)) if replay_rows else None,
            "high_priority_replay_non_prediction_violation_ratio": None,
            "prediction_violation_replay_lift": None,
            "direct_replay_lift_available": False,
        }


def _load_h02_temporal_rows(input_report: dict[str, Any] | None, memory_dir: Path | None) -> list[dict[str, Any]]:
    rows = list(((input_report or {}).get("temporal_milestones") or {}).get("by_game_sampler_seed", []) or [])
    if rows:
        return [dict(item) for item in rows if isinstance(item, dict)]
    if memory_dir is None:
        return []
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return []
    try:
        with sqlite3.connect(current_state) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT game, sampler, seed, first_prediction_violation_step, first_high_replay_priority_step,
                           first_emergent_carrier_step
                    FROM temporal_milestones
                    """
                ).fetchall()
            ]
    except sqlite3.DatabaseError:
        return []


def _evaluate_h02b_pre_carrier_timing(result: dict[str, Any], temporal_rows: list[dict[str, Any]]) -> dict[str, str]:
    if temporal_rows:
        saw_explicit_invalid = False
        saw_explicit_valid = False
        for row in temporal_rows:
            replay_steps = [
                int(value)
                for value in (row.get("first_prediction_violation_step"), row.get("first_high_replay_priority_step"))
                if value is not None
            ]
            carrier_step = row.get("first_emergent_carrier_step")
            if carrier_step is None or not replay_steps:
                continue
            replay_step = min(replay_steps)
            carrier_step_int = int(carrier_step)
            if carrier_step_int < replay_step:
                saw_explicit_invalid = True
            elif carrier_step_int > replay_step:
                saw_explicit_valid = True
        if saw_explicit_invalid:
            return {
                "decision": "INVALID",
                "conclusion": "H02B is not supported because explicit temporal evidence shows carrier emergence preceded replay/attention evidence.",
                "note": "Temporal milestones show emergent carriers before the replay/attention signal.",
            }
        if saw_explicit_valid:
            return {
                "decision": "VALID",
                "conclusion": "H02B is supported because replay/attention evidence appears before emergent carriers in the available temporal milestones.",
                "note": "Temporal milestones show replay/attention before carrier emergence.",
            }
    if result.get("emergent_object_carrier_count") is None or result.get("emergent_context_action_fallback_count") is None:
        return {
            "decision": "INCONCLUSIVE",
            "conclusion": "H02B remains inconclusive because carrier timing fields are unavailable.",
            "note": "Carrier timing evidence is unavailable.",
        }
    emergent_carriers = result.get("emergent_carrier_count")
    emergent_object_carriers = result.get("emergent_object_carrier_count")
    if emergent_carriers is None:
        return {
            "decision": "INCONCLUSIVE",
            "conclusion": "H02B remains inconclusive because carrier timing fields are unavailable.",
            "note": "Carrier timing evidence is unavailable.",
        }
    if _eq(emergent_carriers, 0) is True:
        return {
            "decision": "VALID",
            "conclusion": "H02B is supported at the current measurement point because no emergent carriers are present.",
            "note": "No emergent carriers are present at the evaluated snapshot.",
        }
    if _gt(emergent_carriers, 0) is True and _eq(emergent_object_carriers, 0) is True:
        return {
            "decision": "PARTIALLY_VALID",
            "conclusion": "H02B is partially supported because generic emergent carriers are present, but emergent object carriers are absent.",
            "note": "Generic emergent carriers are present, but object carriers are absent. This supports pre-object timing only, not pre-carrier timing.",
        }
    if _gt(emergent_object_carriers, 0) is True:
        return {
            "decision": "INCONCLUSIVE",
            "conclusion": "H02B remains inconclusive because emergent object carriers are present and no temporal ordering evidence shows replay/attention preceded carrier emergence.",
            "note": "Emergent object carriers are present, but temporal ordering evidence is unavailable.",
        }
    return {
        "decision": "INCONCLUSIVE",
        "conclusion": "H02B remains inconclusive because carriers are present in the final memory snapshot, but no temporal evidence shows they emerged before replay/attention signals.",
        "note": "Carriers present in final memory snapshot; no temporal evidence that carriers preceded replay signal.",
    }


def compute_prediction_violation_replay_lift_from_existing_db(
    run_dir: Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    prefer_db: str | None = None,
    scan_all_dbs: bool = False,
) -> dict:
    return _compute_prediction_violation_replay_lift_from_existing_db(
        Path(run_dir),
        max_rows=max_rows,
        max_db_files=max_db_files,
        prefer_db=prefer_db,
        scan_all_dbs=scan_all_dbs,
    )


def find_h02_ready_runs(runs_root: Path, output_dir: Path) -> dict:
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for report_path in sorted(runs_root.rglob(INPUT_REPORT_NAME)):
        run_dir = report_path.parent
        payload = _load_json(report_path)
        source = _report_source(payload)
        sqlite_paths = _rank_sqlite_paths(run_dir, _find_sqlite_paths(run_dir), prefer_db=None)
        missing_required = [
            field
            for field in H02_READY_REQUIRED_CARRIER_FIELDS
            if source.get(field) is None
        ]
        has_required = not missing_required
        entry = {
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "report_mtime": _isoformat_mtime(report_path),
            "has_required_carrier_fields": has_required,
            "missing_required_carrier_fields": missing_required,
            "has_sqlite_db": bool(sqlite_paths),
            "sqlite_db_count": len(sqlite_paths),
            "sqlite_db_paths": [str(path) for path in sqlite_paths],
            "memory_record_count": _maybe_int(source.get("memory_record_count")),
            "mean_isf_prediction_error": _maybe_float(source.get("mean_isf_prediction_error")),
            "context_contradiction_count": _maybe_int(source.get("context_contradiction_count")),
            "repeated_contradiction_count": _maybe_int(source.get("repeated_contradiction_count")),
            "context_expansion_suggested_count": _maybe_int(source.get("context_expansion_suggested_count")),
            "memory_replay_candidate_count": _maybe_int(source.get("memory_replay_candidate_count")),
            "high_priority_replay_count": _maybe_int(source.get("high_priority_replay_count")),
            "carrier_object_candidate_count": _maybe_int(source.get("carrier_object_candidate_count")),
            "emergent_object_carrier_count": _maybe_int(source.get("emergent_object_carrier_count")),
            "carrier_context_action_fallback_candidate_count": _maybe_int(
                source.get("carrier_context_action_fallback_candidate_count")
            ),
            "emergent_context_action_fallback_count": _maybe_int(source.get("emergent_context_action_fallback_count")),
            "h02_ready": False,
            "recommended_output_dir": str(output_dir),
        }
        entry["h02_ready"] = _h02_ready_entry(entry)
        runs.append(entry)

    runs.sort(key=_h02_ready_sort_key)
    recommended_entry = next((entry for entry in runs if entry["h02_ready"] is True), None)

    result = {
        "runs_root": str(runs_root),
        "candidate_count": len(runs),
        "ready_count": sum(1 for entry in runs if entry["h02_ready"] is True),
        "runs": runs,
        "run_best_parameters": None,
        "recommended_run": None if recommended_entry is None else {
            "run_dir": recommended_entry["run_dir"],
            "recommended_output_dir": recommended_entry["recommended_output_dir"],
        },
    }
    _write_h02_ready_inventory(result, output_dir)
    return result


def run_h02_on_best_ready_run(
    runs_root: Path,
    output_dir: Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_db_files: int = DEFAULT_MAX_DB_FILES,
    prefer_db: str | None = None,
    scan_all_dbs: bool = False,
) -> dict:
    result = find_h02_ready_runs(runs_root, output_dir)
    result["run_best_parameters"] = {
        "max_rows": int(max_rows),
        "max_db_files": int(max_db_files),
        "prefer_db": prefer_db,
        "scan_all_dbs": bool(scan_all_dbs),
    }
    _write_h02_ready_inventory(result, Path(output_dir))
    recommended = result.get("recommended_run")
    if recommended is None:
        return result

    evaluate_h02_prediction_violation_attention(
        run_dir=Path(recommended["run_dir"]),
        output_dir=Path(recommended["recommended_output_dir"]),
        max_rows=max_rows,
        max_db_files=max_db_files,
        prefer_db=prefer_db,
        scan_all_dbs=scan_all_dbs,
    )
    return result


def _compute_prediction_violation_replay_lift_from_existing_db(
    run_dir: Path,
    *,
    max_rows: int,
    max_db_files: int,
    prefer_db: str | None,
    scan_all_dbs: bool,
) -> dict:
    sqlite_paths = _find_sqlite_paths(run_dir)
    ranked_sqlite_paths = _rank_sqlite_paths(run_dir, sqlite_paths, prefer_db=prefer_db)
    if int(max_db_files) <= 0:
        limited_sqlite_paths = list(ranked_sqlite_paths)
    else:
        limited_sqlite_paths = ranked_sqlite_paths[: max(1, int(max_db_files))]
    base_result = {
        "db_found": bool(sqlite_paths),
        "db_path": None,
        "selected_db_path": None,
        "schema_inspected": False,
        "tables_seen": [],
        "candidate_tables_used": [],
        "prediction_violation_metric_source": None,
        "replay_priority_metric_source": None,
        "row_count_available": None,
        "row_count_used": 0,
        "prediction_violation_row_count": None,
        "non_prediction_violation_row_count": None,
        "prediction_violation_base_ratio": None,
        "mean_replay_priority_for_prediction_violating_interactions": None,
        "mean_replay_priority_for_non_prediction_violating_interactions": None,
        "prediction_violation_replay_lift": None,
        "high_priority_replay_threshold": None,
        "high_priority_threshold_method": None,
        "high_priority_replay_prediction_violation_ratio": None,
        "high_priority_replay_non_prediction_violation_ratio": None,
        "direct_replay_lift_available": False,
        "sqlite_db_count_total": len(sqlite_paths),
        "sqlite_db_count_inspected": 0,
        "sqlite_db_inspection_truncated": len(limited_sqlite_paths) < len(ranked_sqlite_paths),
        "sqlite_db_skipped_count": 0,
        "inspected_db_paths": [],
        "missing_evidence": [],
    }
    if not sqlite_paths:
        base_result["missing_evidence"].append(DIRECT_LINKAGE_UNAVAILABLE_MESSAGE)
        return base_result

    tables_seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for sqlite_path in limited_sqlite_paths:
        base_result["inspected_db_paths"].append(str(sqlite_path))
        base_result["sqlite_db_count_inspected"] += 1
        try:
            with sqlite3.connect(sqlite_path) as connection:
                table_map = _load_table_map(connection)
                base_result["schema_inspected"] = True
                tables_seen.update(table_map)
                db_candidates = _collect_db_candidates(connection, sqlite_path, table_map, max_rows=max_rows)
                if not db_candidates:
                    continue
                direct_candidates = [candidate for candidate in db_candidates if candidate["metrics"]["direct_replay_lift_available"] is True]
                if direct_candidates and not scan_all_dbs:
                    selected = max(direct_candidates, key=_direct_candidate_sort_key)
                    base_result.update(selected["metrics"])
                    base_result["tables_seen"] = sorted(tables_seen)
                    base_result["selected_db_path"] = selected["metrics"].get("db_path")
                    base_result["sqlite_db_skipped_count"] = max(
                        0, base_result["sqlite_db_count_total"] - base_result["sqlite_db_count_inspected"]
                    )
                    return base_result
                candidates.extend(db_candidates)
        except sqlite3.DatabaseError:
            continue

    base_result["tables_seen"] = sorted(tables_seen)
    base_result["sqlite_db_skipped_count"] = max(
        0, base_result["sqlite_db_count_total"] - base_result["sqlite_db_count_inspected"]
    )
    if not candidates:
        if base_result["sqlite_db_inspection_truncated"] is True:
            base_result["missing_evidence"].append(DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE)
        else:
            base_result["missing_evidence"].append(DIRECT_LINKAGE_UNAVAILABLE_MESSAGE)
        return base_result

    direct_candidates = [candidate for candidate in candidates if candidate["metrics"]["direct_replay_lift_available"] is True]
    best = (
        max(direct_candidates, key=_scan_all_direct_candidate_sort_key)
        if direct_candidates and scan_all_dbs
        else max(direct_candidates, key=_direct_candidate_sort_key)
        if direct_candidates
        else max(candidates, key=_candidate_sort_key)
    )
    base_result.update(best["metrics"])
    base_result["tables_seen"] = sorted(tables_seen)
    base_result["selected_db_path"] = best["metrics"].get("db_path")
    if base_result["direct_replay_lift_available"] is not True:
        if base_result["sqlite_db_inspection_truncated"] is True:
            base_result["missing_evidence"].append(DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE)
        else:
            base_result["missing_evidence"].append(DIRECT_LINKAGE_UNAVAILABLE_MESSAGE)
    return base_result


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _report_source(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    validation = payload.get("validation")
    if isinstance(validation, dict):
        return validation
    return payload


def _report_has_all_fields(report_metrics: dict[str, Any]) -> bool:
    return all(field in report_metrics and report_metrics[field] is not None for field in _REPORT_FIELD_NAMES)


def _find_sqlite_paths(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".sqlite", ".db", ".sqlite3"}
    ]


def _rank_sqlite_paths(run_dir: Path, sqlite_paths: list[Path], *, prefer_db: str | None) -> list[Path]:
    preferred = _resolve_preferred_db(run_dir, prefer_db)
    return sorted(sqlite_paths, key=lambda path: _sqlite_path_sort_key(run_dir, path, preferred))


def _resolve_preferred_db(run_dir: Path, prefer_db: str | None) -> Path | None:
    if not prefer_db:
        return None
    candidate = Path(prefer_db)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate.resolve()


def _sqlite_path_sort_key(run_dir: Path, path: Path, preferred: Path | None) -> tuple[int, int, int, int, str]:
    resolved = path.resolve()
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
        0 if preferred is not None and resolved == preferred else 1,
        sampler_rank,
        size_key,
        mtime_key,
        rel,
    )


def _sampler_rank(path: Path) -> int:
    lowered = str(path).lower()
    for index, name in enumerate(DB_SAMPLER_PRIORITY):
        if name in lowered:
            return index
    return len(DB_SAMPLER_PRIORITY)


def _extract_report_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    source = _report_source(payload)
    return {field: source.get(field) for field in _REPORT_FIELD_NAMES if field in source}


def _aggregate_db_metrics(sqlite_paths: list[Path]) -> dict[str, Any]:
    values: dict[str, list[Any]] = {field: [] for field in _REPORT_FIELD_NAMES}
    for sqlite_path in sqlite_paths:
        try:
            with sqlite3.connect(sqlite_path) as connection:
                tables = set(_load_table_map(connection))
                if "sampling_metadata" in tables:
                    rows = connection.execute("SELECT key, value FROM sampling_metadata").fetchall()
                    metadata = {str(key): json.loads(value) for key, value in rows}
                    for field in _REPORT_FIELD_NAMES:
                        if field in metadata and metadata[field] is not None:
                            values[field].append(metadata[field])
                _append_interaction_metrics(connection, values)
                _append_prediction_metrics(connection, values)
        except (sqlite3.DatabaseError, OSError, json.JSONDecodeError):
            continue
    aggregate: dict[str, Any] = {}
    for field, items in values.items():
        if not items:
            aggregate[field] = None
        elif field in _SUM_FIELDS:
            aggregate[field] = int(sum(int(item) for item in items))
        elif field in _MAX_FIELDS:
            aggregate[field] = max(float(item) for item in items)
        else:
            aggregate[field] = float(mean(float(item) for item in items))
    return aggregate


def _append_interaction_metrics(connection: sqlite3.Connection, values: dict[str, list[Any]]) -> None:
    columns = _table_columns(connection, "interactions")
    if not columns:
        return
    if {"isf_total", "isf_prediction_error", "isf_learning_value", "isf_transfer_potential", "isf_explanatory_potential"} & columns:
        row = connection.execute(
            """
            SELECT
                AVG(isf_total),
                MAX(isf_total),
                AVG(isf_prediction_error),
                AVG(isf_learning_value),
                AVG(isf_transfer_potential),
                AVG(isf_explanatory_potential),
                SUM(CASE WHEN COALESCE(isf_total, 0.0) >= 1.0 THEN 1 ELSE 0 END)
            FROM interactions
            """
        ).fetchone()
        for name, item in zip(
            (
                "mean_isf_total",
                "max_isf_total",
                "mean_isf_prediction_error",
                "mean_isf_learning_value",
                "mean_isf_transfer_potential",
                "mean_isf_explanatory_potential",
                "high_isf_interaction_count",
            ),
            row,
        ):
            if item is not None:
                values[name].append(item)
    if {"memory_replay_priority", "memory_replay_candidate"} <= columns:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN memory_replay_candidate = 1 THEN 1 ELSE 0 END),
                AVG(CASE WHEN memory_replay_candidate = 1 THEN memory_replay_priority END),
                MAX(memory_replay_priority),
                SUM(CASE WHEN COALESCE(memory_replay_priority, 0.0) >= 0.70 THEN 1 ELSE 0 END)
            FROM interactions
            """
        ).fetchone()
        for name, item in zip(
            (
                "memory_record_count",
                "memory_replay_candidate_count",
                "memory_mean_replay_priority",
                "memory_max_replay_priority",
                "high_priority_replay_count",
            ),
            row,
        ):
            if item is not None:
                values[name].append(item)


def _append_prediction_metrics(connection: sqlite3.Connection, values: dict[str, list[Any]]) -> None:
    columns = _table_columns(connection, "prediction_results")
    if not columns or "context_contradiction" not in columns:
        return
    if "context_contradiction_key" in columns:
        row = connection.execute(
            """
            WITH contradiction_groups AS (
                SELECT context_contradiction_key, COUNT(*) AS contradiction_count
                FROM prediction_results
                WHERE context_contradiction = 1
                GROUP BY context_contradiction_key
            )
            SELECT
                SUM(CASE WHEN context_contradiction = 1 THEN 1 ELSE 0 END),
                COUNT(DISTINCT CASE WHEN context_contradiction = 1 THEN COALESCE(context_contradiction_key, context_signature) END),
                COUNT(DISTINCT CASE WHEN context_contradiction = 1 THEN COALESCE(context_signature, '') || '|' || CAST(COALESCE(action, -1) AS TEXT) END),
                (SELECT COUNT(*) FROM contradiction_groups WHERE contradiction_count > 1),
                SUM(CASE WHEN context_expansion_suggested = 1 THEN 1 ELSE 0 END)
            FROM prediction_results
            """
        ).fetchone()
    else:
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN context_contradiction = 1 THEN 1 ELSE 0 END),
                COUNT(DISTINCT CASE WHEN context_contradiction = 1 THEN context_signature END),
                COUNT(DISTINCT CASE WHEN context_contradiction = 1 THEN COALESCE(context_signature, '') || '|' || CAST(COALESCE(action, -1) AS TEXT) END),
                NULL,
                SUM(CASE WHEN context_expansion_suggested = 1 THEN 1 ELSE 0 END)
            FROM prediction_results
            """
        ).fetchone()
    for name, item in zip(
        (
            "context_contradiction_count",
            "contradicted_context_count",
            "contradicted_context_action_count",
            "repeated_contradiction_count",
            "context_expansion_suggested_count",
        ),
        row,
    ):
        if item is not None:
            values[name].append(item)
    prediction_columns = set(columns)
    confidence_expr = "COALESCE(prediction_confidence, 0.0)" if "prediction_confidence" in prediction_columns else "0.0"
    family_row = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN COALESCE(isf_prediction_error, 0.0) > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN predicted_family IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN actual_family IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN predicted_family IS NOT NULL AND actual_family IS NOT NULL AND predicted_family != actual_family THEN 1 ELSE 0 END),
            SUM(CASE WHEN predicted_family IS NOT NULL AND actual_family IS NOT NULL AND predicted_family != actual_family AND {confidence_expr} >= 0.50 THEN 1 ELSE 0 END),
            SUM(CASE WHEN context_contradiction = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN predicted_family IS NOT NULL AND actual_family IS NOT NULL AND predicted_family != actual_family AND {confidence_expr} < 0.50 THEN 1 ELSE 0 END),
            SUM(CASE WHEN predicted_family IS NULL OR actual_family IS NULL THEN 1 ELSE 0 END)
        FROM prediction_results
        """
    ).fetchone()
    for name, item in zip(
        (
            "prediction_error_positive_count",
            "predicted_family_available_count",
            "actual_family_available_count",
            "wrong_prediction_count",
            "confident_wrong_prediction_count",
            "contradiction_event_count",
            "contradiction_suppressed_low_confidence_count",
            "contradiction_suppressed_missing_prediction_count",
        ),
        family_row,
    ):
        if item is not None:
            values[name].append(item)


def _collect_db_candidates(
    connection: sqlite3.Connection,
    sqlite_path: Path,
    table_map: dict[str, set[str]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    fast_candidate = _collect_interactions_fast_candidate(connection, sqlite_path, table_map, max_rows=max_rows)
    if fast_candidate is not None and fast_candidate["metrics"]["direct_replay_lift_available"] is True:
        return [fast_candidate]

    candidates: list[dict[str, Any]] = []
    if fast_candidate is not None:
        candidates.append(fast_candidate)
    candidates.extend(_collect_single_table_candidates(connection, sqlite_path, table_map, max_rows=max_rows))
    candidates.extend(_collect_join_candidates(connection, sqlite_path, table_map, max_rows=max_rows))
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate["metrics"].get("candidate_tables_used", []))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    return deduped


def _collect_interactions_fast_candidate(
    connection: sqlite3.Connection,
    sqlite_path: Path,
    table_map: dict[str, set[str]],
    *,
    max_rows: int,
) -> dict[str, Any] | None:
    columns = table_map.get("interactions")
    if not columns:
        return None
    if "memory_replay_priority" not in columns or "isf_prediction_error" not in columns:
        return None
    metrics = _compute_single_table_metrics(
        connection,
        sqlite_path=sqlite_path,
        table_name="interactions",
        replay_column="memory_replay_priority",
        violation_info={"expr": _numeric_violation_expr("isf_prediction_error"), "source": "isf_prediction_error", "kind": "numeric"},
        max_rows=max_rows,
    )
    if metrics["row_count_available"] in (None, 0):
        return None
    return {
        "kind": "single",
        "priority_rank": 5,
        "row_count_available": int(metrics["row_count_available"] or 0),
        "metrics": metrics,
    }


def _collect_single_table_candidates(
    connection: sqlite3.Connection,
    sqlite_path: Path,
    table_map: dict[str, set[str]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for table_name, columns in sorted(table_map.items(), key=_table_priority_key):
        replay_column = _first_matching(columns, REPLAY_PRIORITY_COLUMNS)
        violation_info = _violation_info(columns)
        if replay_column is None or violation_info is None:
            continue
        if table_name == "interactions" and replay_column == "memory_replay_priority" and violation_info["source"] == "isf_prediction_error":
            continue
        metrics = _compute_single_table_metrics(
            connection,
            sqlite_path=sqlite_path,
            table_name=table_name,
            replay_column=replay_column,
            violation_info=violation_info,
            max_rows=max_rows,
        )
        if metrics["row_count_available"] in (None, 0):
            continue
        candidates.append(
            {
                "kind": "single",
                "priority_rank": _candidate_priority_rank(replay_column, violation_info["kind"]),
                "row_count_available": int(metrics["row_count_available"] or 0),
                "metrics": metrics,
            }
        )
    return candidates


def _collect_join_candidates(
    connection: sqlite3.Connection,
    sqlite_path: Path,
    table_map: dict[str, set[str]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    replay_tables: list[dict[str, Any]] = []
    prediction_tables: list[dict[str, Any]] = []
    for table_name, columns in table_map.items():
        key_column = _interaction_key_column(columns)
        if key_column is None:
            continue
        replay_column = _first_matching(columns, REPLAY_PRIORITY_COLUMNS)
        if replay_column is not None:
            replay_tables.append(
                {
                    "table_name": table_name,
                    "columns": columns,
                    "key_column": key_column,
                    "replay_column": replay_column,
                }
            )
        violation_info = _violation_info(columns)
        if violation_info is not None:
            prediction_tables.append(
                {
                    "table_name": table_name,
                    "columns": columns,
                    "key_column": key_column,
                    "violation_info": violation_info,
                }
            )

    candidates: list[dict[str, Any]] = []
    for replay_table in replay_tables:
        for prediction_table in prediction_tables:
            if replay_table["table_name"] == prediction_table["table_name"]:
                continue
            metrics = _compute_joined_table_metrics(
                connection,
                sqlite_path=sqlite_path,
                replay_table=replay_table,
                prediction_table=prediction_table,
                max_rows=max_rows,
            )
            if metrics["row_count_available"] in (None, 0):
                continue
            candidates.append(
                {
                    "kind": "join",
                    "priority_rank": 1,
                    "row_count_available": int(metrics["row_count_available"] or 0),
                    "metrics": metrics,
                }
            )
    return candidates


def _compute_single_table_metrics(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    table_name: str,
    replay_column: str,
    violation_info: dict[str, str],
    max_rows: int,
) -> dict[str, Any]:
    replay_expr = f"CAST({_quote_ident(replay_column)} AS REAL)"
    violation_expr = violation_info["expr"]
    high_flag_info = _high_priority_flag_info(_table_columns(connection, table_name))
    high_flag_expr = high_flag_info["expr"] if high_flag_info else "NULL"
    table_ref = _quote_ident(table_name)
    limit = max(1, int(max_rows))

    count_query = (
        f"SELECT COUNT(*) FROM {table_ref} "
        f"WHERE {_quote_ident(replay_column)} IS NOT NULL AND ({violation_expr}) IS NOT NULL"
    )
    row_count_available = int(connection.execute(count_query).fetchone()[0])
    base_cte = (
        "WITH base AS ("
        f"SELECT {replay_expr} AS replay_priority, ({violation_expr}) AS violation, ({high_flag_expr}) AS high_priority_flag "
        f"FROM {table_ref} "
        f"WHERE {_quote_ident(replay_column)} IS NOT NULL AND ({violation_expr}) IS NOT NULL "
        f"LIMIT {limit}"
        ") "
    )
    return _finalize_direct_metrics(
        connection,
        base_cte=base_cte,
        row_count_available=row_count_available,
        db_path=sqlite_path,
        candidate_tables=[table_name],
        prediction_source=f"{table_name}.{violation_info['source']}",
        replay_source=f"{table_name}.{replay_column}",
        explicit_high_priority=high_flag_info is not None,
    )


def _compute_joined_table_metrics(
    connection: sqlite3.Connection,
    *,
    sqlite_path: Path,
    replay_table: dict[str, Any],
    prediction_table: dict[str, Any],
    max_rows: int,
) -> dict[str, Any]:
    replay_table_name = replay_table["table_name"]
    prediction_table_name = prediction_table["table_name"]
    replay_ref = _quote_ident(replay_table_name)
    prediction_ref = _quote_ident(prediction_table_name)
    replay_key_expr = _quote_ident(replay_table["key_column"])
    prediction_key_expr = _quote_ident(prediction_table["key_column"])
    replay_expr = f"CAST({_quote_ident(replay_table['replay_column'])} AS REAL)"
    violation_info = prediction_table["violation_info"]
    violation_expr = violation_info["expr"]
    high_flag_info = _high_priority_flag_info(replay_table["columns"])
    high_flag_expr = high_flag_info["expr"] if high_flag_info else "NULL"
    limit = max(1, int(max_rows))

    count_query = f"""
        WITH replay_agg AS (
            SELECT
                {replay_key_expr} AS interaction_id,
                MAX({replay_expr}) AS replay_priority,
                MAX(({high_flag_expr})) AS high_priority_flag
            FROM {replay_ref}
            WHERE {_quote_ident(replay_table['replay_column'])} IS NOT NULL
            GROUP BY {replay_key_expr}
        ),
        prediction_agg AS (
            SELECT
                {prediction_key_expr} AS interaction_id,
                MAX(({violation_expr})) AS violation,
                MAX(CASE WHEN ({violation_expr}) IS NOT NULL THEN 1 ELSE 0 END) AS has_violation
            FROM {prediction_ref}
            GROUP BY {prediction_key_expr}
        )
        SELECT COUNT(*)
        FROM replay_agg
        JOIN prediction_agg USING (interaction_id)
        WHERE prediction_agg.has_violation = 1
    """
    row_count_available = int(connection.execute(count_query).fetchone()[0])
    base_cte = f"""
        WITH replay_agg AS (
            SELECT
                {replay_key_expr} AS interaction_id,
                MAX({replay_expr}) AS replay_priority,
                MAX(({high_flag_expr})) AS high_priority_flag
            FROM {replay_ref}
            WHERE {_quote_ident(replay_table['replay_column'])} IS NOT NULL
            GROUP BY {replay_key_expr}
        ),
        prediction_agg AS (
            SELECT
                {prediction_key_expr} AS interaction_id,
                MAX(({violation_expr})) AS violation,
                MAX(CASE WHEN ({violation_expr}) IS NOT NULL THEN 1 ELSE 0 END) AS has_violation
            FROM {prediction_ref}
            GROUP BY {prediction_key_expr}
        ),
        base AS (
            SELECT
                replay_agg.replay_priority,
                prediction_agg.violation,
                replay_agg.high_priority_flag
            FROM replay_agg
            JOIN prediction_agg USING (interaction_id)
            WHERE prediction_agg.has_violation = 1
            LIMIT {limit}
        )
    """
    return _finalize_direct_metrics(
        connection,
        base_cte=base_cte,
        row_count_available=row_count_available,
        db_path=sqlite_path,
        candidate_tables=[replay_table_name, prediction_table_name],
        prediction_source=f"{prediction_table_name}.{violation_info['source']}",
        replay_source=f"{replay_table_name}.{replay_table['replay_column']}",
        explicit_high_priority=high_flag_info is not None,
    )


def _finalize_direct_metrics(
    connection: sqlite3.Connection,
    *,
    base_cte: str,
    row_count_available: int,
    db_path: Path,
    candidate_tables: list[str],
    prediction_source: str,
    replay_source: str,
    explicit_high_priority: bool,
) -> dict[str, Any]:
    metrics = {
        "db_found": True,
        "db_path": str(db_path),
        "schema_inspected": True,
        "tables_seen": [],
        "candidate_tables_used": candidate_tables,
        "prediction_violation_metric_source": prediction_source,
        "replay_priority_metric_source": replay_source,
        "row_count_available": row_count_available,
        "row_count_used": 0,
        "prediction_violation_row_count": None,
        "non_prediction_violation_row_count": None,
        "prediction_violation_base_ratio": None,
        "mean_replay_priority_for_prediction_violating_interactions": None,
        "mean_replay_priority_for_non_prediction_violating_interactions": None,
        "prediction_violation_replay_lift": None,
        "high_priority_replay_threshold": None,
        "high_priority_threshold_method": None,
        "high_priority_replay_prediction_violation_ratio": None,
        "high_priority_replay_non_prediction_violation_ratio": None,
        "direct_replay_lift_available": False,
        "missing_evidence": [],
    }
    if row_count_available <= 0:
        return metrics

    row = connection.execute(
        base_cte
        + """
        SELECT
            COUNT(*),
            SUM(CASE WHEN violation = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN violation = 0 THEN 1 ELSE 0 END),
            AVG(CASE WHEN violation = 1 THEN replay_priority END),
            AVG(CASE WHEN violation = 0 THEN replay_priority END)
        FROM base
        """
    ).fetchone()
    row_count_used = int(row[0] or 0)
    violating_count = None if row[1] is None else int(row[1])
    non_violating_count = None if row[2] is None else int(row[2])
    violating_mean = None if row[3] is None else float(row[3])
    non_violating_mean = None if row[4] is None else float(row[4])

    metrics["row_count_used"] = row_count_used
    metrics["prediction_violation_row_count"] = violating_count
    metrics["non_prediction_violation_row_count"] = non_violating_count
    if row_count_used > 0 and violating_count is not None:
        metrics["prediction_violation_base_ratio"] = violating_count / row_count_used
    metrics["mean_replay_priority_for_prediction_violating_interactions"] = violating_mean
    metrics["mean_replay_priority_for_non_prediction_violating_interactions"] = non_violating_mean
    if violating_mean is not None and non_violating_mean is not None and non_violating_mean > 0:
        metrics["prediction_violation_replay_lift"] = violating_mean / non_violating_mean

    if explicit_high_priority:
        metrics["high_priority_replay_threshold"] = None
        metrics["high_priority_threshold_method"] = "explicit_flag"
        high_row = connection.execute(
            base_cte
            + """
            SELECT
                COUNT(*),
                SUM(CASE WHEN violation = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN violation = 0 THEN 1 ELSE 0 END)
            FROM base
            WHERE high_priority_flag = 1
            """
        ).fetchone()
    else:
        threshold, method = _select_percentile_threshold(connection, base_cte, row_count_used)
        metrics["high_priority_replay_threshold"] = threshold
        metrics["high_priority_threshold_method"] = method
        high_row = connection.execute(
            base_cte
            + f"""
            SELECT
                COUNT(*),
                SUM(CASE WHEN violation = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN violation = 0 THEN 1 ELSE 0 END)
            FROM base
            WHERE replay_priority >= {float(threshold)}
            """
        ).fetchone()

    high_count = int(high_row[0] or 0)
    high_violating_count = int(high_row[1] or 0)
    high_non_violating_count = int(high_row[2] or 0)
    if high_count > 0:
        metrics["high_priority_replay_prediction_violation_ratio"] = high_violating_count / high_count
        metrics["high_priority_replay_non_prediction_violation_ratio"] = high_non_violating_count / high_count

    metrics["direct_replay_lift_available"] = all(
        (
            row_count_used > 0,
            violating_count is not None and violating_count > 0,
            non_violating_count is not None and non_violating_count > 0,
            metrics["prediction_violation_replay_lift"] is not None,
            metrics["high_priority_replay_prediction_violation_ratio"] is not None,
            metrics["high_priority_replay_non_prediction_violation_ratio"] is not None,
            metrics["prediction_violation_base_ratio"] is not None,
        )
    )
    return metrics


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


def _table_priority_key(item: tuple[str, set[str]]) -> tuple[int, str]:
    table_name = item[0]
    return (0 if table_name == "interactions" else 1, table_name)


def _interaction_key_column(columns: set[str]) -> str | None:
    if "interaction_id" in columns:
        return "interaction_id"
    if "id" in columns:
        return "id"
    return None


def _violation_info(columns: set[str]) -> dict[str, str] | None:
    for column in PREDICTION_NUMERIC_COLUMNS:
        if column in columns:
            return {
                "expr": _numeric_violation_expr(column),
                "source": column,
                "kind": "numeric",
            }
    for column in PREDICTION_BOOLEAN_COLUMNS:
        if column in columns:
            return {
                "expr": _boolean_violation_expr(column),
                "source": column,
                "kind": "boolean",
            }
    for left, right in PREDICTION_COMPARISON_PAIRS:
        if left in columns and right in columns:
            return {
                "expr": _comparison_violation_expr(left, right),
                "source": f"{left}!={right}",
                "kind": "comparison",
            }
    return None


def _high_priority_flag_info(columns: set[str]) -> dict[str, str] | None:
    for column in HIGH_PRIORITY_FLAG_COLUMNS:
        if column in columns:
            return {"column": column, "expr": _boolean_violation_expr(column)}
    return None


def _numeric_violation_expr(column: str) -> str:
    quoted = _quote_ident(column)
    return f"CASE WHEN {quoted} IS NULL THEN NULL WHEN CAST({quoted} AS REAL) > 0 THEN 1 ELSE 0 END"


def _boolean_violation_expr(column: str) -> str:
    quoted = _quote_ident(column)
    text_value = f"LOWER(TRIM(CAST({quoted} AS TEXT)))"
    return (
        f"CASE WHEN {quoted} IS NULL THEN NULL "
        f"WHEN {text_value} IN ('1', 'true', 'yes') THEN 1 "
        f"WHEN CAST({quoted} AS REAL) = 1 THEN 1 "
        f"ELSE 0 END"
    )


def _comparison_violation_expr(left: str, right: str) -> str:
    left_q = _quote_ident(left)
    right_q = _quote_ident(right)
    return (
        f"CASE WHEN {left_q} IS NULL OR {right_q} IS NULL THEN NULL "
        f"WHEN CAST({left_q} AS TEXT) <> CAST({right_q} AS TEXT) THEN 1 "
        f"ELSE 0 END"
    )


def _candidate_priority_rank(replay_column: str, violation_kind: str) -> int:
    if replay_column == "replay_priority" and violation_kind == "numeric":
        return 4
    if replay_column == "replay_priority" and violation_kind == "boolean":
        return 3
    if violation_kind == "numeric":
        return 2
    if violation_kind == "boolean":
        return 1
    return 0


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(candidate["priority_rank"]),
        1 if candidate["kind"] == "single" else 0,
        int(candidate["row_count_available"]),
    )


def _direct_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = candidate["metrics"]
    return (
        1 if metrics.get("direct_replay_lift_available") is True else 0,
        int(candidate["priority_rank"]),
        int(metrics.get("row_count_used") or 0),
        int(candidate["row_count_available"] or 0),
    )


def _scan_all_direct_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, int]:
    metrics = candidate["metrics"]
    return (
        int(metrics.get("row_count_used") or 0),
        float(metrics.get("prediction_violation_replay_lift") or 0.0),
        int(candidate["priority_rank"]),
        int(candidate["row_count_available"] or 0),
    )


def _select_percentile_threshold(connection: sqlite3.Connection, base_cte: str, row_count_used: int) -> tuple[float, str]:
    if row_count_used <= 0:
        return 0.0, "sql_percentile"
    offset = max(0, math.ceil(0.9 * row_count_used) - 1)
    try:
        row = connection.execute(
            base_cte
            + f"SELECT replay_priority FROM base ORDER BY replay_priority LIMIT 1 OFFSET {int(offset)}"
        ).fetchone()
    except sqlite3.DatabaseError:
        row = None
    if row is not None and row[0] is not None:
        return float(row[0]), "sql_percentile"
    fallback_row = connection.execute(base_cte + "SELECT MAX(replay_priority) FROM base").fetchone()
    max_priority = 0.0 if fallback_row is None or fallback_row[0] is None else float(fallback_row[0])
    return max_priority * 0.9, "fallback_max_0_9"


def _gt(value: Any, threshold: float) -> bool | None:
    if value is None:
        return None
    return float(value) > float(threshold)


def _eq(value: Any, target: float) -> bool | None:
    if value is None:
        return None
    return float(value) == float(target)


def _compare_gt(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return float(left) > float(right)


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


def _isoformat_mtime(path: Path) -> str:
    timestamp = path.stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _h02_ready_entry(entry: dict[str, Any]) -> bool:
    return all(
        (
            entry["has_required_carrier_fields"] is True,
            entry["has_sqlite_db"] is True,
            _gt(entry.get("mean_isf_prediction_error"), 0.0) is True,
            _gt(entry.get("context_contradiction_count"), 0) is True,
            _gt(entry.get("repeated_contradiction_count"), 0) is True,
            _gt(entry.get("context_expansion_suggested_count"), 0) is True,
            _gt(entry.get("memory_replay_candidate_count"), 0) is True,
        )
    )


def _h02_ready_sort_key(entry: dict[str, Any]) -> tuple[int, int, int, float]:
    mtime = _parse_iso_timestamp(entry["report_mtime"])
    return (
        0 if entry["has_required_carrier_fields"] is True else 1,
        0 if entry["has_sqlite_db"] is True else 1,
        -int(entry.get("memory_record_count") or 0),
        -mtime,
    )


def _parse_iso_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _relative_or_absolute_display(run_dir: Path, sqlite_path: Path) -> str:
    try:
        return str(sqlite_path.relative_to(run_dir))
    except ValueError:
        return str(sqlite_path)


def _write_h02_ready_inventory(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / H02_READY_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / H02_READY_TXT_NAME).write_text(_format_h02_ready_text(result), encoding="utf-8")


def _format_h02_ready_text(result: dict[str, Any]) -> str:
    lines = [
        f"Runs root: {result['runs_root']}",
        f"Total reports found: {result['candidate_count']}",
        f"Ready count: {result['ready_count']}",
        "",
    ]
    recommended = result.get("recommended_run")
    if recommended is None:
        lines.append("No H02-ready existing v05c run found. Generate a new v05c run with current code, then rerun find-h02-ready-runs.")
        return "\n".join(lines)

    lines.extend(
        [
            "Top recommended run:",
            recommended["run_dir"],
            "",
            "Run:",
            "PYTHONPATH=src python -m v6.cli hypothesis-h02-report \\",
            f"  --run-dir {recommended['run_dir']} \\",
            f"  --output-dir {recommended['recommended_output_dir']} \\",
            f"  --max-db-files {DEFAULT_MAX_DB_FILES}",
        ]
    )
    top_entry = next((entry for entry in result.get("runs", []) if entry["run_dir"] == recommended["run_dir"]), None)
    if top_entry and top_entry.get("sqlite_db_paths"):
        preferred_path = _relative_or_absolute_display(Path(recommended["run_dir"]), Path(top_entry["sqlite_db_paths"][0]))
        lines.extend(
            [
                "",
                "Preferred DB example:",
                "PYTHONPATH=src python -m v6.cli hypothesis-h02-report \\",
                f"  --run-dir {recommended['run_dir']} \\",
                f"  --output-dir {recommended['recommended_output_dir']} \\",
                "  --max-db-files 1 \\",
                f"  --prefer-db {preferred_path}",
            ]
        )
    lines.extend(["", "Candidates:"])
    for entry in result.get("runs", []):
        lines.extend(
            [
                f"- run_dir: {entry['run_dir']}",
                f"  h02_ready: {entry['h02_ready']}",
                f"  has_required_carrier_fields: {entry['has_required_carrier_fields']}",
                f"  has_sqlite_db: {entry['has_sqlite_db']}",
                f"  memory_record_count: {entry['memory_record_count']}",
                f"  missing_required_carrier_fields: {', '.join(entry['missing_required_carrier_fields']) or 'none'}",
            ]
        )
    return "\n".join(lines)


def _populate_evidence_lists(result: dict[str, Any]) -> None:
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    missing_evidence: list[str] = list(result.get("missing_evidence", []))

    if _gt(result.get("mean_isf_prediction_error"), 0.0) is True:
        evidence_for.append(f"Mean ISF prediction-error signal is positive ({result['mean_isf_prediction_error']:.4f}).")
    else:
        evidence_against.append("Mean ISF prediction-error signal is absent or zero.")

    if _gt(result.get("context_contradiction_count"), 0) is True:
        evidence_for.append(f"Contradictions are present ({int(result['context_contradiction_count'])} contradiction events).")
    elif _gt(result.get("wrong_prediction_count"), 0) is True and _gt(result.get("confident_wrong_prediction_count"), 0) is True:
        evidence_against.append("Confident wrong predictions are present, but contradiction events were not recorded.")
    elif _gt(result.get("prediction_error_positive_count"), 0) is True and _gt(result.get("wrong_prediction_count"), 0) is not True:
        missing_evidence.append("Prediction-error ISF exists, but family-level contradiction was not observable.")
    else:
        evidence_against.append("No contradiction evidence is present in the sampled interactions.")

    if _gt(result.get("repeated_contradiction_count"), 0) is True:
        evidence_for.append(f"Repeated contradictions are present ({int(result['repeated_contradiction_count'])}).")
    else:
        evidence_against.append("Repeated contradiction pressure is not demonstrated.")

    if _gt(result.get("context_expansion_suggested_count"), 0) is True:
        evidence_for.append(
            f"Context expansion is suggested before concept emergence ({int(result['context_expansion_suggested_count'])} cases)."
        )
    else:
        evidence_against.append("Context expansion pressure is not demonstrated.")

    if _gt(result.get("memory_replay_candidate_count"), 0) is True:
        evidence_for.append(f"Replay/attention evidence: replay candidates exist in memory ({int(result['memory_replay_candidate_count'])} candidates).")
    elif _gt(result.get("high_priority_replay_count"), 0) is True or _gt(result.get("memory_max_replay_priority"), 0.0) is True:
        missing_evidence.append("Replay priority is present, but replay candidate counts were not exported consistently.")
    elif result.get("direct_replay_lift_pass") is True:
        missing_evidence.append("Aggregate replay/contradiction counters are unavailable or zero despite direct replay-lift evidence.")
    else:
        evidence_against.append("Replay/attention evidence: no replay candidates are available to show memory centrality.")

    lift = result.get("prediction_violation_replay_lift")
    if lift is not None:
        if result.get("direct_replay_lift_pass") is True:
            evidence_for.append(
                "Direct replay-lift evidence supports H02: prediction-violating interactions receive higher replay priority."
            )
        elif float(lift) > 1.25:
            evidence_for.append(f"Replay/attention evidence: prediction-violating interactions have replay lift {float(lift):.3f} (> 1.25).")
        else:
            evidence_against.append(f"Replay/attention evidence: prediction-violating replay lift is weak ({float(lift):.3f}).")
    elif (
        DIRECT_LINKAGE_UNAVAILABLE_MESSAGE not in missing_evidence
        and DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE not in missing_evidence
    ):
        missing_evidence.append(DIRECT_LINKAGE_UNAVAILABLE_MESSAGE)

    object_carrier_check = _eq(result.get("emergent_object_carrier_count"), 0)
    if object_carrier_check is True:
        evidence_for.append("Temporal pre-carrier evidence: no emergent object carriers are present in the sampled run.")
    elif result.get("emergent_object_carrier_count") is None:
        missing_evidence.append("Temporal pre-carrier evidence: aggregate object-carrier absence evidence is unavailable.")
    else:
        evidence_for.append("Carrier emergence present in final memory snapshot; this does not invalidate replay/attention evidence without temporal ordering evidence.")

    fallback_check = _eq(result.get("emergent_context_action_fallback_count"), 0)
    if fallback_check is False:
        evidence_against.append("Temporal pre-carrier evidence: context-action fallback carriers emerged, which weakens the pre-object interpretation.")
    elif result.get("emergent_context_action_fallback_count") is None:
        missing_evidence.append("Temporal pre-carrier evidence: aggregate context-action fallback evidence is unavailable.")

    if result.get("carrier_timing_note"):
        evidence_for.append(f"Temporal pre-carrier evidence: {result['carrier_timing_note']}")

    result["evidence_for"] = evidence_for
    result["evidence_against"] = evidence_against
    result["missing_evidence"] = missing_evidence


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _finalize_h02_result(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / H02_JSON_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / H02_TXT_NAME).write_text(_format_text_report(result), encoding="utf-8")
    (output_dir / H02_MD_NAME).write_text(_format_markdown_report(result), encoding="utf-8")


def _format_text_report(result: dict[str, Any]) -> str:
    lines = [
        "H02 Hypothesis Report",
        "",
        "Hypothesis statement:",
        "Prediction-violating interactions become central in memory before object concepts emerge.",
        "",
        f"Final Decision: {result['decision']}",
        f"H02A replay/attention: {result['h02a_replay_attention_decision']}",
        f"H02B pre-carrier timing: {result['h02b_pre_carrier_timing_decision']}",
        f"Final decision basis: {result['h02_final_decision_basis']}",
        f"Carrier timing note: {result.get('carrier_timing_note') or 'none'}",
        "",
        "Direct replay-lift evidence:",
        *_format_direct_evidence_text(result),
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


def _format_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# H02: Prediction Violations Drive Attention",
        "",
        "**Hypothesis statement**",
        "",
        "Prediction-violating interactions become central in memory before object concepts emerge.",
        "",
        f"**Final Decision:** `{result['decision']}`",
        f"**H02A replay/attention:** `{result['h02a_replay_attention_decision']}`",
        f"**H02B pre-carrier timing:** `{result['h02b_pre_carrier_timing_decision']}`",
        f"**Final decision basis:** {result['h02_final_decision_basis']}",
        f"**Carrier timing note:** {result.get('carrier_timing_note') or 'none'}",
        "",
        "## Direct replay-lift evidence",
        *_format_direct_evidence_markdown(result),
        "",
        "## Evidence For",
        *_format_markdown_bullets(result["evidence_for"]),
        "",
        "## Evidence Against",
        *_format_markdown_bullets(result["evidence_against"]),
        "",
        "## Missing Evidence",
        *_format_markdown_bullets(result["missing_evidence"]),
        "",
        "## Acceptance Checklist",
        *_format_markdown_acceptance_checks(result["acceptance_checks"]),
        "",
        "## Scientific Conclusion",
        "",
        result["scientific_conclusion"],
        "",
    ]
    return "\n".join(lines)


def _format_direct_evidence_text(result: dict[str, Any]) -> list[str]:
    lines = [
        f"SQLite DB files total: {result.get('sqlite_db_count_total')}",
        f"SQLite DB files inspected: {result.get('sqlite_db_count_inspected')}",
        f"SQLite DB scan truncated: {result.get('sqlite_db_inspection_truncated')}",
        f"SQLite DB files skipped: {result.get('sqlite_db_skipped_count')}",
    ]
    if result.get("direct_replay_lift_available") is not True:
        return lines + [_direct_linkage_missing_message(result)]
    return lines + [
        f"DB used: {result.get('db_path')}",
        f"table(s) used: {', '.join(result.get('candidate_tables_used', [])) or 'none'}",
        f"prediction violation source column: {result.get('prediction_violation_metric_source')}",
        f"replay priority source column: {result.get('replay_priority_metric_source')}",
        f"row count used: {result.get('row_count_used')}",
        f"prediction violation base ratio: {_fmt_number(result.get('prediction_violation_base_ratio'))}",
        f"mean priority violating: {_fmt_number(result.get('mean_replay_priority_for_prediction_violating_interactions'))}",
        f"mean priority non-violating: {_fmt_number(result.get('mean_replay_priority_for_non_prediction_violating_interactions'))}",
        f"replay lift: {_fmt_number(result.get('prediction_violation_replay_lift'))}",
        f"high-priority threshold method: {result.get('high_priority_threshold_method')}",
        f"high-priority violation ratio: {_fmt_number(result.get('high_priority_replay_prediction_violation_ratio'))}",
        f"conclusion: {'direct replay-lift evidence supports H02A' if _gt(result.get('prediction_violation_replay_lift'), 1.25) is True else 'direct replay-lift evidence is present but weak'}",
    ]


def _format_direct_evidence_markdown(result: dict[str, Any]) -> list[str]:
    lines = [
        f"- SQLite DB files total: `{result.get('sqlite_db_count_total')}`",
        f"- SQLite DB files inspected: `{result.get('sqlite_db_count_inspected')}`",
        f"- SQLite DB scan truncated: `{result.get('sqlite_db_inspection_truncated')}`",
        f"- SQLite DB files skipped: `{result.get('sqlite_db_skipped_count')}`",
        f"- selected DB: `{result.get('selected_db_path')}`",
        f"- inspected DB paths: `{', '.join(result.get('inspected_db_paths', [])) or 'none'}`",
    ]
    if result.get("direct_replay_lift_available") is not True:
        return lines + [f"- {_direct_linkage_missing_message(result)}"]
    return lines + [
        f"- DB used: `{result.get('db_path')}`",
        f"- table(s) used: `{', '.join(result.get('candidate_tables_used', [])) or 'none'}`",
        f"- prediction violation source column: `{result.get('prediction_violation_metric_source')}`",
        f"- replay priority source column: `{result.get('replay_priority_metric_source')}`",
        f"- row count used: `{result.get('row_count_used')}`",
        f"- prediction violation base ratio: `{_fmt_number(result.get('prediction_violation_base_ratio'))}`",
        f"- mean priority violating: `{_fmt_number(result.get('mean_replay_priority_for_prediction_violating_interactions'))}`",
        f"- mean priority non-violating: `{_fmt_number(result.get('mean_replay_priority_for_non_prediction_violating_interactions'))}`",
        f"- replay lift: `{_fmt_number(result.get('prediction_violation_replay_lift'))}`",
        f"- base violation ratio: `{_fmt_number(result.get('prediction_violation_base_ratio'))}`",
        f"- high-priority threshold method: `{result.get('high_priority_threshold_method')}`",
        f"- high-priority violation ratio: `{_fmt_number(result.get('high_priority_replay_prediction_violation_ratio'))}`",
        f"- conclusion: {'direct replay-lift evidence supports H02A' if _gt(result.get('prediction_violation_replay_lift'), 1.25) is True else 'direct replay-lift evidence is present but weak'}",
    ]


def _fmt_number(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


def _direct_linkage_missing_message(result: dict[str, Any]) -> str:
    missing = list(result.get("missing_evidence", []))
    for message in (DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE, DIRECT_LINKAGE_UNAVAILABLE_MESSAGE):
        if message in missing:
            return message
    return DIRECT_LINKAGE_UNAVAILABLE_MESSAGE


def _format_bullets(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _format_markdown_bullets(items: list[str]) -> list[str]:
    return _format_bullets(items)


def _format_acceptance_checks(checks: dict[str, Any]) -> list[str]:
    return [f"- {name}: {value}" for name, value in checks.items()]


def _format_markdown_acceptance_checks(checks: dict[str, Any]) -> list[str]:
    return _format_acceptance_checks(checks)
