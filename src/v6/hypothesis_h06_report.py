from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from v6.higher_order_substrate import derive_higher_order_memory
from v6.memory.compact_memory import ensure_memory_layout


def _evidence_diagnostics(memory_dir: Path, run_dir: Path | None, *, missing_target: str) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    raw_db_exists = bool(run_dir is not None and any(Path(run_dir).rglob("*.sqlite")))
    return {
        "expected_current_state_path": str(current_state),
        "compact_memory_exists": bool(current_state.exists()),
        "raw_db_evidence_exists": bool(raw_db_exists),
        "direct_streamed_manifest_exists": bool((Path(memory_dir) / "direct_streaming_fold_manifest.sqlite").exists()),
        "missing_target": str(missing_target),
    }


def evaluate_h06_role_transfer(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_memory_layout(memory_dir)
    if not already_derived:
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        result = _base_result("INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory file: {current_state}"])
        result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target="current_state.sqlite")
        _write_outputs(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing_tables = [name for name in ("role_candidates", "role_transfer_attempts", "higher_order_milestones") if name not in tables]
        if missing_tables:
            result = _base_result("INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"])
            result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target=",".join(missing_tables))
            result["evidence_diagnostics"]["tables_seen"] = sorted(tables)
            _write_outputs(output_dir, result)
            return result
        role_candidate_count = int(conn.execute("SELECT COUNT(*) FROM role_candidates").fetchone()[0])
        rows = conn.execute(
            """
            SELECT attempts.transfer_kind, attempts.source_scope_type,
                   attempts.source_scope_key, attempts.target_scope_type, attempts.target_scope_key,
                   attempts.source_game_key, attempts.target_game_key,
                   attempts.source_context_key, attempts.target_context_key,
                   attempts.source_carrier_signature, attempts.source_role_signature,
                   attempts.predicted_target_role_signature, attempts.observed_target_role_signature,
                   attempts.provenance_mode, attempts.provenance_status,
                   attempts.similarity_score, attempts.transfer_score, attempts.reuse_success,
                   attempts.failure_reason, attempts.best_margin, attempts.source_carrier_count,
                   attempts.candidate_role_count, COALESCE(roles.role_type, 'unknown') AS role_type
            FROM role_transfer_attempts AS attempts
            LEFT JOIN role_candidates AS roles ON roles.role_signature = attempts.source_role_signature
            ORDER BY attempts.source_role_signature ASC, attempts.target_scope_key ASC, attempts.target_carrier_signature ASC
            """
        ).fetchall()
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        transfer_summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'higher_order_transfer_summary'"
        ).fetchone()
        transfer_summary = json.loads(str(transfer_summary_row[0])) if transfer_summary_row and transfer_summary_row[0] else {}
    provenance_errors = [_transfer_provenance_error(row) for row in rows]
    legacy_rows = [row for row, error in zip(rows, provenance_errors, strict=True) if error == "legacy_transfer_provenance"]
    invalid_rows = [row for row, error in zip(rows, provenance_errors, strict=True) if error not in (None, "legacy_transfer_provenance")]
    provenance_valid_rows = [row for row, error in zip(rows, provenance_errors, strict=True) if error is None]
    verified_rows = [row for row in provenance_valid_rows if str(row["provenance_mode"] or "") == "single_source"]
    multi_source_rows = [row for row in provenance_valid_rows if str(row["provenance_mode"] or "") == "multi_source"]
    valid_rows = verified_rows
    success_rows = [row for row in valid_rows if int(row["reuse_success"] or 0) == 1]
    cross_game_rows = [row for row in valid_rows if str(row["transfer_kind"]) == "cross_game"]
    cross_context_rows = [row for row in valid_rows if str(row["transfer_kind"]) == "cross_context"]
    cross_game_success_count = sum(1 for row in cross_game_rows if int(row["reuse_success"] or 0) == 1)
    cross_context_success_count = sum(1 for row in cross_context_rows if int(row["reuse_success"] or 0) == 1)
    transfer_attempt_count = len(valid_rows)
    successful_transfer_count = len(success_rows)
    transfer_success_rate = float(successful_transfer_count / transfer_attempt_count) if transfer_attempt_count else None
    successful_role_count = len({str(row["source_role_signature"]) for row in success_rows})
    failed_rows = [row for row in valid_rows if int(row["reuse_success"] or 0) != 1]
    role_mismatch_count = sum(1 for row in failed_rows if str(row["failure_reason"] or "") == "role_mismatch")
    low_similarity_count = sum(1 for row in failed_rows if str(row["failure_reason"] or "") == "low_similarity")
    insufficient_source_support_count = sum(1 for row in failed_rows if str(row["failure_reason"] or "") == "insufficient_source_support")
    no_source_profile_count = sum(1 for row in failed_rows if str(row["failure_reason"] or "") == "no_source_profile")
    other_failure_count = max(
        0,
        transfer_attempt_count - successful_transfer_count - role_mismatch_count - low_similarity_count
        - insufficient_source_support_count - no_source_profile_count,
    )
    failure_total = successful_transfer_count + role_mismatch_count + low_similarity_count + insufficient_source_support_count + no_source_profile_count + other_failure_count
    consistency_warnings = [] if failure_total == transfer_attempt_count else [
        f"failure breakdown totals {failure_total}, expected {transfer_attempt_count}"
    ]
    mean_transfer_score = (
        sum(float(row["transfer_score"] or 0.0) for row in valid_rows) / max(1, transfer_attempt_count)
        if valid_rows
        else None
    )
    max_transfer_score = max((float(row["transfer_score"] or 0.0) for row in valid_rows), default=None)
    margins = [float(row["best_margin"]) for row in valid_rows if row["best_margin"] is not None]
    source_counts = [int(row["source_carrier_count"] or 0) for row in valid_rows]
    candidate_role_counts = [int(row["candidate_role_count"] or 0) for row in valid_rows]
    cross_game_success_rate = _rate(cross_game_success_count, len(cross_game_rows))
    cross_context_success_rate = _rate(cross_context_success_count, len(cross_context_rows))
    total_possible = int(transfer_summary.get("total_possible_transfer_attempts", len(rows)) or 0)
    sampled_attempts = int(transfer_summary.get("sampled_transfer_attempts", transfer_attempt_count) or 0)
    game_provenance_rows = [row for row in verified_rows if row["source_game_key"] and row["target_game_key"]]
    pair_rows = _aggregate_transfer_rows(game_provenance_rows, key_fn=lambda row: (
        str(row["source_game_key"]), str(row["target_game_key"]), str(row["transfer_kind"]),
    ), labels=("source_game", "target_game", "transfer_kind"), include_role_fields=False)
    source_game_rows = _aggregate_transfer_rows(game_provenance_rows, key_fn=lambda row: (str(row["source_game_key"]),), labels=("source_game",), include_role_fields=False)
    target_game_rows = _aggregate_transfer_rows(game_provenance_rows, key_fn=lambda row: (str(row["target_game_key"]),), labels=("target_game",), include_role_fields=False)
    source_context_rows = _aggregate_transfer_rows(cross_context_rows, key_fn=lambda row: (str(row["source_context_key"]),), labels=("source_context",), include_role_fields=False)
    target_context_rows = _aggregate_transfer_rows(cross_context_rows, key_fn=lambda row: (str(row["target_context_key"]),), labels=("target_context",), include_role_fields=False)
    context_pair_rows = _aggregate_transfer_rows(cross_context_rows, key_fn=lambda row: (
        str(row["source_context_key"]), str(row["target_context_key"]),
    ), labels=("source_context", "target_context"), include_role_fields=False)
    role_rows = _aggregate_transfer_rows(verified_rows, key_fn=lambda row: (str(row["source_role_signature"] or "unknown"), str(row["role_type"] or "unknown")), labels=("role_signature", "role_type"), include_role_fields=True)
    predicted_target_role_rows = _aggregate_transfer_rows(verified_rows, key_fn=lambda row: (str(row["predicted_target_role_signature"] or "unknown"),), labels=("predicted_target_role_signature",), include_role_fields=True)
    observed_target_role_rows = _aggregate_transfer_rows(verified_rows, key_fn=lambda row: (str(row["observed_target_role_signature"] or "unknown"),), labels=("observed_target_role_signature",), include_role_fields=True)
    role_type_rows = _aggregate_transfer_rows(valid_rows, key_fn=lambda row: (str(row["role_type"] or "unknown"),), labels=("role_type",), include_role_fields=True)
    quality = {
        "similarity_score_buckets": _bucket_transfer_rows(valid_rows, "similarity_score", ((0.40, "<0.40"), (0.60, "0.40-0.59"), (0.80, "0.60-0.79"), (float("inf"), ">=0.80"))),
        "best_margin_buckets": _bucket_transfer_rows(valid_rows, "best_margin", ((0.00, "<0.00"), (0.10, "0.00-0.09"), (0.20, "0.10-0.19"), (float("inf"), ">=0.20"))),
        "source_carrier_count_buckets": _bucket_transfer_rows(valid_rows, "source_carrier_count", ((2, "0-1"), (4, "2-3"), (8, "4-7"), (float("inf"), "8+"))),
        "candidate_role_count_buckets": _bucket_transfer_rows(valid_rows, "candidate_role_count", ((2, "0-1"), (4, "2-3"), (8, "4-7"), (float("inf"), "8+"))),
    }
    validity_gates = _h06_validity_gates(
        transfer_attempt_count=transfer_attempt_count,
        transfer_success_rate=transfer_success_rate,
        successful_role_count=successful_role_count,
        mean_best_margin=(sum(margins) / len(margins)) if margins else None,
        cross_game_success_count=cross_game_success_count,
        cross_context_success_count=cross_context_success_count,
    )
    provenance_warning = "invalid_transfer_provenance" if invalid_rows or legacy_rows else None
    if provenance_warning is not None:
        consistency_warnings.append(provenance_warning)
    metrics = {
        "transfer_attempt_count": transfer_attempt_count,
        "recorded_transfer_attempt_count": len(rows),
        "verified_cross_game_attempt_count": len(cross_game_rows),
        "verified_cross_game_success_count": cross_game_success_count,
        "verified_cross_game_success_rate": cross_game_success_rate,
        "verified_cross_context_attempt_count": len(cross_context_rows),
        "verified_cross_context_success_count": cross_context_success_count,
        "verified_cross_context_success_rate": cross_context_success_rate,
        "multi_source_transfer_attempt_count": len(multi_source_rows),
        "missing_source_transfer_attempt_count": sum(
            str(row["provenance_mode"] or "") == "missing_source" for row in rows
        ),
        "legacy_transfer_attempt_count": len(legacy_rows),
        "valid_cross_game_attempt_count": len(cross_game_rows),
        "valid_cross_context_attempt_count": len(cross_context_rows),
        "invalid_provenance_attempt_count": len(invalid_rows),
        "invalid_transfer_provenance_count": len(invalid_rows),
        "legacy_transfer_provenance_count": len(legacy_rows),
        "missing_source_game_count": sum(error == "missing_source_game" for error in provenance_errors),
        "missing_target_game_count": sum(error == "missing_target_game" for error in provenance_errors),
        "same_game_marked_cross_game_count": sum(error == "same_game_marked_cross_game" for error in provenance_errors),
        "same_context_marked_cross_context_count": sum(error == "same_context_marked_cross_context" for error in provenance_errors),
        "distinct_source_game_count": len({str(row["source_game_key"]) for row in game_provenance_rows}),
        "distinct_target_game_count": len({str(row["target_game_key"]) for row in game_provenance_rows}),
        "distinct_game_pair_count": len(pair_rows),
        "distinct_context_pair_count": len(context_pair_rows),
        "total_possible_transfer_attempts": total_possible,
        "sampled_transfer_attempts": sampled_attempts,
        "sampling_fraction": _rate(sampled_attempts, total_possible),
        "skipped_by_cap_count": transfer_summary.get("skipped_by_cap_count", 0),
        "sampled_cross_game_attempt_count": transfer_summary.get("sampled_cross_game_attempt_count", len(cross_game_rows)),
        "sampled_cross_context_attempt_count": transfer_summary.get("sampled_cross_context_attempt_count", len(cross_context_rows)),
        "transfer_sampling_strategy": transfer_summary.get("transfer_sampling_strategy", "persisted_role_transfer_attempts"),
        "max_attempts_per_role": transfer_summary.get("max_attempts_per_role"),
        "max_attempts_per_target_scope": transfer_summary.get("max_attempts_per_target_scope"),
        "successful_transfer_count": successful_transfer_count,
        "transfer_success_rate": transfer_success_rate,
        "cross_game_attempt_count": len(cross_game_rows),
        "cross_game_success_count": cross_game_success_count,
        "cross_game_success_rate": cross_game_success_rate,
        "cross_context_attempt_count": len(cross_context_rows),
        "cross_context_success_count": cross_context_success_count,
        "cross_context_success_rate": cross_context_success_rate,
        "successful_role_count": successful_role_count,
        "role_mismatch_count": role_mismatch_count,
        "low_similarity_count": low_similarity_count,
        "insufficient_source_support_count": insufficient_source_support_count,
        "no_source_profile_count": no_source_profile_count,
        "role_mismatch_rate": _rate(role_mismatch_count, transfer_attempt_count),
        "low_similarity_rate": _rate(low_similarity_count, transfer_attempt_count),
        "insufficient_source_support_rate": _rate(insufficient_source_support_count, transfer_attempt_count),
        "no_source_profile_rate": _rate(no_source_profile_count, transfer_attempt_count),
        "other_failure_count": other_failure_count,
        "other_failure_rate": _rate(other_failure_count, transfer_attempt_count),
        "mean_transfer_score": mean_transfer_score,
        "max_transfer_score": max_transfer_score,
        "mean_best_margin": (sum(margins) / len(margins)) if margins else None,
        "mean_source_carrier_count": (sum(source_counts) / len(source_counts)) if source_counts else None,
        "candidate_role_count_mean": (sum(candidate_role_counts) / len(candidate_role_counts)) if candidate_role_counts else None,
        "first_role_candidate_step": milestone_map.get("first_role_candidate_step"),
        "first_role_transfer_attempt_step": milestone_map.get("first_role_transfer_attempt_step"),
        "first_role_transfer_success_step": milestone_map.get("first_role_transfer_success_step"),
        "transfer_by_source_game": source_game_rows,
        "transfer_by_target_game": target_game_rows,
        "transfer_by_game_pair": pair_rows[:100],
        "transfer_by_source_context": source_context_rows,
        "transfer_by_target_context": target_context_rows,
        "transfer_by_context_pair": context_pair_rows[:100],
        "transfer_by_role": role_rows[:100],
        "transfer_by_predicted_target_role": predicted_target_role_rows[:100],
        "transfer_by_observed_target_role": observed_target_role_rows[:100],
        "transfer_by_role_type": role_type_rows,
        **quality,
        "attempts_per_role_mean": _mean_group_attempts(role_rows),
        "attempts_per_role_median": _median_group_attempts(role_rows),
        "attempts_per_role_max": _max_group_attempts(role_rows),
        "attempts_per_game_pair_mean": _mean_group_attempts(pair_rows),
        "attempts_per_game_pair_max": _max_group_attempts(pair_rows),
        "h06_validity_gates": validity_gates,
        "consistency_warnings": consistency_warnings,
    }
    if role_candidate_count <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no role candidates available"]
    elif transfer_attempt_count <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no role transfer attempts available"]
    elif all(bool(gate["passed"]) for gate in validity_gates.values()):
        decision = "VALID"
        missing = _failed_validity_gate_messages(validity_gates)
    elif transfer_attempt_count < 5:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["too few role transfer attempts for H06 evaluation"]
    elif transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) >= 0.35 and successful_role_count >= 1:
        decision = "PARTIALLY_VALID"
        missing = _failed_validity_gate_messages(validity_gates)
    elif role_candidate_count > 0 and transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) < 0.20:
        decision = "INVALID"
        missing = _failed_validity_gate_messages(validity_gates)
    else:
        decision = "PARTIALLY_VALID"
        missing = _failed_validity_gate_messages(validity_gates)
    result = _base_result(decision, missing)
    if legacy_rows:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append(
            "Legacy transfer provenance remains active; rebuild higher-order transfer artifacts before verified H06 evaluation."
        )
    sampled = int(metrics.get("sampled_transfer_attempts") or 0)
    possible = int(metrics.get("total_possible_transfer_attempts") or 0)
    sample_fraction = (float(sampled) / float(possible)) if possible > 0 else None
    if (
        result["decision"] == "VALID"
        and int(metrics.get("skipped_by_cap_count") or 0) > 0
        and sample_fraction is not None
        and sample_fraction < 0.5
    ):
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append(
            "H06 transfer sampling was capped; sampled attempts are insufficient for robust VALID classification."
        )
    result.update(metrics)
    result["transfer_rate_diagnosis"] = _transfer_rate_diagnosis(metrics, pair_rows, role_type_rows)
    result["core_metrics"] = dict(metrics)
    result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target="none")
    _write_outputs(output_dir, result, full_game_pairs=pair_rows)
    return result


def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H06",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _transfer_provenance_error(row: sqlite3.Row) -> str | None:
    """Return a precise reason when an attempt cannot support H06 claims."""
    mode = str(row["provenance_mode"] or "")
    if mode == "legacy" or mode not in {"single_source", "multi_source", "missing_source"}:
        return "legacy_transfer_provenance"
    kind = str(row["transfer_kind"] or "")
    source_game = row["source_game_key"]
    target_game = row["target_game_key"]
    source_context = row["source_context_key"]
    target_context = row["target_context_key"]
    if int(row["reuse_success"] or 0) == 1 and not row["source_role_signature"]:
        return "missing_source_role"
    if mode == "single_source" and not row["source_carrier_signature"]:
        return "missing_source_carrier"
    if mode == "missing_source":
        return "missing_source_profile"
    if kind == "cross_game":
        if not source_game:
            return "missing_source_game"
        if not target_game:
            return "missing_target_game"
        if str(source_game) == str(target_game):
            return "same_game_marked_cross_game"
        return None
    if kind == "cross_context":
        if not source_context:
            return "missing_source_context"
        if not target_context:
            return "missing_target_context"
        if str(source_context) == str(target_context):
            return "same_context_marked_cross_context"
        if source_game and target_game and str(source_game) != str(target_game):
            return "cross_game_context_transfer_not_supported"
        return None
    return "unknown_transfer_kind"


def _aggregate_transfer_rows(
    rows: list[sqlite3.Row], *, key_fn: Any, labels: tuple[str, ...], include_role_fields: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(value) for value in key_fn(row))].append(row)
    result: list[dict[str, Any]] = []
    for key, items in groups.items():
        attempts = len(items)
        successes = sum(int(item["reuse_success"] or 0) == 1 for item in items)
        entry = {label: value for label, value in zip(labels, key, strict=True)}
        entry.update({"attempt_count": attempts, "success_count": successes, "success_rate": _rate(successes, attempts)})
        if include_role_fields:
            similarity = [float(item["similarity_score"] or 0.0) for item in items]
            margins = [float(item["best_margin"] or 0.0) for item in items]
            failures = [str(item["failure_reason"] or "other") for item in items if int(item["reuse_success"] or 0) != 1]
            entry["mean_similarity_score"] = sum(similarity) / attempts if attempts else 0.0
            entry["mean_best_margin"] = sum(margins) / attempts if attempts else 0.0
            entry["dominant_failure_reason"] = sorted(
                ((failures.count(reason), reason) for reason in set(failures)), key=lambda item: (-item[0], item[1])
            )[0][1] if failures else "none"
        result.append(entry)
    return sorted(result, key=lambda item: (-int(item["attempt_count"]), *(str(item[label]) for label in labels)))


def _bucket_transfer_rows(
    rows: list[sqlite3.Row], column: str, boundaries: tuple[tuple[float, str], ...],
) -> list[dict[str, Any]]:
    buckets = [{"bucket": label, "attempt_count": 0, "success_count": 0} for _limit, label in boundaries]
    for row in rows:
        value = float(row[column] or 0.0)
        for index, (limit, _label) in enumerate(boundaries):
            if value < limit:
                buckets[index]["attempt_count"] += 1
                buckets[index]["success_count"] += int(row["reuse_success"] or 0) == 1
                break
    for bucket in buckets:
        bucket["success_rate"] = _rate(int(bucket["success_count"]), int(bucket["attempt_count"]))
    return buckets


def _mean_group_attempts(groups: list[dict[str, Any]]) -> float:
    values = [int(row["attempt_count"]) for row in groups]
    return sum(values) / len(values) if values else 0.0


def _median_group_attempts(groups: list[dict[str, Any]]) -> float:
    values = [int(row["attempt_count"]) for row in groups]
    return float(median(values)) if values else 0.0


def _max_group_attempts(groups: list[dict[str, Any]]) -> int:
    return max((int(row["attempt_count"]) for row in groups), default=0)


def _h06_validity_gates(*, transfer_attempt_count: int, transfer_success_rate: float | None, successful_role_count: int, mean_best_margin: float | None, cross_game_success_count: int, cross_context_success_count: int) -> dict[str, dict[str, Any]]:
    return {
        "min_transfer_attempt_count": {"required": 20, "actual": transfer_attempt_count, "passed": transfer_attempt_count >= 20},
        "min_transfer_success_rate": {"required": 0.60, "actual": transfer_success_rate or 0.0, "passed": (transfer_success_rate or 0.0) >= 0.60},
        "min_successful_role_count": {"required": 3, "actual": successful_role_count, "passed": successful_role_count >= 3},
        "min_mean_best_margin": {"required": 0.10, "actual": mean_best_margin or 0.0, "passed": (mean_best_margin or 0.0) >= 0.10},
        "cross_scope_success": {
            "required": "cross_game_success_count >= 1 OR cross_context_success_count >= 5",
            "actual": {"cross_game_success_count": cross_game_success_count, "cross_context_success_count": cross_context_success_count},
            "passed": cross_game_success_count >= 1 or cross_context_success_count >= 5,
        },
    }


def _failed_validity_gate_messages(gates: dict[str, dict[str, Any]]) -> list[str]:
    labels = {
        "min_transfer_attempt_count": "Transfer attempts",
        "min_transfer_success_rate": "Transfer success rate",
        "min_successful_role_count": "Successful role count",
        "min_mean_best_margin": "Mean best margin",
        "cross_scope_success": "Cross-scope transfer success",
    }
    return [
        f"{labels[name]} {gate['actual']!r} is below VALID requirement {gate['required']!r}."
        for name, gate in gates.items() if not bool(gate["passed"])
    ]


def _transfer_rate_diagnosis(metrics: dict[str, Any], pairs: list[dict[str, Any]], role_types: list[dict[str, Any]]) -> dict[str, Any]:
    failures = {
        "role_mismatch": float(metrics["role_mismatch_rate"]),
        "low_similarity": float(metrics["low_similarity_rate"]),
        "insufficient_source_support": float(metrics["insufficient_source_support_rate"]),
        "no_source_profile": float(metrics["no_source_profile_rate"]),
        "other": float(metrics["other_failure_rate"]),
    }
    dominant_reason, dominant_rate = sorted(failures.items(), key=lambda item: (-item[1], item[0]))[0]
    diagnoses: list[str] = []
    if int(metrics["transfer_attempt_count"]) < 5:
        diagnoses.append("insufficient_attempts")
    if float(metrics["cross_game_success_rate"]) + 0.10 < float(metrics["cross_context_success_rate"]):
        diagnoses.append("cross_game_transfer_is_primary_bottleneck")
    if dominant_reason == "low_similarity":
        diagnoses.append("low_similarity_pairs_dominate")
    if dominant_reason == "insufficient_source_support":
        diagnoses.append("insufficient_source_support_dominates")
    pair_failures = [row for row in pairs if row["attempt_count"] >= 3 and row["success_rate"] < float(metrics["transfer_success_rate"] or 0.0)]
    role_failures = [row for row in role_types if row["attempt_count"] >= 3 and row["success_rate"] < float(metrics["transfer_success_rate"] or 0.0)]
    if pair_failures:
        diagnoses.append("specific_game_pairs_dominate_failures")
    if role_failures:
        diagnoses.append("specific_role_types_dominate_failures")
    if float(metrics["transfer_success_rate"] or 0.0) < 0.35 and not pair_failures and not role_failures:
        diagnoses.append("aggregate_rate_is_broadly_low")
    return {
        "aggregate_success_rate": float(metrics["transfer_success_rate"] or 0.0),
        "dominant_failure_reason": dominant_reason,
        "dominant_failure_rate": dominant_rate,
        "cross_context_success_rate": float(metrics["cross_context_success_rate"]),
        "cross_game_success_rate": float(metrics["cross_game_success_rate"]),
        "low_rate_concentrated_in_specific_games": bool(pair_failures),
        "low_rate_concentrated_in_specific_role_types": bool(role_failures),
        "sampling_capped": int(metrics["skipped_by_cap_count"] or 0) > 0,
        "diagnosis": diagnoses,
    }


def _write_outputs(output_dir: Path, result: dict[str, Any], *, full_game_pairs: list[dict[str, Any]] | None = None) -> None:
    (output_dir / "h06_role_transfer_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if full_game_pairs is not None:
        with (output_dir / "h06_transfer_by_game_pair.jsonl").open("w", encoding="utf-8") as handle:
            for row in full_game_pairs:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    text = (
        f"H06 decision: {result.get('decision')}\n"
        f"transfer attempts: {result.get('transfer_attempt_count')}\n"
        f"recorded transfer attempts: {result.get('recorded_transfer_attempt_count')}\n"
        f"invalid provenance attempts: {result.get('invalid_provenance_attempt_count')}\n"
        f"legacy provenance attempts: {result.get('legacy_transfer_provenance_count')}\n"
        f"total possible transfer attempts: {result.get('total_possible_transfer_attempts')}\n"
        f"sampled transfer attempts: {result.get('sampled_transfer_attempts')}\n"
        f"skipped by cap count: {result.get('skipped_by_cap_count')}\n"
        f"sampled cross-game attempts: {result.get('sampled_cross_game_attempt_count')}\n"
        f"sampled cross-context attempts: {result.get('sampled_cross_context_attempt_count')}\n"
        f"transfer sampling strategy: {result.get('transfer_sampling_strategy')}\n"
        f"successful transfers: {result.get('successful_transfer_count')}\n"
        f"transfer success rate: {result.get('transfer_success_rate')}\n"
        "\nTransfer type:\n"
        f"- cross-context: {result.get('cross_context_success_count')}/{result.get('cross_context_attempt_count')}, rate {result.get('cross_context_success_rate')}\n"
        f"- cross-game: {result.get('cross_game_success_count')}/{result.get('cross_game_attempt_count')}, rate {result.get('cross_game_success_rate')}\n"
        "\nMain failure causes:\n"
        f"- role mismatch: {result.get('role_mismatch_count')}, rate {result.get('role_mismatch_rate')}\n"
        f"- low similarity: {result.get('low_similarity_count')}, rate {result.get('low_similarity_rate')}\n"
        f"- insufficient source support: {result.get('insufficient_source_support_count')}, rate {result.get('insufficient_source_support_rate')}\n"
        f"- no source profile: {result.get('no_source_profile_count')}, rate {result.get('no_source_profile_rate')}\n"
        f"mean best margin: {result.get('mean_best_margin')}\n"
        f"mean source carrier count: {result.get('mean_source_carrier_count')}\n"
        f"candidate role count mean: {result.get('candidate_role_count_mean')}\n"
    )
    gates = result.get("h06_validity_gates", {})
    text += "\nVALID gates:\n" + "".join(
        f"- {name}: {'pass' if gate.get('passed') else 'fail'}\n"
        for name, gate in gates.items() if isinstance(gate, dict)
    )
    diagnosis = (result.get("transfer_rate_diagnosis") or {}).get("diagnosis", [])
    text += "\nDiagnosis:\n" + "".join(f"- {item}\n" for item in diagnosis)
    (output_dir / "h06_role_transfer_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h06_role_transfer.md").write_text("```\n" + text + "```\n", encoding="utf-8")
