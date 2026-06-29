from __future__ import annotations

import json
import sqlite3
from pathlib import Path
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
            SELECT role_signature, transfer_kind, similarity_score, transfer_score, reuse_success,
                   failure_reason, best_margin, source_carrier_count, candidate_role_count
            FROM role_transfer_attempts
            ORDER BY role_signature ASC, target_scope_key ASC, target_carrier_signature ASC
            """
        ).fetchall()
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        transfer_summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'higher_order_transfer_summary'"
        ).fetchone()
        transfer_summary = json.loads(str(transfer_summary_row[0])) if transfer_summary_row and transfer_summary_row[0] else {}
    success_rows = [row for row in rows if int(row["reuse_success"] or 0) == 1]
    cross_game_rows = [row for row in rows if str(row["transfer_kind"]) == "cross_game"]
    cross_context_rows = [row for row in rows if str(row["transfer_kind"]) == "cross_context"]
    cross_game_success_count = sum(1 for row in cross_game_rows if int(row["reuse_success"] or 0) == 1)
    cross_context_success_count = sum(1 for row in cross_context_rows if int(row["reuse_success"] or 0) == 1)
    transfer_attempt_count = len(rows)
    successful_transfer_count = len(success_rows)
    transfer_success_rate = float(successful_transfer_count / transfer_attempt_count) if transfer_attempt_count else None
    successful_role_count = len({str(row["role_signature"]) for row in success_rows})
    role_mismatch_count = sum(1 for row in rows if str(row["failure_reason"] or "") == "role_mismatch")
    low_similarity_count = sum(1 for row in rows if str(row["failure_reason"] or "") == "low_similarity")
    insufficient_source_support_count = sum(1 for row in rows if str(row["failure_reason"] or "") == "insufficient_source_support")
    no_source_profile_count = sum(1 for row in rows if str(row["failure_reason"] or "") == "no_source_profile")
    mean_transfer_score = (
        sum(float(row["transfer_score"] or 0.0) for row in rows) / max(1, transfer_attempt_count)
        if rows
        else None
    )
    max_transfer_score = max((float(row["transfer_score"] or 0.0) for row in rows), default=None)
    margins = [float(row["best_margin"]) for row in rows if row["best_margin"] is not None]
    source_counts = [int(row["source_carrier_count"] or 0) for row in rows]
    candidate_role_counts = [int(row["candidate_role_count"] or 0) for row in rows]
    metrics = {
        "transfer_attempt_count": transfer_attempt_count,
        "total_possible_transfer_attempts": transfer_summary.get("total_possible_transfer_attempts", len(rows)),
        "sampled_transfer_attempts": transfer_summary.get("sampled_transfer_attempts", transfer_attempt_count),
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
        "cross_context_attempt_count": len(cross_context_rows),
        "cross_context_success_count": cross_context_success_count,
        "successful_role_count": successful_role_count,
        "role_mismatch_count": role_mismatch_count,
        "low_similarity_count": low_similarity_count,
        "insufficient_source_support_count": insufficient_source_support_count,
        "no_source_profile_count": no_source_profile_count,
        "mean_transfer_score": mean_transfer_score,
        "max_transfer_score": max_transfer_score,
        "mean_best_margin": (sum(margins) / len(margins)) if margins else None,
        "mean_source_carrier_count": (sum(source_counts) / len(source_counts)) if source_counts else None,
        "candidate_role_count_mean": (sum(candidate_role_counts) / len(candidate_role_counts)) if candidate_role_counts else None,
        "first_role_candidate_step": milestone_map.get("first_role_candidate_step"),
        "first_role_transfer_attempt_step": milestone_map.get("first_role_transfer_attempt_step"),
        "first_role_transfer_success_step": milestone_map.get("first_role_transfer_success_step"),
    }
    if role_candidate_count <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no role candidates available"]
    elif transfer_attempt_count <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no role transfer attempts available"]
    elif (
        transfer_attempt_count >= 20
        and (transfer_success_rate or 0.0) >= 0.60
        and successful_role_count >= 3
        and (metrics["mean_best_margin"] or 0.0) >= 0.10
        and (cross_game_success_count >= 1 or cross_context_success_count >= 5)
    ):
        decision = "VALID"
        missing = []
    elif transfer_attempt_count < 5:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["too few role transfer attempts for H06 evaluation"]
    elif transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) >= 0.35 and successful_role_count >= 1:
        decision = "PARTIALLY_VALID"
        missing = []
    elif role_candidate_count > 0 and transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) < 0.20:
        decision = "INVALID"
        missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []
    result = _base_result(decision, missing)
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
    result["core_metrics"] = dict(metrics)
    result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target="none")
    _write_outputs(output_dir, result)
    return result


def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H06",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h06_role_transfer_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H06 decision: {result.get('decision')}\n"
        f"transfer attempts: {result.get('transfer_attempt_count')}\n"
        f"total possible transfer attempts: {result.get('total_possible_transfer_attempts')}\n"
        f"sampled transfer attempts: {result.get('sampled_transfer_attempts')}\n"
        f"skipped by cap count: {result.get('skipped_by_cap_count')}\n"
        f"sampled cross-game attempts: {result.get('sampled_cross_game_attempt_count')}\n"
        f"sampled cross-context attempts: {result.get('sampled_cross_context_attempt_count')}\n"
        f"transfer sampling strategy: {result.get('transfer_sampling_strategy')}\n"
        f"successful transfers: {result.get('successful_transfer_count')}\n"
        f"transfer success rate: {result.get('transfer_success_rate')}\n"
        f"role mismatch count: {result.get('role_mismatch_count')}\n"
        f"low similarity count: {result.get('low_similarity_count')}\n"
        f"insufficient source support count: {result.get('insufficient_source_support_count')}\n"
        f"no source profile count: {result.get('no_source_profile_count')}\n"
        f"mean best margin: {result.get('mean_best_margin')}\n"
        f"mean source carrier count: {result.get('mean_source_carrier_count')}\n"
        f"candidate role count mean: {result.get('candidate_role_count_mean')}\n"
    )
    (output_dir / "h06_role_transfer_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h06_role_transfer.md").write_text("```\n" + text + "```\n", encoding="utf-8")
