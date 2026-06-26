from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def evaluate_h06_role_transfer(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    del run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        result = _base_result("INCONCLUSIVE", ["compact memory missing current_state.sqlite"])
        _write_outputs(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        role_candidate_count = int(conn.execute("SELECT COUNT(*) FROM role_candidates").fetchone()[0])
        rows = conn.execute(
            """
            SELECT role_signature, transfer_kind, similarity_score, transfer_score, reuse_success
            FROM role_transfer_attempts
            ORDER BY role_signature ASC, target_scope_key ASC, target_carrier_signature ASC
            """
        ).fetchall()
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
    success_rows = [row for row in rows if int(row["reuse_success"] or 0) == 1]
    cross_game_rows = [row for row in rows if str(row["transfer_kind"]) == "cross_game"]
    cross_context_rows = [row for row in rows if str(row["transfer_kind"]) == "cross_context"]
    cross_game_success_count = sum(1 for row in cross_game_rows if int(row["reuse_success"] or 0) == 1)
    cross_context_success_count = sum(1 for row in cross_context_rows if int(row["reuse_success"] or 0) == 1)
    transfer_attempt_count = len(rows)
    successful_transfer_count = len(success_rows)
    transfer_success_rate = float(successful_transfer_count / transfer_attempt_count) if transfer_attempt_count else None
    successful_role_count = len({str(row["role_signature"]) for row in success_rows})
    mean_transfer_score = (
        sum(float(row["transfer_score"] or 0.0) for row in rows) / max(1, transfer_attempt_count)
        if rows
        else None
    )
    max_transfer_score = max((float(row["transfer_score"] or 0.0) for row in rows), default=None)
    metrics = {
        "transfer_attempt_count": transfer_attempt_count,
        "successful_transfer_count": successful_transfer_count,
        "transfer_success_rate": transfer_success_rate,
        "cross_game_attempt_count": len(cross_game_rows),
        "cross_game_success_count": cross_game_success_count,
        "cross_context_attempt_count": len(cross_context_rows),
        "cross_context_success_count": cross_context_success_count,
        "successful_role_count": successful_role_count,
        "mean_transfer_score": mean_transfer_score,
        "max_transfer_score": max_transfer_score,
        "first_role_candidate_step": milestone_map.get("first_role_candidate_step"),
        "first_role_transfer_attempt_step": milestone_map.get("first_role_transfer_attempt_step"),
        "first_role_transfer_success_step": milestone_map.get("first_role_transfer_success_step"),
    }
    if role_candidate_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no role candidates available"]
    elif transfer_attempt_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no role transfer attempts available"]
    elif transfer_attempt_count >= 20 and (transfer_success_rate or 0.0) >= 0.60 and successful_role_count >= 3 and (cross_game_success_count >= 1 or cross_context_success_count >= 5):
        decision = "VALID"
        missing = []
    elif transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) >= 0.35:
        decision = "PARTIALLY_VALID"
        missing = []
    elif role_candidate_count > 0 and transfer_attempt_count >= 5 and (transfer_success_rate or 0.0) < 0.20:
        decision = "INVALID"
        missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []
    result = _base_result(decision, missing)
    result.update(metrics)
    result["core_metrics"] = dict(metrics)
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
        f"successful transfers: {result.get('successful_transfer_count')}\n"
        f"transfer success rate: {result.get('transfer_success_rate')}\n"
    )
    (output_dir / "h06_role_transfer_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h06_role_transfer.md").write_text("```\n" + text + "```\n", encoding="utf-8")
