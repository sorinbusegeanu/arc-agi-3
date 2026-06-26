from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def evaluate_h07_concept_emergence(
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
        transfer_attempt_count = int(conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0])
        successful_transfers = int(conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1").fetchone()[0])
        concept_rows = conn.execute(
            """
            SELECT concept_signature, compression_gain, promotion_score, transfer_success_count,
                   cross_context_count, cross_game_count, is_promoted
            FROM concept_candidates
            ORDER BY concept_signature ASC
            """
        ).fetchall()
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
    concept_candidate_count = len(concept_rows)
    promoted_concept_count = sum(1 for row in concept_rows if int(row["is_promoted"] or 0) == 1)
    mean_compression_gain = (
        sum(float(row["compression_gain"] or 0.0) for row in concept_rows) / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_compression_gain = max((float(row["compression_gain"] or 0.0) for row in concept_rows), default=None)
    mean_promotion_score = (
        sum(float(row["promotion_score"] or 0.0) for row in concept_rows) / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_promotion_score = max((float(row["promotion_score"] or 0.0) for row in concept_rows), default=None)
    cross_context_concept_count = sum(1 for row in concept_rows if int(row["cross_context_count"] or 0) >= 1)
    cross_game_concept_count = sum(1 for row in concept_rows if int(row["cross_game_count"] or 0) >= 1)
    concept_transfer_success_count = successful_transfers
    metrics = {
        "concept_candidate_count": concept_candidate_count,
        "promoted_concept_count": promoted_concept_count,
        "mean_compression_gain": mean_compression_gain,
        "max_compression_gain": max_compression_gain,
        "mean_promotion_score": mean_promotion_score,
        "max_promotion_score": max_promotion_score,
        "concept_transfer_success_count": concept_transfer_success_count,
        "cross_context_concept_count": cross_context_concept_count,
        "cross_game_concept_count": cross_game_concept_count,
        "first_concept_candidate_step": milestone_map.get("first_concept_candidate_step"),
        "first_promoted_concept_step": milestone_map.get("first_promoted_concept_step"),
        "first_role_transfer_success_step": milestone_map.get("first_role_transfer_success_step"),
    }
    if transfer_attempt_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no role transfer attempts available"]
    elif successful_transfers > 0 and concept_candidate_count == 0:
        decision = "INVALID"
        missing = []
    elif promoted_concept_count >= 1 and concept_transfer_success_count >= 2 and (max_compression_gain or 0.0) >= 1.50 and (max_promotion_score or 0.0) >= 0.55 and (cross_context_concept_count >= 1 or cross_game_concept_count >= 1):
        decision = "VALID"
        missing = []
    elif concept_candidate_count > 0 and promoted_concept_count == 0:
        decision = "PARTIALLY_VALID"
        missing = []
    elif transfer_attempt_count > 0 and promoted_concept_count == 0:
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
        "hypothesis_id": "H07",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h07_concept_emergence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H07 decision: {result.get('decision')}\n"
        f"concept candidates: {result.get('concept_candidate_count')}\n"
        f"promoted concepts: {result.get('promoted_concept_count')}\n"
        f"max compression gain: {result.get('max_compression_gain')}\n"
    )
    (output_dir / "h07_concept_emergence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h07_concept_emergence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
