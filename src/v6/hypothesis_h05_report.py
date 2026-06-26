from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def evaluate_h05_role_emergence(
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
        role_rows = conn.execute(
            """
            SELECT role_signature, linked_carrier_count, cross_context_count, cross_game_count, role_stability_score, is_emergent
            FROM role_candidates
            ORDER BY role_signature ASC
            """
        ).fetchall()
        carrier_count = int(conn.execute("SELECT COUNT(*) FROM carrier_candidates").fetchone()[0])
        emergent_carrier_count = int(conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0])
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        h04_first = conn.execute("SELECT MIN(first_emergent_carrier_step) FROM temporal_milestones").fetchone()[0]
    role_candidate_count = len(role_rows)
    emergent_role_count = sum(1 for row in role_rows if int(row["is_emergent"] or 0) == 1)
    stable_role_count = sum(1 for row in role_rows if float(row["role_stability_score"] or 0.0) >= 0.50)
    singleton_role_count = sum(1 for row in role_rows if int(row["linked_carrier_count"] or 0) <= 1)
    multi_carrier_role_count = sum(1 for row in role_rows if int(row["linked_carrier_count"] or 0) >= 2)
    cross_context_role_count = sum(1 for row in role_rows if int(row["cross_context_count"] or 0) >= 2)
    cross_game_role_count = sum(1 for row in role_rows if int(row["cross_game_count"] or 0) >= 2)
    mean_carriers_per_role = (
        sum(int(row["linked_carrier_count"] or 0) for row in role_rows) / max(1, role_candidate_count)
        if role_rows
        else None
    )
    max_carriers_per_role = max((int(row["linked_carrier_count"] or 0) for row in role_rows), default=0)
    singleton_role_ratio = float(singleton_role_count / role_candidate_count) if role_candidate_count else None
    first_role_candidate_step = milestone_map.get("first_role_candidate_step")
    first_emergent_role_step = milestone_map.get("first_emergent_role_step")
    h04_before_h05_cases = (
        0
        if h04_first is None or first_emergent_role_step is None or int(h04_first) > int(first_emergent_role_step)
        else 1
    )
    metrics = {
        "role_candidate_count": role_candidate_count,
        "emergent_role_count": emergent_role_count,
        "stable_role_count": stable_role_count,
        "singleton_role_count": singleton_role_count,
        "singleton_role_ratio": singleton_role_ratio,
        "multi_carrier_role_count": multi_carrier_role_count,
        "cross_context_role_count": cross_context_role_count,
        "cross_game_role_count": cross_game_role_count,
        "mean_carriers_per_role": mean_carriers_per_role,
        "max_carriers_per_role": max_carriers_per_role,
        "first_emergent_carrier_step": h04_first,
        "first_role_candidate_step": first_role_candidate_step,
        "first_emergent_role_step": first_emergent_role_step,
        "h04_before_h05_cases": h04_before_h05_cases,
    }
    if carrier_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no carrier candidates available"]
    elif emergent_carrier_count > 0 and role_candidate_count == 0:
        decision = "INVALID"
        missing = ["emergent carriers present but no role candidates"]
    elif emergent_role_count >= 1 and multi_carrier_role_count >= 1 and (cross_context_role_count >= 1 or cross_game_role_count >= 1) and (singleton_role_ratio is None or singleton_role_ratio <= 0.75) and (
        h04_first is None or first_emergent_role_step is None or int(h04_first) <= int(first_emergent_role_step)
    ):
        decision = "VALID"
        missing = []
    elif role_candidate_count > 0 and not (singleton_role_count == role_candidate_count and emergent_carrier_count >= 5):
        decision = "PARTIALLY_VALID"
        missing = []
    else:
        decision = "INVALID"
        missing = []
    result = _base_result(decision, missing)
    result.update(metrics)
    result["core_metrics"] = dict(metrics)
    _write_outputs(output_dir, result)
    return result


def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H05",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h05_functional_role_emergence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H05 decision: {result.get('decision')}\n"
        f"role candidates: {result.get('role_candidate_count')}\n"
        f"emergent roles: {result.get('emergent_role_count')}\n"
        f"singleton role ratio: {result.get('singleton_role_ratio')}\n"
    )
    (output_dir / "h05_functional_role_emergence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h05_functional_role_emergence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
