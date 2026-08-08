from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import derive_role_candidates_only
from v6.memory.compact_memory import ensure_memory_layout

MAX_CARRIERS_PER_ROLE_FOR_VALIDITY = 100
MAX_SINGLETON_ROLE_RATIO_FOR_VALID = 0.50
MAX_OVERCONNECTED_ROLE_RATIO_FOR_VALID = 0.10
MAX_CARRIERS_PER_ROLE_HARD_INVALID = 4 * MAX_CARRIERS_PER_ROLE_FOR_VALIDITY

def _resolve_temporal_value(primary: Any, fallback: Any) -> tuple[int | None, str]:
    if primary is not None:
        return int(primary), "temporal_milestones"
    if fallback is not None:
        return int(fallback), "compact_table_fallback"
    return None, "missing"


def evaluate_h05_role_emergence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not already_derived:
        ensure_memory_layout(memory_dir)
        derive_role_candidates_only(memory_dir=memory_dir, run_dir=run_dir)
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
        role_link_rows = conn.execute(
            """
            SELECT rl.role_signature, cc.carrier_timing_source
            FROM role_links rl
            LEFT JOIN carrier_candidates cc
              ON cc.carrier_signature = rl.linked_key
            WHERE rl.linked_type = 'carrier'
            ORDER BY rl.role_signature ASC
            """
        ).fetchall()
        carrier_count = int(conn.execute("SELECT COUNT(*) FROM carrier_candidates").fetchone()[0])
        emergent_carrier_count = int(conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0])
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        milestone_h04_first = conn.execute("SELECT MIN(first_emergent_carrier_step) FROM temporal_milestones").fetchone()[0]
        fallback_h04_first = conn.execute(
            "SELECT MIN(first_seen_global_step) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1"
        ).fetchone()[0]
        fallback_role_candidate_step = conn.execute(
            "SELECT MIN(first_seen_global_step) FROM role_candidates"
        ).fetchone()[0]
        fallback_emergent_role_step = conn.execute(
            "SELECT MIN(first_seen_global_step) FROM role_candidates WHERE COALESCE(is_emergent, 0) = 1"
        ).fetchone()[0]
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
    overconnected_role_count = sum(1 for row in role_rows if int(row["linked_carrier_count"] or 0) > MAX_CARRIERS_PER_ROLE_FOR_VALIDITY)
    usable_role_rows = [row for row in role_rows if int(row["linked_carrier_count"] or 0) <= MAX_CARRIERS_PER_ROLE_FOR_VALIDITY]
    usable_emergent_role_rows = [row for row in usable_role_rows if int(row["is_emergent"] or 0) == 1]
    usable_role_count = len(usable_role_rows)
    usable_emergent_role_count = len(usable_emergent_role_rows)
    usable_multi_carrier_role_count = sum(1 for row in usable_role_rows if int(row["linked_carrier_count"] or 0) >= 2)
    usable_cross_context_role_count = sum(1 for row in usable_role_rows if int(row["cross_context_count"] or 0) >= 2)
    usable_cross_game_role_count = sum(1 for row in usable_role_rows if int(row["cross_game_count"] or 0) >= 2)
    h04_first, first_emergent_carrier_step_source = _resolve_temporal_value(milestone_h04_first, fallback_h04_first)
    first_role_candidate_step, first_role_candidate_step_source = _resolve_temporal_value(
        milestone_map.get("first_role_candidate_step"),
        fallback_role_candidate_step,
    )
    first_emergent_role_step, first_emergent_role_step_source = _resolve_temporal_value(
        milestone_map.get("first_emergent_role_step"),
        fallback_emergent_role_step,
    )
    temporal_sources = {
        first_emergent_carrier_step_source,
        first_role_candidate_step_source,
        first_emergent_role_step_source,
    }
    if "temporal_milestones" in temporal_sources:
        h05_temporal_source = "temporal_milestones"
    elif "compact_table_fallback" in temporal_sources:
        h05_temporal_source = "compact_table_fallback"
    else:
        h05_temporal_source = "missing"
    carrier_sources_by_role: dict[str, list[str]] = {}
    for row in role_link_rows:
        source = str(row["carrier_timing_source"] or "unknown")
        if source not in {"real_evidence", "fold_start_fallback", "mixed", "unknown"}:
            source = "unknown"
        carrier_sources_by_role.setdefault(str(row["role_signature"]), []).append(source)
    role_source_counts = {"real_evidence": 0, "fold_start_fallback": 0, "mixed": 0, "unknown": 0}
    considered_role_sources: set[str] = set()
    for row in usable_emergent_role_rows:
        if int(row["is_emergent"] or 0) != 1:
            continue
        sources = carrier_sources_by_role.get(str(row["role_signature"]), [])
        if not sources:
            role_source_counts["unknown"] += 1
            considered_role_sources.add("unknown")
            continue
        if all(source == "real_evidence" for source in sources):
            source = "real_evidence"
        elif any(source == "real_evidence" for source in sources):
            source = "mixed"
        elif all(source == "fold_start_fallback" for source in sources):
            source = "fold_start_fallback"
        elif any(source == "mixed" for source in sources) or len(set(sources)) > 1:
            source = "mixed"
        else:
            source = "unknown"
        role_source_counts[source] += 1
        considered_role_sources.add(source)
    if considered_role_sources == {"real_evidence"}:
        role_timing_source = "real_evidence"
    elif "real_evidence" in considered_role_sources:
        role_timing_source = "mixed"
    elif considered_role_sources == {"fold_start_fallback"}:
        role_timing_source = "fold_start_fallback"
    elif considered_role_sources:
        role_timing_source = "mixed" if "mixed" in considered_role_sources or len(considered_role_sources) > 1 else "unknown"
    else:
        role_timing_source = "unknown"
    h04_before_h05 = (
        None
        if h04_first is None or first_emergent_role_step is None
        else int(h04_first) <= int(first_emergent_role_step)
    )
    h04_before_h05_cases = (
        0
        if h04_first is None or first_emergent_role_step is None or int(h04_first) > int(first_emergent_role_step)
        else 1
    )
    role_count = max(1, int(role_candidate_count or 0))
    overconnected_role_ratio = float(overconnected_role_count) / role_count
    h05_role_quality_pass = (
        (singleton_role_ratio is None or singleton_role_ratio <= MAX_SINGLETON_ROLE_RATIO_FOR_VALID)
        and overconnected_role_ratio <= MAX_OVERCONNECTED_ROLE_RATIO_FOR_VALID
        and max_carriers_per_role <= MAX_CARRIERS_PER_ROLE_HARD_INVALID
    )
    metrics = {
        "role_candidate_count": role_candidate_count,
        "emergent_role_count": emergent_role_count,
        "stable_role_count": stable_role_count,
        "overconnected_role_count": overconnected_role_count,
        "usable_role_count": usable_role_count,
        "usable_emergent_role_count": usable_emergent_role_count,
        "singleton_role_count": singleton_role_count,
        "singleton_role_ratio": singleton_role_ratio,
        "multi_carrier_role_count": multi_carrier_role_count,
        "cross_context_role_count": cross_context_role_count,
        "cross_game_role_count": cross_game_role_count,
        "mean_carriers_per_role": mean_carriers_per_role,
        "max_carriers_per_role": max_carriers_per_role,
        "overconnected_role_ratio": overconnected_role_ratio,
        "h05_role_quality_pass": h05_role_quality_pass,
        "first_emergent_carrier_step": h04_first,
        "first_role_candidate_step": first_role_candidate_step,
        "first_emergent_role_step": first_emergent_role_step,
        "h05_temporal_source": h05_temporal_source,
        "first_emergent_carrier_step_source": first_emergent_carrier_step_source,
        "first_role_candidate_step_source": first_role_candidate_step_source,
        "first_emergent_role_step_source": first_emergent_role_step_source,
        "h04_before_h05_cases": h04_before_h05_cases,
        "h04_before_h05": h04_before_h05,
        "temporal_order_required_for_valid": True,
        "role_timing_source": role_timing_source,
        "role_real_timing_count": role_source_counts["real_evidence"],
        "role_fallback_timing_count": role_source_counts["fold_start_fallback"],
        "role_mixed_timing_count": role_source_counts["mixed"],
        "role_unknown_timing_count": role_source_counts["unknown"],
    }
    if carrier_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no carrier candidates available"]
    elif emergent_carrier_count > 0 and role_candidate_count == 0:
        decision = "INVALID"
        missing = ["emergent carriers present but no role candidates"]
    elif (
        usable_emergent_role_count >= 1
        and usable_multi_carrier_role_count >= 1
        and (usable_cross_context_role_count >= 1 or usable_cross_game_role_count >= 1)
        and (singleton_role_ratio is None or singleton_role_ratio <= 0.75)
        and h04_before_h05 is False
        and role_timing_source == "real_evidence"
    ):
        decision = "INVALID"
        missing = []
    elif usable_emergent_role_count >= 1 and usable_multi_carrier_role_count >= 1 and (usable_cross_context_role_count >= 1 or usable_cross_game_role_count >= 1) and (singleton_role_ratio is None or singleton_role_ratio <= 0.75) and (
        h04_before_h05 is True
    ) and role_timing_source == "real_evidence" and h05_role_quality_pass is True:
        decision = "VALID"
        missing = []
    elif (
        usable_emergent_role_count >= 1
        and usable_multi_carrier_role_count >= 1
        and (usable_cross_context_role_count >= 1 or usable_cross_game_role_count >= 1)
        and (singleton_role_ratio is None or singleton_role_ratio <= 0.75)
        and h04_before_h05 is None
    ):
        decision = "PARTIALLY_VALID"
        missing = ["explicit H04-before-H05 temporal evidence unavailable"]
    elif role_candidate_count > 0 and not (singleton_role_count == role_candidate_count and emergent_carrier_count >= 5):
        decision = "PARTIALLY_VALID"
        missing = []
    else:
        decision = "INVALID"
        missing = []
    result = _base_result(decision, missing)
    result.update(metrics)
    if role_timing_source != "real_evidence":
        result["missing_evidence"] = list(
            dict.fromkeys(
                list(result.get("missing_evidence", []))
                + ["Role timing is not fully grounded in real carrier evidence timing."]
            )
        )
    if result["decision"] == "VALID" and role_timing_source != "real_evidence":
        result["decision"] = "PARTIALLY_VALID"
    elif result["decision"] == "INVALID" and role_timing_source != "real_evidence" and h04_before_h05 is False:
        result["decision"] = "PARTIALLY_VALID"
    if h05_role_quality_pass is not True and result["decision"] in {"VALID", "PARTIALLY_VALID"}:
        if result["decision"] == "VALID":
            result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = list(
            dict.fromkeys(
                list(result.get("missing_evidence", []))
                + ["H05 role graph is noisy or overconnected; remapping quality prevents robust VALID classification."]
            )
        )
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
