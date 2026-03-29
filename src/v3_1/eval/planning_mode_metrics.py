from __future__ import annotations


def compute_planning_mode_metrics(decision_rows: list[dict]) -> dict:
    rows = [dict(row) for row in list(decision_rows or []) if isinstance(row, dict)]
    total = max(1, len(rows))
    structure_rows = [row for row in rows if str(row.get("planning_mode") or "") == "structure_acquisition"]
    default_rows = [row for row in rows if str(row.get("planning_mode") or "") == "default_progress"]
    premature_regressions = [
        row for row in rows
        if str(row.get("planning_mode") or "") == "default_progress"
        and str(row.get("previous_planning_mode") or "") == "structure_acquisition"
        and float(row.get("structure_acquisition_score", 0.0) or 0.0) >= float(row.get("default_progress_score", 0.0) or 0.0)
    ]
    consecutive_structure = 0
    current = 0
    for row in rows:
        if str(row.get("planning_mode") or "") == "structure_acquisition":
            current += 1
            consecutive_structure = max(consecutive_structure, current)
        else:
            current = 0
    previous_mode_match_rows = [
        row for row in rows
        if int(row.get("round_id", 0) or 0) > 1
    ]
    both_match_count = sum(
        1
        for row in previous_mode_match_rows
        if bool(row.get("trace_previous_mode_matches_committed", False))
        and bool(row.get("ledger_previous_mode_matches_committed", False))
    )
    trace_repaired_count = sum(1 for row in rows if bool(row.get("mode_trace_correction_applied", False)))
    ledger_repaired_count = sum(
        1
        for row in previous_mode_match_rows
        if not bool(row.get("ledger_previous_mode_matches_committed", False))
    )
    return {
        "structure_acquisition_round_rate": float(len(structure_rows)) / float(total),
        "default_progress_round_rate": float(len(default_rows)) / float(total),
        "premature_mode_regression_rate": float(len(premature_regressions)) / float(total),
        "consecutive_structure_rounds": consecutive_structure,
        "mode_switch_count": sum(1 for row in rows if bool(row.get("mode_switch_applied", False))),
        "verification_candidate_selection_rate_by_mode": {
            "structure_acquisition": float(sum(1 for row in structure_rows if str(row.get("selected_objective_type") or "") in {"verify_trigger_contact", "reobserve_remote_change", "verify_panel_state", "verify_gate_match", "trigger_then_target"})) / float(max(1, len(structure_rows))),
            "default_progress": float(sum(1 for row in default_rows if str(row.get("selected_objective_type") or "") in {"verify_trigger_contact", "reobserve_remote_change", "verify_panel_state", "verify_gate_match", "trigger_then_target"})) / float(max(1, len(default_rows))),
        },
        "progress_candidate_selection_rate_by_mode": {
            "structure_acquisition": float(sum(1 for row in structure_rows if str(row.get("selected_objective_type") or "") in {"probe_route", "explore_frontier", "interact"})) / float(max(1, len(structure_rows))),
            "default_progress": float(sum(1 for row in default_rows if str(row.get("selected_objective_type") or "") in {"probe_route", "explore_frontier", "interact"})) / float(max(1, len(default_rows))),
        },
        "structure_support_gain_by_mode": {
            "structure_acquisition": float(sum(float(row.get("recent_structure_support_gain", 0.0) or 0.0) for row in structure_rows)) / float(max(1, len(structure_rows))),
            "default_progress": float(sum(float(row.get("recent_structure_support_gain", 0.0) or 0.0) for row in default_rows)) / float(max(1, len(default_rows))),
        },
        "trace_and_ledger_both_match_prior_committed_rate": float(both_match_count) / float(max(1, len(previous_mode_match_rows))),
        "trace_repaired_before_persist_count": trace_repaired_count,
        "ledger_repaired_before_append_count": ledger_repaired_count,
    }
