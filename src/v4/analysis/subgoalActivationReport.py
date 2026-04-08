from __future__ import annotations


def build_subgoal_activation_report(
    step8_trace_rows: tuple[dict[str, object], ...],
    success: bool,
    game_id: str,
) -> dict[str, object]:
    total_steps = len(step8_trace_rows)
    counts: dict[str, int] = {}
    step8_kinds = {
        "enable_construction_path",
        "manage_construction_budget",
        "build_under_time_pressure",
        "complete_construction_path",
    }
    for row in step8_trace_rows:
        selected = row.get("selected_subgoal_kind")
        if selected is None:
            continue
        counts[str(selected)] = counts.get(str(selected), 0) + 1
    return {
        "game_id": game_id,
        "success": bool(success),
        "total_steps": total_steps,
        "selected_subgoal_kind_counts": counts,
        "selected_subgoal_non_immediate_count": sum(
            1
            for row in step8_trace_rows
            if row.get("selected_subgoal_kind") is not None and row.get("selected_subgoal_kind") != "immediate_progress"
        ),
        "extracted_step6_subgoal_count": sum(
            1 for row in step8_trace_rows if "disambiguate_hypothesis" in tuple(row.get("extracted_subgoal_kinds", ()))
        ),
        "extracted_step7_subgoal_count": sum(
            1 for row in step8_trace_rows if "preserve_safety_margin" in tuple(row.get("extracted_subgoal_kinds", ()))
        ),
        "extracted_step8_subgoal_count": sum(
            1 for row in step8_trace_rows if any(kind in step8_kinds for kind in tuple(row.get("extracted_subgoal_kinds", ())))
        ),
        "selected_step6_subgoal_count": sum(1 for row in step8_trace_rows if row.get("selected_subgoal_kind") == "disambiguate_hypothesis"),
        "selected_step7_subgoal_count": sum(1 for row in step8_trace_rows if row.get("selected_subgoal_kind") == "preserve_safety_margin"),
        "selected_step8_subgoal_count": sum(1 for row in step8_trace_rows if row.get("selected_subgoal_kind") in step8_kinds),
        "failed_last_5_selected_subgoal_kinds": (
            [row.get("selected_subgoal_kind") for row in step8_trace_rows[-5:]]
            if not success
            else []
        ),
        "failed_last_5_extracted_subgoal_kinds": (
            [tuple(row.get("extracted_subgoal_kinds", ())) for row in step8_trace_rows[-5:]]
            if not success
            else []
        ),
        "failed_last_5_subgoal_progress_rows": (
            [tuple(row.get("subgoal_progress_rows", ())) for row in step8_trace_rows[-5:]]
            if not success
            else []
        ),
    }
