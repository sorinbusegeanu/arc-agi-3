from __future__ import annotations


def build_step8_trace_report(
    step8_trace_rows: tuple[dict[str, object], ...],
    success: bool,
    game_id: str,
) -> dict[str, object]:
    total_steps = len(step8_trace_rows)
    last_row = step8_trace_rows[-1] if step8_trace_rows else {}
    selected_goal_kind_non_immediate_count = sum(
        1
        for row in step8_trace_rows
        if row.get("selected_goal_kind") is not None and row.get("selected_goal_kind") != "immediate_progress"
    )
    changed_cells_values = tuple(row.get("changed_cells") for row in step8_trace_rows if row.get("changed_cells") is not None)
    return {
        "game_id": game_id,
        "success": bool(success),
        "total_steps": total_steps,
        "trace_row_count": total_steps,
        "ended_without_trace_rows": total_steps == 0,
        "last_executed_action_name": last_row.get("executed_action_name") if step8_trace_rows else None,
        "last_selected_goal_kind": last_row.get("selected_goal_kind") if step8_trace_rows else None,
        "last_selected_subgoal_kind": last_row.get("selected_subgoal_kind") if step8_trace_rows else None,
        "changed_cells_missing_despite_observation_change": False,
        "first_failure_step_index": None,
        "first_failure_bucket": None,
        "first_failure_stop_reason": None,
        "first_failure_abort_site": None,
        "first_failure_abort_message": None,
        "first_failure_missing_field": None,
        "first_failure_required_fields": "",
        "first_failure_current_visible_fields": "",
        "first_failure_previous_state_available": None,
        "first_failure_reconstruction_attempted": None,
        "first_failure_parsed_state_summary": {},
        "first_failure_decision_summary": {},
        "first_failure_step_result_summary": {},
        "selected_goal_kind_non_immediate_count": selected_goal_kind_non_immediate_count,
        "selected_goal_kind_non_immediate_rate": (
            float(selected_goal_kind_non_immediate_count) / float(total_steps) if total_steps > 0 else 0.0
        ),
        "changed_cells_field_present_count": len(changed_cells_values),
        "positive_changed_cells_count": sum(1 for value in changed_cells_values if int(value) > 0),
        "changed_cells_total": sum(int(value) for value in changed_cells_values),
        "generated_step6_total": sum(int(row.get("generated_step6_count", 0)) for row in step8_trace_rows),
        "generated_step7_total": sum(int(row.get("generated_step7_count", 0)) for row in step8_trace_rows),
        "generated_step8_total": sum(int(row.get("generated_step8_count", 0)) for row in step8_trace_rows),
        "accepted_step6_total": sum(int(row.get("accepted_step6_count", 0)) for row in step8_trace_rows),
        "accepted_step7_total": sum(int(row.get("accepted_step7_count", 0)) for row in step8_trace_rows),
        "accepted_step8_total": sum(int(row.get("accepted_step8_count", 0)) for row in step8_trace_rows),
        "selected_step6_total": sum(1 for row in step8_trace_rows if bool(row.get("selected_is_step6", False))),
        "selected_step7_total": sum(1 for row in step8_trace_rows if bool(row.get("selected_is_step7", False))),
        "selected_step8_total": sum(1 for row in step8_trace_rows if bool(row.get("selected_is_step8", False))),
        "failed_last_5_goal_kinds": (
            [row.get("selected_goal_kind") for row in step8_trace_rows[-5:]]
            if not success
            else []
        ),
    }
