from __future__ import annotations


def compute_planner_chain_preference_metrics(decision_rows: list[dict]) -> dict:
    rows = [dict(row or {}) for row in list(decision_rows or [])]
    selected = [dict(row.get("selected_candidate", row) or {}) for row in rows]
    total = max(1, len(selected))
    executable_chain = [
        row for row in selected
        if bool(list(row.get("candidate_step_plan", []) or [])) and float(row.get("candidate_execution_feasibility", row.get("execution_feasibility_score", 0.0)) or 0.0) >= 0.5
    ]
    shallow_targets = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"frontier_move", "target_interaction", "local_probe"}
        and not bool(list(row.get("candidate_step_plan", []) or []))
    ]
    direct_exit_without_prereq = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"unlock_then_exit", "mechanic_chain_deterministic", "mechanic_chain_llm"}
        and not bool(list(row.get("prerequisite_chain", []) or []))
    ]
    trigger_only = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"unlock_trigger", "trigger_probe"}
        and not bool(row.get("target_exit_id"))
    ]
    completed_chain_before_exit = [
        row for row in selected
        if bool(row.get("selected_subgoal_chain_id")) and str(row.get("selected_subgoal_step_kind") or "") == "attempt_exit"
    ]
    planner_usable = [
        row for row in selected
        if float(dict(row.get("score_breakdown", {}) or {}).get("planner_usable_hypothesis_bonus", 0.0) or 0.0) > 0.0
    ]
    return {
        "executable_chain_selection_rate": float(len(executable_chain)) / float(total),
        "shallow_target_selection_rate": float(len(shallow_targets)) / float(total),
        "direct_exit_attempt_rate_without_prerequisites": float(len(direct_exit_without_prereq)) / float(total),
        "trigger_only_candidate_selection_rate": float(len(trigger_only)) / float(total),
        "completed_chain_before_exit_success_rate": float(len(completed_chain_before_exit)) / float(total),
        "planner_usable_hypothesis_utilization_rate": float(len(planner_usable)) / float(total),
    }
