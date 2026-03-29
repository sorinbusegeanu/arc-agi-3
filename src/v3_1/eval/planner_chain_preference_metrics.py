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
    premature_exit = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"unlock_then_exit", "mechanic_chain_deterministic", "mechanic_chain_llm"}
        and float(row.get("exit_readiness_score", 0.0) or 0.0) < 0.72
    ]
    without_trigger_verification = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"unlock_then_exit", "mechanic_chain_deterministic", "mechanic_chain_llm"}
        and not bool(row.get("has_verified_trigger_contact", False))
    ]
    without_panel_gate_verification = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"unlock_then_exit", "mechanic_chain_deterministic", "mechanic_chain_llm"}
        and not bool(row.get("has_panel_or_gate_confirmation", False))
    ]
    verification_before_exit = [
        row for row in selected
        if str(row.get("candidate_class") or "") in {"verify_trigger_contact", "reobserve_remote_change", "verify_panel_state", "verify_gate_match"}
    ]
    failed_without_new_support = [
        row for row in rows
        if bool(dict(row.get("selected_outcome", row.get("outcome", {})) or {}).get("exit_attempt_failed_without_new_support", False))
    ]
    rewritten_to_verification = [
        row for row in rows
        if bool(dict(row.get("chain_snapshot", {}) or {}).get("chain_rewritten_to_verification", False))
    ]
    return {
        "executable_chain_selection_rate": float(len(executable_chain)) / float(total),
        "shallow_target_selection_rate": float(len(shallow_targets)) / float(total),
        "trigger_only_selection_rate": float(len(trigger_only)) / float(total),
        "verification_candidate_selection_rate": float(len(verification_before_exit)) / float(total),
        "planner_usable_hypothesis_utilization_rate": float(len(planner_usable)) / float(total),
        "premature_exit_attempt_rate": float(len(premature_exit)) / float(total),
        "exit_attempt_without_trigger_verification_rate": float(len(without_trigger_verification)) / float(total),
        "exit_attempt_without_panel_or_gate_verification_rate": float(len(without_panel_gate_verification)) / float(total),
        "verification_before_exit_rate": float(len(verification_before_exit)) / float(total),
        "completed_chain_before_exit_success_rate": float(len(completed_chain_before_exit)) / float(total),
        "direct_exit_attempt_rate_without_prerequisites": float(len(direct_exit_without_prereq)) / float(total),
        "failed_exit_without_new_support_rate": float(len(failed_without_new_support)) / float(total),
        "chain_rewrite_to_verification_rate": float(len(rewritten_to_verification)) / float(total),
    }
