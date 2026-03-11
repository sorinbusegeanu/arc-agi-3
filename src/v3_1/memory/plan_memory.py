from __future__ import annotations


def update_plan_memory(plan_memory: dict, *, decision: dict | None, outcome: dict | None, blackboard_state: dict) -> dict:
    next_state = {
        "history": list(plan_memory.get("history", [])),
        "repeated_failures": dict(plan_memory.get("repeated_failures", {})),
        "movement_memory": dict(plan_memory.get("movement_memory", {})),
        "recovery_memory": dict(plan_memory.get("recovery_memory", {})),
        "no_progress_rounds": int(plan_memory.get("no_progress_rounds", 0)),
    }
    if decision is None:
        return next_state

    selected = dict(decision.get("metadata", {}).get("selected_candidate", {}))
    target_entity_id = selected.get("target_entity_id")
    target_area_id = selected.get("target_area_id")
    candidate_class = selected.get("candidate_class")
    entry = {
        "decision": decision,
        "outcome": outcome or {},
        "target_entity_id": target_entity_id,
        "target_area_id": target_area_id,
        "candidate_class": candidate_class,
    }
    next_state["history"].append(entry)
    next_state["history"] = next_state["history"][-80:]

    outcome_payload = dict(outcome or {})
    outcome_summary = dict(outcome_payload.get("outcome", {}))
    success = bool(outcome_payload.get("success") or outcome_summary.get("success"))
    progress = float(outcome_summary.get("progress", 0.0))
    termination_reason = outcome_payload.get("termination_reason") or outcome_summary.get("termination_reason")

    if not success:
        if target_entity_id:
            next_state["repeated_failures"][str(target_entity_id)] = int(next_state["repeated_failures"].get(str(target_entity_id), 0)) + 1
        if target_area_id and candidate_class in {"frontier_move", "recovery_move"}:
            next_state["movement_memory"][str(target_area_id)] = int(next_state["movement_memory"].get(str(target_area_id), 0)) + 1
    else:
        if target_entity_id:
            next_state["repeated_failures"][str(target_entity_id)] = 0

    if progress <= 0.0 and not success:
        next_state["no_progress_rounds"] += 1
    else:
        next_state["no_progress_rounds"] = 0

    if candidate_class == "recovery_move" and not success:
        key = str(target_area_id or target_entity_id or "global")
        next_state["recovery_memory"][key] = {
            "attempts": int(next_state["recovery_memory"].get(key, {}).get("attempts", 0)) + 1,
            "last_termination_reason": termination_reason,
        }
    return next_state
