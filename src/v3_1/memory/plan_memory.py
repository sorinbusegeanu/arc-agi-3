from __future__ import annotations


def _decay_count_map(values: dict, *, amount: int = 1) -> dict:
    decayed = {}
    for key, value in dict(values).items():
        next_value = max(0, int(value) - amount)
        if next_value > 0:
            decayed[str(key)] = next_value
    return decayed


def _decay_stat_rows(values: dict, *, factor: float = 0.9) -> dict:
    decayed = {}
    for key, row in dict(values).items():
        payload = dict(row)
        attempts = int(round(float(payload.get("attempts", 0) or 0) * factor))
        successes = int(round(float(payload.get("successes", 0) or 0) * factor))
        failures = int(round(float(payload.get("failures", 0) or 0) * factor))
        progress_total = float(payload.get("progress_total", 0.0) or 0.0) * factor if "progress_total" in payload else None
        if attempts <= 0 and successes <= 0 and failures <= 0 and (progress_total is None or abs(progress_total) < 1e-9):
            continue
        payload["attempts"] = max(0, attempts)
        if "successes" in payload:
            payload["successes"] = max(0, successes)
        if "failures" in payload:
            payload["failures"] = max(0, failures)
        if progress_total is not None:
            payload["progress_total"] = progress_total
        decayed[str(key)] = payload
    return decayed


def update_plan_memory(plan_memory: dict, *, decision: dict | None, outcome: dict | None, blackboard_state: dict) -> dict:
    next_state = {
        "history": list(plan_memory.get("history", [])),
        "repeated_failures": _decay_count_map(plan_memory.get("repeated_failures", {})),
        "movement_memory": _decay_count_map(plan_memory.get("movement_memory", {})),
        "recovery_memory": _decay_stat_rows(plan_memory.get("recovery_memory", {}), factor=0.92),
        "blocked_patterns": _decay_count_map(plan_memory.get("blocked_patterns", {})),
        "route_patterns": _decay_stat_rows(plan_memory.get("route_patterns", {}), factor=0.92),
        "candidate_class_performance": _decay_stat_rows(plan_memory.get("candidate_class_performance", {}), factor=0.95),
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
    route_failed = bool(outcome_summary.get("route_failed")) or str(termination_reason or "").startswith("route")

    if not success:
        if target_entity_id:
            next_state["repeated_failures"][str(target_entity_id)] = int(next_state["repeated_failures"].get(str(target_entity_id), 0)) + 1
        if target_area_id and candidate_class in {"frontier_move", "recovery_move"}:
            next_state["movement_memory"][str(target_area_id)] = int(next_state["movement_memory"].get(str(target_area_id), 0)) + 1
        if termination_reason in {"blocked", "stalled", "noop"}:
            blocked_key = f"{candidate_class}:{termination_reason}:{target_area_id or target_entity_id or 'global'}"
            next_state["blocked_patterns"][blocked_key] = int(next_state["blocked_patterns"].get(blocked_key, 0)) + 1
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
            "successes": int(next_state["recovery_memory"].get(key, {}).get("successes", 0)),
            "failures": int(next_state["recovery_memory"].get(key, {}).get("failures", 0)) + 1,
            "last_termination_reason": termination_reason,
        }
    elif candidate_class == "recovery_move":
        key = str(target_area_id or target_entity_id or "global")
        next_state["recovery_memory"][key] = {
            "attempts": int(next_state["recovery_memory"].get(key, {}).get("attempts", 0)) + 1,
            "successes": int(next_state["recovery_memory"].get(key, {}).get("successes", 0)) + 1,
            "failures": int(next_state["recovery_memory"].get(key, {}).get("failures", 0)),
            "last_termination_reason": termination_reason,
        }
    if route_failed:
        route_key = f"{candidate_class}:{target_area_id or target_entity_id or 'global'}"
        next_state["route_patterns"][route_key] = {
            "attempts": int(next_state["route_patterns"].get(route_key, {}).get("attempts", 0)) + 1,
            "failures": int(next_state["route_patterns"].get(route_key, {}).get("failures", 0)) + 1,
            "successes": int(next_state["route_patterns"].get(route_key, {}).get("successes", 0)),
        }
    elif candidate_class in {"frontier_move", "recovery_move", "route_probe"}:
        route_key = f"{candidate_class}:{target_area_id or target_entity_id or 'global'}"
        next_state["route_patterns"][route_key] = {
            "attempts": int(next_state["route_patterns"].get(route_key, {}).get("attempts", 0)) + 1,
            "failures": int(next_state["route_patterns"].get(route_key, {}).get("failures", 0)),
            "successes": int(next_state["route_patterns"].get(route_key, {}).get("successes", 0)) + (1 if success else 0),
        }
    if candidate_class:
        perf = dict(next_state["candidate_class_performance"].get(str(candidate_class), {}))
        perf["attempts"] = int(perf.get("attempts", 0)) + 1
        perf["successes"] = int(perf.get("successes", 0)) + (1 if success else 0)
        perf["failures"] = int(perf.get("failures", 0)) + (0 if success else 1)
        perf["progress_total"] = float(perf.get("progress_total", 0.0)) + progress
        next_state["candidate_class_performance"][str(candidate_class)] = perf
    return next_state


def derive_durable_pattern_updates(plan_memory: dict) -> dict[str, tuple[dict, ...]]:
    blocked_patterns = tuple(
        {
            "pattern_key": str(key),
            "count": int(count),
            "metadata": {"pattern_type": "blocked"},
        }
        for key, count in sorted(dict(plan_memory.get("blocked_patterns", {})).items())
    )
    repeated_failures = tuple(
        {
            "pattern_key": f"target_failure:{key}",
            "count": int(count),
            "metadata": {"pattern_type": "repeated_failed_target"},
        }
        for key, count in sorted(dict(plan_memory.get("repeated_failures", {})).items())
        if int(count) > 0
    )
    route_patterns = tuple(
        {
            "pattern_key": str(key),
            "attempts": int(row.get("attempts", 0)),
            "successes": int(row.get("successes", 0)),
            "failures": int(row.get("failures", 0)),
            "metadata": {"pattern_type": "route"},
        }
        for key, row in sorted(dict(plan_memory.get("route_patterns", {})).items())
    )
    recovery_patterns = tuple(
        {
            "pattern_key": str(key),
            "attempts": int(row.get("attempts", 0)),
            "successes": int(row.get("successes", 0)),
            "failures": int(row.get("failures", 0)),
            "metadata": {"last_termination_reason": row.get("last_termination_reason")},
        }
        for key, row in sorted(dict(plan_memory.get("recovery_memory", {})).items())
    )
    candidate_outcomes = tuple(
        {
            "candidate_class": str(key),
            "attempts": int(row.get("attempts", 0)),
            "successes": int(row.get("successes", 0)),
            "failures": int(row.get("failures", 0)),
            "progress_total": float(row.get("progress_total", 0.0)),
            "route_failures": int(dict(plan_memory.get("route_patterns", {})).get(f"{key}:global", {}).get("failures", 0)),
            "metadata": {"pattern_type": "candidate_class_performance"},
        }
        for key, row in sorted(dict(plan_memory.get("candidate_class_performance", {})).items())
    )
    return {
        "failure_patterns": blocked_patterns + repeated_failures,
        "recovery_patterns": recovery_patterns + route_patterns,
        "candidate_outcomes": candidate_outcomes,
    }
