from __future__ import annotations


def _avatar_from_observation(observation):
    if not isinstance(observation, list):
        return None
    for y, row in enumerate(observation):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if int(value) == 1:
                return [x, y]
    return None


def _distance(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 2 or len(b) != 2:
        return None
    return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))


def _evidence_cell(value, provenance: str) -> dict:
    return {"value": value, "provenance": provenance}


def _latest_analysis_hint(steps, key: str):
    for step in reversed(list(steps or [])):
        info = dict(step.info) if isinstance(step.info, dict) else {}
        if key in info:
            return info.get(key)
    return None


def summarize_outcome(*, steps, request, routed_history: list[dict], rewards: list[float]) -> dict:
    avatar_positions = [
        list(step.info.get("avatar_cell_after"))
        if isinstance(step.info, dict) and isinstance(step.info.get("avatar_cell_after"), list)
        else _avatar_from_observation(step.observation)
        for step in steps
    ]
    avatar_positions = [position for position in avatar_positions if position is not None]
    target = request.navigation.get("route_target") or request.target_centroid
    first_info = dict(steps[0].info) if steps and isinstance(steps[0].info, dict) else {}
    last_info = dict(steps[-1].info) if steps and isinstance(steps[-1].info, dict) else {}
    initial_distance = _distance(first_info.get("avatar_cell_before") or (avatar_positions[0] if avatar_positions else None), target)
    final_distance = _distance(last_info.get("avatar_cell_after") or (avatar_positions[-1] if avatar_positions else None), target)
    analysis_route_progress = _latest_analysis_hint(steps, "analysis_route_progress")
    route_progress = 0.0
    if initial_distance is not None and final_distance is not None:
        route_progress = initial_distance - final_distance
    if analysis_route_progress is not None:
        route_progress = float(analysis_route_progress)
    unique_positions = {tuple(position) for position in avatar_positions}
    noop_steps = max(0, sum(1 for previous, current in zip(avatar_positions, avatar_positions[1:]) if previous == current))
    stalled = len(unique_positions) <= 2 and len(avatar_positions) >= 3
    blocked = stalled and route_progress <= 0.0
    route_success = final_distance is not None and final_distance <= float(request.terminal_action.get("terminal_distance", 0) or 0)
    explicit_failure = next((row for row in routed_history if row.get("failed")), None)
    termination_reason = str(last_info.get("terminal_stop_reason") or ("done" if steps and steps[-1].done else "step_budget_exhausted"))
    if explicit_failure is not None:
        termination_reason = str(explicit_failure.get("failure_reason", "execution_failed"))
    elif blocked:
        termination_reason = "blocked"
    elif stalled:
        termination_reason = "stalled"
    elif noop_steps > 0 and route_progress <= 0.0:
        termination_reason = "noop"

    environment_terminal_success = bool(steps and steps[-1].done)
    objective_success = bool(environment_terminal_success and (route_success or float(sum(rewards)) > 0.0 or request.objective.get("objective_type") == "fallback"))
    execution_success = bool(not explicit_failure and not blocked and not stalled)
    partial_success_modes = list(request.stop_conditions.get("allowed_partial_success_modes", []))
    partial_success = bool("route_progress" in partial_success_modes and route_progress > 0.0)
    objective_contact_observed = None
    if steps:
        objective_contact_observed = bool(
            any(bool(dict(step.info).get("terminal_action_marker")) for step in steps if isinstance(step.info, dict))
            or route_success
        )
    target_presence_observed = None
    if steps or routed_history:
        target_presence_observed = bool(
            isinstance(first_info.get("target_cell_before"), list)
            or isinstance(last_info.get("target_cell_after"), list)
            or any(isinstance(dict(row).get("target_centroid"), list) for row in routed_history)
        )
    terminal_action_executed = bool(
        any(bool(dict(step.info).get("terminal_action_marker")) for step in steps if isinstance(step.info, dict))
    )
    terminal_reward_observed = float(sum(rewards))
    done_observed = bool(steps and steps[-1].done)
    blocked_by_boundary_observed = None if not steps else bool(any(bool(dict(step.info).get("boundary_hit")) for step in steps if isinstance(step.info, dict)))
    blocked_by_unavailable_action_observed = None if not routed_history else bool(
        any("unavailable" in str(row.get("failure_reason", "")).lower() for row in routed_history if row.get("failed"))
    )
    analysis_effect_region = _latest_analysis_hint(steps, "analysis_effect_region")
    effect_region_observed = next(
        (
            dict(dict(step.info).get("effect_region"))
            for step in reversed(steps)
            if isinstance(step.info, dict) and isinstance(dict(step.info).get("effect_region"), dict)
        ),
        None,
    )
    if isinstance(analysis_effect_region, dict):
        effect_region_observed = dict(analysis_effect_region)
    analysis_effect_changed_cells = _latest_analysis_hint(steps, "analysis_effect_changed_cells")
    effect_changed_cells_observed = max(
        [int(dict(step.info).get("effect_changed_cells", 0) or 0) for step in steps if isinstance(step.info, dict)] or [0]
    )
    if analysis_effect_changed_cells is not None:
        effect_changed_cells_observed = int(analysis_effect_changed_cells or 0)
    success_certainty = 0.0
    if done_observed:
        success_certainty += 0.45
    if route_success:
        success_certainty += 0.25
    if terminal_action_executed:
        success_certainty += 0.15
    if terminal_reward_observed > 0.0:
        success_certainty += 0.15
    success_certainty = min(1.0, success_certainty)
    outcome_evidence = {
        "objective_contact_observed": _evidence_cell(objective_contact_observed, "execution_derived" if objective_contact_observed is not None else "unknown"),
        "target_presence_observed": _evidence_cell(target_presence_observed, "execution_derived" if target_presence_observed is not None else "unknown"),
        "avatar_target_distance_before": _evidence_cell(initial_distance, "execution_derived" if initial_distance is not None else "unknown"),
        "avatar_target_distance_after": _evidence_cell(final_distance, "execution_derived" if final_distance is not None else "unknown"),
        "route_progress_observed": _evidence_cell(route_progress, "analysis_derived" if analysis_route_progress is not None else "execution_derived" if initial_distance is not None and final_distance is not None else "unknown"),
        "terminal_action_executed": _evidence_cell(terminal_action_executed, "execution_derived" if steps else "unknown"),
        "terminal_reward_observed": _evidence_cell(terminal_reward_observed, "env_native" if steps else "unknown"),
        "done_observed": _evidence_cell(done_observed, "env_native" if steps else "unknown"),
        "blocked_by_boundary_observed": _evidence_cell(blocked_by_boundary_observed, "execution_derived" if blocked_by_boundary_observed is not None else "unknown"),
        "blocked_by_unavailable_action_observed": _evidence_cell(blocked_by_unavailable_action_observed, "execution_derived" if blocked_by_unavailable_action_observed is not None else "unknown"),
        "effect_region_observed": _evidence_cell(effect_region_observed, "analysis_derived" if isinstance(analysis_effect_region, dict) else "execution_derived" if effect_region_observed is not None else "unknown"),
        "effect_changed_cells_observed": _evidence_cell(effect_changed_cells_observed, "analysis_derived" if analysis_effect_changed_cells is not None else "execution_derived" if steps else "unknown"),
        "success_certainty": _evidence_cell(success_certainty if any(step.done for step in steps) or terminal_action_executed else min(success_certainty, 0.5), "execution_derived"),
    }
    env_native_support_count = sum(1 for cell in outcome_evidence.values() if str(cell.get("provenance")) == "env_native")
    derived_support_count = sum(1 for cell in outcome_evidence.values() if str(cell.get("provenance")) in {"execution_derived", "analysis_derived"})
    experiment_id = dict(request.metadata or {}).get("experiment_id")
    expected_observations = dict(dict(request.metadata or {}).get("expected_observations", {}) or {})
    expected_edge_ids = list(expected_observations.get("expected_edge_ids", []) or [])
    experiment_completed = bool(experiment_id)
    experiment_expected_observation_seen = bool(effect_changed_cells_observed > 0 or environment_terminal_success)
    origin_hypothesis_ids = list(dict(request.metadata or {}).get("origin_hypothesis_ids", []) or [])
    return {
        "success": execution_success and objective_success,
        "execution_success": execution_success,
        "objective_success": objective_success,
        "environment_terminal_success": environment_terminal_success,
        "reward_delta": float(sum(rewards)),
        "termination_reason": termination_reason,
        "progress": route_progress,
        "route_progress": route_progress,
        "blocked": blocked,
        "stalled": stalled,
        "noop": bool(noop_steps > 0),
        "noop_steps": noop_steps,
        "route_success": route_success,
        "route_failure": bool(request.mode == "directed" and (blocked or explicit_failure is not None) and not route_success),
        "partial_success": partial_success,
        "partial_success_modes": partial_success_modes,
        "outcome_evidence": outcome_evidence,
        "env_native_support_count": env_native_support_count,
        "derived_support_count": derived_support_count,
        "avatar_positions": avatar_positions,
        "routed_actions": routed_history,
        "telemetry": {
            "route_failure_reasons": [str(row.get("failure_reason")) for row in routed_history if row.get("failed")],
            "target_missing_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "missing_target"),
            "avatar_missing_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "missing_avatar"),
            "terminal_action_unavailable_count": sum(1 for row in routed_history if "terminal action unavailable" in str(row.get("failure_reason", "")).lower()),
            "local_reroute_attempts": sum(1 for row in routed_history if row.get("routing_mode") == "bounded_bfs"),
            "stall_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "stalled"),
        },
        "consequence_summary": {
            "reward_total": float(sum(rewards)),
            "positive_reward_steps": sum(1 for reward in rewards if reward > 0),
            "terminal": environment_terminal_success,
        },
        "mechanic_graph_evidence": {
            "trigger_contact_achieved": bool(objective_contact_observed and str(request.required_action_family or "") == "interact"),
            "remote_region_changed": bool(effect_region_observed and effect_changed_cells_observed > 0),
            "gate_region_state_changed": bool(effect_region_observed and str(request.objective.get("candidate_class") or "") in {"verify_gate_match", "unlock_then_exit"}),
            "exit_attempt_succeeded_after_prerequisite": bool(environment_terminal_success and str(request.objective.get("candidate_class") or "") in {"unlock_then_exit", "trigger_then_target"}),
            "expected_match_observed": bool(str(request.objective.get("candidate_class") or "") == "verify_panel_state" and effect_changed_cells_observed > 0),
            "expected_match_not_observed": bool(str(request.objective.get("candidate_class") or "") == "verify_panel_state" and effect_changed_cells_observed <= 0),
        },
        "experiment_result": {
            "experiment_completed": experiment_completed,
            "experiment_expected_observation_seen": experiment_expected_observation_seen,
            "experiment_supports_hypothesis_ids": origin_hypothesis_ids if experiment_completed and experiment_expected_observation_seen else [],
            "experiment_contradicts_hypothesis_ids": origin_hypothesis_ids if experiment_completed and not experiment_expected_observation_seen else [],
            "experiment_id": experiment_id,
            "experiment_kind": dict(request.metadata or {}).get("experiment_kind"),
            "experiment_target_node_ids": list(dict(request.metadata or {}).get("experiment_target_node_ids", []) or []),
            "expected_edge_ids": expected_edge_ids,
        },
    }
