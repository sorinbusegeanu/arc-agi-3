from __future__ import annotations

from v3_1.contracts.messages import ExecutorRequest, PlannerDecision
from v3_1.config.defaults import DEFAULT_CONFIG


def _expected_effect_contract(selected: dict, active_step: dict, objective: dict) -> dict:
    step_kind = str(active_step.get("step_kind") or "")
    objective_type = str(selected.get("objective_type") or objective.get("objective_type") or "")
    candidate_class = str(selected.get("candidate_class") or "")
    if step_kind in {"go_to_trigger", "verify_trigger_contact"} or objective_type in {"trigger_then_target"}:
        return {"expected_effect_type": "trigger_contact", "expected_relation_type": "trigger_contact"}
    if step_kind in {"reobserve_region", "reobserve_remote_change"} or candidate_class == "reobserve_remote_change":
        return {"expected_effect_type": "remote_change", "expected_relation_type": "causes_remote_change"}
    if step_kind in {"verify_panel"} or candidate_class == "verify_panel_state":
        return {"expected_effect_type": "panel_state_change", "expected_relation_type": "panel_state"}
    if step_kind in {"verify_gate"} or candidate_class == "verify_gate_match":
        return {"expected_effect_type": "gate_state_change", "expected_relation_type": "gate_state"}
    if step_kind == "attempt_exit" or objective_type in {"attempt_exit", "unlock_then_exit"} or candidate_class == "unlock_then_exit":
        return {"expected_effect_type": "exit_transition", "expected_relation_type": "exit"}
    return {
        "expected_effect_type": str(selected.get("expected_effect_type") or objective.get("expected_effect_type") or ""),
        "expected_relation_type": str(selected.get("expected_relation_type") or objective.get("expected_effect_relation") or ""),
    }


def _expected_boundary_hint(*, target_centroid, click_target_coordinates, target_area_id: str | None, relation_type: str) -> str | None:
    target = click_target_coordinates or target_centroid
    if isinstance(target, (list, tuple)) and len(target) == 2:
        x = float(target[0])
        y = float(target[1])
        if x <= 1:
            return "left_boundary"
        if x >= 62:
            return "right_boundary"
        if y <= 1:
            return "top_boundary"
        if y >= 62:
            return "bottom_boundary"
    if relation_type in {"exit", "leave", "escape", "terminal"}:
        return str(target_area_id or "boundary_region")
    return None


def _terminal_distance_from_action(selected: dict) -> int:
    execution_cfg = DEFAULT_CONFIG.execution
    execution_mode = str(selected.get("execution_mode") or selected.get("required_action_family") or "move")
    if execution_mode == "click_at":
        return int(getattr(execution_cfg, "click_terminal_distance_cells", 0))
    if execution_mode == "interact":
        return int(getattr(execution_cfg, "interact_terminal_distance_cells", 0))
    return int(getattr(execution_cfg, "move_terminal_distance_cells", 0))


def build_executor_request(decision: PlannerDecision, *, max_steps: int, mode: str, seed: int | None = None) -> ExecutorRequest:
    metadata = dict(decision.metadata) if isinstance(decision.metadata, dict) else {}
    selected_raw = metadata.get("selected_candidate", {})
    selected = dict(selected_raw) if isinstance(selected_raw, dict) else {}
    runtime_chain_state = dict(metadata.get("runtime_subgoal_chain_state", {}) or {}) if isinstance(metadata.get("runtime_subgoal_chain_state"), dict) else {}
    selected_subgoal_chain = dict(runtime_chain_state.get("active_chain", {}) or {}) if isinstance(runtime_chain_state.get("active_chain"), dict) else {}
    if not selected_subgoal_chain:
        selected_subgoal_chain = dict(metadata.get("selected_subgoal_chain", {}) or {}) if isinstance(metadata.get("selected_subgoal_chain"), dict) else {}
    selected_subgoal_step = dict(runtime_chain_state.get("active_step", {}) or {}) if isinstance(runtime_chain_state.get("active_step"), dict) else {}
    if not selected_subgoal_step:
        selected_subgoal_step = dict(metadata.get("selected_subgoal_step", {}) or {}) if isinstance(metadata.get("selected_subgoal_step"), dict) else {}
    chain_steps = list(selected_subgoal_chain.get("steps", []) or [])
    chain_status_before = str(selected_subgoal_chain.get("status") or "")
    chain_step_index = int(selected_subgoal_chain.get("current_step_index", 0) or 0)
    active_step = dict(selected_subgoal_step or (chain_steps[chain_step_index] if 0 <= chain_step_index < len(chain_steps) else {}))
    selected_action = dict(decision.selected_action or {})
    target_centroid = selected_action.get("centroid")
    click_target_coordinates = selected_action.get("click_target_coordinates") or selected_action.get("coordinates")
    objective_type = str(selected.get("objective_type") or "fallback")
    execution_mode = str(selected.get("execution_mode") or selected.get("required_action_family") or "unknown")
    navigation_mode = str(selected.get("navigation_mode") or ("hold" if objective_type == "fallback" else "direct"))
    if objective_type == "fallback" or navigation_mode == "hold":
        selected_action["type"] = "hold_position"
        selected_action.pop("centroid", None)
        selected_action.pop("target", None)
        selected_action.pop("target_entity_id", None)
    route_required = bool(selected.get("route_required", navigation_mode in {"direct", "routed"}))
    route_signature = selected.get("route_signature")
    target_entity_id = selected.get("target_entity_id")
    target_area_id = selected.get("target_area_id")
    objective_contract = {
        "objective_type": objective_type,
        "candidate_class": selected.get("candidate_class"),
        "target_entity_id": target_entity_id,
        "target_area_id": target_area_id,
        "expected_progress_type": selected.get("expected_progress_type"),
        "rationale": selected.get("rationale") or decision.rationale,
        "supporting_graph_node_ids": list(selected.get("supporting_graph_node_ids", []) or []),
        "supporting_graph_edge_ids": list(selected.get("supporting_graph_edge_ids", []) or []),
        "prerequisite_chain": list(selected.get("prerequisite_chain", []) or []),
        "hop_count": int(selected.get("hop_count", 0) or 0),
        "selected_subgoal_chain": dict(selected_subgoal_chain or {}),
        "selected_subgoal_step": dict(active_step or {}),
        "runtime_subgoal_chain_state": dict(runtime_chain_state or {}),
        "active_chain_id": selected_subgoal_chain.get("chain_id"),
        "current_step_id": active_step.get("step_id"),
        "current_step_kind": active_step.get("step_kind"),
        "current_step_expected_evidence": list(active_step.get("expected_evidence", []) or []),
        "expected_target_id": active_step.get("target_node_id") or target_entity_id,
        "expected_trigger_id": active_step.get("target_node_id") if str(active_step.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact"} else None,
        "chain_local_stop_markers": list(selected_subgoal_chain.get("required_verification_steps", []) or []),
    }
    expected_effect = _expected_effect_contract(selected, active_step, objective_contract)
    expected_boundary = _expected_boundary_hint(
        target_centroid=target_centroid,
        click_target_coordinates=click_target_coordinates,
        target_area_id=str(target_area_id) if target_area_id is not None else None,
        relation_type=str(expected_effect.get("expected_relation_type") or ""),
    )
    objective_contract.update(
        {
            "expected_effect_type": expected_effect.get("expected_effect_type"),
            "expected_effect_relation": expected_effect.get("expected_relation_type"),
            "expected_trigger_target_id": active_step.get("target_node_id") if str(active_step.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact"} else None,
            "expected_terminal_or_exit_target_id": target_entity_id if str(expected_effect.get("expected_relation_type") or "") == "exit" else None,
            "expected_contact_region_or_boundary": expected_boundary,
        }
    )
    navigation_contract = {
        "route_required": route_required,
        "navigation_mode": navigation_mode,
        "route_signature": route_signature,
        "route_target": target_centroid,
        "route_target_entity_id": target_entity_id,
        "avatar_area": dict(selected.get("candidate_context", {}) or {}).get("avatar_area"),
        "local_area": dict(selected.get("candidate_context", {}) or {}).get("local_area"),
    }
    terminal_action_contract = {
        "must_execute_terminal_action": bool(execution_mode in {"interact", "click_at"}),
        "terminal_action_family": execution_mode,
        "terminal_action_name": execution_mode,
        "target_coordinates": click_target_coordinates if execution_mode == "click_at" else target_centroid,
        "terminal_distance": _terminal_distance_from_action(selected),
    }
    constraints_contract = {
        "allowed_action_families": [execution_mode] if execution_mode != "unknown" else ["move"],
        "route_required": route_required,
        "allow_local_reroute": True,
        "allow_avatar_reacquire_once": True,
        "allow_target_reacquire_once": True,
        "max_local_reroute_attempts": int(getattr(DEFAULT_CONFIG.execution, "blocked_repeat_limit", 1)),
        "allow_partial_success_modes": ["route_progress"],
    }
    stop_conditions_contract = {
        "stop_on_terminal_action": True,
        "stop_on_route_failure": True,
        "stop_on_stall": True,
        "stop_on_blocked": True,
        "allowed_partial_success_modes": ["route_progress"],
        "failure_modes": ["missing_target", "missing_avatar", "avatar_localization_low_confidence", "blocked", "unreachable", "stalled", "terminal_action_unavailable"],
        "reset_mode": "reset_each_episode",
        "stall_limit": int(getattr(DEFAULT_CONFIG.execution, "stall_limit", 3)),
    }
    return ExecutorRequest(
        session_id=decision.session_id,
        run_id=decision.run_id,
        game_id=decision.game_id,
        round_id=decision.round_id,
        pass_id=decision.pass_id,
        plan_context_id=decision.plan_context_id,
        candidate_id=decision.selected_candidate_id,
        action=selected_action,
        max_steps=max_steps,
        mode=mode,
        action_id=selected.get("action_id"),
        action_name=selected.get("action_name"),
        action_family=execution_mode,
        required_action_family=execution_mode,
        target_entity_id=target_entity_id,
        target_centroid=target_centroid,
        click_target_coordinates=click_target_coordinates,
        objective=objective_contract,
        navigation=navigation_contract,
        terminal_action=terminal_action_contract,
        constraints=constraints_contract,
        stop_conditions=stop_conditions_contract,
        metadata={
            "planner_candidate": selected,
            "avatar_tracking_mode": "live_motion",
            "avatar_localization_policy": "tracker_first",
            "selected_subgoal_chain": dict(selected_subgoal_chain or {}),
            "selected_subgoal_step": dict(active_step or {}),
            "runtime_subgoal_chain_state": dict(runtime_chain_state or {}),
            "chain_id": selected_subgoal_chain.get("chain_id"),
            "step_id": active_step.get("step_id"),
            "step_kind": active_step.get("step_kind"),
            "step_index": chain_step_index if active_step else None,
            "chain_status_before": chain_status_before or None,
            "chain_status_after": "executing_step" if active_step else chain_status_before or None,
            "step_target_node_id": active_step.get("target_node_id"),
            "expected_target_id": active_step.get("target_node_id") or target_entity_id,
            "expected_trigger_id": active_step.get("target_node_id") if str(active_step.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact"} else None,
            "expected_effect_type": expected_effect.get("expected_effect_type") or None,
            "expected_effect_relation": expected_effect.get("expected_relation_type") or None,
            "expected_relation_type": expected_effect.get("expected_relation_type") or None,
            "expected_trigger_target_id": active_step.get("target_node_id") if str(active_step.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact"} else None,
            "expected_terminal_or_exit_target_id": target_entity_id if str(expected_effect.get("expected_relation_type") or "") == "exit" else None,
            "expected_contact_region_or_boundary": expected_boundary,
            "expected_evidence": list(active_step.get("expected_evidence", []) or []),
            "step_result_contract": {
                "emit_step_success": True,
                "emit_expected_evidence_seen_missing": True,
                "emit_chain_progress_decision": True,
            },
            "fallback_candidates": list(metadata.get("fallback_candidates", []) or []),
            "seed": seed,
            "mechanic_subgoal": {
                "supporting_graph_node_ids": list(selected.get("supporting_graph_node_ids", []) or []),
                "supporting_graph_edge_ids": list(selected.get("supporting_graph_edge_ids", []) or []),
                "prerequisite_chain": list(selected.get("prerequisite_chain", []) or []),
            },
            "experiment_id": selected.get("test_proposal_id"),
            "experiment_kind": selected.get("candidate_class") if str(selected.get("candidate_class") or "").startswith("mechanic_test_") else None,
            "experiment_target_node_ids": list(selected.get("target_node_ids", []) or []),
            "expected_observations": {
                "expected_edge_ids": list(selected.get("supporting_graph_edge_ids", []) or []),
                "expected_information_gain": float(selected.get("expected_information_gain", 0.0) or 0.0),
            },
            "origin_hypothesis_ids": list(selected.get("supporting_hypothesis_ids", []) or []),
        },
    )
