from __future__ import annotations


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


def _matches_expected_effect(expectation: dict, observed_rows: list[dict]) -> tuple[bool, str, int, int]:
    expected_effect_type = str(expectation.get("expected_effect_type") or "").lower()
    expected_relation_type = str(expectation.get("expected_relation_type") or "").lower()
    expected_target_kind = str(expectation.get("expected_target_kind") or "region").lower()
    expected_target_value = str(expectation.get("expected_target_value") or "").lower()
    candidate_count = 0
    matching_count = 0
    saw_wrong_target = False
    saw_wrong_relation = False
    saw_wrong_effect_type = False
    for row in list(observed_rows or []):
        payload = dict(row or {})
        effect_type = str(payload.get("effect_type") or "").lower()
        relation_type = str(payload.get("relation_type") or "").lower()
        target_kind = str(payload.get("target_kind") or "region").lower()
        target_value = str(payload.get("target_value") or "").lower()
        if not any([effect_type, relation_type, target_value, bool(payload.get("has_effect"))]):
            continue
        candidate_count += 1
        effect_ok = not expected_effect_type or effect_type == expected_effect_type
        relation_ok = not expected_relation_type or relation_type == expected_relation_type
        target_ok = not expected_target_value or (target_kind == expected_target_kind and target_value == expected_target_value)
        if effect_ok and relation_ok and target_ok:
            matching_count += 1
            return True, (
                "matched_expected_target_relation"
                if expected_target_kind == "entity"
                else "matched_expected_target_region"
            ), candidate_count, matching_count
        if not target_ok and expected_target_value:
            saw_wrong_target = True
        elif not relation_ok and expected_relation_type:
            saw_wrong_relation = True
        elif not effect_ok and expected_effect_type:
            saw_wrong_effect_type = True
    if saw_wrong_target:
        return False, "observed_effect_wrong_target", candidate_count, matching_count
    if saw_wrong_relation:
        return False, "observed_effect_wrong_relation", candidate_count, matching_count
    if saw_wrong_effect_type:
        return False, "observed_effect_wrong_effect_type", candidate_count, matching_count
    return False, "no_matching_effect_in_window", candidate_count, matching_count


def summarize_outcome(*, steps, request, routed_history: list[dict], rewards: list[float]) -> dict:
    last_routed = dict(routed_history[-1]) if routed_history else {}
    avatar_positions = [
        list(step.info.get("avatar_cell_after"))
        if isinstance(step.info, dict) and isinstance(step.info.get("avatar_cell_after"), list)
        else None
        for step in steps
    ]
    avatar_positions = [position for position in avatar_positions if position is not None]
    target = request.navigation.get("route_target") or request.target_centroid
    first_info = dict(steps[0].info) if steps and isinstance(steps[0].info, dict) else {}
    last_info = dict(steps[-1].info) if steps and isinstance(steps[-1].info, dict) else {}
    avatar_confidence_before = float(first_info.get("avatar_confidence_before", 0.0) or first_info.get("avatar_confidence", 0.0) or 0.0)
    avatar_confidence_after = float(last_info.get("avatar_confidence_after", 0.0) or last_info.get("avatar_confidence", 0.0) or last_routed.get("avatar_confidence", 0.0) or 0.0)
    avatar_source_after = str(last_info.get("avatar_source_after") or last_info.get("avatar_source") or last_routed.get("avatar_source") or "unknown")
    avatar_ambiguous_after = bool(last_info.get("avatar_ambiguous_after", last_info.get("avatar_ambiguous", last_routed.get("avatar_ambiguous", False))))
    avatar_mode_status = str(last_info.get("avatar_mode_status") or first_info.get("avatar_mode_status") or "unknown")
    avatar_status = str(last_info.get("avatar_status") or first_info.get("avatar_status") or "unknown")
    avatar_localization_confident = bool(avatar_mode_status == "movement_avatar" and avatar_status == "present" and avatar_confidence_after >= 0.6 and not avatar_ambiguous_after)
    initial_distance = _distance(first_info.get("avatar_cell_before") or (avatar_positions[0] if avatar_positions else None), target)
    final_distance = _distance(last_info.get("avatar_cell_after") or (avatar_positions[-1] if avatar_positions else None), target)
    analysis_route_progress = _latest_analysis_hint(steps, "analysis_route_progress")
    route_progress = 0.0
    if initial_distance is not None and final_distance is not None:
        route_progress = initial_distance - final_distance
    if analysis_route_progress is not None:
        route_progress = float(analysis_route_progress)
    route_progress_based_on_confident_avatar = bool(avatar_localization_confident and initial_distance is not None and final_distance is not None)
    unique_positions = {tuple(position) for position in avatar_positions}
    noop_steps = max(0, sum(1 for previous, current in zip(avatar_positions, avatar_positions[1:]) if previous == current))
    stalled = len(unique_positions) <= 2 and len(avatar_positions) >= 3
    blocked = stalled and route_progress <= 0.0
    route_success = final_distance is not None and final_distance <= float(request.terminal_action.get("terminal_distance", 0) or 0)
    explicit_failure = next((row for row in routed_history if row.get("failed")), None)
    termination_reason = str(last_info.get("terminal_stop_reason") or ("done" if steps and steps[-1].done else "step_budget_exhausted"))
    if explicit_failure is not None:
        termination_reason = str(explicit_failure.get("failure_reason", "execution_failed"))
    elif bool(last_routed.get("stop")):
        termination_reason = "position_hold"
    elif blocked:
        termination_reason = "blocked"
    elif stalled:
        termination_reason = "stalled"
    elif noop_steps > 0 and route_progress <= 0.0:
        termination_reason = "noop"

    environment_terminal_success = bool(steps and steps[-1].done)
    objective_success = bool(environment_terminal_success and (route_success or float(sum(rewards)) > 0.0 or request.objective.get("objective_type") == "fallback"))
    execution_success = bool((bool(last_routed.get("stop")) and not explicit_failure) or (not explicit_failure and not blocked and not stalled))
    partial_success_modes = list(request.stop_conditions.get("allowed_partial_success_modes", []))
    partial_success = bool("route_progress" in partial_success_modes and route_progress > 0.0)
    objective_contact_observed = None
    if steps:
        objective_contact_observed = bool(
            any(bool(dict(step.info).get("terminal_action_marker")) for step in steps if isinstance(step.info, dict))
            or route_success
        )
    trigger_contact_based_on_confident_avatar = bool(avatar_localization_confident and objective_contact_observed)
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
    target_approach_based_on_confident_avatar = bool(avatar_localization_confident and route_progress > 0.0)
    objective = dict(request.objective or {})
    metadata = dict(request.metadata or {})
    execution_intent = dict(getattr(request, "metadata", {}) or {}).get("execution_intent", {})
    execution_intent = dict(execution_intent or {}) if isinstance(execution_intent, dict) else {}
    action = dict(request.action or {})
    expected_effect_type = str(metadata.get("expected_effect_type") or execution_intent.get("expected_effect_type") or objective.get("expected_effect_type") or "")
    expected_effect_relation = str(metadata.get("expected_effect_relation") or metadata.get("expected_relation_type") or execution_intent.get("expected_effect_relation") or objective.get("expected_effect_relation") or "")
    directed_outcome_relation_supported = bool(route_progress > 0.0 or effect_changed_cells_observed > 0 or done_observed or terminal_reward_observed > 0.0)
    target_entity_id = str(metadata.get("target_entity_id") or request.target_entity_id or objective.get("target_entity_id") or action.get("target_entity_id") or "") or None
    expected_target_id = str(metadata.get("expected_target_id") or execution_intent.get("expected_target_id") or target_entity_id or "") or None
    expected_effect_target_id = str(metadata.get("expected_effect_target_id") or execution_intent.get("expected_target_id") or expected_target_id or "") or None
    exit_attempt_target_id = str(metadata.get("exit_target_id") or execution_intent.get("expected_terminal_or_exit_target_id") or expected_target_id or "") or None
    step_kind_hint = str(metadata.get("step_kind") or "")
    objective_type_hint = str(objective.get("objective_type") or metadata.get("objective_type") or "")
    action_name_hint = str(request.action_name or action.get("type") or "")
    attempted_boundary_contact = bool(
        any(bool(dict(step.info).get("attempted_boundary_contact") or dict(step.info).get("boundary_contact_observed") or dict(step.info).get("boundary_hit")) for step in steps if isinstance(step.info, dict))
    )
    attempted_portal_contact = bool(any(bool(dict(step.info).get("attempted_portal_contact") or dict(step.info).get("portal_like_contact_observed")) for step in steps if isinstance(step.info, dict)))
    attempted_terminal_affordance_contact = bool(any(bool(dict(step.info).get("attempted_terminal_affordance_contact") or dict(step.info).get("terminal_affordance_contact_observed") or dict(step.info).get("terminal_action_marker")) for step in steps if isinstance(step.info, dict)))
    attempted_escape_direction = next(
        (
            dict(step.info).get("attempted_escape_direction")
            for step in reversed(steps)
            if isinstance(step.info, dict) and dict(step.info).get("attempted_escape_direction") is not None
        ),
        None,
    )
    exit_attempt_observed = bool(step_kind_hint == "attempt_exit")
    exit_attempt_evidence_observed = bool(
        exit_attempt_observed
        or bool(metadata.get("candidate_class") in {"unlock_then_exit"})
        or objective_type_hint in {"attempt_exit", "unlock_then_exit"}
        or "exit" in action_name_hint.lower()
        or str(expected_effect_relation or "").lower() in {"exit", "leave", "escape", "terminal"}
    )
    exit_attempt_boundary_contact = bool(
        exit_attempt_evidence_observed
        and (
            bool(last_info.get("terminal_action_marker"))
            or bool(last_info.get("boundary_hit"))
            or bool(last_info.get("target_contact_observed"))
            or bool(last_info.get("objective_contact_observed"))
        )
    )
    exit_attempt_action_type = action_name_hint or str(request.action_family or "unknown")
    trigger_contact_observed = bool(trigger_contact_based_on_confident_avatar or objective_contact_observed)
    target_contact_observed = bool(target_presence_observed)
    expected_trigger_contact_observed = bool(trigger_contact_observed or objective_contact_observed)
    expected_region_reached = bool(
        target_approach_based_on_confident_avatar
        or target_contact_observed
        or objective_contact_observed
        or (final_distance is not None and final_distance <= 1.5)
    )
    observed_effect_change = bool(effect_changed_cells_observed > 0 or done_observed or terminal_reward_observed > 0.0)
    weak_expectation_basis: list[str] = []
    repeated_blocked_boundary = any(bool(dict(step.info).get("repeated_boundary_facing_movement")) for step in steps if isinstance(step.info, dict))
    stationary_salient_attempt = any(
        bool(
            (
                dict(step.info).get("movement_stayed_in_place_after_attempted_boundary_move")
                or (
                    dict(step.info).get("avatar_cell_before") == dict(step.info).get("avatar_cell_after")
                    and (
                        dict(step.info).get("attempted_boundary_contact")
                        or dict(step.info).get("attempted_portal_contact")
                        or dict(step.info).get("attempted_terminal_affordance_contact")
                        or dict(step.info).get("objective_contact_observed")
                        or dict(step.info).get("target_contact_observed")
                    )
                )
            )
        )
        for step in steps
        if isinstance(step.info, dict)
    )
    attempted_salient_contact = bool(attempted_portal_contact or attempted_terminal_affordance_contact or target_contact_observed or objective_contact_observed)
    counterfactual_target_scope = None
    expected_target_region_class = None
    if not expected_effect_type and not expected_effect_relation:
        if attempted_boundary_contact or any(bool(dict(step.info).get("attempted_move_into_blocked_boundary")) for step in steps if isinstance(step.info, dict)) or repeated_blocked_boundary:
            weak_expectation_basis.append("boundary_crossing_attempt")
            expected_effect_type = expected_effect_type or "boundary_transition"
            expected_effect_relation = expected_effect_relation or "cross_boundary"
            counterfactual_target_scope = counterfactual_target_scope or "region"
            expected_target_region_class = expected_target_region_class or "boundary_region"
        if attempted_portal_contact or attempted_terminal_affordance_contact or attempted_salient_contact:
            weak_expectation_basis.append("salient_contact")
            expected_effect_type = expected_effect_type or "contact_effect"
            expected_effect_relation = expected_effect_relation or "activate_contact_target"
            counterfactual_target_scope = counterfactual_target_scope or ("entity" if expected_target_id else "region")
            expected_target_region_class = expected_target_region_class or (
                "portal_region" if attempted_portal_contact else "terminal_affordance_region" if attempted_terminal_affordance_contact else "salient_area"
            )
        if trigger_contact_observed or objective_contact_observed or target_contact_observed:
            weak_expectation_basis.append("trigger_zone_entry")
            expected_effect_type = expected_effect_type or "trigger_effect"
            expected_effect_relation = expected_effect_relation or "enter_trigger_zone"
            counterfactual_target_scope = counterfactual_target_scope or ("entity" if expected_target_id else "region")
            expected_target_region_class = expected_target_region_class or "trigger_zone"
    expectation_basis = "directed_expectation" if (expected_effect_type or expected_effect_relation) else ("weak_probe_expectation" if weak_expectation_basis else None)
    if counterfactual_target_scope is None:
        counterfactual_target_scope = "entity" if (expected_target_id or expected_effect_target_id) else "region"
    no_state_hash_change = bool(
        len(step_rows := [
            dict(step.info) for step in steps if isinstance(step.info, dict)
        ]) > 0
        and all(dict_info.get("state_hash_before") == dict_info.get("state_hash_after") for dict_info in step_rows if dict_info.get("state_hash_before") is not None and dict_info.get("state_hash_after") is not None)
    )
    repeated_noop_after_attempt = bool(
        stationary_salient_attempt
        or ((blocked or stalled or noop_steps > 0) and (attempted_boundary_contact or attempted_salient_contact or expected_region_reached))
    )
    counterfactual_expectation_basis_present = bool(expectation_basis)
    counterfactual_attempt_context_present = bool(
        expected_trigger_contact_observed
        or expected_region_reached
        or attempted_boundary_contact
        or attempted_portal_contact
        or attempted_terminal_affordance_contact
        or attempted_salient_contact
        or repeated_blocked_boundary
    )
    attempt_anchor_idx = None
    for idx, step in enumerate(steps):
        info = dict(step.info) if isinstance(step.info, dict) else {}
        if bool(
            info.get("attempted_boundary_contact")
            or info.get("boundary_hit")
            or info.get("attempted_portal_contact")
            or info.get("attempted_terminal_affordance_contact")
            or info.get("objective_contact_observed")
            or info.get("target_contact_observed")
            or info.get("attempted_move_into_blocked_boundary")
        ):
            attempt_anchor_idx = idx
            break
    if attempt_anchor_idx is None and counterfactual_attempt_context_present:
        attempt_anchor_idx = 0 if steps else None
    post_attempt_steps = list(steps[attempt_anchor_idx: min(len(steps), attempt_anchor_idx + 3)]) if attempt_anchor_idx is not None else []
    counterfactual_post_attempt_window_steps = len(post_attempt_steps)
    post_attempt_infos = [dict(step.info) for step in post_attempt_steps if isinstance(step.info, dict)]
    observed_effect_candidates = []
    for info in post_attempt_infos:
        if bool(info.get("effect_region")):
            observed_effect_candidates.append(
                {
                    "effect_type": "region_effect",
                    "relation_type": "enter_trigger_zone" if str(expected_effect_relation or "").lower() == "enter_trigger_zone" else "activate_contact_target",
                    "target_kind": "region",
                    "target_value": str(expected_target_id or expected_effect_target_id or expected_target_region_class or "effect_region"),
                    "has_effect": True,
                }
            )
        if bool(info.get("terminal_action_marker")) or bool(info.get("done")):
            observed_effect_candidates.append(
                {
                    "effect_type": "boundary_transition",
                    "relation_type": "cross_boundary",
                    "target_kind": "region",
                    "target_value": str(expected_target_region_class or "boundary_region"),
                    "has_effect": True,
                }
            )
    expected_target_value = str(expected_target_id or expected_effect_target_id or expected_target_region_class or "").lower()
    expected_target_kind = "entity" if (expected_target_id or expected_effect_target_id) else "region"
    counterfactual_matched_expected_effect, counterfactual_match_reason_code, counterfactual_observed_effect_candidate_count, counterfactual_matching_effect_candidate_count = _matches_expected_effect(
        {
            "expected_effect_type": expected_effect_type,
            "expected_relation_type": expected_effect_relation,
            "expected_target_kind": expected_target_kind,
            "expected_target_value": expected_target_value,
        },
        observed_effect_candidates,
    )
    observed_effect_absent = bool(
        counterfactual_expectation_basis_present
        and counterfactual_attempt_context_present
        and counterfactual_post_attempt_window_steps > 0
        and not counterfactual_matched_expected_effect
        and (
            stationary_salient_attempt
            or repeated_blocked_boundary
            or repeated_noop_after_attempt
            or no_state_hash_change
            or expected_region_reached
            or expected_trigger_contact_observed
        )
    )
    counterfactual_non_effect_confirmed = bool(counterfactual_expectation_basis_present and counterfactual_attempt_context_present and observed_effect_absent)
    counterfactual_classifier_block_reason_codes: list[str] = []
    if not counterfactual_expectation_basis_present:
        counterfactual_classifier_block_reason_codes.append("no_expectation_basis")
    if not counterfactual_attempt_context_present:
        counterfactual_classifier_block_reason_codes.append("no_attempt_context")
    if attempt_anchor_idx is None:
        counterfactual_classifier_block_reason_codes.append("no_salient_attempt_anchor")
    if counterfactual_post_attempt_window_steps <= 0:
        counterfactual_classifier_block_reason_codes.append("no_post_attempt_window")
    if not (expected_effect_type or expected_effect_relation or expectation_basis):
        counterfactual_classifier_block_reason_codes.append("missing_expected_effect_context")
    if counterfactual_matched_expected_effect:
        counterfactual_classifier_block_reason_codes.append("matched_expected_effect")
    if not observed_effect_absent:
        counterfactual_classifier_block_reason_codes.append("attempt_present_but_non_effect_not_confirmed")
    if not (expected_target_id or expected_effect_target_id or expected_region_reached or attempted_boundary_contact or attempted_salient_contact):
        counterfactual_classifier_block_reason_codes.append("target_or_region_missing")
    counterfactual_evidence_observed = bool(
        counterfactual_non_effect_confirmed
        and counterfactual_attempt_context_present
    )
    if attempted_escape_direction is None and isinstance(target, (list, tuple)) and len(target) == 2:
        x = float(target[0]); y = float(target[1])
        if x <= 1:
            attempted_escape_direction = "left"
        elif x >= 62:
            attempted_escape_direction = "right"
        elif y <= 1:
            attempted_escape_direction = "up"
        elif y >= 62:
            attempted_escape_direction = "down"
    exit_attempt_evidence_observed = bool(
        exit_attempt_evidence_observed
        or attempted_boundary_contact
        or attempted_portal_contact
        or attempted_terminal_affordance_contact
        or attempted_escape_direction is not None
        or any(bool(dict(step.info).get("repeated_boundary_facing_movement")) for step in steps if isinstance(step.info, dict))
    )
    success_certainty = 0.0
    if done_observed:
        success_certainty += 0.45
    if route_success:
        success_certainty += 0.25
    if terminal_action_executed:
        success_certainty += 0.15
    if terminal_reward_observed > 0.0:
        success_certainty += 0.15
    if not avatar_localization_confident:
        success_certainty *= 0.7
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
        "directed_outcome_relation_supported": _evidence_cell(directed_outcome_relation_supported, "execution_derived"),
        "exit_attempt_observed": _evidence_cell(exit_attempt_observed, "execution_derived"),
        "exit_attempt_evidence_observed": _evidence_cell(exit_attempt_evidence_observed, "execution_derived"),
        "exit_attempt_target_id": _evidence_cell(exit_attempt_target_id, "execution_derived" if exit_attempt_target_id is not None else "unknown"),
        "exit_attempt_boundary_contact": _evidence_cell(exit_attempt_boundary_contact, "execution_derived"),
        "exit_attempt_action_type": _evidence_cell(exit_attempt_action_type, "execution_derived"),
        "attempted_boundary_contact": _evidence_cell(attempted_boundary_contact, "execution_derived"),
        "attempted_portal_contact": _evidence_cell(attempted_portal_contact, "execution_derived"),
        "attempted_terminal_affordance_contact": _evidence_cell(attempted_terminal_affordance_contact, "execution_derived"),
        "attempted_escape_direction": _evidence_cell(attempted_escape_direction, "execution_derived" if attempted_escape_direction is not None else "unknown"),
        "counterfactual_evidence_observed": _evidence_cell(counterfactual_evidence_observed, "execution_derived"),
        "counterfactual_expectation_basis_present": _evidence_cell(counterfactual_expectation_basis_present, "execution_derived"),
        "counterfactual_attempt_context_present": _evidence_cell(counterfactual_attempt_context_present, "execution_derived"),
        "counterfactual_non_effect_confirmed": _evidence_cell(counterfactual_non_effect_confirmed, "execution_derived"),
        "counterfactual_matched_expected_effect": _evidence_cell(counterfactual_matched_expected_effect, "execution_derived"),
        "counterfactual_match_reason_code": _evidence_cell(counterfactual_match_reason_code, "execution_derived"),
        "counterfactual_observed_effect_candidate_count": _evidence_cell(counterfactual_observed_effect_candidate_count, "execution_derived"),
        "counterfactual_matching_effect_candidate_count": _evidence_cell(counterfactual_matching_effect_candidate_count, "execution_derived"),
        "counterfactual_expected_target_kind": _evidence_cell(expected_target_kind, "execution_derived"),
        "counterfactual_expected_target_value": _evidence_cell(expected_target_value, "execution_derived"),
        "counterfactual_post_attempt_window_steps": _evidence_cell(counterfactual_post_attempt_window_steps, "execution_derived"),
        "counterfactual_classifier_block_reason_codes": _evidence_cell(list(counterfactual_classifier_block_reason_codes), "execution_derived"),
        "expectation_basis": _evidence_cell(expectation_basis, "execution_derived" if expectation_basis is not None else "unknown"),
        "expected_effect_absent": _evidence_cell(observed_effect_absent, "execution_derived"),
        "observed_effect_absent": _evidence_cell(observed_effect_absent, "execution_derived"),
        "expected_effect_target_id": _evidence_cell(expected_effect_target_id, "execution_derived" if expected_effect_target_id is not None else "unknown"),
        "expected_effect_type": _evidence_cell(expected_effect_type, "execution_derived" if expected_effect_type else "unknown"),
        "expected_effect_relation": _evidence_cell(expected_effect_relation, "execution_derived" if expected_effect_relation else "unknown"),
        "expected_relation_type": _evidence_cell(expected_effect_relation, "execution_derived" if expected_effect_relation else "unknown"),
        "expected_trigger_contact_observed": _evidence_cell(expected_trigger_contact_observed, "execution_derived"),
        "expected_region_reached": _evidence_cell(expected_region_reached, "execution_derived"),
        "observed_effect_change": _evidence_cell(observed_effect_change, "execution_derived"),
        "trigger_contact_observed": _evidence_cell(trigger_contact_observed, "execution_derived"),
        "target_contact_observed": _evidence_cell(target_contact_observed, "execution_derived"),
        "weak_expectation_basis": _evidence_cell(list(weak_expectation_basis), "execution_derived" if weak_expectation_basis else "unknown"),
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
    selected_chain = dict(dict(request.metadata or {}).get("selected_subgoal_chain", {}) or {})
    selected_step = dict(dict(request.metadata or {}).get("selected_subgoal_step", {}) or {})
    planner_candidate = dict(dict(request.metadata or {}).get("planner_candidate", {}) or {})
    chain_id = str(dict(request.metadata or {}).get("chain_id") or selected_chain.get("chain_id") or "") or None
    step_id = str(dict(request.metadata or {}).get("step_id") or selected_step.get("step_id") or "") or None
    step_kind = str(dict(request.metadata or {}).get("step_kind") or selected_step.get("step_kind") or "") or None
    current_step = dict(selected_step or {})
    if selected_chain and not current_step:
        step_index = int(dict(request.metadata or {}).get("step_index", selected_chain.get("current_step_index", 0)) or 0)
        chain_steps = list(selected_chain.get("steps", []) or [])
        if 0 <= step_index < len(chain_steps):
            current_step = dict(chain_steps[step_index])
    expected_evidence = list(current_step.get("expected_evidence", []) or [])
    expected_evidence_seen = [
        key for key in expected_evidence
        if (
            ("contact" in key and bool(objective_contact_observed))
            or ("done" in key and bool(done_observed))
            or ("match" in key and int(effect_changed_cells_observed or 0) > 0)
            or ("gate_state" in key and int(effect_changed_cells_observed or 0) > 0)
            or ("exit_attempt" in key and (environment_terminal_success or terminal_action_executed))
            or ("reobserved" in key and bool(target_presence_observed))
        )
    ]
    expected_evidence_missing = [key for key in expected_evidence if key not in expected_evidence_seen]
    normalized_step_kind = "reobserve_remote_change" if step_kind == "reobserve_region" else step_kind
    step_success = bool(
        (step_kind == "attempt_exit" and environment_terminal_success)
        or (step_kind in {"go_to_trigger", "retry_trigger", "verify_trigger_contact"} and objective_contact_observed)
        or (step_kind in {"verify_panel", "verify_gate"} and (target_presence_observed or int(effect_changed_cells_observed or 0) > 0))
        or (normalized_step_kind == "reobserve_remote_change" and target_presence_observed)
    ) if step_kind else False
    step_failure_reason = None
    if step_kind and not step_success:
        if blocked:
            step_failure_reason = "blocked"
        elif stalled:
            step_failure_reason = "stalled"
        elif explicit_failure is not None:
            step_failure_reason = str(explicit_failure.get("failure_reason", "execution_failed"))
        elif expected_evidence_missing:
            step_failure_reason = "expected_evidence_missing"
        else:
            step_failure_reason = termination_reason
    chain_should_advance = bool(chain_id and step_success)
    chain_should_retry = bool(chain_id and not step_success and normalized_step_kind in {"go_to_trigger", "verify_trigger_contact", "verify_panel", "verify_gate", "reobserve_remote_change"} and not blocked and not environment_terminal_success)
    chain_should_abort = bool(chain_id and not chain_should_advance and not chain_should_retry and step_kind is not None)
    chain_completion_reason = None
    if chain_should_advance and step_kind == "attempt_exit" and environment_terminal_success:
        chain_completion_reason = "exit_reached"
    elif chain_should_advance:
        chain_completion_reason = "step_verified"
    elif chain_should_abort:
        chain_completion_reason = str(step_failure_reason or termination_reason or "chain_aborted")
    exit_attempt_failed = bool(step_kind == "attempt_exit" and not environment_terminal_success)
    position_hold_detected = bool(step_kind == "attempt_exit" and termination_reason == "position_hold")
    new_support_since_previous_exit_attempt = bool(planner_candidate.get("has_new_support_since_last_exit_attempt", False))
    exit_attempt_failed_without_new_support = bool(exit_attempt_failed and not new_support_since_previous_exit_attempt)
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
        "avatar_cell_before": first_info.get("avatar_cell_before"),
        "avatar_cell_after": last_info.get("avatar_cell_after") or last_routed.get("avatar"),
        "avatar_confidence_before": avatar_confidence_before,
        "avatar_confidence_after": avatar_confidence_after,
        "avatar_source_before": str(first_info.get("avatar_source_before") or first_info.get("avatar_source") or "unknown"),
        "avatar_source_after": avatar_source_after,
        "avatar_localization_confident": avatar_localization_confident,
        "avatar_localization_source": avatar_source_after,
        "avatar_localization_ambiguous": avatar_ambiguous_after,
        "avatar_mode_status": avatar_mode_status,
        "avatar_status": avatar_status,
        "route_progress_based_on_confident_avatar": route_progress_based_on_confident_avatar,
        "trigger_contact_based_on_confident_avatar": trigger_contact_based_on_confident_avatar,
        "target_approach_based_on_confident_avatar": target_approach_based_on_confident_avatar,
        "directed_outcome_relation_supported": directed_outcome_relation_supported,
        "exit_attempt_observed": exit_attempt_observed,
        "exit_attempt_evidence_observed": exit_attempt_evidence_observed,
        "exit_attempt_target_id": exit_attempt_target_id,
        "exit_attempt_boundary_contact": exit_attempt_boundary_contact,
        "exit_attempt_action_type": exit_attempt_action_type,
        "counterfactual_evidence_observed": counterfactual_evidence_observed,
        "counterfactual_expectation_basis_present": counterfactual_expectation_basis_present,
        "counterfactual_attempt_context_present": counterfactual_attempt_context_present,
        "counterfactual_non_effect_confirmed": counterfactual_non_effect_confirmed,
        "counterfactual_matched_expected_effect": counterfactual_matched_expected_effect,
        "counterfactual_match_reason_code": counterfactual_match_reason_code,
        "counterfactual_observed_effect_candidate_count": int(counterfactual_observed_effect_candidate_count),
        "counterfactual_matching_effect_candidate_count": int(counterfactual_matching_effect_candidate_count),
        "counterfactual_expected_target_kind": expected_target_kind,
        "counterfactual_expected_target_value": expected_target_value or None,
        "counterfactual_post_attempt_window_steps": int(counterfactual_post_attempt_window_steps),
        "counterfactual_classifier_block_reason_codes": list(counterfactual_classifier_block_reason_codes),
        "expectation_basis": expectation_basis,
        "expected_effect_absent": observed_effect_absent,
        "expected_effect_target_id": expected_effect_target_id,
        "expected_effect_type": expected_effect_type or None,
        "expected_effect_relation": expected_effect_relation or None,
        "expected_relation_type": expected_effect_relation or None,
        "expected_target_id": expected_target_id,
        "counterfactual_target_scope": counterfactual_target_scope,
        "expected_trigger_contact_observed": expected_trigger_contact_observed,
        "expected_region_reached": expected_region_reached,
        "observed_effect_change": observed_effect_change,
        "observed_effect_absent": observed_effect_absent,
        "weak_expectation_basis": list(weak_expectation_basis),
        "attempted_boundary_contact": attempted_boundary_contact,
        "attempted_portal_contact": attempted_portal_contact,
        "attempted_terminal_affordance_contact": attempted_terminal_affordance_contact,
        "attempted_escape_direction": attempted_escape_direction,
        "trigger_contact_observed": trigger_contact_observed,
        "target_contact_observed": target_contact_observed,
        "avatar_positions": avatar_positions,
        "routed_actions": routed_history,
        "telemetry": {
            "route_failure_reasons": [str(row.get("failure_reason")) for row in routed_history if row.get("failed")],
            "target_missing_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "missing_target"),
            "avatar_missing_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "missing_avatar"),
            "avatar_uncertainty_failure_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "avatar_localization_low_confidence"),
            "avatar_mode_unsupported_count": sum(1 for row in routed_history if str(row.get("failure_reason")) == "avatar_mode_unsupported"),
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
            "trigger_contact_achieved": bool(trigger_contact_based_on_confident_avatar and str(request.required_action_family or "") == "interact"),
            "remote_region_changed": bool(effect_region_observed and effect_changed_cells_observed > 0),
            "gate_region_state_changed": bool(target_approach_based_on_confident_avatar and effect_region_observed and str(request.objective.get("candidate_class") or "") in {"verify_gate_match", "unlock_then_exit"}),
            "exit_attempt_succeeded_after_prerequisite": bool(avatar_localization_confident and environment_terminal_success and str(request.objective.get("candidate_class") or "") in {"unlock_then_exit", "trigger_then_target"}),
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
        "chain_id": chain_id,
        "step_id": step_id,
        "step_kind": step_kind,
        "step_result": {
            "step_kind": step_kind,
            "step_success": step_success,
            "step_failure_reason": step_failure_reason,
            "expected_evidence_seen": expected_evidence_seen,
            "expected_evidence_missing": expected_evidence_missing,
            "chain_should_advance": chain_should_advance,
            "chain_should_retry": chain_should_retry,
            "chain_should_abort": chain_should_abort,
        },
        "step_success": step_success,
        "step_failure_reason": step_failure_reason,
        "expected_evidence_seen": expected_evidence_seen,
        "expected_evidence_missing": expected_evidence_missing,
        "chain_should_advance": chain_should_advance,
        "chain_should_retry": chain_should_retry,
        "chain_should_abort": chain_should_abort,
        "chain_completion_reason": chain_completion_reason,
        "exit_attempt_failed": exit_attempt_failed,
        "exit_attempt_failure_reason": step_failure_reason if exit_attempt_failed else None,
        "exit_attempt_failed_without_new_support": exit_attempt_failed_without_new_support,
        "position_hold_detected": position_hold_detected,
        "new_support_since_previous_exit_attempt": new_support_since_previous_exit_attempt,
        "exit_readiness_score": float(planner_candidate.get("exit_readiness_score", 0.0) or 0.0),
        "missing_prerequisites": list(planner_candidate.get("missing_prerequisite_types", []) or []),
        "chain_status_before": dict(request.metadata or {}).get("chain_status_before"),
        "chain_status_after": "completed" if chain_should_advance and step_kind == "attempt_exit" and environment_terminal_success else "advanced" if chain_should_advance else "retrying" if chain_should_retry else "aborted" if chain_should_abort else dict(request.metadata or {}).get("chain_status_before"),
    }
