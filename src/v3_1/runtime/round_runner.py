from __future__ import annotations

from dataclasses import dataclass

from v3_1.contracts.messages import PlannerDecision
from v3_1.contracts.versions import CompatibilityStamp
from v3_1.execution.executor_service import build_executor_request
from v3_1.planning.decision import final_action_from_candidate
from v3_1.planning.subgoal_chain_manager import SubgoalChainManager
from v3_1.runtime.export_assembler import (
    actual_effect_mode,
    available_families_from_blackboard,
    build_live_candidate_bridge_rows,
    build_round_analysis_summary,
    decision_export_payload,
    episode_export_row,
)
from v3_1.world.blackboard import export_strict_snapshot
from v3_1.runtime.invalidation import invalidate_if_needed
from v3_1.runtime.session_ledger import (
    AnalysisCompletedPayload,
    EpisodeExecutedPayload,
    HypothesisGenerationPayload,
    LLMOperationPayload,
    MechanicGraphMergeCompletedPayload,
    MemoryReconcilePayload,
    MergeCompletedPayload,
    PlanSelectedPayload,
    RoundStartPayload,
    DetectorPoiEscalationPayload,
    SubgoalChainAbandonedPayload,
    SubgoalChainAbortedPayload,
    SubgoalChainAdvancedPayload,
    SubgoalChainCompletedPayload,
    SubgoalChainStartedPayload,
    SubgoalChainStepActivatedPayload,
    SubgoalChainStepProgressedPayload,
    SubgoalChainStepPayload,
)
from v3_1.visualization.heatmaps import (
    build_poi_heatmap,
    build_visit_heatmap,
    render_combined_overlay_png,
    render_overlay_png,
)


def _target_effect_payload(step_rows: list[dict], target_entity_id: str | None) -> dict:
    if not target_entity_id:
        return {}
    candidate_rows = [dict(row) for row in list(step_rows or []) if str(row.get("target_entity_id") or "") == str(target_entity_id)]
    rows_to_use = candidate_rows or [dict(row) for row in list(step_rows or [])]
    movement_attempts = 0
    interact_attempts = 0
    click_attempts = 0
    movement_effect_sum = 0
    interact_effect_sum = 0
    click_effect_sum = 0
    for row in rows_to_use:
        family = str(row.get("action_family") or "unknown").strip().lower()
        changed_cells = int(row.get("changed_cells", 0) or 0)
        if family == "move":
            movement_attempts += 1
            movement_effect_sum += changed_cells
        elif family == "interact":
            interact_attempts += 1
            interact_effect_sum += changed_cells
        elif family == "click_at":
            click_attempts += 1
            click_effect_sum += changed_cells
    movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts) / 50.0) if movement_attempts > 0 else 0.0)
    interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts) / 50.0) if interact_attempts > 0 else 0.0)
    click_effect_score = min(1.0, (click_effect_sum / float(click_attempts) / 50.0) if click_attempts > 0 else 0.0)
    if interact_attempts > 0:
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif click_attempts > 0:
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": movement_attempts,
        "interact_attempts": interact_attempts,
        "click_attempts": click_attempts,
        "movement_effect_sum": movement_effect_sum,
        "interact_effect_sum": interact_effect_sum,
        "click_effect_sum": click_effect_sum,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


def _analysis_effect_payload(analyzed_episodes: list[object]) -> dict:
    step_rows = [
        dict(row)
        for episode in list(analyzed_episodes or [])
        for row in list(getattr(episode, "summary", {}).get("step_rows", []) or [])
    ]
    move_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "move")
    interact_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "interact")
    click_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "click_at")
    movement_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "move" and int(row.get("changed_cells", 0) or 0) > 0)
    interact_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "interact" and int(row.get("changed_cells", 0) or 0) > 0)
    click_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "click_at" and int(row.get("changed_cells", 0) or 0) > 0)
    movement_effect_score = (float(movement_steps_with_change) / float(move_steps)) if move_steps > 0 else 0.0
    interact_effect_score = (float(interact_steps_with_change) / float(interact_steps)) if interact_steps > 0 else 0.0
    click_effect_score = (float(click_steps_with_change) / float(click_steps)) if click_steps > 0 else 0.0
    if interact_effect_score > 0.0:
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif click_effect_score > 0.0:
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": move_steps,
        "interact_attempts": interact_steps,
        "click_attempts": click_steps,
        "movement_effect_sum": movement_steps_with_change,
        "interact_effect_sum": interact_steps_with_change,
        "click_effect_sum": click_steps_with_change,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


def _episode_classifier_payload(outcome) -> dict:
    payload = dict(getattr(outcome, "outcome", {}) or {})
    return {
        "counterfactual_evidence_observed": payload.get("counterfactual_evidence_observed"),
        "exit_attempt_evidence_observed": payload.get("exit_attempt_evidence_observed"),
        "expected_effect_type": payload.get("expected_effect_type"),
        "expected_relation_type": payload.get("expected_relation_type") or payload.get("expected_effect_relation"),
        "expected_target_id": payload.get("expected_target_id") or payload.get("expected_effect_target_id"),
        "expected_trigger_contact_observed": payload.get("expected_trigger_contact_observed"),
        "expected_region_reached": payload.get("expected_region_reached"),
        "observed_effect_change": payload.get("observed_effect_change"),
        "observed_effect_absent": payload.get("observed_effect_absent") if "observed_effect_absent" in payload else payload.get("expected_effect_absent"),
        "attempted_boundary_contact": payload.get("attempted_boundary_contact"),
        "attempted_portal_contact": payload.get("attempted_portal_contact"),
        "attempted_terminal_affordance_contact": payload.get("attempted_terminal_affordance_contact"),
        "attempted_escape_direction": payload.get("attempted_escape_direction"),
        "exit_attempt_target_id": payload.get("exit_attempt_target_id"),
    }


def _count_exit_attempt_family_rows(deltas: list[dict]) -> int:
    count = 0
    for delta in list(deltas or []):
        for row in list(dict(delta or {}).get("consequences", ()) or []):
            payload = dict(row or {})
            if bool(payload.get("supports_exit_attempt_relation", False)) or str(payload.get("support_family") or "") == "exit_attempt":
                count += 1
    return int(count)


def _sync_analysis_poi_debug_with_observed_store(analysis_summary: dict, blackboard_state: dict) -> dict:
    payload = dict(analysis_summary or {})
    poi_debug = dict(payload.get("poi_detection_debug", {}) or {})
    collapsed_peer_ids = {str(value) for value in list(poi_debug.get("collapsed_peer_ids", []) or []) if value}
    observed_entities = dict(blackboard_state.get("observed_entities", {}) or {})
    observed_pois = [
        dict(row)
        for row in observed_entities.values()
        if isinstance(row, dict) and str(row.get("kind") or "") == "poi" and bool(row.get("planner_visible", True))
    ]
    remaining_collapsed = sorted(
        str(row.get("entity_id") or row.get("poi_id") or "")
        for row in observed_pois
        if str(row.get("entity_id") or row.get("poi_id") or "") in collapsed_peer_ids
    )
    if remaining_collapsed:
        raise AssertionError(f"collapsed peers still present in observed store: {remaining_collapsed}")
    observed_pois.sort(
        key=lambda row: (
            -float(row.get("confidence", 0.0) or 0.0),
            -float(row.get("utility", 0.0) or 0.0),
            str(row.get("entity_id") or ""),
        )
    )
    poi_debug["final_exported_canonical_pois"] = [
        {
            "poi_id": str(row.get("entity_id") or row.get("poi_id") or ""),
            "entity_id": str(row.get("entity_id") or row.get("poi_id") or ""),
            "planner_visible": bool(row.get("planner_visible", True)),
            "poi_bucket": str(row.get("poi_bucket") or ""),
            "poi_class": str(row.get("poi_class") or row.get("kind") or ""),
            "bbox": dict(row.get("bbox", {}) or {}),
            "area": int(row.get("area", 0) or 0),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "poi_source_provenance": list(row.get("poi_source_provenance", []) or []),
            "merged_input_poi_ids": list(row.get("merged_input_poi_ids", []) or []),
            "planner_targetable": bool(row.get("planner_targetable", False)),
            "poi_hierarchy_level": int(row.get("poi_hierarchy_level", 0) or 0),
            "parent_poi_id": row.get("parent_poi_id"),
            "child_poi_ids": list(row.get("child_poi_ids", []) or []),
            "hierarchy_role": str(row.get("hierarchy_role") or "parent_region"),
        }
        for row in observed_pois
    ]
    poi_debug["final_exported_count"] = len(observed_pois)
    poi_debug["final_post_merge_observed_poi_count"] = len(observed_pois)
    payload["poi_detection_debug"] = poi_debug
    return payload


def _merge_effect_payload(primary: dict, fallback: dict) -> dict:
    merged = dict(primary or {})
    if float(merged.get("candidate_effect_score", 0.0) or 0.0) > 0.0:
        return merged
    for key, value in dict(fallback or {}).items():
        merged[key] = value
    return merged


def _apply_target_effect_to_blackboard(snapshot, *, target_entity_id: str | None, step_rows: list[dict]) -> dict:
    payload = _target_effect_payload(step_rows, target_entity_id)
    if not payload:
        return {}
    state = getattr(snapshot, "state", None)
    if not isinstance(state, dict):
        return payload
    entities = state.get("entities")
    if not isinstance(entities, dict) or not target_entity_id or target_entity_id not in entities:
        return payload
    entity = dict(entities.get(target_entity_id, {}))
    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
        entity[field] = int(entity.get(field, 0) or 0) + int(payload.get(field, 0) or 0)
    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(payload.get(field, 0.0) or 0.0))
    if float(payload.get("candidate_effect_score", 0.0) or 0.0) >= float(entity.get("candidate_effect_score", 0.0) or 0.0):
        entity["candidate_effect_mode"] = payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
    elif "candidate_effect_mode" not in entity:
        entity["candidate_effect_mode"] = payload.get("candidate_effect_mode")
    entities[str(target_entity_id)] = entity
    return payload


def _apply_target_effect_to_decision_export(payload: dict, *, target_entity_id: str | None, effect_payload: dict) -> dict:
    updated = dict(payload or {})
    if not target_entity_id or not effect_payload:
        return updated
    selected_action = dict(updated.get("selected_action", {}) or {})
    if str(selected_action.get("target_entity_id") or selected_action.get("target") or "") == str(target_entity_id):
        selected_action["candidate_effect_score"] = max(float(selected_action.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
        selected_action["effect_action_family"] = effect_payload.get("candidate_effect_mode", selected_action.get("effect_action_family"))
        updated["selected_action"] = selected_action
    metadata = dict(updated.get("metadata", {}) or {})
    selected_candidate = dict(metadata.get("selected_candidate", {}) or {})
    if str(selected_candidate.get("target_entity_id") or selected_candidate.get("target") or "") == str(target_entity_id):
        selected_candidate["candidate_effect_score"] = max(float(selected_candidate.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
        selected_candidate["effect_action_family"] = effect_payload.get("candidate_effect_mode", selected_candidate.get("effect_action_family"))
        metadata["selected_candidate"] = selected_candidate
    ranked = []
    for row in list(updated.get("ranked_candidates", []) or []):
        candidate = dict(row)
        if str(candidate.get("target_entity_id") or candidate.get("target") or "") == str(target_entity_id):
            candidate["candidate_effect_score"] = max(float(candidate.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
            candidate["effect_action_family"] = effect_payload.get("candidate_effect_mode", candidate.get("effect_action_family"))
        ranked.append(candidate)
    updated["ranked_candidates"] = ranked
    updated["metadata"] = metadata
    return updated


def _round_effect_payload(analysis_summary: dict, actual_mode: str) -> dict:
    move_steps = int(analysis_summary.get("move_steps_count", 0) or 0)
    move_changes = int(analysis_summary.get("movement_steps_with_change", 0) or 0)
    interact_steps = int(analysis_summary.get("interact_steps_count", 0) or 0)
    interact_changes = int(analysis_summary.get("interact_steps_with_change", 0) or 0)
    click_steps = int(analysis_summary.get("click_steps_count", 0) or 0)
    click_changes = int(analysis_summary.get("click_steps_with_change", 0) or 0)
    movement_effect_score = (float(move_changes) / float(move_steps)) if move_steps > 0 else 0.0
    interact_effect_score = (float(interact_changes) / float(interact_steps)) if interact_steps > 0 else 0.0
    click_effect_score = (float(click_changes) / float(click_steps)) if click_steps > 0 else 0.0
    if actual_mode == "interact":
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif actual_mode == "click_at":
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": move_steps,
        "interact_attempts": interact_steps,
        "click_attempts": click_steps,
        "movement_effect_sum": move_changes,
        "interact_effect_sum": interact_changes,
        "click_effect_sum": click_changes,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


def _detector_backed_selected_poi(decision_export: dict) -> dict:
    selected = dict(dict(decision_export.get("metadata", {}) or {}).get("selected_candidate", {}) or {})
    provenance = {str(value) for value in list(selected.get("poi_source_provenance", []) or []) if value}
    if str(selected.get("candidate_class") or "") != "route_probe":
        return {}
    if "detector" not in provenance:
        return {}
    return selected


def _update_poi_followthrough_state(*, latest_memory, selected_candidate: dict, mg_counts: dict, hypothesis_registry_snapshot: dict, round_id: int, selected_subgoal_chain: dict | None = None, chain_outcome_update: dict | None = None, selected_outcome: dict | None = None) -> tuple[dict, list[tuple[str, DetectorPoiEscalationPayload]]]:
    state = getattr(latest_memory, "state", None)
    if not isinstance(state, dict):
        return {}, []
    working = dict(state.get("working_memory", {}) or {})
    plan_memory = dict(working.get("plan_memory", {}) or {})
    followthrough = {
        str(key): dict(value)
        for key, value in dict(plan_memory.get("poi_followthrough", {}) or {}).items()
        if isinstance(value, dict)
    }
    poi_id = str(selected_candidate.get("target_entity_id") or "")
    events: list[tuple[str, DetectorPoiEscalationPayload]] = []
    if not poi_id:
        plan_memory["poi_followthrough"] = followthrough
        working["plan_memory"] = plan_memory
        state["working_memory"] = working
        return followthrough, events
    row = dict(followthrough.get(poi_id, {}) or {})
    first_selection_round = int(row.get("selection_round", round_id) or round_id)
    revisit_count = int(row.get("revisit_count", 0) or 0) + (1 if row else 0)
    selection_rounds = int(row.get("selection_rounds", 0) or 0) + 1
    new_graph_edges = int(mg_counts.get("edge_count_added", 0) or 0)
    graph_support_gain = int(mg_counts.get("observed_edge_count_added", 0) or 0)
    validation_state = dict(hypothesis_registry_snapshot.get("validation_state", {}) or {})
    new_hypothesis_support = sum(1 for state_value in validation_state.values() if str(state_value or "") in {"supported", "validated", "planner_usable"})
    new_verification_candidates = int(mg_counts.get("observed_edge_count_added", 0) or 0)
    changed_exit_linked_evidence = int(mg_counts.get("observed_edge_count_added", 0) or 0)
    identity_gain = int(mg_counts.get("node_count_added", 0) or 0)
    usable_trigger_gain = int(mg_counts.get("observed_edge_count_added", 0) or 0)
    durable_gain = sum(1 for state_value in validation_state.values() if str(state_value or "") in {"planner_usable", "durable_ready"})
    contradiction_gain = 0
    new_support_delta = graph_support_gain + identity_gain + usable_trigger_gain + durable_gain + new_hypothesis_support + new_verification_candidates
    escalated_candidate_ids = list(dict(row).get("escalated_candidate_ids", []) or [])
    if selected_subgoal_chain:
        escalated_candidate_ids.append(str(dict(selected_subgoal_chain).get("source_candidate_id") or ""))
    chain_failure_count = int(row.get("chain_step_failure_count", 0) or 0)
    last_chain_failure_round = int(row.get("last_chain_failure_round", 0) or 0)
    stale_dead_end_count = int(row.get("stale_dead_end_count", 0) or 0)
    chain_outcome_update = dict(chain_outcome_update or {})
    selected_outcome = dict(selected_outcome or {})
    if bool(chain_outcome_update.get("runtime_progress_updated", False)) and any(str(dict(evt).get("event_type") or "") in {"subgoal chain abandoned", "subgoal chain step failed"} for evt in list(chain_outcome_update.get("runtime_chain_rows", []) or [])):
        chain_failure_count += 1
        last_chain_failure_round = int(round_id)
    if new_support_delta <= 0:
        stale_dead_end_count += 1
    probe_stale = selection_rounds >= 2 and new_support_delta <= 0
    next_row = {
        **row,
        "poi_id": poi_id,
        "selection_round": first_selection_round,
        "last_selection_round": int(round_id),
        "selection_rounds": selection_rounds,
        "revisit_count": revisit_count,
        "new_graph_edges": new_graph_edges,
        "graph_support_gain": graph_support_gain,
        "support_gain_since_last_visit": new_support_delta,
        "identity_gain_since_last_visit": identity_gain,
        "contradiction_gain_since_last_visit": contradiction_gain,
        "usable_trigger_gain_since_last_visit": usable_trigger_gain,
        "durable_gain_since_last_visit": durable_gain,
        "new_hypothesis_support": new_hypothesis_support,
        "new_verification_candidates": new_verification_candidates,
        "changed_exit_linked_evidence": changed_exit_linked_evidence,
        "new_support_delta": new_support_delta,
        "probe_stale": probe_stale,
        "stale_dead_end_count": stale_dead_end_count,
        "last_chain_failure_round": last_chain_failure_round if last_chain_failure_round > 0 else None,
        "chain_step_failure_count": chain_failure_count,
        "detector_backed": True,
        "detector_strength": float(selected_candidate.get("confidence", 0.0) or 0.0),
        "reachable": bool(selected_candidate.get("reachable_now")) or bool(selected_candidate.get("reachable_later")),
        "approachable": str(selected_candidate.get("approachable_status") or "") in {"approachable_now", "approachable_later"},
        "supporting_refs": list(selected_candidate.get("supporting_evidence_refs", []) or []),
        "escalated_candidate_ids": list(dict.fromkeys(str(value) for value in escalated_candidate_ids if value)),
    }
    followthrough[poi_id] = next_row
    plan_memory["poi_followthrough"] = followthrough
    working["plan_memory"] = plan_memory
    state["working_memory"] = working
    event_type = "detector poi revisited" if row else "detector poi selected"
    events.append((event_type, DetectorPoiEscalationPayload(
        poi_id=poi_id,
        detector_strength=float(next_row.get("detector_strength", 0.0) or 0.0),
        selection_round=int(first_selection_round),
        revisit_count=int(next_row.get("revisit_count", 0) or 0),
        downstream_support_gained={
            "new_graph_edges": new_graph_edges,
            "graph_support_gain": graph_support_gain,
            "identity_gain_since_last_visit": identity_gain,
            "durable_gain_since_last_visit": durable_gain,
            "new_hypothesis_support": new_hypothesis_support,
            "new_verification_candidates": new_verification_candidates,
            "changed_exit_linked_evidence": changed_exit_linked_evidence,
        },
        escalation_target_candidate_id=(next_row.get("escalated_candidate_ids") or [None])[-1],
        stale_reason="repeated_probe_without_new_effect" if probe_stale else None,
    )))
    if selected_subgoal_chain:
        events.append(("detector poi escalated to chain", DetectorPoiEscalationPayload(
            poi_id=poi_id,
            detector_strength=float(next_row.get("detector_strength", 0.0) or 0.0),
            selection_round=int(first_selection_round),
            revisit_count=int(next_row.get("revisit_count", 0) or 0),
            downstream_support_gained={"new_graph_edges": new_graph_edges, "new_hypothesis_support": new_hypothesis_support},
            escalation_target_candidate_id=str(dict(selected_subgoal_chain).get("source_candidate_id") or "") or None,
        )))
    elif new_support_delta > 0:
        events.append(("detector poi escalated to verification", DetectorPoiEscalationPayload(
            poi_id=poi_id,
            detector_strength=float(next_row.get("detector_strength", 0.0) or 0.0),
            selection_round=int(first_selection_round),
            revisit_count=int(next_row.get("revisit_count", 0) or 0),
            downstream_support_gained={"new_graph_edges": new_graph_edges, "new_hypothesis_support": new_hypothesis_support, "new_verification_candidates": new_verification_candidates},
            escalation_target_candidate_id=None,
        )))
    if probe_stale:
        events.append(("detector poi marked stale", DetectorPoiEscalationPayload(
            poi_id=poi_id,
            detector_strength=float(next_row.get("detector_strength", 0.0) or 0.0),
            selection_round=int(first_selection_round),
            revisit_count=int(next_row.get("revisit_count", 0) or 0),
            downstream_support_gained={"new_support_delta": new_support_delta},
            stale_reason="no_new_support_after_repeated_probe",
        )))
    return followthrough, events


def _update_mode_persistence_state(*, latest_memory, decision, round_id: int, mg_counts: dict, selected_outcome: dict | None = None, prior_mode_state: dict | None = None) -> dict:
    state = getattr(latest_memory, "state", None)
    if not isinstance(state, dict):
        return {}
    working = dict(state.get("working_memory", {}) or {})
    plan_memory = dict(working.get("plan_memory", {}) or {})
    metadata = dict(getattr(decision, "metadata", {}) or {})
    planner_trace = dict(metadata.get("planner_trace", {}) or {})
    current_mode = str(planner_trace.get("planning_mode") or "default_progress")
    previous_mode = str(planner_trace.get("previous_planning_mode") or plan_memory.get("mode_persistence", {}).get("last_planning_mode") or "")
    prior = dict(plan_memory.get("mode_persistence", {}) or {})
    if not prior:
        prior = dict(prior_mode_state or {})
    prior_history = list(prior.get("history", []) or [])
    selected_candidate = dict(metadata.get("selected_candidate", {}) or {})
    objective_type = str(selected_candidate.get("objective_type") or "")
    structure_support_gain = float(int(dict(mg_counts or {}).get("edge_count_added", 0) or 0) > 0)
    structure_support_gain += float(int(dict(mg_counts or {}).get("observed_edge_count_added", 0) or 0) > 0)
    if objective_type in {"trigger_then_target", "verify_trigger_contact", "reobserve_remote_change", "verify_panel_state", "verify_gate_match"}:
        structure_support_gain += 0.5
    progress_support_gain = float(bool(dict(selected_outcome or {}).get("progress", 0.0)))
    if objective_type in {"explore_frontier", "probe_route", "interact"}:
        progress_support_gain += 0.25
    last_structure_gain_round = int(prior.get("last_structure_support_gain_round", 0) or 0)
    last_progress_gain_round = int(prior.get("last_progress_support_gain_round", 0) or 0)
    if structure_support_gain > 0.0:
        last_structure_gain_round = int(round_id)
    if progress_support_gain > 0.0:
        last_progress_gain_round = int(round_id)
    switch_applied = bool(planner_trace.get("mode_switch_applied", False))
    next_state = {
        "last_planning_mode": current_mode,
        "last_planning_mode_committed": current_mode,
        "last_mode_reason": str(planner_trace.get("mode_switch_reason") or planner_trace.get("mode_switch_block_reason") or ""),
        "last_mode_round_id": int(round_id),
        "consecutive_structure_acquisition_rounds": int(planner_trace.get("mode_persistence_summary", {}).get("consecutive_structure_acquisition_rounds", prior.get("consecutive_structure_acquisition_rounds", 0)) or 0),
        "consecutive_default_progress_rounds": int(planner_trace.get("mode_persistence_summary", {}).get("consecutive_default_progress_rounds", prior.get("consecutive_default_progress_rounds", 0)) or 0),
        "recent_structure_support_gain": float(planner_trace.get("recent_structure_support_gain", structure_support_gain) or 0.0),
        "recent_progress_evidence_gain": float(planner_trace.get("recent_progress_evidence_gain", progress_support_gain) or 0.0),
        "last_structure_support_gain_round": last_structure_gain_round,
        "last_progress_support_gain_round": last_progress_gain_round,
        "mode_switch_applied_last_round": switch_applied,
        "previous_planning_mode": previous_mode,
        "history": [
            *prior_history[-5:],
            {
                "round_id": int(round_id),
                "planning_mode": current_mode,
                "previous_planning_mode": previous_mode,
                "mode_switch_applied": switch_applied,
                "mode_switch_reason": str(planner_trace.get("mode_switch_reason") or planner_trace.get("mode_switch_block_reason") or ""),
                "structure_support_gain": structure_support_gain,
                "progress_support_gain": progress_support_gain,
            },
        ][-8:],
    }
    plan_memory["mode_persistence"] = next_state
    working["plan_memory"] = plan_memory
    state["working_memory"] = working
    return next_state


def _inject_mode_persistence(memory_state: dict, mode_state: dict | None) -> dict:
    payload = dict(memory_state or {})
    if not mode_state:
        return payload
    working = dict(payload.get("working_memory", {}) or {})
    plan_memory = dict(working.get("plan_memory", {}) or {})
    plan_memory["mode_persistence"] = dict(mode_state)
    working["plan_memory"] = plan_memory
    payload["working_memory"] = working
    return payload


def _with_committed_previous_mode(decision, prior_mode_state: dict | None):
    committed_previous = dict(prior_mode_state or {}).get("last_planning_mode_committed")
    metadata = dict(getattr(decision, "metadata", {}) or {})
    planner_trace = dict(metadata.get("planner_trace", {}) or {})
    if committed_previous is not None and planner_trace.get("previous_planning_mode") in {"", None}:
        planner_trace["previous_planning_mode"] = committed_previous
    metadata["planner_trace"] = planner_trace
    metadata["previous_planning_mode"] = planner_trace.get("previous_planning_mode")
    return type(decision)(**{**decision.__dict__, "metadata": metadata})


def _build_final_mode_payload(*, round_id: int, decision, prior_committed_mode: str | None, prior_committed_mode_source: str) -> dict:
    metadata = dict(getattr(decision, "metadata", {}) or {})
    planner_trace = dict(metadata.get("planner_trace", {}) or {})
    final_payload = {
        "previous_planning_mode": planner_trace.get("previous_planning_mode"),
        "planning_mode_before_hysteresis": str(planner_trace.get("planning_mode_before_hysteresis") or ""),
        "planning_mode_after_hysteresis": str(planner_trace.get("planning_mode_after_hysteresis") or planner_trace.get("planning_mode") or ""),
        "planning_mode_committed": str(planner_trace.get("planning_mode_committed") or planner_trace.get("planning_mode") or "default_progress"),
        "payload_correction_applied": False,
        "payload_correction_reason": "",
        "prior_committed_mode_loaded": prior_committed_mode,
        "prior_committed_mode_source": prior_committed_mode_source,
        "final_mode_payload_built": True,
    }
    if int(round_id) > 1 and prior_committed_mode not in {"", None} and final_payload["previous_planning_mode"] in {"", None}:
        final_payload["previous_planning_mode"] = prior_committed_mode
        final_payload["payload_correction_applied"] = True
        final_payload["payload_correction_reason"] = "missing_previous_committed_mode_before_persist"
    return final_payload


def _apply_final_mode_payload(decision, final_mode_payload: dict):
    metadata = dict(getattr(decision, "metadata", {}) or {})
    planner_trace = dict(metadata.get("planner_trace", {}) or {})
    metadata["previous_planning_mode_committed_for_trace"] = final_mode_payload.get("previous_planning_mode")
    for key, value in dict(final_mode_payload or {}).items():
        if key in {
            "previous_planning_mode",
            "planning_mode_before_hysteresis",
            "planning_mode_after_hysteresis",
            "planning_mode_committed",
            "payload_correction_applied",
            "payload_correction_reason",
            "prior_committed_mode_loaded",
            "prior_committed_mode_source",
            "final_mode_payload_built",
        }:
            planner_trace[key] = value
            metadata[key] = value
    metadata["planner_trace"] = planner_trace
    return type(decision)(**{**decision.__dict__, "metadata": metadata})


@dataclass
class RoundRunResult:
    latest_blackboard: object
    latest_memory: object
    latest_mechanic_graph: object
    hypothesis_registry_snapshot: dict
    first_observation: list[list[int]] | None
    analyzed_rows: list[dict]
    round_record: dict
    memory_snapshot_path: str | None
    selected_target_entity_id: str | None
    stop_outcome: dict


@dataclass
class RoundRunner:
    config: object
    context: object
    services: dict[str, object]
    snapshot_registry: object
    task_registry: object
    helper_coordinator: object
    pick: object
    submit: object
    resolve: object
    call: object
    planning_context_builder: object
    session_ledger: object | None = None
    subgoal_chain_manager: object | None = None
    last_planning_mode_committed: str | None = None

    def _ensure_chain_manager(self) -> SubgoalChainManager:
        if self.subgoal_chain_manager is None:
            self.subgoal_chain_manager = SubgoalChainManager()
        return self.subgoal_chain_manager

    def _with_decision_metadata(self, decision, *, updates: dict) -> PlannerDecision:
        payload = dict(decision.__dict__)
        metadata = dict(payload.get("metadata", {}) or {})
        metadata.update(dict(updates or {}))
        payload["metadata"] = metadata
        return PlannerDecision(**payload)

    def _materialize_selected_chain_runtime(
        self,
        *,
        decision,
        round_id: int,
        blackboard_version: str,
        memory_version: str,
        plan_context_id: str,
    ) -> tuple[PlannerDecision, dict[str, object]]:
        manager = self._ensure_chain_manager()
        metadata = dict(getattr(decision, "metadata", {}) or {})
        selected_chain = dict(metadata.get("selected_subgoal_chain", {}) or {}) if isinstance(metadata.get("selected_subgoal_chain"), dict) else {}
        selected_step = dict(metadata.get("selected_subgoal_step", {}) or {}) if isinstance(metadata.get("selected_subgoal_step"), dict) else {}
        selected_candidate = dict(metadata.get("selected_candidate", {}) or {}) if isinstance(metadata.get("selected_candidate"), dict) else {}
        runtime_rows: list[dict] = []
        materialized = False
        runtime_chain_event_count = 0
        runtime_step_event_count = 0
        payload_correction_applied = False
        if selected_chain:
            current_active = dict(manager.current_chain() or {})
            current_active_id = str(current_active.get("chain_id") or "")
            selected_chain_id = str(selected_chain.get("chain_id") or "")
            if not current_active or current_active_id != selected_chain_id:
                manager.active_chain = dict(selected_chain)
                manager.active_chain["status"] = str(manager.active_chain.get("status") or "planned")
                manager.active_chain["last_updated_round_id"] = int(round_id)
                manager.retry_counts = {}
                manager._should_replan = False
                manager.chain_history.append({"event": "runtime_materialized", "chain_id": selected_chain_id, "round_id": int(round_id)})
                materialized = True
                current_step = dict(selected_step or manager.current_step() or {})
                if current_step:
                    current_step["step_status"] = str(current_step.get("step_status") or "ready")
                    manager._replace_current_step(current_step)
                runtime_rows.append({"event_type": "subgoal chain started", "round_id": int(round_id), "payload": {"chain_id": selected_chain_id}})
                runtime_chain_event_count += 1
                if self.session_ledger is not None:
                    self.session_ledger.append_subgoal_chain_started(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=blackboard_version,
                        memory_version=memory_version,
                        plan_context_id=plan_context_id,
                        decision_id=str(getattr(decision, "selected_candidate_id", None) or ""),
                        payload=SubgoalChainStartedPayload(
                            chain_id=selected_chain_id,
                            selected_subgoal_chain_id=selected_chain_id or None,
                            selected_subgoal_step_id=str(current_step.get("step_id") or "") or None,
                            current_step_id=str(current_step.get("step_id") or "") or None,
                            step_kind=str(current_step.get("step_kind") or "") or None,
                            expected_evidence=tuple(str(value) for value in list(current_step.get("expected_evidence", []) or []) if value),
                            observed_evidence=(),
                            advancement_reason="runtime_chain_materialized",
                            exit_readiness_score=float(selected_chain.get("exit_readiness_score_at_creation", 0.0) or 0.0),
                            missing_prerequisites=tuple(str(value) for value in list(selected_chain.get("required_verification_steps", []) or []) if value),
                        ),
                    )
            current_step = dict(selected_step or manager.current_step() or {})
            if current_step:
                runtime_rows.append({"event_type": "subgoal chain step activated", "round_id": int(round_id), "payload": {"chain_id": str(selected_chain.get("chain_id") or ""), "step_id": str(current_step.get("step_id") or "")}})
                runtime_step_event_count += 1
                if self.session_ledger is not None:
                    self.session_ledger.append_subgoal_chain_step_activated(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=blackboard_version,
                        memory_version=memory_version,
                        plan_context_id=plan_context_id,
                        decision_id=str(getattr(decision, "selected_candidate_id", None) or ""),
                        payload=SubgoalChainStepActivatedPayload(
                            chain_id=str(selected_chain.get("chain_id") or ""),
                            selected_subgoal_chain_id=str(selected_chain.get("chain_id") or "") or None,
                            selected_subgoal_step_id=str(current_step.get("step_id") or "") or None,
                            current_step_id=str(current_step.get("step_id") or "") or None,
                            step_kind=str(current_step.get("step_kind") or "") or None,
                            expected_evidence=tuple(str(value) for value in list(current_step.get("expected_evidence", []) or []) if value),
                            observed_evidence=(),
                            advancement_reason="step_activated_for_execution",
                            exit_readiness_score=float(selected_chain.get("exit_readiness_score_at_creation", 0.0) or 0.0),
                            missing_prerequisites=tuple(str(value) for value in list(selected_chain.get("required_verification_steps", []) or []) if value),
                        ),
                    )
            runtime_state = manager.snapshot()
            runtime_state["selected_candidate_id"] = str(getattr(decision, "selected_candidate_id", None) or "")
            runtime_state["materialized_this_round"] = bool(materialized)
            runtime_state["runtime_chain_event_count"] = int(runtime_chain_event_count)
            runtime_state["runtime_step_event_count"] = int(runtime_step_event_count)
            runtime_state["prior_runtime_source"] = "round_runner"
            if selected_chain and not runtime_rows:
                raise AssertionError("selected chain present in decision but no chain runtime row materialized")
            if selected_step and runtime_step_event_count <= 0:
                raise AssertionError("selected step present in decision but no chain step runtime row materialized")
            decision = self._with_decision_metadata(
                decision,
                updates={
                    "runtime_subgoal_chain_state": runtime_state,
                    "selected_subgoal_chain": dict(runtime_state.get("active_chain", {}) or selected_chain),
                    "selected_subgoal_step": dict(runtime_state.get("active_step", {}) or selected_step),
                },
            )
        else:
            runtime_state = manager.snapshot()
            decision = self._with_decision_metadata(decision, updates={"runtime_subgoal_chain_state": runtime_state})
        return decision, {
            "runtime_subgoal_chain_state": runtime_state,
            "runtime_chain_rows": runtime_rows,
            "runtime_chain_event_count": int(runtime_chain_event_count),
            "runtime_step_event_count": int(runtime_step_event_count),
            "payload_correction_applied": bool(payload_correction_applied),
        }

    def _maybe_start_chain(self, *, decision, round_id: int, blackboard_version: str, memory_version: str, plan_context_id: str) -> None:
        manager = self._ensure_chain_manager()
        if manager.active_chain:
            return
        metadata = dict(getattr(decision, "metadata", {}) or {})
        selected_chain = dict(metadata.get("selected_subgoal_chain", {}) or {}) if isinstance(metadata.get("selected_subgoal_chain"), dict) else {}
        selected_candidate = dict(metadata.get("selected_candidate", {}) or {}) if isinstance(metadata.get("selected_candidate"), dict) else {}
        if not selected_chain or not selected_candidate:
            return
        started = manager.start_chain(selected_candidate=selected_candidate, round_id=round_id)
        if not started or self.session_ledger is None:
            return
        current_step = dict(manager.current_step() or {})
        self.session_ledger.append_subgoal_chain_started(
            round_id=round_id,
            pass_id=1,
            blackboard_version=blackboard_version,
            memory_version=memory_version,
            plan_context_id=plan_context_id,
            decision_id=str(getattr(decision, "selected_candidate_id", None) or ""),
            payload=SubgoalChainStartedPayload(
                chain_id=str(started.get("chain_id") or ""),
                current_step_id=str(current_step.get("step_id") or "") or None,
                step_kind=str(current_step.get("step_kind") or "") or None,
                expected_evidence=tuple(str(value) for value in list(current_step.get("expected_evidence", []) or []) if value),
                observed_evidence=(),
                advancement_reason="chain_planned",
                exit_readiness_score=float(started.get("exit_readiness_score_at_creation", 0.0) or 0.0),
                missing_prerequisites=tuple(str(value) for value in list(started.get("required_verification_steps", []) or []) if value),
            ),
        )

    def _update_chain_from_outcome(self, *, round_id: int, pass_id: int, plan_context_id: str, blackboard_version: str, memory_version: str, outcome_payload: dict) -> dict[str, object]:
        manager = self._ensure_chain_manager()
        snapshot_before = manager.snapshot()
        if not snapshot_before.get("active_chain"):
            return {"runtime_chain_rows": [], "runtime_chain_event_count": 0, "runtime_step_event_count": 0, "runtime_progress_updated": False}
        manager.update_from_outcome({**dict(outcome_payload or {}), "round_id": int(round_id)})
        snapshot_after = manager.snapshot()
        active_chain = dict(snapshot_before.get("active_chain", {}) or {})
        active_step = dict(snapshot_before.get("active_step", {}) or {})
        runtime_rows: list[dict] = []
        runtime_chain_event_count = 0
        runtime_step_event_count = 0
        common_kwargs = {
            "round_id": round_id,
            "pass_id": pass_id,
            "blackboard_version": blackboard_version,
            "memory_version": memory_version,
            "plan_context_id": plan_context_id,
        }
        if self.session_ledger is not None:
            payload = SubgoalChainStepPayload(
                chain_id=str(outcome_payload.get("chain_id") or active_chain.get("chain_id") or ""),
                selected_subgoal_chain_id=str(active_chain.get("chain_id") or "") or None,
                selected_subgoal_step_id=str(outcome_payload.get("step_id") or active_step.get("step_id") or "") or None,
                current_step_id=str(outcome_payload.get("step_id") or active_step.get("step_id") or "") or None,
                step_kind=str(outcome_payload.get("step_kind") or active_step.get("step_kind") or "") or None,
                expected_evidence=tuple(str(value) for value in list(active_step.get("expected_evidence", []) or []) if value),
                observed_evidence=tuple(str(value) for value in list(outcome_payload.get("expected_evidence_seen", []) or []) if value),
                failure_reason=str(outcome_payload.get("step_failure_reason") or "") or None,
                advancement_reason="step_success" if bool(outcome_payload.get("step_success")) else None,
                exit_readiness_score=float(dict(active_chain).get("exit_readiness_score_at_creation", 0.0) or 0.0),
                missing_prerequisites=tuple(str(value) for value in list(outcome_payload.get("missing_prerequisites", []) or []) if value),
                failed_without_new_support=bool(outcome_payload.get("exit_attempt_failed_without_new_support", False)),
                position_hold_detected=bool(outcome_payload.get("position_hold_detected", False)),
            )
            runtime_rows.append({"event_type": "subgoal chain step progressed", "round_id": int(round_id), "payload": {"chain_id": payload.chain_id, "step_id": payload.current_step_id, "step_success": bool(outcome_payload.get("step_success"))}})
            runtime_step_event_count += 1
            self.session_ledger.append_subgoal_chain_step_progressed(
                payload=SubgoalChainStepProgressedPayload(
                    chain_id=payload.chain_id,
                    selected_subgoal_chain_id=payload.selected_subgoal_chain_id,
                    selected_subgoal_step_id=payload.selected_subgoal_step_id,
                    current_step_id=payload.current_step_id,
                    step_kind=payload.step_kind,
                    expected_evidence=payload.expected_evidence,
                    observed_evidence=payload.observed_evidence,
                    failure_reason=payload.failure_reason,
                    advancement_reason=payload.advancement_reason,
                    exit_readiness_score=payload.exit_readiness_score,
                    missing_prerequisites=payload.missing_prerequisites,
                    premature_exit_penalty_applied=payload.premature_exit_penalty_applied,
                    failed_without_new_support=payload.failed_without_new_support,
                    position_hold_detected=payload.position_hold_detected,
                ),
                decision_id=str(outcome_payload.get("candidate_id") or ""),
                outcome_id=str(outcome_payload.get("candidate_id") or ""),
                **common_kwargs,
            )
            if bool(outcome_payload.get("step_success")):
                self.session_ledger.append_subgoal_chain_step_completed(payload=payload, **common_kwargs)
            else:
                self.session_ledger.append_subgoal_chain_step_failed(payload=payload, **common_kwargs)
            if snapshot_after.get("active_chain") and int(dict(snapshot_after.get("active_chain", {}) or {}).get("current_step_index", 0) or 0) != int(active_chain.get("current_step_index", 0) or 0):
                next_step = dict(snapshot_after.get("active_step", {}) or {})
                runtime_rows.append({"event_type": "subgoal chain advanced", "round_id": int(round_id), "payload": {"chain_id": str(active_chain.get("chain_id") or ""), "step_id": str(next_step.get("step_id") or "")}})
                runtime_chain_event_count += 1
                self.session_ledger.append_subgoal_chain_advanced(
                    payload=SubgoalChainAdvancedPayload(
                        chain_id=str(active_chain.get("chain_id") or ""),
                        selected_subgoal_chain_id=str(active_chain.get("chain_id") or "") or None,
                        selected_subgoal_step_id=str(next_step.get("step_id") or "") or None,
                        current_step_id=str(next_step.get("step_id") or "") or None,
                        step_kind=str(next_step.get("step_kind") or "") or None,
                        expected_evidence=tuple(str(value) for value in list(next_step.get("expected_evidence", []) or []) if value),
                        observed_evidence=tuple(str(value) for value in list(outcome_payload.get("expected_evidence_seen", []) or []) if value),
                        advancement_reason=str(outcome_payload.get("chain_completion_reason") or "advance_on_success"),
                        exit_readiness_score=float(dict(active_chain).get("exit_readiness_score_at_creation", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(snapshot_after.get("missing_verification_step_kind") and [snapshot_after.get("missing_verification_step_kind")] or []) if value),
                    ),
                    **common_kwargs,
                )
            if str(dict(snapshot_after.get("active_chain", {}) or {}).get("status") or "") == "completed":
                runtime_rows.append({"event_type": "subgoal chain completed", "round_id": int(round_id), "payload": {"chain_id": str(active_chain.get("chain_id") or "")}})
                runtime_chain_event_count += 1
                self.session_ledger.append_subgoal_chain_completed(
                    payload=SubgoalChainCompletedPayload(
                        chain_id=str(active_chain.get("chain_id") or ""),
                        selected_subgoal_chain_id=str(active_chain.get("chain_id") or "") or None,
                        selected_subgoal_step_id=str(active_step.get("step_id") or "") or None,
                        current_step_id=str(active_step.get("step_id") or "") or None,
                        step_kind=str(active_step.get("step_kind") or "") or None,
                        expected_evidence=tuple(str(value) for value in list(active_step.get("expected_evidence", []) or []) if value),
                        observed_evidence=tuple(str(value) for value in list(outcome_payload.get("expected_evidence_seen", []) or []) if value),
                        advancement_reason=str(outcome_payload.get("chain_completion_reason") or "all_steps_completed"),
                        exit_readiness_score=float(dict(active_chain).get("exit_readiness_score_at_creation", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(outcome_payload.get("missing_prerequisites", []) or []) if value),
                        failed_without_new_support=bool(outcome_payload.get("exit_attempt_failed_without_new_support", False)),
                        position_hold_detected=bool(outcome_payload.get("position_hold_detected", False)),
                    ),
                    **common_kwargs,
                )
                manager.active_chain = None
            elif str(dict(snapshot_after.get("active_chain", {}) or {}).get("status") or "") == "aborted":
                runtime_rows.append({"event_type": "subgoal chain abandoned", "round_id": int(round_id), "payload": {"chain_id": str(active_chain.get("chain_id") or ""), "reason": str(outcome_payload.get("step_failure_reason") or "")}})
                runtime_chain_event_count += 1
                self.session_ledger.append_subgoal_chain_aborted(
                    payload=SubgoalChainAbortedPayload(
                        chain_id=str(active_chain.get("chain_id") or ""),
                        selected_subgoal_chain_id=str(active_chain.get("chain_id") or "") or None,
                        selected_subgoal_step_id=str(active_step.get("step_id") or "") or None,
                        current_step_id=str(active_step.get("step_id") or "") or None,
                        step_kind=str(active_step.get("step_kind") or "") or None,
                        expected_evidence=tuple(str(value) for value in list(active_step.get("expected_evidence", []) or []) if value),
                        observed_evidence=tuple(str(value) for value in list(outcome_payload.get("expected_evidence_seen", []) or []) if value),
                        failure_reason=str(outcome_payload.get("step_failure_reason") or "") or None,
                        advancement_reason=str(outcome_payload.get("chain_completion_reason") or "chain_aborted"),
                        exit_readiness_score=float(dict(active_chain).get("exit_readiness_score_at_creation", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(outcome_payload.get("missing_prerequisites", []) or []) if value),
                        failed_without_new_support=bool(outcome_payload.get("exit_attempt_failed_without_new_support", False)),
                        position_hold_detected=bool(outcome_payload.get("position_hold_detected", False)),
                    ),
                    **common_kwargs,
                )
                self.session_ledger.append_subgoal_chain_abandoned(
                    payload=SubgoalChainAbandonedPayload(
                        chain_id=str(active_chain.get("chain_id") or ""),
                        selected_subgoal_chain_id=str(active_chain.get("chain_id") or "") or None,
                        selected_subgoal_step_id=str(active_step.get("step_id") or "") or None,
                        current_step_id=str(active_step.get("step_id") or "") or None,
                        step_kind=str(active_step.get("step_kind") or "") or None,
                        expected_evidence=tuple(str(value) for value in list(active_step.get("expected_evidence", []) or []) if value),
                        observed_evidence=tuple(str(value) for value in list(outcome_payload.get("expected_evidence_seen", []) or []) if value),
                        failure_reason=str(outcome_payload.get("step_failure_reason") or "") or None,
                        advancement_reason=str(outcome_payload.get("chain_completion_reason") or "chain_abandoned"),
                        exit_readiness_score=float(dict(active_chain).get("exit_readiness_score_at_creation", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(outcome_payload.get("missing_prerequisites", []) or []) if value),
                        failed_without_new_support=bool(outcome_payload.get("exit_attempt_failed_without_new_support", False)),
                        position_hold_detected=bool(outcome_payload.get("position_hold_detected", False)),
                    ),
                    **common_kwargs,
                )
        terminal_status = str(dict(snapshot_after.get("active_chain", {}) or {}).get("status") or "")
        if terminal_status in {"completed", "aborted"}:
            manager.chain_history.append({"event": "cleared_terminal_chain", "chain_id": active_chain.get("chain_id"), "status": terminal_status})
            manager.active_chain = None
            manager._should_replan = terminal_status == "aborted"
        return {
            "runtime_chain_rows": runtime_rows,
            "runtime_chain_event_count": int(runtime_chain_event_count),
            "runtime_step_event_count": int(runtime_step_event_count),
            "runtime_progress_updated": bool(runtime_rows),
            "runtime_subgoal_chain_state_after_outcome": manager.snapshot(),
        }

    def _parallelism(self, pool_name: str, requested: int) -> int:
        return max(1, min(int(requested or 1), len(list(self.services.get(pool_name, []))) or 1))

    def _strict_blackboard_snapshot_ref(self, snapshot) -> dict:
        state = dict(getattr(snapshot, "state", {}) or {})
        split_indexes = dict(state.get("split_indexes", {}) or {})
        observed_indexes = dict(split_indexes.get("observed", {}) or {})
        hypothesized_indexes = dict(split_indexes.get("hypothesized", {}) or {})
        return {
            "snapshot_kind": "strict_split_world",
            "snapshot_version": getattr(snapshot, "blackboard_version", None),
            "contract_mode": "strict_split_native",
            "observed_row_counts": {
                "entities": len(dict(state.get("observed_entities", {}))),
                "consequences": len(dict(state.get("observed_consequences", {}))),
                "trigger_zones": len(dict(state.get("observed_trigger_zones", {}))),
                "topology_nodes": len(dict(state.get("observed_topology", {}).get("nodes", {}))),
                "topology_edges": len(dict(state.get("observed_topology", {}).get("edges", {}))),
            },
            "hypothesized_row_counts": {
                "entities": len(dict(state.get("hypothesized_entities", {}))),
                "consequences": len(dict(state.get("hypothesized_consequences", {}))),
                "trigger_zones": len(dict(state.get("hypothesized_trigger_zones", {}))),
                "topology_nodes": len(dict(state.get("hypothesized_topology", {}).get("nodes", {}))),
                "topology_edges": len(dict(state.get("hypothesized_topology", {}).get("edges", {}))),
            },
            "strict_index_counts": {
                "observed_entities_by_area_rows": sum(len(list(rows or [])) for rows in dict(observed_indexes.get("entities_by_area_rows", {})).values()),
                "hypothesized_entities_by_area_rows": sum(len(list(rows or [])) for rows in dict(hypothesized_indexes.get("entities_by_area_rows", {})).values()),
                "observed_evidence_index_rows": sum(len(list(rows or [])) for rows in dict(observed_indexes.get("evidence_index_rows", {})).values()),
                "hypothesized_evidence_index_rows": sum(len(list(rows or [])) for rows in dict(hypothesized_indexes.get("evidence_index_rows", {})).values()),
            },
        }

    def _strict_memory_snapshot_ref(self, snapshot) -> dict:
        state = dict(getattr(snapshot, "state", {}) or {})
        working = dict(state.get("working_memory", {}) or {})
        return {
            "snapshot_kind": "working_memory",
            "snapshot_version": getattr(snapshot, "memory_version", None),
            "contract_mode": "working_memory_native",
            "snapshot_handle": getattr(snapshot, "snapshot_handle", None),
            "observed_row_counts": {
                "cooldowns": len(dict(working.get("cooldowns", {}) or {})),
                "retries": len(dict(working.get("retries", {}) or {})),
                "skills": len(dict(state.get("skill_library", {}) or {})),
            },
            "hypothesized_row_counts": {
                "durable_priors": len(dict(state.get("durable_priors", {}) or {})),
            },
            "strict_index_counts": {
                "exhausted_keys": len(list(working.get("exhausted_keys", []) or [])),
            },
            "durable_checkpoint_id": getattr(snapshot, "durable_checkpoint_id", None),
        }

    def _outcome_evidence_provenance_summary(self, outcome) -> dict:
        evidence = dict(dict(getattr(outcome, "outcome", {}) or {}).get("outcome_evidence", {}) or {})
        counts: dict[str, int] = {}
        for cell in evidence.values():
            provenance = str(dict(cell or {}).get("provenance", "unknown"))
            counts[provenance] = counts.get(provenance, 0) + 1
        return counts

    def _candidate_decision(self, decision, candidate: dict) -> PlannerDecision:
        payload = dict(decision.__dict__)
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["selected_candidate"] = dict(candidate)
        payload["metadata"] = metadata
        payload["selected_candidate_id"] = candidate.get("candidate_id")
        payload["selected_action"] = final_action_from_candidate(candidate)
        return PlannerDecision(**payload)

    def _trial_candidates(self, decision, *, branch_count: int) -> list[dict]:
        ranked = [dict(row) for row in list(getattr(decision, "ranked_candidates", ()) or []) if row.get("candidate_id")]
        metadata = dict(getattr(decision, "metadata", {}) or {}) if isinstance(getattr(decision, "metadata", {}), dict) else {}
        selected = dict(metadata.get("selected_candidate", {}) or {}) if isinstance(metadata.get("selected_candidate", {}), dict) else {}
        fallback = [
            dict(row)
            for row in list(metadata.get("fallback_candidates", []) or [])
            if isinstance(row, dict) and row.get("candidate_id")
        ]
        candidates: list[dict] = []
        seen: set[str] = set()
        for row in ([selected] if selected else []) + ranked + fallback:
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(dict(row))
            if len(candidates) >= branch_count:
                break
        if candidates and len(candidates) < branch_count:
            base_rows = [dict(row) for row in candidates]
            idx = 0
            while len(candidates) < branch_count:
                variant = dict(base_rows[idx % len(base_rows)])
                variant["trial_variant_index"] = idx + 1
                variant["trial_variant_of"] = variant.get("candidate_id")
                candidates.append(variant)
                idx += 1
        return candidates

    def _submit_execution_batch(self, *, decisions: list, max_steps: int, mode: str, round_id: int) -> list:
        submitted = []
        for idx, branch_decision in enumerate(decisions):
            worker = self.pick("env_workers")
            request = build_executor_request(
                branch_decision,
                max_steps=max_steps,
                mode=mode,
                seed=self.config.environment.seed + round_id + (idx * 1000),
            )
            task_id = f"exec:{mode}:{round_id}:{idx}"
            submitted.append((task_id, self.submit(worker, "execute", request, task_id=task_id)))
        return [self.resolve(task_id, ref) for task_id, ref in submitted]

    def _submit_analysis_batch(self, *, outcomes: list, round_id: int, mode: str, blackboard_snapshot: object | None = None, mechanic_graph_snapshot: object | None = None) -> list:
        submitted = []
        analysis_mode = "probe" if mode == "probe" else "directed_outcome"
        for idx, outcome in enumerate(outcomes):
            task_id = f"analysis:{mode}:{round_id}:{idx}"
            task = self.services["analysis_task"]
            classifier_truth_surface = _episode_classifier_payload(outcome)
            ref = task.options(num_cpus=self.config.ray.worker_cpus).remote(
                outcome.episode,
                analysis_mode,
                getattr(blackboard_snapshot, "state", blackboard_snapshot),
                getattr(mechanic_graph_snapshot, "state", mechanic_graph_snapshot),
                getattr(self.config, "hypothesis_generation", None),
                self.services.get("llm_reasoner_adapter"),
                self.services["hypothesis_registry"].snapshot(),
                classifier_truth_surface,
            )
            self.task_registry.put(task_id, ref)
            submitted.append((task_id, ref))
        return [self.resolve(task_id, ref) for task_id, ref in submitted]

    def _trial_rank_key(self, *, outcome, analysis) -> tuple:
        summary = dict(getattr(analysis, "summary", {}) or {})
        changed_steps = sum(1 for row in list(summary.get("step_rows", []) or []) if int(row.get("changed_cells", 0) or 0) > 0)
        outcome_payload = dict(getattr(outcome, "outcome", {}) or {})
        return (
            int(bool(getattr(outcome, "success", False))),
            int(bool(getattr(outcome.episode, "won", False))),
            float(outcome_payload.get("progress", 0.0) or 0.0),
            int(changed_steps),
            float(getattr(outcome, "reward_delta", 0.0) or 0.0),
            -int(bool(outcome_payload.get("termination_reason") == "missing_avatar")),
        )

    def _choose_trial_winner(self, *, decisions: list, outcomes: list, analyses: list) -> int:
        scored = [
            (self._trial_rank_key(outcome=outcome, analysis=analysis), idx)
            for idx, (outcome, analysis) in enumerate(zip(outcomes, analyses))
        ]
        scored.sort()
        return scored[-1][1] if scored else 0

    def _append_llm_operation_events(self, *, round_id: int, pass_id: int, plan_context, bundle) -> None:
        if self.session_ledger is None or bundle is None:
            return
        metadata = dict(getattr(bundle, "metadata", {}) or {})
        proposals = [*list(getattr(bundle, "edge_proposals", ()) or []), *list(getattr(bundle, "path_proposals", ()) or []), *list(getattr(bundle, "test_proposals", ()) or [])]
        payload = LLMOperationPayload(
            gating_reason=str(metadata.get("reason") or metadata.get("gating_reason") or ""),
            provider_name=str(metadata.get("llm_adapter_name") or ""),
            model_name=str(metadata.get("llm_model_name") or ""),
            latency_ms=int(metadata.get("llm_latency_ms", 0) or 0),
            proposal_count=len(proposals),
            prompt_char_count=int(metadata.get("prompt_char_count", 0) or 0),
            prompt_approx_token_count=int(metadata.get("prompt_approx_token_count", 0) or 0),
            prompt_trim_applied=bool(metadata.get("prompt_trim_applied", False)),
            prompt_mode=str(metadata.get("prompt_mode") or ""),
            query_target_id=str(metadata.get("query_target_id") or ""),
            skip_reason=str(metadata.get("reason") or metadata.get("gating_reason") or ""),
            temperature=float(metadata.get("temperature", 0.0) or 0.0),
            top_p=float(metadata.get("top_p", 0.0) or 0.0),
            top_k=int(metadata.get("top_k", 0) or 0),
            presence_penalty=float(metadata.get("presence_penalty", 0.0) or 0.0),
            repetition_penalty=float(metadata.get("repetition_penalty", 0.0) or 0.0),
            max_output_tokens=int(metadata.get("max_output_tokens", 0) or 0),
            enable_thinking=bool(metadata.get("enable_thinking", False)),
            stream=bool(metadata.get("stream", False)),
            error_code=metadata.get("error_code"),
        )
        if "debug_prompt_payload" in metadata or "raw_text" in metadata:
            debug_index = int(getattr(self, "_llm_debug_counter", 0) or 0) + 1
            self._llm_debug_counter = debug_index
            self.call(
                self.services["storage"],
                "persist",
                session_id=self.context.session_id,
                round_id=round_id,
                kind="report",
                name=f"llm_debug_pass{int(pass_id):01d}_{debug_index:02d}.json",
                payload={
                    "round_id": int(round_id),
                    "pass_id": int(pass_id),
                    "prompt_mode": str(metadata.get("prompt_mode") or ""),
                    "query_target_id": str(metadata.get("query_target_id") or ""),
                    "prompt_char_count": int(metadata.get("prompt_char_count", 0) or 0),
                    "prompt_approx_token_count": int(metadata.get("prompt_approx_token_count", 0) or 0),
                    "prompt_trim_applied": bool(metadata.get("prompt_trim_applied", False)),
                    "Prompt": str(metadata.get("debug_prompt_string") or ""),
                    "Response": str(metadata.get("raw_text") or ""),
                    "SystemInstruction": str(metadata.get("debug_system_instruction") or ""),
                    "PromptPayload": dict(metadata.get("debug_prompt_payload", {}) or {}),
                    "llm_model_name": str(metadata.get("llm_model_name") or ""),
                    "llm_latency_ms": int(metadata.get("llm_latency_ms", 0) or 0),
                    "error_code": metadata.get("error_code"),
                    "error_message": metadata.get("error_message"),
                },
            )
        if not bool(metadata.get("llm_call_attempted", False)):
            self.session_ledger.append_llm_call_skipped(
                round_id=round_id,
                pass_id=pass_id,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=payload,
            )
        else:
            self.session_ledger.append_llm_call_attempted(
                round_id=round_id,
                pass_id=pass_id,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=payload,
            )
            if bool(metadata.get("llm_call_succeeded", False)):
                self.session_ledger.append_llm_call_succeeded(
                    round_id=round_id,
                    pass_id=pass_id,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    payload=payload,
                )
            else:
                self.session_ledger.append_llm_call_failed(
                    round_id=round_id,
                    pass_id=pass_id,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    payload=payload,
                )

    def export_round_debug_heatmaps(
        self,
        *,
        round_id: int,
        analyzed_rows: list[dict],
        blackboard_state: dict,
        first_observation: list[list[int]] | None,
    ) -> None:
        visit_bundle = build_visit_heatmap(
            analyzed_rows,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
        )
        poi_bundle = build_poi_heatmap(
            blackboard_state,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
        )
        self.call(
            self.services["storage"],
            "persist_bytes",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="visualization",
            name="visit_heatmap_debug.png",
            payload=render_overlay_png(
                first_observation,
                visit_bundle["counts"],
                overlay_kind="visit",
                width=self.config.visualization.grid_width,
                height=self.config.visualization.grid_height,
                scale=15,
                start=visit_bundle.get("start"),
                end=visit_bundle.get("end"),
            ),
        )
        self.call(
            self.services["storage"],
            "persist_bytes",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="visualization",
            name="poi_heatmap_debug.png",
            payload=render_overlay_png(
                first_observation,
                poi_bundle["accepted_counts"],
                overlay_kind="poi",
                width=self.config.visualization.grid_width,
                height=self.config.visualization.grid_height,
                scale=15,
            ),
        )
        self.call(
            self.services["storage"],
            "persist_bytes",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="visualization",
            name="combined_heatmap_overlay.png",
            payload=render_combined_overlay_png(
                first_observation,
                visit_bundle["counts"],
                poi_bundle["accepted_counts"],
                width=self.config.visualization.grid_width,
                height=self.config.visualization.grid_height,
                scale=15,
                start=visit_bundle.get("start"),
                end=visit_bundle.get("end"),
            ),
        )

    def run_round(
        self,
        *,
        round_id: int,
        latest_blackboard,
        latest_memory,
        latest_mechanic_graph,
        hypothesis_registry_snapshot: dict | None = None,
        first_observation: list[list[int]] | None,
        analyzed_rows_so_far: list[dict],
        current_stamp: object | None,
    ) -> tuple[RoundRunResult, object | None]:
        round_progress = 0.0
        if self.session_ledger is not None:
            self.session_ledger.append_round_start(
                round_id=round_id,
                pass_id=0,
                blackboard_version=getattr(latest_blackboard, "blackboard_version", None),
                memory_version=getattr(latest_memory, "memory_version", None),
                payload=RoundStartPayload(game_id=self.context.game_id),
            )
        blackboard_state = self.snapshot_registry.get(latest_blackboard.snapshot_handle).state
        memory_state = self.snapshot_registry.get(latest_memory.snapshot_handle).state
        incoming_mode_persistence = dict(dict(dict(memory_state or {}).get("working_memory", {}) or {}).get("plan_memory", {}) or {}).get("mode_persistence", {})
        prior_committed_mode = self.last_planning_mode_committed
        prior_committed_mode_source = "round_runner.last_planning_mode_committed"
        if prior_committed_mode in {"", None}:
            prior_committed_mode = dict(incoming_mode_persistence or {}).get("last_planning_mode_committed")
            prior_committed_mode_source = "working_memory.plan_memory.mode_persistence.last_planning_mode_committed"
        if int(round_id) <= 1:
            prior_committed_mode = None
            prior_committed_mode_source = "initial_round"
        memory_state = _inject_mode_persistence(memory_state, incoming_mode_persistence)
        memory_state = {
            **dict(memory_state or {}),
            "subgoal_chain_state": self._ensure_chain_manager().snapshot(),
            "previous_planning_mode_committed_for_trace": prior_committed_mode,
        }
        hypothesis_registry_snapshot = dict(hypothesis_registry_snapshot or self.services["hypothesis_registry"].snapshot())
        deterministic_hypotheses = dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))
        llm_hypotheses = dict(hypothesis_registry_snapshot.get("llm_proposals", {}))

        probe_context = self.planning_context_builder(round_id=round_id, pass_id=0, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
        probe_decision_ref = self.submit(self.services["planner"], "decide", probe_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:probe:{round_id}")
        probe_decision = self.resolve(f"planner:probe:{round_id}", probe_decision_ref)
        probe_final_mode_payload = _build_final_mode_payload(
            round_id=round_id,
            decision=probe_decision,
            prior_committed_mode=prior_committed_mode,
            prior_committed_mode_source=prior_committed_mode_source,
        )
        probe_decision = _apply_final_mode_payload(probe_decision, probe_final_mode_payload)
        if self.session_ledger is not None:
            self.session_ledger.append_probe_plan_selected(
                round_id=round_id,
                pass_id=0,
                blackboard_version=probe_context.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                decision_id=str(getattr(probe_decision, "selected_candidate_id", None) or f"probe-plan:{round_id}"),
                    final_mode_payload=probe_final_mode_payload,
                    payload=PlanSelectedPayload(
                        selected_candidate_id=getattr(probe_decision, "selected_candidate_id", None),
                        selected_candidate_count=len(list(getattr(probe_decision, "ranked_candidates", ()) or [])),
                        planner_contract_mode=str(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_pipeline_contract_mode") or "split_world_native_partial"),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                        strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                        planning_mode=str(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_mode") or "default_progress"),
                        previous_planning_mode=str(probe_final_mode_payload.get("previous_planning_mode") or ""),
                        planning_mode_before_hysteresis=str(probe_final_mode_payload.get("planning_mode_before_hysteresis") or ""),
                        planning_mode_after_hysteresis=str(probe_final_mode_payload.get("planning_mode_after_hysteresis") or ""),
                        planning_mode_committed=str(probe_final_mode_payload.get("planning_mode_committed") or "default_progress"),
                        structure_acquisition_score=float(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("structure_acquisition_score", 0.0) or 0.0),
                        default_progress_score=float(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("default_progress_score", 0.0) or 0.0),
                        mode_switch_applied=bool(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_applied", False)),
                        mode_switch_reason=str(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_reason") or ""),
                        mode_persistence_hysteresis_applied=bool(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_persistence_hysteresis_applied", False)),
                        mode_switch_block_reason=str(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_block_reason") or ""),
                        payload_correction_applied=bool(probe_final_mode_payload.get("payload_correction_applied", False)),
                        payload_correction_reason=str(probe_final_mode_payload.get("payload_correction_reason") or ""),
                        prior_committed_mode_loaded=probe_final_mode_payload.get("prior_committed_mode_loaded"),
                        prior_committed_mode_source=str(probe_final_mode_payload.get("prior_committed_mode_source") or ""),
                        final_mode_payload_built=bool(probe_final_mode_payload.get("final_mode_payload_built", False)),
                        exit_readiness_score=float(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_exit_readiness_score", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_missing_prerequisites", []) or []) if value),
                        premature_exit_penalty_applied=bool(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_candidate_blocked_by_low_exit_readiness", False)),
                    ),
            )
        probe_branch_count = self._parallelism("env_workers", getattr(self.config.planning, "probe_branch_count", 1))
        probe_candidates = self._trial_candidates(probe_decision, branch_count=probe_branch_count)
        probe_decisions = [self._candidate_decision(probe_decision, candidate) for candidate in probe_candidates] or [probe_decision]
        probe_outcomes = self._submit_execution_batch(
            decisions=probe_decisions,
            max_steps=self.config.environment.probe_steps,
            mode="probe",
            round_id=round_id,
        )
        probe_ledger_classifier_null_count = 0
        if self.session_ledger is not None:
            for idx, outcome in enumerate(probe_outcomes):
                classifier_payload = _episode_classifier_payload(outcome)
                if any(value is not None for value in classifier_payload.values()) and any(value is None for value in classifier_payload.values()):
                    probe_ledger_classifier_null_count += 1
                self.session_ledger.append_probe_episode_executed(
                    round_id=round_id,
                    pass_id=0,
                    blackboard_version=probe_context.blackboard_version,
                    memory_version=probe_context.memory_version,
                    plan_context_id=probe_context.plan_context_id,
                    episode_id=str(getattr(outcome.episode, "episode_id", f"probe:{round_id}:{idx}")),
                    decision_id=str(getattr(probe_decisions[idx], "selected_candidate_id", None) or f"probe:{round_id}:{idx}"),
                    outcome_id=str(getattr(outcome, "candidate_id", None) or f"probe-outcome:{round_id}:{idx}"),
                    payload=EpisodeExecutedPayload(
                        termination_reason=getattr(outcome, "termination_reason", None),
                        mode="probe",
                        reward_delta=float(getattr(outcome, "reward_delta", 0.0) or 0.0),
                        outcome_evidence_provenance_summary=self._outcome_evidence_provenance_summary(outcome),
                        avatar_cell=dict(getattr(outcome, "outcome", {}) or {}).get("avatar_cell_after"),
                        avatar_confidence=float(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_confidence_after", 0.0) or 0.0),
                        avatar_source=str(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_source_after") or "unknown"),
                        avatar_ambiguous=bool(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_localization_ambiguous", False)),
                        avatar_tracker_status="confident" if bool(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_localization_confident", False)) else "uncertain",
                        exit_readiness_score=float(dict(getattr(outcome, "outcome", {}) or {}).get("exit_readiness_score", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(dict(getattr(outcome, "outcome", {}) or {}).get("missing_prerequisites", []) or []) if value),
                        failed_without_new_support=bool(dict(getattr(outcome, "outcome", {}) or {}).get("exit_attempt_failed_without_new_support", False)),
                        position_hold_detected=bool(dict(getattr(outcome, "outcome", {}) or {}).get("position_hold_detected", False)),
                        **classifier_payload,
                    ),
                )
        if first_observation is None and probe_outcomes:
            metadata = dict(getattr(probe_outcomes[0].episode, "metadata", {}) or {})
            initial_observation = metadata.get("initial_observation")
            if isinstance(initial_observation, list):
                first_observation = initial_observation

        probe_analyses = self._submit_analysis_batch(outcomes=probe_outcomes, round_id=round_id, mode="probe", blackboard_snapshot=latest_blackboard, mechanic_graph_snapshot=latest_mechanic_graph)
        for analysis in probe_analyses:
            if getattr(analysis, "deterministic_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.deterministic_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.deterministic_hypothesis_bundle.to_dict() if hasattr(analysis.deterministic_hypothesis_bundle, "to_dict") else analysis.deterministic_hypothesis_bundle.__dict__)
                if self.session_ledger is not None:
                    bundle = analysis.deterministic_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_deterministic_hypotheses_generated(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
                    self.session_ledger.append_hypothesis_validation_completed(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if not bool(getattr(row, "requires_validation", True))),
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
            if getattr(analysis, "llm_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.llm_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.llm_hypothesis_bundle.to_dict() if hasattr(analysis.llm_hypothesis_bundle, "to_dict") else analysis.llm_hypothesis_bundle.__dict__)
                self._append_llm_operation_events(round_id=round_id, pass_id=0, plan_context=probe_context, bundle=analysis.llm_hypothesis_bundle)
                if self.session_ledger is not None:
                    bundle = analysis.llm_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_llm_hypotheses_generated(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if str(dict(bundle.metadata or {}).get("validator_status", "")) == "validated"),
                            source_provenance="llm_hypothesis",
                        ),
                    )
        if self.session_ledger is not None:
            for idx, analysis in enumerate(probe_analyses):
                self.session_ledger.append_probe_analysis_completed(
                    round_id=round_id,
                    pass_id=0,
                    blackboard_version=probe_context.blackboard_version,
                    memory_version=probe_context.memory_version,
                    plan_context_id=probe_context.plan_context_id,
                    episode_id=str(getattr(analysis, "episode_id", f"probe-analysis:{round_id}:{idx}")),
                    outcome_id=str(getattr(probe_outcomes[idx], "candidate_id", None) or f"probe-outcome:{round_id}:{idx}"),
                    payload=AnalysisCompletedPayload(
                        analysis_mode=str(dict(getattr(analysis, "summary", {}) or {}).get("analysis_mode") or "probe"),
                        delta_count=len(list(getattr(analysis, "blackboard_deltas", ()) or [])),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                    ),
                )
        new_analyzed_rows = list(analyzed_rows_so_far)
        new_analyzed_rows.extend(episode_export_row(analysis) for analysis in probe_analyses)
        probe_winner_idx = self._choose_trial_winner(decisions=probe_decisions, outcomes=probe_outcomes, analyses=probe_analyses)
        probe_outcome = probe_outcomes[probe_winner_idx]

        probe_analysis_deltas = [delta.__dict__ for analysis in probe_analyses for delta in analysis.blackboard_deltas]
        probe_exit_attempt_rows_present_in_analysis_delta_count = _count_exit_attempt_family_rows(probe_analysis_deltas)
        bb_probe_ref = self.submit(
            self.services["blackboard"],
            "merge",
            round_id=round_id,
            pass_id=0,
            deltas=probe_analysis_deltas,
            task_id=f"blackboard:probe:{round_id}",
        )
        latest_blackboard = self.resolve(f"blackboard:probe:{round_id}", bb_probe_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        if self.session_ledger is not None:
            self.session_ledger.append_probe_blackboard_merge_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                payload=MergeCompletedPayload(
                    material_change=bool(getattr(latest_blackboard, "material_change", False)),
                    strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                ),
            )

        mg_probe_ref = self.submit(
            self.services["mechanic_graph"],
            "merge",
            round_id=round_id,
            pass_id=0,
            deltas=[analysis.mechanic_graph_delta.__dict__ for analysis in probe_analyses if getattr(analysis, "mechanic_graph_delta", None) is not None],
            task_id=f"mechanic-graph:probe:{round_id}",
        )
        mg_probe_result = self.resolve(f"mechanic-graph:probe:{round_id}", mg_probe_ref)
        latest_mechanic_graph = mg_probe_result["snapshot"]
        self.snapshot_registry.register(latest_mechanic_graph.snapshot_handle, latest_mechanic_graph)
        hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()
        if self.session_ledger is not None:
            top_supported_probe = sorted(
                [dict(row) for row in list(latest_mechanic_graph.state.get("edges_by_id", {}).values())],
                key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("edge_id", ""))),
            )[:5]
            self.session_ledger.append_probe_mechanic_graph_merge_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                payload=MechanicGraphMergeCompletedPayload(
                    mechanic_graph_version_before=probe_context.mechanic_graph_version,
                    mechanic_graph_version_after=latest_mechanic_graph.mechanic_graph_version,
                    node_count_added=int(mg_probe_result["counts"].get("node_count_added", 0) or 0),
                    edge_count_added=int(mg_probe_result["counts"].get("edge_count_added", 0) or 0),
                    observed_edge_count_added=int(mg_probe_result["counts"].get("observed_edge_count_added", 0) or 0),
                    hypothesized_edge_count_added=int(mg_probe_result["counts"].get("hypothesized_edge_count_added", 0) or 0),
                    top_supported_new_relations_summary=tuple(top_supported_probe),
                ),
            )
            self.session_ledger.append_hypothesis_validation_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                payload=HypothesisGenerationPayload(
                    proposal_count=sum(len(list(rows or [])) for rows in dict(hypothesis_registry_snapshot).values() if isinstance(rows, dict)),
                    edge_proposal_count=0,
                    path_proposal_count=0,
                    test_proposal_count=0,
                    top_confidence=0.0,
                    top_support_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("supported_count", 0) or 0),
                    contradicted_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("contradicted_count", 0) or 0),
                    validated_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("validated_count", 0) or 0),
                    source_provenance="registry_feedback",
                ),
            )

        mem_probe_ref = self.submit(
            self.services["memory"],
            "reconcile",
            round_id=round_id,
            pass_id=0,
            blackboard_state=latest_blackboard.state,
            mechanic_graph_state=latest_mechanic_graph.state,
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            decision=None,
            outcome=probe_outcome.__dict__,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id=f"memory:probe:{round_id}",
        )
        latest_memory = self.resolve(f"memory:probe:{round_id}", mem_probe_ref)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)
        probe_durable_status = self.call(self.services["memory"], "get_pending_durable_status")
        if self.session_ledger is not None:
            self.session_ledger.append_probe_memory_reconcile_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                outcome_id=str(getattr(probe_outcome, "candidate_id", None) or f"probe-outcome:{round_id}"),
                payload=MemoryReconcilePayload(
                    memory_snapshot_handle=latest_memory.snapshot_handle,
                    strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    durable_eligibility_summary=dict(probe_durable_status or {}),
                ),
            )

        plan_context = self.planning_context_builder(round_id=round_id, pass_id=1, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
        new_stamp = CompatibilityStamp(
            plan_context_id=plan_context.plan_context_id,
            blackboard_version=plan_context.blackboard_version,
            memory_version=plan_context.memory_version,
            policy_version=plan_context.policy_version,
            ranker_version=plan_context.ranker_version,
        )
        if current_stamp is not None:
            invalidate_if_needed(
                stale=current_stamp,
                current=new_stamp,
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                game_id=self.context.game_id,
                round_id=round_id,
                pass_id=1,
                stale_task_ids=[task_id for task_id, row in self.task_registry.tasks.items() if row.get("status") != "completed"],
            )
        current_stamp = new_stamp

        blackboard_state = self.snapshot_registry.get(latest_blackboard.snapshot_handle).state
        memory_state = self.snapshot_registry.get(latest_memory.snapshot_handle).state
        memory_state = _inject_mode_persistence(memory_state, incoming_mode_persistence)
        memory_state = {
            **dict(memory_state or {}),
            "subgoal_chain_state": self._ensure_chain_manager().snapshot(),
            "previous_planning_mode_committed_for_trace": prior_committed_mode,
        }
        deterministic_hypotheses = dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))
        llm_hypotheses = dict(hypothesis_registry_snapshot.get("llm_proposals", {}))
        use_helpers = bool(self.config.feature_flags.enable_helper_workers)
        if use_helpers:
            seed_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:seed:{round_id}")
            seed_decision = self.resolve(f"planner:seed:{round_id}", seed_decision_ref)
            helper_results, _ = self.helper_coordinator.dispatch(
                planning_context=plan_context,
                decision_seed=seed_decision,
                blackboard_state=blackboard_state,
                memory_state=memory_state,
            )
            final_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, helper_results, latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:final:{round_id}")
            decision = self.resolve(f"planner:final:{round_id}", final_decision_ref)
        else:
            final_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:final:{round_id}")
            decision = self.resolve(f"planner:final:{round_id}", final_decision_ref)
        final_mode_payload = _build_final_mode_payload(
            round_id=round_id,
            decision=decision,
            prior_committed_mode=prior_committed_mode,
            prior_committed_mode_source=prior_committed_mode_source,
        )
        decision = _apply_final_mode_payload(decision, final_mode_payload)
        decision, chain_runtime_materialization = self._materialize_selected_chain_runtime(
            decision=decision,
            round_id=round_id,
            blackboard_version=plan_context.blackboard_version,
            memory_version=plan_context.memory_version,
            plan_context_id=plan_context.plan_context_id,
        )
        if self.session_ledger is not None:
            self.session_ledger.append_directed_plan_selected(
                round_id=round_id,
                pass_id=1,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                    final_mode_payload=final_mode_payload,
                    payload=PlanSelectedPayload(
                        selected_candidate_id=getattr(decision, "selected_candidate_id", None),
                        selected_candidate_count=len(list(getattr(decision, "ranked_candidates", ()) or [])),
                        planner_contract_mode=str(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_pipeline_contract_mode") or "split_world_native_partial"),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                        strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                        planning_mode=str(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_mode") or "default_progress"),
                        previous_planning_mode=str(final_mode_payload.get("previous_planning_mode") or ""),
                        planning_mode_before_hysteresis=str(final_mode_payload.get("planning_mode_before_hysteresis") or ""),
                        planning_mode_after_hysteresis=str(final_mode_payload.get("planning_mode_after_hysteresis") or ""),
                        planning_mode_committed=str(final_mode_payload.get("planning_mode_committed") or "default_progress"),
                        structure_acquisition_score=float(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("structure_acquisition_score", 0.0) or 0.0),
                        default_progress_score=float(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("default_progress_score", 0.0) or 0.0),
                        mode_switch_applied=bool(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_applied", False)),
                        mode_switch_reason=str(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_reason") or ""),
                        mode_persistence_hysteresis_applied=bool(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_persistence_hysteresis_applied", False)),
                        mode_switch_block_reason=str(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("mode_switch_block_reason") or ""),
                        payload_correction_applied=bool(final_mode_payload.get("payload_correction_applied", False)),
                        payload_correction_reason=str(final_mode_payload.get("payload_correction_reason") or ""),
                        prior_committed_mode_loaded=final_mode_payload.get("prior_committed_mode_loaded"),
                        prior_committed_mode_source=str(final_mode_payload.get("prior_committed_mode_source") or ""),
                        final_mode_payload_built=bool(final_mode_payload.get("final_mode_payload_built", False)),
                        exit_readiness_score=float(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_exit_readiness_score", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_missing_prerequisites", []) or []) if value),
                        premature_exit_penalty_applied=bool(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("selected_candidate_blocked_by_low_exit_readiness", False)),
                    ),
            )

        directed_trial_count = self._parallelism("env_workers", getattr(self.config.planning, "directed_trial_count", 1))
        directed_candidates = self._trial_candidates(decision, branch_count=directed_trial_count)
        directed_decisions = [self._candidate_decision(decision, candidate) for candidate in directed_candidates] or [decision]
        directed_outcomes = self._submit_execution_batch(
            decisions=directed_decisions,
            max_steps=self.config.environment.directed_steps,
            mode="directed",
            round_id=round_id,
        )
        directed_ledger_classifier_null_count = 0
        if self.session_ledger is not None:
            for idx, outcome in enumerate(directed_outcomes):
                classifier_payload = _episode_classifier_payload(outcome)
                if any(value is not None for value in classifier_payload.values()) and any(value is None for value in classifier_payload.values()):
                    directed_ledger_classifier_null_count += 1
                self.session_ledger.append_directed_episode_executed(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    episode_id=str(getattr(outcome.episode, "episode_id", f"directed:{round_id}:{idx}")),
                    decision_id=str(getattr(directed_decisions[idx], "selected_candidate_id", None) or f"directed:{round_id}:{idx}"),
                    outcome_id=str(getattr(outcome, "candidate_id", None) or f"directed-outcome:{round_id}:{idx}"),
                    payload=EpisodeExecutedPayload(
                        termination_reason=getattr(outcome, "termination_reason", None),
                        mode="directed",
                        reward_delta=float(getattr(outcome, "reward_delta", 0.0) or 0.0),
                        outcome_evidence_provenance_summary=self._outcome_evidence_provenance_summary(outcome),
                        avatar_cell=dict(getattr(outcome, "outcome", {}) or {}).get("avatar_cell_after"),
                        avatar_confidence=float(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_confidence_after", 0.0) or 0.0),
                        avatar_source=str(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_source_after") or "unknown"),
                        avatar_ambiguous=bool(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_localization_ambiguous", False)),
                        avatar_tracker_status="confident" if bool(dict(getattr(outcome, "outcome", {}) or {}).get("avatar_localization_confident", False)) else "uncertain",
                        exit_readiness_score=float(dict(getattr(outcome, "outcome", {}) or {}).get("exit_readiness_score", 0.0) or 0.0),
                        missing_prerequisites=tuple(str(value) for value in list(dict(getattr(outcome, "outcome", {}) or {}).get("missing_prerequisites", []) or []) if value),
                        failed_without_new_support=bool(dict(getattr(outcome, "outcome", {}) or {}).get("exit_attempt_failed_without_new_support", False)),
                        position_hold_detected=bool(dict(getattr(outcome, "outcome", {}) or {}).get("position_hold_detected", False)),
                        **classifier_payload,
                    ),
                )
        directed_analyses = self._submit_analysis_batch(outcomes=directed_outcomes, round_id=round_id, mode="directed_outcome", blackboard_snapshot=latest_blackboard, mechanic_graph_snapshot=latest_mechanic_graph)
        for analysis in directed_analyses:
            if getattr(analysis, "deterministic_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.deterministic_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.deterministic_hypothesis_bundle.to_dict() if hasattr(analysis.deterministic_hypothesis_bundle, "to_dict") else analysis.deterministic_hypothesis_bundle.__dict__)
                if self.session_ledger is not None:
                    bundle = analysis.deterministic_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_deterministic_hypotheses_generated(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
            if getattr(analysis, "llm_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.llm_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.llm_hypothesis_bundle.to_dict() if hasattr(analysis.llm_hypothesis_bundle, "to_dict") else analysis.llm_hypothesis_bundle.__dict__)
                self._append_llm_operation_events(round_id=round_id, pass_id=1, plan_context=plan_context, bundle=analysis.llm_hypothesis_bundle)
                if self.session_ledger is not None:
                    bundle = analysis.llm_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_llm_hypotheses_generated(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="llm_hypothesis",
                        ),
                    )
                    self.session_ledger.append_hypothesis_validation_completed(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if str(dict(getattr(row, "metadata", {}) or {}).get("validator_status", "")) == "accepted"),
                            source_provenance="llm_hypothesis",
                        ),
                    )
        if self.session_ledger is not None:
            for idx, analysis in enumerate(directed_analyses):
                self.session_ledger.append_directed_analysis_completed(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    episode_id=str(getattr(analysis, "episode_id", f"directed-analysis:{round_id}:{idx}")),
                    outcome_id=str(getattr(directed_outcomes[idx], "candidate_id", None) or f"directed-outcome:{round_id}:{idx}"),
                    payload=AnalysisCompletedPayload(
                        analysis_mode=str(dict(getattr(analysis, "summary", {}) or {}).get("analysis_mode") or "directed_outcome"),
                        delta_count=len(list(getattr(analysis, "blackboard_deltas", ()) or [])),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                    ),
                )
        new_analyzed_rows.extend(episode_export_row(analysis) for analysis in directed_analyses)
        directed_winner_idx = self._choose_trial_winner(decisions=directed_decisions, outcomes=directed_outcomes, analyses=directed_analyses)
        decision = directed_decisions[directed_winner_idx]
        exec_outcome = directed_outcomes[directed_winner_idx]
        directed_analysis = directed_analyses[directed_winner_idx]
        chain_outcome_update = self._update_chain_from_outcome(
            round_id=round_id,
            pass_id=1,
            plan_context_id=plan_context.plan_context_id,
            blackboard_version=plan_context.blackboard_version,
            memory_version=plan_context.memory_version,
            outcome_payload=dict(getattr(exec_outcome, "outcome", {}) or {}),
        )
        selected_target_entity_id = None
        if isinstance(decision.selected_action, dict):
            selected_target_entity_id = decision.selected_action.get("target_entity_id") or decision.selected_action.get("target")
        if not selected_target_entity_id and isinstance(getattr(decision, "metadata", None), dict):
            selected_candidate = dict(decision.metadata.get("selected_candidate", {}) or {})
            selected_target_entity_id = selected_candidate.get("target_entity_id") or selected_candidate.get("target")
        round_progress += float(exec_outcome.outcome.get("progress", 0.0))

        directed_analysis_deltas = [delta.__dict__ for analysis in directed_analyses for delta in analysis.blackboard_deltas]
        directed_exit_attempt_rows_present_in_analysis_delta_count = _count_exit_attempt_family_rows(directed_analysis_deltas)
        bb_directed_ref = self.submit(
            self.services["blackboard"],
            "merge",
            round_id=round_id,
            pass_id=1,
            deltas=directed_analysis_deltas,
            task_id=f"blackboard:directed:{round_id}",
        )
        latest_blackboard = self.resolve(f"blackboard:directed:{round_id}", bb_directed_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        if self.session_ledger is not None:
            self.session_ledger.append_directed_blackboard_merge_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MergeCompletedPayload(
                    material_change=bool(getattr(latest_blackboard, "material_change", False)),
                    strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                ),
            )

        mg_directed_ref = self.submit(
            self.services["mechanic_graph"],
            "merge",
            round_id=round_id,
            pass_id=1,
            deltas=[analysis.mechanic_graph_delta.__dict__ for analysis in directed_analyses if getattr(analysis, "mechanic_graph_delta", None) is not None],
            task_id=f"mechanic-graph:directed:{round_id}",
        )
        mg_directed_result = self.resolve(f"mechanic-graph:directed:{round_id}", mg_directed_ref)
        latest_mechanic_graph = mg_directed_result["snapshot"]
        self.snapshot_registry.register(latest_mechanic_graph.snapshot_handle, latest_mechanic_graph)
        hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()
        if self.session_ledger is not None:
            validated_count = sum(1 for state in dict(hypothesis_registry_snapshot.get("validation_state", {})).values() if str(state) == "validated")
            self.session_ledger.append_hypothesis_validation_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=HypothesisGenerationPayload(
                    proposal_count=len(dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))) + len(dict(hypothesis_registry_snapshot.get("llm_proposals", {}))),
                    edge_proposal_count=0,
                    path_proposal_count=0,
                    test_proposal_count=0,
                    top_confidence=0.0,
                    top_support_count=0,
                    contradicted_count=sum(1 for state in dict(hypothesis_registry_snapshot.get("validation_state", {})).values() if str(state) == "contradicted"),
                    validated_count=validated_count,
                    source_provenance="mixed",
                ),
            )
        if self.session_ledger is not None:
            top_supported_directed = sorted(
                [dict(row) for row in list(latest_mechanic_graph.state.get("edges_by_id", {}).values())],
                key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("edge_id", ""))),
            )[:5]
            self.session_ledger.append_directed_mechanic_graph_merge_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MechanicGraphMergeCompletedPayload(
                    mechanic_graph_version_before=plan_context.mechanic_graph_version,
                    mechanic_graph_version_after=latest_mechanic_graph.mechanic_graph_version,
                    node_count_added=int(mg_directed_result["counts"].get("node_count_added", 0) or 0),
                    edge_count_added=int(mg_directed_result["counts"].get("edge_count_added", 0) or 0),
                    observed_edge_count_added=int(mg_directed_result["counts"].get("observed_edge_count_added", 0) or 0),
                    hypothesized_edge_count_added=int(mg_directed_result["counts"].get("hypothesized_edge_count_added", 0) or 0),
                    top_supported_new_relations_summary=tuple(top_supported_directed),
                ),
            )

        mem_directed_ref = self.submit(
            self.services["memory"],
            "reconcile",
            round_id=round_id,
            pass_id=1,
            blackboard_state=latest_blackboard.state,
            mechanic_graph_state=latest_mechanic_graph.state,
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            decision=decision.__dict__,
            outcome=exec_outcome.__dict__,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id=f"memory:directed:{round_id}",
        )
        latest_memory = self.resolve(f"memory:directed:{round_id}", mem_directed_ref)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)
        selected_candidate_metadata = dict(getattr(decision, "metadata", {}) or {}).get("selected_candidate", {})
        selected_candidate_metadata = dict(selected_candidate_metadata or {}) if isinstance(selected_candidate_metadata, dict) else {}
        active_chain_snapshot = dict(getattr(decision, "metadata", {}) or {}).get("selected_subgoal_chain", {})
        active_chain_snapshot = dict(active_chain_snapshot or {}) if isinstance(active_chain_snapshot, dict) else {}
        _, poi_escalation_events = _update_poi_followthrough_state(
            latest_memory=latest_memory,
            selected_candidate=selected_candidate_metadata,
            mg_counts=dict(mg_directed_result.get("counts", {}) or {}),
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            round_id=round_id,
            selected_subgoal_chain=active_chain_snapshot or None,
            chain_outcome_update=chain_outcome_update,
            selected_outcome=dict(getattr(exec_outcome, "outcome", {}) or {}),
        )
        mode_persistence_state = _update_mode_persistence_state(
            latest_memory=latest_memory,
            decision=decision,
            round_id=round_id,
            mg_counts=dict(mg_directed_result.get("counts", {}) or {}),
            selected_outcome=dict(getattr(exec_outcome, "outcome", {}) or {}),
            prior_mode_state=incoming_mode_persistence,
        )
        self.last_planning_mode_committed = str(final_mode_payload.get("planning_mode_committed") or "")
        directed_durable_status = self.call(self.services["memory"], "get_pending_durable_status")
        if self.session_ledger is not None:
            self.session_ledger.append_directed_memory_reconcile_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MemoryReconcilePayload(
                    memory_snapshot_handle=latest_memory.snapshot_handle,
                    strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    durable_eligibility_summary=dict(directed_durable_status or {}),
                ),
            )
            ledger_method_map = {
                "detector poi selected": self.session_ledger.append_detector_poi_selected,
                "detector poi revisited": self.session_ledger.append_detector_poi_revisited,
                "detector poi escalated to verification": self.session_ledger.append_detector_poi_escalated_to_verification,
                "detector poi escalated to chain": self.session_ledger.append_detector_poi_escalated_to_chain,
                "detector poi marked stale": self.session_ledger.append_detector_poi_marked_stale,
            }
            for event_type, payload in poi_escalation_events:
                ledger_method = ledger_method_map.get(str(event_type))
                if ledger_method is None:
                    continue
                ledger_method(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=latest_blackboard.blackboard_version,
                    memory_version=latest_memory.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                    payload=payload,
                )

        directed_step_rows = list(directed_analysis.summary.get("step_rows", []) or [])
        effect_payload = _merge_effect_payload(
            _target_effect_payload(directed_step_rows, selected_target_entity_id),
            _analysis_effect_payload(directed_analyses),
        )
        if selected_target_entity_id:
            _apply_target_effect_to_blackboard(
                latest_blackboard,
                target_entity_id=selected_target_entity_id,
                step_rows=[],
            )
            state = getattr(latest_blackboard, "state", None)
            if isinstance(state, dict):
                entities = state.get("entities")
                if isinstance(entities, dict) and selected_target_entity_id in entities:
                    entity = dict(entities[selected_target_entity_id])
                    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
                        entity[field] = int(entity.get(field, 0) or 0) + int(effect_payload.get(field, 0) or 0)
                    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
                        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(effect_payload.get(field, 0.0) or 0.0))
                    entity["candidate_effect_mode"] = effect_payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
                    entities[selected_target_entity_id] = entity

        self.call(
            self.services["storage"],
            "persist",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="snapshot",
            name=f"blackboard_pass1_round{round_id:03d}.json",
            payload=export_strict_snapshot(latest_blackboard),
        )
        memory_snapshot_path = self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"memory_pass1_round{round_id:03d}.json", payload=latest_memory)
        mechanic_graph_snapshot_path = self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"mechanic_graph_pass1_round{round_id:03d}.json", payload=latest_mechanic_graph)
        planned_effect_mode = str((decision.selected_action or {}).get("required_action_family") or "unknown") if isinstance(decision.selected_action, dict) else "unknown"
        actual_mode = actual_effect_mode(directed_step_rows, planned_effect_mode)
        available_families = available_families_from_blackboard(latest_blackboard.state)
        decision_export = decision_export_payload(decision, available_families=available_families, executed_family=actual_mode)
        decision_export = _apply_target_effect_to_decision_export(
            decision_export,
            target_entity_id=selected_target_entity_id,
            effect_payload=effect_payload,
        )
        live_bridge_rows = build_live_candidate_bridge_rows(decision_export, round_id=round_id)
        registry_promotion_input = {
            str(row.get("stable_id") or ""): dict(row)
            for row in live_bridge_rows
            if str(row.get("stable_id") or "")
        }
        live_bridge_diagnostics = {
            "live_bridge_row_count": len(live_bridge_rows),
            "live_bridge_usable_row_count": sum(1 for row in live_bridge_rows if bool(row.get("planner_usable", False))),
            "live_bridge_durable_row_count": sum(1 for row in live_bridge_rows if bool(row.get("durable_ready", False))),
            "registry_promotion_input_row_count": len(registry_promotion_input),
            "registry_promotion_input_usable_row_count": sum(1 for row in registry_promotion_input.values() if bool(row.get("planner_usable", False))),
            "registry_promotion_input_durable_row_count": sum(1 for row in registry_promotion_input.values() if bool(row.get("durable_ready", False))),
            "live_bridge_rows_missing_registry_input_count": max(0, len(live_bridge_rows) - len(registry_promotion_input)),
        }
        promotion_result = {}
        if registry_promotion_input:
            promotion_result = self.services["hypothesis_registry"].promote_target_usability(
                target_rows=registry_promotion_input,
                round_id=round_id,
            )
            hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()
        self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision_export)
        analysis_summary = build_round_analysis_summary(
            round_id=round_id,
            analyzed_episodes=[*probe_analyses, *directed_analyses],
            candidate_effect_mode_used=actual_mode,
        )
        selected_outcome_payload = dict(getattr(exec_outcome, "outcome", {}) or {})
        selected_ledger_classifier = _episode_classifier_payload(exec_outcome)
        analysis_summary["executed_episode_ledger_classifier_diagnostics"] = {
            "outcome_has_classifier_fields": any(value is not None for value in selected_ledger_classifier.values()),
            "ledger_classifier_null_despite_outcome_count": int(probe_ledger_classifier_null_count + directed_ledger_classifier_null_count),
        }
        analysis_summary = _sync_analysis_poi_debug_with_observed_store(analysis_summary, latest_blackboard.state)
        analysis_summary["live_bridge_diagnostics"] = {**live_bridge_diagnostics, **dict(promotion_result or {})}
        analysis_summary["family_handoff_diagnostics"] = {
            "exit_attempt_rows_present_in_analysis_delta_count": int(locals().get("probe_exit_attempt_rows_present_in_analysis_delta_count", 0) or 0)
            + int(locals().get("directed_exit_attempt_rows_present_in_analysis_delta_count", 0) or 0),
            "exit_attempt_rows_present_in_merge_request_count": int(locals().get("probe_exit_attempt_rows_present_in_analysis_delta_count", 0) or 0)
            + int(locals().get("directed_exit_attempt_rows_present_in_analysis_delta_count", 0) or 0),
        }
        round_effect_payload = _round_effect_payload(analysis_summary, actual_mode)
        if selected_target_entity_id:
            state = getattr(latest_blackboard, "state", None)
            if isinstance(state, dict):
                entities = state.get("entities")
                if isinstance(entities, dict) and selected_target_entity_id in entities:
                    entity = dict(entities[selected_target_entity_id])
                    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
                        entity[field] = max(int(entity.get(field, 0) or 0), int(round_effect_payload.get(field, 0) or 0))
                    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
                        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(round_effect_payload.get(field, 0.0) or 0.0))
                    entity["candidate_effect_mode"] = round_effect_payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
                    entities[selected_target_entity_id] = entity
            decision_export = _apply_target_effect_to_decision_export(
                decision_export,
                target_entity_id=selected_target_entity_id,
                effect_payload=_merge_effect_payload(effect_payload, round_effect_payload),
            )
            self.call(
                self.services["storage"],
                "persist",
                session_id=self.context.session_id,
                round_id=round_id,
                kind="snapshot",
                name=f"blackboard_pass1_round{round_id:03d}.json",
                payload=export_strict_snapshot(latest_blackboard),
            )
            self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision_export)
        round_record = {
            "round_id": int(round_id),
            "decision": decision_export,
            "outcome": exec_outcome.__dict__,
            "selected_outcome": dict(getattr(exec_outcome, "outcome", {}) or {}),
            "family_handoff_diagnostics": dict(analysis_summary.get("family_handoff_diagnostics", {}) or {}),
            "pre_memory_version": plan_context.memory_version,
            "post_memory_version": latest_memory.memory_version,
            "mechanic_graph_version": latest_mechanic_graph.mechanic_graph_version,
            "pre_memory_state": memory_state,
            "post_memory_state": latest_memory.state,
            "mechanic_graph_state": latest_mechanic_graph.state,
            "mechanic_graph_snapshot_path": mechanic_graph_snapshot_path,
            "hypothesis_registry_snapshot": hypothesis_registry_snapshot,
            "analysis_summary": analysis_summary,
            "live_bridge_rows": [dict(row) for row in live_bridge_rows],
            "live_bridge_diagnostics": dict(analysis_summary.get("live_bridge_diagnostics", {}) or {}),
            "mode_persistence_state": mode_persistence_state,
            "runtime_subgoal_chain_state": dict(chain_runtime_materialization.get("runtime_subgoal_chain_state", {}) or {}),
            "runtime_subgoal_chain_state_after_outcome": dict(chain_outcome_update.get("runtime_subgoal_chain_state_after_outcome", {}) or {}),
            "runtime_chain_rows": [dict(row) for row in list(chain_runtime_materialization.get("runtime_chain_rows", []) or [])] + [dict(row) for row in list(chain_outcome_update.get("runtime_chain_rows", []) or [])],
            "runtime_chain_event_count": len([dict(row) for row in list(chain_runtime_materialization.get("runtime_chain_rows", []) or [])] + [dict(row) for row in list(chain_outcome_update.get("runtime_chain_rows", []) or [])]),
            "runtime_chain_step_event_count": int(chain_runtime_materialization.get("runtime_step_event_count", 0) or 0) + int(chain_outcome_update.get("runtime_step_event_count", 0) or 0),
        }
        selected_chain = dict(dict(decision_export.get("metadata", {}) or {}).get("selected_subgoal_chain", {}) or {})
        selected_step = dict(dict(decision_export.get("metadata", {}) or {}).get("selected_subgoal_step", {}) or {})
        if selected_chain and int(round_record.get("runtime_chain_event_count", 0) or 0) <= 0:
            raise AssertionError("selected chain present in decision => no runtime chain row exists in this round")
        if selected_step and int(round_record.get("runtime_chain_step_event_count", 0) or 0) <= 0:
            raise AssertionError("selected step present in decision => no runtime chain step row exists in this round")
        if selected_step and dict(getattr(exec_outcome, "outcome", {}) or {}).get("step_kind") and not bool(chain_outcome_update.get("runtime_progress_updated", False)):
            raise AssertionError("executor outcome finished selected step without same-round runtime chain progress update")
        self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name="analysis_summary.json", payload=analysis_summary)
        self.export_round_debug_heatmaps(
            round_id=round_id,
            analyzed_rows=new_analyzed_rows,
            blackboard_state=latest_blackboard.state,
            first_observation=first_observation,
        )
        selected_candidate_export = decision_export.get("metadata", {}).get("selected_candidate", {})
        stop_outcome = {
            "round_progress": round_progress,
            "won": bool(exec_outcome.episode.won),
            "selected_candidate": dict(selected_candidate_export) if isinstance(selected_candidate_export, dict) else {},
            "outcome": exec_outcome.__dict__,
            "analysis_summary": analysis_summary,
        }
        return (
            RoundRunResult(
                latest_blackboard=latest_blackboard,
                latest_memory=latest_memory,
                latest_mechanic_graph=latest_mechanic_graph,
                hypothesis_registry_snapshot=hypothesis_registry_snapshot,
                first_observation=first_observation,
                analyzed_rows=new_analyzed_rows,
                round_record=round_record,
                memory_snapshot_path=memory_snapshot_path,
                selected_target_entity_id=selected_target_entity_id,
                stop_outcome=stop_outcome,
            ),
            current_stamp,
        )
