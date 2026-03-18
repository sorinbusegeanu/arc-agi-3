from __future__ import annotations

from v3_1.planning.subgoal_chain import build_step
from v3_1.planning.belief_builder import normalized_route_signature, normalized_target_key, normalized_trigger_zone_key
from v3_1.utils.ids import stable_digest

MAX_SUPPORTING_EVIDENCE_REFS = 12
GENERATION_QUOTAS = {
    "target_interaction": 6,
    "local_probe": 4,
    "frontier_move": 4,
    "route_probe": 3,
    "trigger_probe": 4,
    "recovery_move": 3,
    "fallback_action": 1,
    "unlock_trigger": 3,
    "verify_panel_state": 3,
    "verify_gate_match": 3,
    "unlock_then_exit": 2,
    "trigger_then_target": 3,
    "state_sync_probe": 3,
    "mechanic_test_deterministic": 3,
    "mechanic_chain_deterministic": 3,
    "mechanic_test_llm": 2,
    "mechanic_chain_llm": 2,
}


def _candidate_id(payload: dict) -> str:
    return f"candidate:{stable_digest(payload)}"


def _movement_avatar_enabled(belief: dict) -> bool:
    return str(belief.get("control_mode") or "unknown") == "movement_avatar" and str(belief.get("avatar_runtime_status") or "unknown") == "present"


def _available_action_families(belief: dict) -> set[str]:
    families = set(belief.get("available_action_families", []) or [])
    return families or {"move"}


def _match_skill_id(skill_library: dict[str, dict], *, entity_id: str, area_id: str | None, objective_type: str, execution_mode: str) -> tuple[str | None, str | None]:
    preferred_skill_types = (
        ("inspect_target", "trigger_probe") if objective_type in {"interact", "test_trigger", "gather_local_info"} and execution_mode in {"interact", "click_at"}
        else ("recover_route",) if objective_type == "recover"
        else ()
    )
    for skill_type in preferred_skill_types:
        for skill_id, skill in skill_library.items():
            if str(skill.get("skill_type") or "") != skill_type:
                continue
            if entity_id and str(skill.get("entity_id") or "") == entity_id:
                return str(skill_id), skill_type
            if area_id is not None and str(skill.get("target_area_id") or "") == str(area_id):
                return str(skill_id), skill_type
    return None, None


def _compress_supporting_evidence_refs(refs: list[str] | tuple[str, ...]) -> tuple[list[str], dict]:
    ordered = [str(ref) for ref in refs if ref]
    unique = list(dict.fromkeys(ordered))
    sample = unique[:MAX_SUPPORTING_EVIDENCE_REFS]
    return sample, {
        "supporting_evidence_ref_count": len(unique),
        "supporting_evidence_ref_sample": sample,
        "supporting_evidence_signature": stable_digest(unique),
        "supporting_evidence_truncated": len(unique) > len(sample),
        "full_supporting_evidence_refs": unique,
    }


def _derived_candidate_class(*, objective_type: str, execution_mode: str, navigation_mode: str) -> str:
    if objective_type == "unlock_trigger":
        return "unlock_trigger"
    if objective_type == "verify_panel_state":
        return "verify_panel_state"
    if objective_type == "verify_gate_match":
        return "verify_gate_match"
    if objective_type == "unlock_then_exit":
        return "unlock_then_exit"
    if objective_type == "trigger_then_target":
        return "trigger_then_target"
    if objective_type == "state_sync_probe":
        return "state_sync_probe"
    if objective_type == "mechanic_test_deterministic":
        return "mechanic_test_deterministic"
    if objective_type == "mechanic_chain_deterministic":
        return "mechanic_chain_deterministic"
    if objective_type == "mechanic_test_llm":
        return "mechanic_test_llm"
    if objective_type == "mechanic_chain_llm":
        return "mechanic_chain_llm"
    if objective_type == "interact" and execution_mode == "interact":
        return "target_interaction"
    if objective_type == "interact" and execution_mode == "click_at":
        return "click_target"
    if objective_type == "gather_local_info":
        return "local_probe"
    if objective_type == "explore_frontier":
        return "frontier_move"
    if objective_type == "test_trigger":
        return "trigger_probe"
    if objective_type == "recover":
        return "recovery_move"
    if objective_type == "probe_route":
        return "route_probe"
    return "fallback_action"


def _candidate_schema(*, target: dict | None, belief: dict, objective_type: str, execution_mode: str, navigation_mode: str, rationale: str, generation_source: str, trigger_zone_id: str | None = None, support_refs: list[str] | None = None, action_overrides: dict | None = None, fallback_baseline_penalty: float = 0.0, supporting_graph_node_ids: list[str] | None = None, supporting_graph_edge_ids: list[str] | None = None, prerequisite_chain: list[str] | None = None, graph_hop_count: int = 0, hypothesized_only_dependency: bool = False) -> dict:
    target = dict(target or {})
    centroid = list(target.get("centroid", [0, 0]))
    target_entity_id = str(target.get("entity_id")) if target.get("entity_id") is not None else None
    target_area_id = str(target.get("area_id")) if target.get("area_id") is not None else belief.get("current_area_id")
    target_entity_class = str(target.get("kind") or target.get("poi_class") or "unknown")
    identity_confidence = float(target.get("identity_confidence", 0.0) or 0.0)
    identity_status = str(target.get("identity_status") or "unknown")
    route_signature = normalized_route_signature(
        objective_type=objective_type,
        navigation_mode=navigation_mode,
        area_id=target_area_id,
        target_entity_id=target_entity_id,
        centroid=centroid,
        action_hint=str(target.get("action_hint")) if target.get("action_hint") is not None else None,
    )
    target_key = normalized_target_key(target_entity_id, target_area_id, target_class=target_entity_class)
    supporting_refs, support_meta = _compress_supporting_evidence_refs(list(support_refs or target.get("evidence_refs", [])))
    skill_id, skill_type = _match_skill_id(
        belief.get("_skill_library", {}),
        entity_id=target_entity_id or "",
        area_id=target_area_id,
        objective_type=objective_type,
        execution_mode=execution_mode,
    )
    candidate_class = _derived_candidate_class(objective_type=objective_type, execution_mode=execution_mode, navigation_mode=navigation_mode)
    stable_key = {
        "objective_type": objective_type,
        "execution_mode": execution_mode,
        "navigation_mode": navigation_mode,
        "target_key": target_key,
        "route_signature": route_signature,
        "trigger_zone_id": trigger_zone_id or "none",
    }
    contradiction_flags = dict(target.get("contradiction_markers", {}))
    stale_support_flags = {
        "support_refs_missing": bool(supporting_refs) and len(supporting_refs) < int(support_meta.get("supporting_evidence_ref_count", 0)),
        "target_stale": contradiction_flags.get("stale_target", False),
    }
    action = {
        "type": execution_mode,
        "target": target_entity_id,
        "centroid": centroid,
        "skill_id": skill_id,
    }
    if action_overrides:
        action.update(action_overrides)
    direct_support = min(1.0, float(target.get("confidence", 0.0)) + (0.1 * len(supporting_refs)))
    indirect_support = min(1.0, float(target.get("utility", 0.0)) + (0.25 if trigger_zone_id else 0.0))
    prior_key = normalized_target_key(target_entity_id, target_area_id, target_class=target_entity_class)
    prior_row = dict(belief.get("durable_prior_view", {}).get("per_target", {}).get(prior_key, {}))
    prior_support = min(1.0, float(prior_row.get("poi_pattern", {}).get("observations", 0) or 0) / 10.0) if prior_row else 0.0
    expected_progress_type = "state_change" if objective_type in {"interact", "test_trigger"} else "evidence_gain" if objective_type in {"gather_local_info", "probe_route"} else "route_progress" if objective_type in {"explore_frontier", "recover"} else "fallback"
    candidate_intent_mode = (
        "validation" if objective_type in {"test_trigger", "verify_panel_state", "verify_gate_match", "state_sync_probe"}
        else "information_gathering" if objective_type in {"gather_local_info", "probe_route", "unlock_trigger"}
        else "progress"
    )
    return {
        "candidate_id": _candidate_id(stable_key),
        "candidate_class": candidate_class,
        "objective_type": objective_type,
        "execution_mode": execution_mode,
        "navigation_mode": navigation_mode,
        "target_entity_id": target_entity_id,
        "target_area_id": target_area_id,
        "target_key": target_key,
        "route_signature": route_signature,
        "trigger_zone_id": trigger_zone_id,
        "target_entity_class": target_entity_class,
        "required_action_family": execution_mode,
        "effect_action_family": str(target.get("candidate_effect_mode") or execution_mode),
        "candidate_intent_mode": candidate_intent_mode,
        "candidate_context": {
            "avatar_area": belief.get("local_context_view", {}).get("current_area_id"),
            "local_area": belief.get("current_area_id"),
            "route_signature": route_signature,
            "trigger_zone_id": trigger_zone_id,
            "target_entity_class": target_entity_class,
        },
        "expected_progress_type": expected_progress_type,
        "expected_outcomes": {
            "expected_state_change": float(target.get("candidate_effect_score", target.get("interact_effect_score", 0.0))),
            "expected_evidence_gain": min(1.0, float(target.get("novelty", 0.0)) + float(target.get("motion_score", 0.0))),
            "expected_route_progress": float(target.get("distance_score", 0.0)) if navigation_mode in {"direct", "routed"} else 0.0,
        },
        "support_strength": {
            "direct_support": direct_support,
            "indirect_support": indirect_support,
            "prior_support": prior_support,
        },
        "contradiction_flags": contradiction_flags,
        "stale_support_flags": stale_support_flags,
        "supporting_evidence_refs": supporting_refs,
        "full_supporting_evidence_refs": list(support_meta["full_supporting_evidence_refs"]),
        "generation_source": generation_source,
        "supporting_graph_node_ids": list(dict.fromkeys(str(value) for value in list(supporting_graph_node_ids or []) if value)),
        "supporting_graph_edge_ids": list(dict.fromkeys(str(value) for value in list(supporting_graph_edge_ids or []) if value)),
        "hop_count": int(graph_hop_count),
        "prerequisite_chain": list(prerequisite_chain or []),
        "depends_on_hypothesized_only_edges": bool(hypothesized_only_dependency),
        "skill_id": skill_id,
        "skill_type": skill_type,
        **{key: value for key, value in support_meta.items() if key != "full_supporting_evidence_refs"},
        "action": action,
        "confidence": float(target.get("confidence", 0.0)),
        "utility": float(target.get("utility", 0.0)),
        "novelty": float(target.get("novelty", 0.0)),
        "movement_effect_score": float(target.get("movement_effect_score", 0.0)),
        "interact_effect_score": float(target.get("interact_effect_score", 0.0)),
        "click_effect_score": float(target.get("click_effect_score", 0.0)),
        "candidate_effect_score": float(target.get("candidate_effect_score", 0.0)),
        "distance_from_avatar": float(target.get("distance_from_avatar", 0.0)),
        "distance_score": float(target.get("distance_score", 0.0)),
        "motion_variance": float(target.get("motion_variance", 0.0)),
        "motion_score": float(target.get("motion_score", 0.0)),
        "identity_confidence": identity_confidence,
        "identity_status": identity_status,
        "reachable_now": bool(target.get("reachable_now")),
        "reachable_later": bool(target.get("reachable_later")),
        "rationale": rationale,
        "route_required": bool(navigation_mode in {"direct", "routed"}),
        "fallback_baseline_penalty": float(fallback_baseline_penalty),
        "blocked_reasons": [],
        "blocked_reason_details": [],
    }


def _seeded_candidate(row: dict, *, seed_contract: str, observed_row_ids: list[str] | None = None, hypothesis_row_ids: list[str] | None = None) -> dict:
    payload = dict(row)
    payload["seed_contract"] = seed_contract
    payload["seed_observed_row_ids"] = list(dict.fromkeys(str(value) for value in list(observed_row_ids or []) if value))
    payload["seed_hypothesis_row_ids"] = list(dict.fromkeys(str(value) for value in list(hypothesis_row_ids or []) if value))
    payload["seed_requires_hypothesis"] = bool(payload["seed_hypothesis_row_ids"] and not payload["seed_observed_row_ids"])
    return payload


def _node_step_kind(node_id: str) -> str:
    text = str(node_id or "").lower()
    if ":trigger" in text or "trigger" in text or "button" in text:
        return "go_to_trigger"
    if ":panel" in text or "panel" in text or "symbol" in text:
        return "verify_panel"
    if ":gate" in text or "gate" in text or "door" in text:
        return "verify_gate"
    if ":exit" in text or "exit" in text:
        return "attempt_exit"
    return "reobserve_region"


def _is_executable_node_id(node_id: str) -> bool:
    step_kind = _node_step_kind(node_id)
    return step_kind in {"go_to_trigger", "verify_panel", "verify_gate", "attempt_exit"}


def _executable_chain_nodes(node_ids: list[str]) -> list[str]:
    return [str(node_id) for node_id in list(node_ids or []) if node_id and _is_executable_node_id(str(node_id))]


def _step_expectations(step_kind: str, target_node_id: str) -> tuple[list[str], list[str], list[str]]:
    if step_kind == "go_to_trigger":
        return (["objective_contact_observed", f"node:{target_node_id}:contact"], ["contact_observed"], ["blocked", "missing_target"])
    if step_kind == "verify_panel":
        return ([f"node:{target_node_id}:pattern_seen", "target_presence_observed"], ["expected_match_seen", "panel_state_seen"], ["expected_match_missing", "missing_target"])
    if step_kind == "verify_gate":
        return ([f"node:{target_node_id}:gate_state", "target_presence_observed"], ["gate_state_changed_after_trigger", "gate_presence_seen"], ["missing_target", "contradiction_seen"])
    if step_kind == "attempt_exit":
        return ([f"node:{target_node_id}:exit_attempt", "done_observed"], ["objective_contact_observed", "done_observed"], ["blocked", "exit_failure"])
    return ([f"node:{target_node_id}:reobserved"], ["target_presence_observed"], ["missing_target"])


def _build_candidate_step_plan(*, chain_nodes: list[str], source_label: str) -> tuple[list[dict], list[str], list[str], list[str], list[str], float]:
    steps = []
    verification_points: list[str] = []
    step_kinds: list[str] = []
    aggregated_expected_evidence: list[str] = []
    fallback_targets: list[str] = []
    previous_step_ids: list[str] = []
    for node_id in list(chain_nodes or []):
        if not node_id:
            continue
        step_kind = _node_step_kind(str(node_id))
        step_expected_evidence, success_conditions, failure_conditions = _step_expectations(step_kind, str(node_id))
        fallback_target = chain_nodes[min(len(chain_nodes) - 1, len(previous_step_ids))] if chain_nodes else None
        step = build_step(
            step_kind=step_kind,
            target_node_id=str(node_id),
            expected_evidence=step_expected_evidence,
            success_conditions=success_conditions,
            failure_conditions=failure_conditions,
            retry_budget=2 if step_kind in {"go_to_trigger", "retry_trigger", "reobserve_region"} else 1,
            depends_on_step_ids=previous_step_ids,
            verification_points=[f"{source_label}:{node_id}:{step_kind}"],
            fallback_targets=[fallback_target] if fallback_target else [],
        )
        steps.append(step.to_dict())
        previous_step_ids.append(step.step_id)
        step_kinds.append(step_kind)
        verification_points.extend(list(step.verification_points))
        aggregated_expected_evidence.extend(list(step.expected_evidence))
        if fallback_target:
            fallback_targets.append(str(fallback_target))
    execution_feasibility = min(1.0, (0.25 * len(steps)) + (0.08 * len(verification_points)) - (0.05 * max(0, len(steps) - 4)))
    return steps, step_kinds, verification_points, list(dict.fromkeys(aggregated_expected_evidence)), list(dict.fromkeys(fallback_targets)), max(0.0, execution_feasibility)


def _graph_target_from_node_id(belief: dict, node_id: str) -> dict:
    lookup = dict(belief.get("mechanic_graph_node_lookup", {}) or {})
    row = dict(lookup.get(str(node_id), {}) or {})
    metadata = dict(row.get("metadata", {}) or {})
    centroid = list(metadata.get("centroid", [0, 0]) or [0, 0])
    bbox = dict(metadata.get("bbox", {}) or {})
    object_ref = str(row.get("object_ref") or "") or str(node_id or "")
    node_kind = str(row.get("node_kind") or "poi")
    return {
        "entity_id": object_ref,
        "area_id": belief.get("current_area_id"),
        "kind": node_kind,
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "utility": float(row.get("confidence", 0.0) or 0.0),
        "centroid": centroid,
        "bbox": bbox,
        "target_label": node_kind,
    }


def _graph_target_exists_in_world(belief: dict, node_id: str) -> bool:
    target = _graph_target_from_node_id(belief, node_id)
    entity_id = str(target.get("entity_id") or "")
    if not entity_id:
        return False
    observed_world = dict(belief.get("observed_world", {}) or {})
    hypothesized_world = dict(belief.get("hypothesized_world", {}) or {})
    observed_entities = dict(observed_world.get("entities", {}) or {})
    hypothesized_entities = dict(hypothesized_world.get("entities", {}) or {})
    return entity_id in observed_entities or entity_id in hypothesized_entities


def _synthetic_trigger_only_node(belief: dict, node_id: str) -> bool:
    lookup = dict(belief.get("mechanic_graph_node_lookup", {}) or {})
    row = dict(lookup.get(str(node_id), {}) or {})
    if str(row.get("node_kind") or "") != "trigger":
        return False
    object_ref = str(row.get("object_ref") or "")
    evidence_tier = str(row.get("evidence_tier") or "hypothesized")
    support_count = int(row.get("support_count", 0) or 0)
    return object_ref.startswith("trigger:") and evidence_tier != "observed" and support_count <= 1


def _strong_full_chain_seed(belief: dict, node_id: str) -> bool:
    lookup = dict(belief.get("mechanic_graph_node_lookup", {}) or {})
    row = dict(lookup.get(str(node_id), {}) or {})
    if not row:
        return False
    if str(row.get("node_kind") or "") != "trigger":
        return True
    return bool(
        row.get("object_backed", False)
        and not bool(row.get("synthetic_region_only", False))
        and int(row.get("observed_support_count", 0) or 0) > 0
    )


def generate_candidates(
    skill_library: dict[str, dict],
    belief: dict,
    limit: int,
    *,
    observed_world: dict | None = None,
    hypothesized_world: dict | None = None,
) -> list[dict]:
    belief = dict(belief)
    belief["_skill_library"] = skill_library
    candidates: list[dict] = []
    diagnostics = {"count_by_class": {}, "dropped_during_generation": 0, "unsupported_template_count": 0}
    class_counts: dict[str, int] = {}
    available_families = _available_action_families(belief)
    movement_avatar_enabled = _movement_avatar_enabled(belief)
    planning_mode = str(belief.get("planning_mode") or "default_progress")

    def _admit(row: dict | None) -> None:
        if row is None:
            diagnostics["unsupported_template_count"] += 1
            return
        candidate_class = str(row.get("candidate_class") or "unknown")
        quota = GENERATION_QUOTAS.get(candidate_class, limit)
        count = class_counts.get(candidate_class, 0)
        if count >= quota:
            diagnostics["dropped_during_generation"] += 1
            return
        class_counts[candidate_class] = count + 1
        diagnostics["count_by_class"][candidate_class] = class_counts[candidate_class]
        row["generation_diagnostics"] = diagnostics
        candidates.append(row)

    observed_world = dict(observed_world or belief.get("observed_world", {}))
    hypothesized_world = dict(hypothesized_world or belief.get("hypothesized_world", {}))
    observed_seeds = {
        "reachable_targets": list(observed_world.get("reachable_targets", [])),
        "promising_pois": list(observed_world.get("promising_pois", [])),
        "frontier_targets": list(observed_world.get("frontier_targets", [])),
        "trigger_candidates": list(observed_world.get("trigger_candidates", [])),
        "recovery_candidates": list(observed_world.get("recovery_candidates", [])),
    }
    hypothesis_seeds = {
        "reachable_targets": list(hypothesized_world.get("reachable_targets", [])),
        "promising_pois": list(hypothesized_world.get("promising_pois", [])),
        "frontier_targets": list(hypothesized_world.get("frontier_targets", [])),
        "trigger_candidates": list(hypothesized_world.get("trigger_candidates", [])),
        "recovery_candidates": list(hypothesized_world.get("recovery_candidates", [])),
    }

    def _row_id(target: dict) -> str | None:
        return str(target.get("entity_id") or target.get("trigger_id") or target.get("node_id") or target.get("edge_id") or "") or None

    mechanic_graph_view = dict(belief.get("mechanic_graph_view", {}) or {})
    mechanic_paths = list(belief.get("mechanic_graph_paths_to_exit", []) or mechanic_graph_view.get("paths_to_exit", []) or [])
    mechanic_match_relations = list(belief.get("mechanic_graph_match_relations", []) or mechanic_graph_view.get("match_relations", []) or [])
    mechanic_trigger_candidates = list(belief.get("mechanic_graph_trigger_candidates", []) or mechanic_graph_view.get("trigger_candidates", []) or [])
    deterministic_hypotheses = dict(belief.get("deterministic_hypotheses", {}) or {})
    llm_hypotheses = dict(belief.get("llm_hypotheses", {}) or {})
    active_chain_summary = dict(belief.get("active_chain_summary", {}) or {})
    chain_progress_summary = dict(belief.get("chain_progress_summary", {}) or {})

    if active_chain_summary and movement_avatar_enabled:
        active_step = dict(chain_progress_summary.get("current_step", {}) or {})
        chain_steps = list(active_chain_summary.get("steps", []) or [])
        target_node_id = str(active_step.get("target_node_id") or "") or None
        candidate = _candidate_schema(
            target={"entity_id": target_node_id, "area_id": belief.get("current_area_id"), "kind": "poi", "confidence": 0.8, "utility": 0.85},
            belief=belief,
            objective_type="unlock_then_exit" if str(active_step.get("step_kind") or "") == "attempt_exit" else "trigger_then_target",
            execution_mode="interact" if str(active_step.get("step_kind") or "") in {"go_to_trigger", "retry_trigger"} and "interact" in available_families else "move",
            navigation_mode="routed",
            rationale="active_subgoal_chain_step",
            generation_source="subgoal_chain.active_step",
            supporting_graph_node_ids=[target_node_id] if target_node_id else [],
            prerequisite_chain=[str(row.get("target_node_id") or "") for row in chain_steps if isinstance(row, dict) and row.get("target_node_id")],
        )
        candidate["candidate_step_plan"] = chain_steps
        candidate["candidate_step_kinds"] = [str(row.get("step_kind") or "") for row in chain_steps if isinstance(row, dict)]
        candidate["candidate_verification_points"] = list(active_step.get("verification_points", []) or [])
        candidate["candidate_expected_evidence"] = list(active_step.get("expected_evidence", []) or [])
        candidate["candidate_fallback_targets"] = list(active_step.get("fallback_targets", []) or [])
        candidate["candidate_execution_feasibility"] = 1.0
        candidate["source_path_id"] = str(active_chain_summary.get("source_path_id") or "") or None
        candidate["target_exit_id"] = str(active_chain_summary.get("target_exit_id") or "") or None
        candidate["expected_outcome_ids"] = list(active_chain_summary.get("expected_outcome_ids", []) or [])
        candidate["fallback_policy"] = str(active_chain_summary.get("fallback_policy") or "replan")
        _admit(candidate)

    for row in mechanic_paths:
        if not movement_avatar_enabled:
            continue
        node_ids = list(row.get("node_ids", []) or [])
        edge_ids = list(row.get("edge_ids", []) or [])
        if node_ids and _synthetic_trigger_only_node(belief, str(node_ids[0])) and bool(row.get("hypothesis_only", True)):
            continue
        if node_ids and not _strong_full_chain_seed(belief, str(node_ids[0])):
            continue
        executable_nodes = _executable_chain_nodes(node_ids)
        if not executable_nodes:
            continue
        if not _graph_target_exists_in_world(belief, executable_nodes[0]):
            diagnostics["dropped_during_generation"] += 1
            continue
        step_plan, step_kinds, verification_points, candidate_expected_evidence, candidate_fallback_targets, candidate_execution_feasibility = _build_candidate_step_plan(chain_nodes=executable_nodes, source_label="mechanic_path")
        first_step_executability = float(row.get("first_step_executability_score", 0.0) or 0.0)
        if first_step_executability < 0.45:
            continue
        target = _graph_target_from_node_id(belief, executable_nodes[0])
        target["confidence"] = max(float(target.get("confidence", 0.0) or 0.0), float(row.get("support_strength", 0.0) or 0.0))
        target["utility"] = max(float(target.get("utility", 0.0) or 0.0), float(row.get("support_strength", 0.0) or 0.0))
        payload = _candidate_schema(
                target=target,
                belief=belief,
                objective_type="unlock_then_exit",
                execution_mode="move",
                navigation_mode="routed",
                rationale="mechanic_path_to_exit",
                generation_source="mechanic_graph.paths_to_exit",
                supporting_graph_node_ids=executable_nodes,
                supporting_graph_edge_ids=edge_ids,
                graph_hop_count=max(0, len(edge_ids)),
                prerequisite_chain=executable_nodes,
                hypothesized_only_dependency=bool(row.get("hypothesis_only")),
            )
        payload["candidate_step_plan"] = step_plan
        payload["candidate_step_kinds"] = step_kinds
        payload["candidate_verification_points"] = verification_points
        payload["candidate_expected_evidence"] = candidate_expected_evidence
        payload["candidate_fallback_targets"] = candidate_fallback_targets
        payload["candidate_execution_feasibility"] = candidate_execution_feasibility
        payload["first_step_executability_score"] = first_step_executability
        payload["evidence_diversity_score"] = float(row.get("evidence_diversity_score", 0.0) or 0.0)
        payload["counterfactual_strength"] = float(row.get("counterfactual_strength", 0.0) or 0.0)
        payload["execution_feasibility_score"] = float(row.get("execution_feasibility_score", candidate_execution_feasibility) or candidate_execution_feasibility)
        payload["source_path_id"] = str(row.get("path_id") or stable_digest((tuple(executable_nodes), tuple(edge_ids))))
        payload["target_exit_id"] = str(executable_nodes[-1]) if executable_nodes else None
        payload["expected_outcome_ids"] = [f"edge:{edge_id}" for edge_id in edge_ids]
        payload["fallback_policy"] = "replan"
        _admit(payload)

    for row in mechanic_trigger_candidates:
        if not movement_avatar_enabled:
            continue
        trigger_node_id = row.get("trigger_node_id")
        paths = list(row.get("paths_to_exit", []) or [])
        if not _is_executable_node_id(str(trigger_node_id or "")):
            continue
        if not _graph_target_exists_in_world(belief, str(trigger_node_id or "")):
            diagnostics["dropped_during_generation"] += 1
            continue
        step_plan, step_kinds, verification_points, candidate_expected_evidence, candidate_fallback_targets, candidate_execution_feasibility = _build_candidate_step_plan(chain_nodes=[trigger_node_id], source_label="trigger_candidate")
        target = _graph_target_from_node_id(belief, str(trigger_node_id))
        target["confidence"] = max(float(target.get("confidence", 0.0) or 0.0), 0.5)
        target["utility"] = max(float(target.get("utility", 0.0) or 0.0), 0.6)
        payload = _candidate_schema(
                target=target,
                belief=belief,
                objective_type="unlock_trigger",
                execution_mode="interact" if "interact" in available_families else "move",
                navigation_mode="routed",
                rationale="mechanic_trigger_candidate",
                generation_source="mechanic_graph.trigger_candidates",
                supporting_graph_node_ids=[trigger_node_id],
                supporting_graph_edge_ids=[edge_id for path in paths for edge_id in list(path.get("edge_ids", []) or [])],
                graph_hop_count=max([len(list(path.get("edge_ids", []) or [])) for path in paths] or [0]),
                prerequisite_chain=[trigger_node_id],
                hypothesized_only_dependency=all(bool(path.get("hypothesis_only")) for path in paths) if paths else False,
            )
        payload["candidate_step_plan"] = step_plan
        payload["candidate_step_kinds"] = step_kinds
        payload["candidate_verification_points"] = verification_points
        payload["candidate_expected_evidence"] = candidate_expected_evidence
        payload["candidate_fallback_targets"] = candidate_fallback_targets
        payload["candidate_execution_feasibility"] = candidate_execution_feasibility
        payload["source_path_id"] = str(row.get("trigger_node_id") or "")
        payload["target_exit_id"] = None
        payload["expected_outcome_ids"] = [f"path:{idx}" for idx, _ in enumerate(paths)]
        payload["fallback_policy"] = "retry_or_replan"
        _admit(payload)

    for row in mechanic_match_relations:
        if not movement_avatar_enabled:
            continue
        panel_node_id = row.get("panel_node_id")
        edges = list(row.get("match_edges", []) or [])
        if not panel_node_id or not edges:
            continue
        if not _graph_target_exists_in_world(belief, str(panel_node_id)):
            diagnostics["dropped_during_generation"] += 1
            continue
        step_plan, step_kinds, verification_points, candidate_expected_evidence, candidate_fallback_targets, candidate_execution_feasibility = _build_candidate_step_plan(chain_nodes=[panel_node_id], source_label="match_relation")
        payload = _candidate_schema(
                target={**_graph_target_from_node_id(belief, str(panel_node_id)), "confidence": max(float(_graph_target_from_node_id(belief, str(panel_node_id)).get("confidence", 0.0) or 0.0), 0.45), "utility": max(float(_graph_target_from_node_id(belief, str(panel_node_id)).get("utility", 0.0) or 0.0), 0.5)},
                belief=belief,
                objective_type="verify_panel_state",
                execution_mode="move",
                navigation_mode="routed",
                rationale="mechanic_panel_match",
                generation_source="mechanic_graph.match_relations",
                supporting_graph_node_ids=[panel_node_id],
                supporting_graph_edge_ids=[str(edge.get("edge_id")) for edge in edges],
                graph_hop_count=1,
                prerequisite_chain=[panel_node_id],
                hypothesized_only_dependency=all(str(edge.get("evidence_tier") or "") != "observed" for edge in edges),
            )
        payload["candidate_step_plan"] = step_plan
        payload["candidate_step_kinds"] = step_kinds
        payload["candidate_verification_points"] = verification_points
        payload["candidate_expected_evidence"] = candidate_expected_evidence
        payload["candidate_fallback_targets"] = candidate_fallback_targets
        payload["candidate_execution_feasibility"] = candidate_execution_feasibility
        payload["source_path_id"] = str(panel_node_id)
        payload["target_exit_id"] = None
        payload["expected_outcome_ids"] = [str(edge.get("edge_id") or "") for edge in edges if edge.get("edge_id")]
        payload["fallback_policy"] = "reobserve"
        _admit(payload)

    for proposal in deterministic_hypotheses.values():
        if not movement_avatar_enabled:
            continue
        proposal_kind = str(proposal.get("proposal_kind") or "")
        objective_type = "mechanic_test_deterministic" if proposal_kind == "test" else "mechanic_chain_deterministic"
        source_ids = [str(proposal.get("proposal_id") or "")]
        chain = [str(proposal.get("src_node_id") or ""), str(proposal.get("dst_node_id") or "")]
        executable_nodes = _executable_chain_nodes(chain)
        if not executable_nodes:
            diagnostics["dropped_during_generation"] += 1
            continue
        if not _graph_target_exists_in_world(belief, executable_nodes[0]):
            diagnostics["dropped_during_generation"] += 1
            continue
        if not _strong_full_chain_seed(belief, executable_nodes[0]):
            diagnostics["dropped_during_generation"] += 1
            continue
        step_plan, step_kinds, verification_points, candidate_expected_evidence, candidate_fallback_targets, candidate_execution_feasibility = _build_candidate_step_plan(chain_nodes=executable_nodes, source_label="deterministic")
        payload = _candidate_schema(
                target={**_graph_target_from_node_id(belief, executable_nodes[0]), "confidence": max(float(_graph_target_from_node_id(belief, executable_nodes[0]).get("confidence", 0.0) or 0.0), float(proposal.get("confidence", 0.0) or 0.0)), "utility": max(float(_graph_target_from_node_id(belief, executable_nodes[0]).get("utility", 0.0) or 0.0), float(proposal.get("confidence", 0.0) or 0.0))},
                belief=belief,
                objective_type=objective_type,
                execution_mode="move",
                navigation_mode="routed",
                rationale="deterministic_hypothesis_candidate",
                generation_source="hypothesis.deterministic",
                supporting_graph_node_ids=executable_nodes,
                supporting_graph_edge_ids=list(proposal.get("expected_edge_ids", []) or []),
                graph_hop_count=max(0, len(executable_nodes) - 1),
                prerequisite_chain=executable_nodes,
                hypothesized_only_dependency=True,
                action_overrides={
                    "hypothesis_source": "deterministic",
                    "supporting_hypothesis_ids": source_ids,
                    "requires_validation": bool(proposal.get("requires_validation", True)),
                    "expected_information_gain": float(proposal.get("expected_information_gain", 0.4) or 0.4),
                    "chain_length": max(0, len(executable_nodes) - 1),
                    "dependency_path_ids": source_ids,
                    "test_proposal_id": str(proposal.get("test_id") or proposal.get("proposal_id") or "") if proposal_kind == "test" else None,
                    "target_node_ids": list(proposal.get("target_node_ids", []) or []),
                    "estimated_cost": float(proposal.get("estimated_cost", 1.0) or 1.0),
                    "discriminates_between_proposal_ids": list(proposal.get("discriminates_between_proposal_ids", []) or []),
                },
            )
        payload["candidate_step_plan"] = step_plan
        payload["candidate_step_kinds"] = step_kinds
        payload["candidate_verification_points"] = verification_points
        payload["candidate_expected_evidence"] = candidate_expected_evidence
        payload["candidate_fallback_targets"] = candidate_fallback_targets
        payload["candidate_execution_feasibility"] = candidate_execution_feasibility
        payload["source_path_id"] = str(proposal.get("proposal_id") or "")
        payload["target_exit_id"] = str(executable_nodes[-1]) if executable_nodes else None
        payload["expected_outcome_ids"] = list(proposal.get("expected_edge_ids", []) or [])
        payload["fallback_policy"] = "replan"
        _admit(payload)

    for proposal in llm_hypotheses.values():
        if not movement_avatar_enabled:
            continue
        proposal_kind = str(proposal.get("proposal_kind") or "")
        validation_state = str(dict(belief.get("hypothesis_registry_snapshot", {}) or {}).get("validation_state", {}).get(str(proposal.get("proposal_id") or ""), "new"))
        if proposal_kind == "test" and validation_state != "validated":
            continue
        objective_type = "mechanic_test_llm" if proposal_kind == "test" else "mechanic_chain_llm"
        source_ids = [str(proposal.get("proposal_id") or "")]
        chain = [str(proposal.get("src_node_id") or ""), str(proposal.get("dst_node_id") or "")]
        executable_nodes = _executable_chain_nodes(chain)
        if not executable_nodes:
            diagnostics["dropped_during_generation"] += 1
            continue
        if not _graph_target_exists_in_world(belief, executable_nodes[0]):
            diagnostics["dropped_during_generation"] += 1
            continue
        if not _strong_full_chain_seed(belief, executable_nodes[0]):
            diagnostics["dropped_during_generation"] += 1
            continue
        step_plan, step_kinds, verification_points, candidate_expected_evidence, candidate_fallback_targets, candidate_execution_feasibility = _build_candidate_step_plan(chain_nodes=executable_nodes, source_label="llm")
        payload = _candidate_schema(
                target={**_graph_target_from_node_id(belief, executable_nodes[0]), "confidence": max(float(_graph_target_from_node_id(belief, executable_nodes[0]).get("confidence", 0.0) or 0.0), float(proposal.get("confidence", 0.0) or 0.0)), "utility": max(float(_graph_target_from_node_id(belief, executable_nodes[0]).get("utility", 0.0) or 0.0), float(proposal.get("confidence", 0.0) or 0.0))},
                belief=belief,
                objective_type=objective_type,
                execution_mode="move",
                navigation_mode="routed",
                rationale="llm_hypothesis_candidate",
                generation_source="hypothesis.llm",
                supporting_graph_node_ids=executable_nodes,
                supporting_graph_edge_ids=list(proposal.get("expected_edge_ids", []) or []),
                graph_hop_count=max(0, len(executable_nodes) - 1),
                prerequisite_chain=executable_nodes,
                hypothesized_only_dependency=True,
                action_overrides={
                    "hypothesis_source": "llm",
                    "supporting_hypothesis_ids": source_ids,
                    "requires_validation": bool(proposal.get("requires_validation", True)),
                    "expected_information_gain": float(proposal.get("expected_information_gain", 0.3) or 0.3),
                    "chain_length": max(0, len(executable_nodes) - 1),
                    "dependency_path_ids": source_ids,
                    "test_proposal_id": str(proposal.get("test_id") or proposal.get("proposal_id") or "") if proposal_kind == "test" else None,
                    "target_node_ids": list(proposal.get("target_node_ids", []) or []),
                    "estimated_cost": float(proposal.get("estimated_cost", 1.0) or 1.0),
                    "discriminates_between_proposal_ids": list(proposal.get("discriminates_between_proposal_ids", []) or []),
                },
            )
        payload["candidate_step_plan"] = step_plan
        payload["candidate_step_kinds"] = step_kinds
        payload["candidate_verification_points"] = verification_points
        payload["candidate_expected_evidence"] = candidate_expected_evidence
        payload["candidate_fallback_targets"] = candidate_fallback_targets
        payload["candidate_execution_feasibility"] = candidate_execution_feasibility
        payload["source_path_id"] = str(proposal.get("proposal_id") or "")
        payload["target_exit_id"] = str(executable_nodes[-1]) if executable_nodes else None
        payload["expected_outcome_ids"] = list(proposal.get("expected_edge_ids", []) or [])
        payload["fallback_policy"] = "replan"
        _admit(payload)

    for target in observed_seeds["reachable_targets"]:
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="reachable_target", generation_source="observed.reachable_targets"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))
        if "click_at" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="click_at", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="reachable_click_target", generation_source="observed.reachable_targets", action_overrides={"click_target_coordinates": list(target.get("centroid", [0, 0]))}), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in observed_seeds["promising_pois"]:
        if target.get("area_id") != belief.get("current_area_id"):
            continue
        if "interact" in available_families:
            candidate = _seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="gather_local_info", execution_mode="interact", navigation_mode="local", rationale="local_probe", generation_source="observed.promising_pois"), seed_contract="observed_only", observed_row_ids=[_row_id(target)])
            if planning_mode == "structure_acquisition" and str(target.get("poi_class") or target.get("kind") or "") == "structure":
                candidate["candidate_intent_mode"] = "information_gathering"
                candidate["utility"] = max(float(candidate.get("utility", 0.0) or 0.0), 0.65)
                candidate["novelty"] = max(float(candidate.get("novelty", 0.0) or 0.0), 0.35)
            _admit(candidate)

    for target in observed_seeds["frontier_targets"]:
        if movement_avatar_enabled and "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="explore_frontier", execution_mode="move", navigation_mode="routed" if not target.get("reachable_now") else "direct", rationale="frontier_target", generation_source="observed.frontier_targets"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in observed_seeds["trigger_candidates"]:
        if movement_avatar_enabled and "interact" in available_families:
            trigger_zone_id = normalized_trigger_zone_key(entity_id=str(target.get("entity_id")), area_id=str(target.get("area_id")) if target.get("area_id") is not None else None)
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="test_trigger", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="trigger_supported", generation_source="observed.trigger_candidates", trigger_zone_id=trigger_zone_id), seed_contract="observed_only", observed_row_ids=[_row_id(target), trigger_zone_id]))

    for target in observed_seeds["recovery_candidates"]:
        if movement_avatar_enabled and "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="recover", execution_mode="move", navigation_mode="routed", rationale="recovery_target", generation_source="observed.recovery_candidates"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["reachable_targets"]:
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_reachable_target", generation_source="hypothesis.reachable_targets"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))
        if "click_at" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="click_at", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_reachable_click_target", generation_source="hypothesis.reachable_targets", action_overrides={"click_target_coordinates": list(target.get("centroid", [0, 0]))}), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["promising_pois"]:
        if target.get("area_id") != belief.get("current_area_id"):
            continue
        if "interact" in available_families:
            candidate = _seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="gather_local_info", execution_mode="interact", navigation_mode="local", rationale="hypothesis_local_probe", generation_source="hypothesis.promising_pois"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)])
            if planning_mode == "structure_acquisition" and str(target.get("poi_class") or target.get("kind") or "") == "structure":
                candidate["candidate_intent_mode"] = "information_gathering"
                candidate["utility"] = max(float(candidate.get("utility", 0.0) or 0.0), 0.5)
                candidate["novelty"] = max(float(candidate.get("novelty", 0.0) or 0.0), 0.25)
            _admit(candidate)

    for target in hypothesis_seeds["frontier_targets"]:
        if movement_avatar_enabled and "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="explore_frontier", execution_mode="move", navigation_mode="routed" if not target.get("reachable_now") else "direct", rationale="hypothesis_frontier_target", generation_source="hypothesis.frontier_targets"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["trigger_candidates"]:
        if movement_avatar_enabled and "interact" in available_families:
            trigger_zone_id = normalized_trigger_zone_key(entity_id=str(target.get("entity_id")), area_id=str(target.get("area_id")) if target.get("area_id") is not None else None)
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="test_trigger", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_trigger_supported", generation_source="hypothesis.trigger_candidates", trigger_zone_id=trigger_zone_id), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target), trigger_zone_id]))

    for target in hypothesis_seeds["recovery_candidates"]:
        if movement_avatar_enabled and "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="recover", execution_mode="move", navigation_mode="routed", rationale="hypothesis_recovery_target", generation_source="hypothesis.recovery_candidates"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    if movement_avatar_enabled and "move" in available_families:
        observed_consequences = list(observed_world.get("consequences", {}).values()) if isinstance(observed_world.get("consequences"), dict) else list(observed_world.get("consequences", []))
        hypothesized_consequences = list(hypothesized_world.get("consequences", {}).values()) if isinstance(hypothesized_world.get("consequences"), dict) else list(hypothesized_world.get("consequences", []))
        consequence_groups: dict[str, dict[str, list[dict]]] = {}
        for row in observed_consequences:
            action_text = str(row.get("action_name") or row.get("action_key") or "").strip().lower()
            if action_text:
                consequence_groups.setdefault(action_text, {"observed": [], "hypothesized": []})["observed"].append(dict(row))
        for row in hypothesized_consequences:
            action_text = str(row.get("action_name") or row.get("action_key") or "").strip().lower()
            if action_text:
                consequence_groups.setdefault(action_text, {"observed": [], "hypothesized": []})["hypothesized"].append(dict(row))
        for action_text, grouped_rows in consequence_groups.items():
            consequence_rows = list(grouped_rows["observed"]) + list(grouped_rows["hypothesized"])
            if not consequence_rows:
                continue
            observed_ids = [str(row.get("consequence_id")) for row in grouped_rows["observed"] if row.get("consequence_id")]
            hypothesis_ids = [str(row.get("consequence_id")) for row in grouped_rows["hypothesized"] if row.get("consequence_id")]
            seed_mode = "observed_only" if observed_ids and not hypothesis_ids else "hypothesis_backfill" if hypothesis_ids and not observed_ids else "observed_plus_hypothesis"
            route_candidate = _seeded_candidate(_candidate_schema(
                    target={"area_id": belief.get("current_area_id"), "action_hint": action_text, "novelty": 0.1, "utility": min(0.5, 0.1 * len(consequence_rows)), "confidence": 0.25},
                    belief=belief,
                    objective_type="probe_route",
                    execution_mode="move",
                    navigation_mode="routed",
                    rationale="consequence_supported_route_probe",
                    generation_source="observed.consequences" if observed_ids else "hypothesis.consequences",
                    support_refs=[str(row.get("consequence_id")) for row in consequence_rows if row.get("consequence_id")],
                    action_overrides={"action_hint": action_text},
                ), seed_contract="observed_only" if observed_ids else "hypothesis_backfill", observed_row_ids=observed_ids, hypothesis_row_ids=hypothesis_ids)
            route_candidate["route_probe_seed_mode"] = seed_mode
            route_candidate["route_probe_observed_consequence_ids"] = observed_ids
            route_candidate["route_probe_hypothesis_consequence_ids"] = hypothesis_ids
            route_candidate["route_probe_used_compatibility_fallback"] = False
            if planning_mode == "structure_acquisition":
                route_candidate["candidate_intent_mode"] = "information_gathering"
            _admit(route_candidate)

    if planning_mode == "structure_acquisition":
        for target in observed_seeds["frontier_targets"][:4]:
            if movement_avatar_enabled and "move" in available_families:
                candidate = _seeded_candidate(
                    _candidate_schema(
                        target={**dict(target), "utility": max(float(target.get("utility", 0.0) or 0.0), 0.55), "novelty": max(float(target.get("novelty", 0.0) or 0.0), 0.4)},
                        belief=belief,
                        objective_type="explore_frontier",
                        execution_mode="move",
                        navigation_mode="routed" if not target.get("reachable_now") else "direct",
                        rationale="structure_acquisition_frontier",
                        generation_source="observed.frontier_targets.structure_acquisition",
                    ),
                    seed_contract="observed_only",
                    observed_row_ids=[_row_id(target)],
                )
                candidate["candidate_intent_mode"] = "information_gathering"
                _admit(candidate)

    _admit(
        _seeded_candidate(_candidate_schema(
            target={"area_id": belief.get("current_area_id"), "utility": 0.0, "confidence": 0.0, "novelty": 0.0},
            belief=belief,
            objective_type="fallback",
            execution_mode="move",
            navigation_mode="hold",
            rationale="always_available_fallback",
            generation_source="fallback.template",
            action_overrides={"type": "hold_position", "area_id": belief.get("current_area_id")},
            fallback_baseline_penalty=2.0,
        ), seed_contract="split_world_native_default", observed_row_ids=[], hypothesis_row_ids=[])
    )

    deduped: dict[tuple[str, str | None, str, str], dict] = {}
    for row in candidates:
        key = (str(row["objective_type"]), row.get("target_key"), str(row.get("execution_mode")), str(row.get("navigation_mode")))
        if key not in deduped or float(row.get("confidence", 0.0)) > float(deduped[key].get("confidence", 0.0)):
            deduped[key] = row

    ranked = sorted(
        deduped.values(),
        key=lambda row: (
            row.get("objective_type") == "fallback",
            not bool(row.get("reachable_now")),
            not bool(row.get("reachable_later")),
            -float(row.get("utility", 0.0)),
            -float(row.get("novelty", 0.0)),
            -float(row.get("confidence", 0.0)),
            row["candidate_id"],
        ),
    )
    return ranked[:limit]
