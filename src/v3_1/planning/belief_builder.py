from __future__ import annotations

from v3_1.world.consequences import normalized_consequence_action_key
from v3_1.world.mechanic_graph_queries import best_supported_paths_to_exit, find_edges_by_kind, find_nodes_by_kind
from v3_1.world.queries import area_local_pois, frontier_candidates, reachable_targets, unreachable_targets


def normalized_target_key(entity_id: str | None, area_id: str | None = None, *, target_class: str | None = None) -> str:
    return f"target:{target_class or 'unknown'}:{area_id or 'none'}:{entity_id or 'none'}"


def normalized_route_signature(*, objective_type: str, navigation_mode: str, area_id: str | None, target_entity_id: str | None = None, centroid: list | tuple | None = None, action_hint: str | None = None) -> str:
    if action_hint:
        return f"route:{objective_type}:{navigation_mode}:{area_id or 'none'}:{action_hint}"
    if centroid and len(centroid) >= 2:
        return f"route:{objective_type}:{navigation_mode}:{area_id or 'none'}:{int(centroid[0])}:{int(centroid[1])}:{target_entity_id or 'none'}"
    return f"route:{objective_type}:{navigation_mode}:{area_id or 'none'}:{target_entity_id or 'none'}"


def normalized_trigger_zone_key(*, trigger_zone_id: str | None = None, entity_id: str | None = None, area_id: str | None = None) -> str:
    return f"trigger:{trigger_zone_id or 'none'}:{area_id or 'none'}:{entity_id or 'none'}"


def _row_provenance(*, source_section: str, refs: list[str] | None = None, derived: bool = False) -> dict:
    return {
        "source_section": source_section,
        "supporting_refs": list(dict.fromkeys(str(ref) for ref in list(refs or []) if ref)),
        "derived": bool(derived),
    }


def _local_context_view(*, current_area_id, blackboard_snapshot: dict, plan_memory: list[dict]) -> dict:
    recent_decisions = plan_memory[-5:]
    recent_target_ids = []
    for row in recent_decisions:
        decision = dict(row.get("decision", {}))
        selected = dict(decision.get("selected_action", {})) if isinstance(decision.get("selected_action"), dict) else {}
        target_id = selected.get("target_entity_id") or selected.get("target")
        if target_id:
            recent_target_ids.append(str(target_id))
    return {
        "current_area_id": current_area_id,
        "area_entity_ids": list(blackboard_snapshot.get("indexes", {}).get("entities_by_area", {}).get(str(current_area_id), [])) if current_area_id is not None else [],
        "recent_target_entity_ids": recent_target_ids,
        "recent_decisions": recent_decisions,
    }


def _compact_history_row(row: dict) -> dict:
    decision = dict(row.get("decision", {}))
    outcome = dict(row.get("outcome", {}))
    metadata = dict(decision.get("metadata", {})) if isinstance(decision.get("metadata"), dict) else {}
    selected = dict(metadata.get("selected_candidate", {})) if isinstance(metadata.get("selected_candidate"), dict) else {}
    selected_action = dict(decision.get("selected_action", {})) if isinstance(decision.get("selected_action"), dict) else {}
    outcome_row = dict(outcome.get("outcome", {})) if isinstance(outcome.get("outcome"), dict) else {}
    return {
        "round_id": decision.get("round_id"),
        "pass_id": decision.get("pass_id"),
        "candidate_class": row.get("candidate_class") or selected.get("candidate_class") or selected_action.get("candidate_class"),
        "candidate_id": selected.get("candidate_id") or decision.get("selected_candidate_id"),
        "target_entity_id": row.get("target_entity_id") or selected.get("target_entity_id") or selected_action.get("target_entity_id") or selected_action.get("target"),
        "target_area_id": row.get("target_area_id") or selected.get("target_area_id") or selected_action.get("target_area_id"),
        "required_action_family": selected.get("required_action_family") or selected_action.get("required_action_family") or selected_action.get("type"),
        "termination_reason": outcome.get("termination_reason") or outcome_row.get("termination_reason"),
        "success": bool(outcome.get("success") or outcome_row.get("success")),
        "progress": float(outcome_row.get("progress", 0.0) or 0.0),
    }


def _tactical_context_view(plan_memory: list[dict], *, current_area_id: str | None) -> dict:
    recent_local_outcomes: dict[str, dict] = {}
    repeat_pattern_state = {
        "by_candidate_area": {},
        "by_target_area": {},
        "by_route_signature": {},
        "by_trigger_zone": {},
    }
    recovery_state = {"attempts": 0, "successes": 0, "failures": 0}
    route_pattern_state: dict[str, dict] = {}
    candidate_history_state: dict[str, dict] = {}
    for row in plan_memory:
        decision = dict(row.get("decision", {}))
        outcome = dict(row.get("outcome", {}))
        selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision.get("metadata"), dict) else {}
        area_id = str(selected.get("target_area_id") or current_area_id or "global")
        target_entity_id = str(selected.get("target_entity_id") or "none")
        candidate_id = str(selected.get("candidate_id") or decision.get("selected_candidate_id") or "none")
        route_signature = str(selected.get("route_signature") or "none")
        trigger_zone_id = str(selected.get("trigger_zone_id") or "none")
        success = bool(outcome.get("success"))
        objective_type = str(selected.get("objective_type") or "unknown")

        local_key = f"{area_id}:{target_entity_id}"
        local_row = recent_local_outcomes.setdefault(local_key, {"successes": 0, "failures": 0, "candidate_ids": []})
        local_row["successes" if success else "failures"] += 1
        local_row["candidate_ids"].append(candidate_id)

        for key_name, key_value in (
            ("by_candidate_area", f"{area_id}:{selected.get('candidate_class') or 'unknown'}"),
            ("by_target_area", f"{area_id}:{target_entity_id}"),
            ("by_route_signature", route_signature),
            ("by_trigger_zone", trigger_zone_id),
        ):
            bucket = repeat_pattern_state[key_name].setdefault(str(key_value), {"attempts": 0, "failures": 0, "successes": 0})
            bucket["attempts"] += 1
            bucket["successes" if success else "failures"] += 1

        candidate_state = candidate_history_state.setdefault(candidate_id, {"attempts": 0, "successes": 0, "failures": 0, "objective_type": objective_type})
        candidate_state["attempts"] += 1
        candidate_state["successes" if success else "failures"] += 1

        route_state = route_pattern_state.setdefault(route_signature, {"attempts": 0, "successes": 0, "failures": 0})
        route_state["attempts"] += 1
        route_state["successes" if success else "failures"] += 1

        if objective_type == "recover":
            recovery_state["attempts"] += 1
            recovery_state["successes" if success else "failures"] += 1

    return {
        "recent_local_outcomes": recent_local_outcomes,
        "repeat_pattern_state": repeat_pattern_state,
        "recovery_state": recovery_state,
        "route_pattern_state": route_pattern_state,
        "candidate_history_state": candidate_history_state,
    }


def _reachable_bucket(row: dict) -> str:
    if bool(row.get("reachable_now")):
        return "directly_reachable_now"
    if bool(row.get("reachable_later")) and float(row.get("distance_score", 0.0)) >= 0.35:
        return "reachable_with_route"
    return "reachable_high_risk"


def _frontier_ranked_rows(frontier: list[dict]) -> list[dict]:
    ranked = []
    for row in frontier:
        candidate = dict(row)
        novelty = float(candidate.get("novelty", 0.0))
        route_cost = max(0.0, 1.0 - float(candidate.get("distance_score", 0.0)))
        candidate["frontier_type"] = str(candidate.get("frontier_type") or ("adjacent_frontier" if candidate.get("reachable_now") else "route_frontier"))
        candidate["novelty"] = novelty
        candidate["expected_information_gain"] = min(1.0, novelty + (0.35 * float(candidate.get("motion_score", 0.0))) + (0.2 * float(candidate.get("utility", 0.0))))
        candidate["route_cost"] = route_cost
        ranked.append(candidate)
    ranked.sort(key=lambda item: (-float(item.get("expected_information_gain", 0.0)), -float(item.get("novelty", 0.0)), float(item.get("route_cost", 0.0)), str(item.get("entity_id", ""))))
    return ranked


def _suppress_redundant_poi_siblings(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in list(rows or []):
        grouped.setdefault(str(row.get("parent_poi_id") or row.get("entity_id") or ""), []).append(row)
    selected: list[dict] = []
    for siblings in grouped.values():
        children = [row for row in siblings if int(row.get("poi_hierarchy_level", 0) or 0) > 0]
        if children:
            children.sort(
                key=lambda row: (
                    not bool(row.get("reachable_now")),
                    not bool(row.get("reachable_later")),
                    -float(row.get("interact_effect_score", 0.0) or row.get("candidate_effect_score", 0.0) or 0.0),
                    -float(row.get("utility", 0.0) or 0.0),
                    -float(row.get("confidence", 0.0) or 0.0),
                    str(row.get("entity_id", "")),
                )
            )
            selected.append(children[0])
            continue
        siblings.sort(key=lambda row: (-float(row.get("utility", 0.0) or 0.0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("entity_id", ""))))
        if siblings:
            selected.append(siblings[0])
    return selected


def _detector_backed_poi(row: dict) -> bool:
    provenance = set(str(value) for value in list(row.get("poi_source_provenance", []) or []))
    return str(row.get("kind") or "") == "poi" and "detector" in provenance and bool(row.get("planner_targetable", True))


def _poi_persistence(row: dict) -> float:
    return float(row.get("canonical_track_persistence", row.get("persistence", 0.0)) or 0.0)


def _strong_detector_backed_poi(row: dict) -> bool:
    return _detector_backed_poi(row) and _poi_persistence(row) >= 0.65 and float(row.get("confidence", 0.0) or 0.0) >= 0.5


def _poi_seed_score(row: dict) -> float:
    provenance_bonus = 0.2 if _detector_backed_poi(row) else 0.0
    reachability_bonus = 0.25 if bool(row.get("reachable_now")) else 0.15 if bool(row.get("reachable_later")) else 0.12 if str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"} else 0.0
    locality_bonus = 0.12 if bool(row.get("returned_by_area_local_pois")) else 0.0
    return float(row.get("utility", 0.0) or 0.0) + float(row.get("confidence", 0.0) or 0.0) + min(0.4, _poi_persistence(row)) + provenance_bonus + reachability_bonus + locality_bonus


def _annotate_poi_debug(row: dict, *, returned_by_area_local_pois: bool, current_area_id: str | None) -> dict:
    payload = dict(row)
    payload["returned_by_area_local_pois"] = bool(returned_by_area_local_pois)
    payload["reachable_status"] = str(payload.get("reachable_status") or ("reachable_now" if payload.get("reachable_now") else "reachable_later" if payload.get("reachable_later") else "blocked"))
    payload["approachable_status"] = str(payload.get("approachable_status") or "not_approachable")
    payload["score_if_considered"] = _poi_seed_score(payload)
    payload["displaced_by_frontier_entity_id"] = payload.get("displaced_by_frontier_entity_id")
    payload["rejected_from_promising_reason"] = payload.get("rejected_from_promising_reason")
    payload["mode_gate_reason"] = payload.get("mode_gate_reason")
    payload["poi_debug_local_match"] = bool(returned_by_area_local_pois) or str(payload.get("area_id") or "") == str(current_area_id or "")
    return payload


def _durable_prior_view(durable_priors: dict, *, reachable: list[dict], local_pois: list[dict], trigger_support: dict[str, list[dict]], current_area_id: str | None) -> dict:
    candidate_outcomes = dict(durable_priors.get("candidate_outcomes", {}))
    poi_patterns = dict(durable_priors.get("poi_patterns", {}))
    trigger_patterns = dict(durable_priors.get("trigger_patterns", {}))
    recovery_patterns = dict(durable_priors.get("recovery_patterns", {}))
    consequence_patterns = dict(durable_priors.get("consequence_patterns", {}))
    per_target = {}
    for row in reachable:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            key = normalized_target_key(entity_id, str(row.get("area_id")) if row.get("area_id") is not None else current_area_id, target_class=str(row.get("kind") or "entity"))
            per_target[key] = {
                "candidate_outcomes": dict(candidate_outcomes.get(str(row.get("candidate_class") or "target"), {})),
                "poi_pattern": dict(poi_patterns.get(str(row.get("signature") or entity_id), {})),
            }
    per_poi_class = {}
    for poi in local_pois:
        poi_class = str(poi.get("kind") or poi.get("poi_class") or "unknown")
        entry = per_poi_class.setdefault(poi_class, {"count": 0, "prior": {}})
        entry["count"] += 1
        if not entry["prior"]:
            entry["prior"] = dict(poi_patterns.get(str(poi.get("signature") or poi_class), {}))
    per_trigger_type = {}
    for entity_id, rows in trigger_support.items():
        trigger_key = normalized_trigger_zone_key(entity_id=entity_id, area_id=current_area_id)
        per_trigger_type[trigger_key] = dict(trigger_patterns.get(entity_id, {}))
        per_trigger_type[trigger_key]["support_count"] = len(rows)
    return {
        "version": str(durable_priors.get("durable_prior_version") or "durable:none"),
        "precedence": "advisory_only",
        "per_target": per_target,
        "per_poi_class": per_poi_class,
        "per_trigger_type": per_trigger_type,
        "per_route_pattern": {str(key): dict(value) for key, value in recovery_patterns.items()},
        "candidate_outcomes": candidate_outcomes,
        "poi_patterns": poi_patterns,
        "trigger_patterns": trigger_patterns,
        "recovery_patterns": recovery_patterns,
        "consequence_patterns": consequence_patterns,
    }


def build_belief(blackboard_snapshot: dict, memory_snapshot: dict, mechanic_graph_snapshot: dict | None = None) -> dict:
    working_memory = dict(memory_snapshot.get("working_memory", memory_snapshot))
    durable_priors = dict(memory_snapshot.get("durable_priors", {}))
    subgoal_chain_state = dict(memory_snapshot.get("subgoal_chain_state", {}) or {})
    indexes = dict(blackboard_snapshot.get("indexes", {}))
    topology_nodes = dict(blackboard_snapshot.get("topology_nodes", {}))
    topology_edges = dict(blackboard_snapshot.get("topology_edges", {}))
    retries = dict(working_memory.get("retries", {}))
    cooldowns = dict(working_memory.get("cooldowns", {}))
    exhaustion_map = dict(working_memory.get("exhaustion_map", {}))
    exhausted = set(working_memory.get("exhausted", []))
    exhausted_keys = set(exhausted)
    for rows in exhaustion_map.values():
        for key in list(rows or []):
            if key is not None:
                exhausted_keys.add(str(key))
    raw_plan_memory = working_memory.get("plan_memory", {})
    plan_memory = list(raw_plan_memory.get("history", [])) if isinstance(raw_plan_memory, dict) else list(raw_plan_memory)
    poi_followthrough = dict(raw_plan_memory.get("poi_followthrough", {})) if isinstance(raw_plan_memory, dict) else {}
    compact_plan_memory = [_compact_history_row(row) for row in plan_memory]
    avatar_mode_counts: dict[str, int] = {}
    avatar_status_counts: dict[str, int] = {}
    for row in plan_memory[-6:]:
        outcome = dict(row.get("outcome", {}))
        outcome_payload = dict(outcome.get("outcome", {})) if isinstance(outcome.get("outcome"), dict) else {}
        mode_status = str(outcome_payload.get("avatar_mode_status") or "unknown")
        status = str(outcome_payload.get("avatar_status") or "unknown")
        avatar_mode_counts[mode_status] = avatar_mode_counts.get(mode_status, 0) + 1
        avatar_status_counts[status] = avatar_status_counts.get(status, 0) + 1

    current_area_id = None
    if compact_plan_memory:
        last_decision = dict(plan_memory[-1].get("decision", {}))
        selected_candidate = dict(last_decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(last_decision.get("metadata"), dict) else {}
        current_area_id = selected_candidate.get("target_area_id")
    if current_area_id is None and indexes.get("entities_by_area"):
        current_area_id = next(iter(indexes["entities_by_area"]))

    versions = {
        "blackboard_version": str(blackboard_snapshot.get("blackboard_version", "bb:unknown")),
        "memory_version": str(memory_snapshot.get("memory_version", "mem:unknown")),
        "durable_prior_version": str(memory_snapshot.get("durable_checkpoint_id") or durable_priors.get("durable_prior_version") or "durable:none"),
    }

    reachable = reachable_targets(blackboard_snapshot)
    frontier = _frontier_ranked_rows(frontier_candidates(blackboard_snapshot))
    blocked = unreachable_targets(blackboard_snapshot)
    local_pois = area_local_pois(blackboard_snapshot, current_area_id) if current_area_id is not None else []
    local_poi_ids = {str(row.get("entity_id") or "") for row in local_pois}

    consequence_support: dict[str, list[dict]] = {}
    for consequence in blackboard_snapshot.get("consequences", {}).values():
        action_text = normalized_consequence_action_key(consequence)
        consequence_support.setdefault(action_text, []).append(consequence)
    trigger_support: dict[str, list[dict]] = {}
    for trigger in blackboard_snapshot.get("trigger_zones", {}).values():
        entity_id = trigger.get("entity_id")
        if entity_id:
            trigger_support.setdefault(str(entity_id), []).append(trigger)
    mechanic_graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    graph_supported_exits = [dict(row) for row in find_nodes_by_kind(mechanic_graph_state, "exit").nodes]
    graph_supported_triggers = [dict(row) for row in find_nodes_by_kind(mechanic_graph_state, "trigger").nodes]
    graph_match_relations = [dict(row) for row in find_edges_by_kind(mechanic_graph_state, "matches").edges]
    graph_paths_to_exit = [path.__dict__ for path in best_supported_paths_to_exit(mechanic_graph_state).paths]
    graph_nodes_by_id = dict(mechanic_graph_state.get("nodes_by_id", {}))

    reachable_split = {"directly_reachable_now": [], "reachable_with_route": [], "reachable_high_risk": []}
    for row in reachable:
        enriched = dict(row)
        area_id = str(enriched.get("area_id") or current_area_id or "none")
        entity_id = str(enriched.get("entity_id") or "")
        enriched["target_key"] = normalized_target_key(entity_id, area_id, target_class=str(enriched.get("kind") or "entity"))
        enriched["freshness"] = dict(versions)
        enriched["contradiction_markers"] = {
            "stale_target": str(enriched.get("lifecycle_state") or "") == "stale",
            "stale_trigger_support": bool(entity_id) and not bool(trigger_support.get(entity_id)),
            "topology_invalidation": not bool(enriched.get("reachable_now") or enriched.get("reachable_later")),
            "evidence_decay": int(enriched.get("observations", 0) or 0) <= 1,
        }
        enriched["provenance"] = _row_provenance(source_section="world_view.reachable_targets", refs=list(enriched.get("evidence_refs", [])), derived=False)
        reachable_split[_reachable_bucket(enriched)].append(enriched)

    blocked_rows = []
    for row in blocked:
        enriched = dict(row)
        area_id = str(enriched.get("area_id") or current_area_id or "none")
        entity_id = str(enriched.get("entity_id") or "")
        enriched["target_key"] = normalized_target_key(entity_id, area_id, target_class=str(enriched.get("kind") or "entity"))
        enriched["freshness"] = dict(versions)
        enriched["contradiction_markers"] = {
            "stale_target": str(enriched.get("lifecycle_state") or "") == "stale",
            "stale_trigger_support": bool(entity_id) and not bool(trigger_support.get(entity_id)),
            "topology_invalidation": True,
            "evidence_decay": int(enriched.get("observations", 0) or 0) <= 1,
        }
        enriched["provenance"] = _row_provenance(source_section="world_view.blocked_targets", refs=list(enriched.get("evidence_refs", [])), derived=True)
        blocked_rows.append(enriched)

    world_view = {
        "reachable_targets": list(reachable_split["directly_reachable_now"]) + list(reachable_split["reachable_with_route"]) + list(reachable_split["reachable_high_risk"]),
        "reachable_targets_split": reachable_split,
        "frontier_targets": frontier,
        "blocked_targets": blocked_rows,
        "local_pois": local_pois,
        "topology": {
            "nodes": topology_nodes,
            "edges": topology_edges,
            "reachable_node_count": len([node for node in topology_nodes.values() if node.get("status") != "blocked"]),
        },
        "versions": dict(versions),
        "precedence": "authoritative",
    }

    local_context_view = _local_context_view(current_area_id=current_area_id, blackboard_snapshot=blackboard_snapshot, plan_memory=compact_plan_memory)
    tactical_context = _tactical_context_view(plan_memory, current_area_id=current_area_id)
    tactical_memory_view = {
        "cooldowns": cooldowns,
        "exhausted": exhausted,
        "exhaustion_map": exhaustion_map,
        "exhausted_keys": exhausted_keys,
        "retries": retries,
        "failed_candidates": {str(k): int(v) for k, v in working_memory.get("failed_candidates", {}).items()} if isinstance(working_memory.get("failed_candidates"), dict) else {},
        "tactical_context": tactical_context,
        "precedence": "overrides_durable_priors",
    }

    support_view = {
        "consequence_support": consequence_support,
        "trigger_support": trigger_support,
        "indexes": indexes,
        "precedence": "derived_from_blackboard",
    }
    mechanic_graph_view = {
        "version": (mechanic_graph_snapshot or {}).get("mechanic_graph_version"),
        "supported_exits": graph_supported_exits,
        "supported_prerequisites": sum(1 for path in graph_paths_to_exit if len(list(path.get("node_ids", []))) > 2),
        "trigger_candidates": graph_supported_triggers,
        "contradiction_count": sum(int(row.get("contradiction_count", 0) or 0) for row in dict(mechanic_graph_state.get("edges_by_id", {})).values()),
        "paths_to_exit": graph_paths_to_exit,
        "match_relations": graph_match_relations,
    }
    durable_prior_view = _durable_prior_view(durable_priors, reachable=reachable, local_pois=local_pois, trigger_support=trigger_support, current_area_id=str(current_area_id) if current_area_id is not None else None)
    active_chain_summary = dict(subgoal_chain_state.get("active_chain", {}) or {})
    active_step_summary = dict(subgoal_chain_state.get("active_step", {}) or {})
    contradicted_paths = [path for path in graph_paths_to_exit if int(path.get("contradiction_count", 0) or 0) > 0]
    strongest_executable_paths = sorted(
        [dict(path) for path in graph_paths_to_exit],
        key=lambda row: (
            -float(row.get("execution_feasibility_score", row.get("support_strength", 0.0)) or 0.0),
            -float(row.get("counterfactual_strength", 0.0) or 0.0),
            int(row.get("contradiction_count", 0) or 0),
        ),
    )[:8]
    pending_verification_nodes = list(dict.fromkeys(
        str(value)
        for value in list(active_step_summary.get("verification_points", []) or [])
        + [str(node_id) for path in strongest_executable_paths for node_id in list(path.get("verification_nodes", []) or [])]
        if value
    ))
    registry_snapshot = dict(memory_snapshot.get("hypothesis_registry_snapshot", {}) or {})
    planner_usable_hypothesis_summary = {
        "planner_usable_count": sum(1 for state in dict(registry_snapshot.get("planner_usable_state", {})).values() if str(state or "") == "planner_usable"),
        "durable_ready_count": sum(1 for state in dict(registry_snapshot.get("durable_ready_state", {})).values() if str(state or "") == "durable_ready"),
        "validated_count": sum(1 for state in dict(registry_snapshot.get("validation_state", {})).values() if str(state or "") == "validated"),
    }

    all_targetable_pois = []
    for row in blackboard_snapshot.get("entities", {}).values():
        if str(row.get("kind") or "") != "poi":
            continue
        if not bool(row.get("planner_visible", True)) or not bool(row.get("planner_targetable", True)):
            continue
        entity_id = str(row.get("entity_id") or "")
        if not entity_id:
            continue
        all_targetable_pois.append(_annotate_poi_debug(row, returned_by_area_local_pois=entity_id in local_poi_ids, current_area_id=current_area_id))
    local_pois = [_annotate_poi_debug(row, returned_by_area_local_pois=True, current_area_id=current_area_id) for row in local_pois]
    reachable_pois = []
    approachable_pois = []
    detector_backed_local_pois = []
    detector_backed_global_fallback = []
    for row in all_targetable_pois:
        payload = dict(row)
        if bool(payload.get("reachable_now")) or bool(payload.get("reachable_later")):
            payload["mode_gate_reason"] = "reachable_poi"
            reachable_pois.append(payload)
        elif (
            _strong_detector_backed_poi(payload)
            and (
                str(payload.get("approachable_status") or "") in {"approachable_now", "approachable_later"}
                or bool(payload.get("returned_by_area_local_pois"))
                or bool(dict(payload.get("access_profile", {}) or {}).get("frontier_adjacent"))
            )
        ):
            payload["mode_gate_reason"] = "approachable_detector_backed"
            if str(payload.get("approachable_status") or "") not in {"approachable_now", "approachable_later"}:
                payload["approachable_status"] = "approachable_now" if bool(payload.get("returned_by_area_local_pois")) else "approachable_later"
            approachable_pois.append(payload)
        elif _strong_detector_backed_poi(payload) and bool(payload.get("returned_by_area_local_pois")):
            payload["mode_gate_reason"] = "strong_local_detector_backed"
            detector_backed_local_pois.append(payload)
        elif _strong_detector_backed_poi(payload):
            payload["mode_gate_reason"] = "global_detector_backed_fallback"
            detector_backed_global_fallback.append(payload)
        else:
            if not _detector_backed_poi(payload):
                payload["rejected_from_promising_reason"] = "not_detector_backed"
            elif _poi_persistence(payload) < 0.65:
                payload["rejected_from_promising_reason"] = "low_canonical_persistence"
            elif not bool(payload.get("returned_by_area_local_pois")):
                payload["rejected_from_promising_reason"] = "not_local_or_nearby"
            else:
                payload["rejected_from_promising_reason"] = "soft_reachability_missing"
    promising_ordered = list(reachable_pois) + list(approachable_pois) + list(detector_backed_local_pois)
    if not promising_ordered:
        promising_ordered.extend(detector_backed_global_fallback)
    promising_pois = sorted(
        _suppress_redundant_poi_siblings(promising_ordered),
        key=lambda row: (
            not _detector_backed_poi(row),
            not bool(row.get("reachable_now")),
            not bool(row.get("reachable_later")),
            not str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"},
            -float(row.get("score_if_considered", 0.0) or 0.0),
            str(row.get("entity_id", "")),
        ),
    )[:12]
    trigger_candidates = sorted([row for row in reachable if str(row.get("entity_id")) in trigger_support], key=lambda row: (-len(trigger_support.get(str(row.get("entity_id")), [])), -float(row.get("utility", 0.0)), row.get("entity_id", "")))
    recovery_candidates = sorted([row for row in blocked_rows if bool(row.get("reachable_later")) or float(row.get("distance_score", 0.0)) > 0.0], key=lambda row: (-float(row.get("distance_score", 0.0)), -float(row.get("motion_score", 0.0)), row.get("entity_id", "")))
    structure_candidate_count = sum(
        1 for row in list(local_pois) + list(promising_pois)
        if str(row.get("poi_class") or row.get("kind") or "") == "structure"
    )
    mechanic_object_backed_node_count = sum(
        1
        for row in graph_nodes_by_id.values()
        if bool(dict(row).get("object_backed", False))
        and (
            str(dict(row).get("node_kind") or "") in {"trigger", "panel", "gate", "exit"}
            or bool(dict(dict(row).get("metadata", {}) or {}).get("pattern_id"))
        )
    )
    object_backed_node_count = sum(1 for row in graph_nodes_by_id.values() if bool(dict(row).get("object_backed", False)))
    region_backed_trigger_count = sum(1 for row in graph_nodes_by_id.values() if str(dict(row).get("node_kind") or "") == "trigger" and bool(dict(row).get("synthetic_region_only", False)))
    chainworthy_path_count = sum(
        1
        for row in strongest_executable_paths
        if len(list(row.get("node_ids", []) or [])) >= 3
        and float(row.get("execution_feasibility_score", 0.0) or 0.0) >= 0.5
        and float(row.get("support_strength", 0.0) or 0.0) >= 0.45
    )
    structure_recall_gap = max(0.0, 1.0 - min(1.0, (0.2 * structure_candidate_count) + (0.35 * mechanic_object_backed_node_count) + (0.25 * chainworthy_path_count)))
    stable_targetable_detector_pois = [
        row for row in all_targetable_pois
        if _strong_detector_backed_poi(row)
        and (
            int(row.get("last_seen_round", row.get("round_id", 0)) or 0) - int(row.get("first_seen_round", row.get("round_id", 0)) or 0) >= 1
            or int(row.get("observations", 0) or 0) >= 20
            or _poi_persistence(row) >= 0.9
        )
    ]
    detector_poi_progress_ready = bool(stable_targetable_detector_pois) and any(
        bool(row.get("reachable_now")) or bool(row.get("reachable_later")) or str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"} or bool(row.get("returned_by_area_local_pois"))
        for row in stable_targetable_detector_pois
    )
    escalation_ready = any(
        isinstance(row, dict)
        and bool(row.get("detector_backed", False))
        and int(row.get("selection_rounds", 0) or 0) >= 2
        and (
            int(row.get("new_graph_edges", 0) or 0) > 0
            or int(row.get("new_hypothesis_support", 0) or 0) > 0
            or int(row.get("new_verification_candidates", 0) or 0) > 0
            or int(row.get("changed_exit_linked_evidence", 0) or 0) > 0
            or bool(row.get("approachable", False))
            or bool(row.get("reachable", False))
        )
        for row in poi_followthrough.values()
    )
    planning_mode = "default_progress" if (detector_poi_progress_ready or escalation_ready) else ("structure_acquisition" if mechanic_object_backed_node_count <= 12 or structure_candidate_count <= 4 or chainworthy_path_count <= 1 else "default_progress")
    candidate_seed_sets = {
        "promising_pois": promising_pois,
        "approachable_pois": approachable_pois,
        "trigger_candidates": trigger_candidates,
        "recovery_candidates": recovery_candidates,
        "frontier_targets": frontier,
        "blocked_targets": blocked_rows,
        "reachable_targets": world_view["reachable_targets"],
        "structure_candidates": [row for row in promising_pois if str(row.get("poi_class") or row.get("kind") or "") == "structure"],
    }

    available_action_families = {"move"}
    targets_for_actions = reachable + local_pois
    if any(float(target.get("interact_effect_score", 0.0)) > 0.0 or int(target.get("interact_attempts", 0) or 0) > 0 for target in targets_for_actions):
        available_action_families.add("interact")
    if any(float(target.get("click_effect_score", 0.0)) > 0.0 or int(target.get("click_attempts", 0) or 0) > 0 for target in targets_for_actions):
        available_action_families.add("click_at")
    control_mode = "unknown"
    directional_consequence_count = sum(
        len(rows)
        for action_key, rows in consequence_support.items()
        if str(action_key or "").lower() in {"up", "down", "left", "right", "action1", "action2", "action3", "action4"}
    )
    move_consequence_count = sum(
        1
        for consequence in blackboard_snapshot.get("consequences", {}).values()
        if str(consequence.get("action_family") or consequence.get("action_key") or "").lower() == "move"
        or str(consequence.get("action_name") or "").lower() in {"up", "down", "left", "right"}
    )
    movement_capability_evidence = max(directional_consequence_count, move_consequence_count)
    if avatar_mode_counts.get("movement_avatar", 0) > 0 or movement_capability_evidence >= 2:
        control_mode = "movement_avatar"
    elif avatar_mode_counts.get("cursor_or_click", 0) > 0:
        control_mode = "cursor_or_click"
    elif avatar_mode_counts.get("global_action_only", 0) > 0:
        control_mode = "global_action_only"
    avatar_runtime_status = "present" if avatar_status_counts.get("present", 0) > 0 or movement_capability_evidence >= 2 else "absent" if avatar_status_counts.get("absent", 0) > 0 else "unknown"

    return {
        "world_view": world_view,
        "tactical_memory_view": tactical_memory_view,
        "durable_prior_view": durable_prior_view,
        "candidate_seed_sets": candidate_seed_sets,
        "local_context_view": local_context_view,
        "support_view": support_view,
        "mechanic_graph_view": mechanic_graph_view,
        "planning_mode": planning_mode,
        "structure_recall_gap": structure_recall_gap,
        "object_backed_node_count": object_backed_node_count,
        "mechanic_object_backed_node_count": mechanic_object_backed_node_count,
        "region_backed_trigger_count": region_backed_trigger_count,
        "structure_candidate_count": structure_candidate_count,
        "active_chain_summary": active_chain_summary,
        "chain_progress_summary": {
            "active_chain_id": active_chain_summary.get("chain_id"),
            "status": active_chain_summary.get("status"),
            "current_step_index": active_chain_summary.get("current_step_index"),
            "current_step": active_step_summary,
            "should_replan": bool(subgoal_chain_state.get("should_replan", False)),
        },
        "strongest_executable_paths": strongest_executable_paths,
        "contradicted_paths": contradicted_paths,
        "pending_verification_nodes": pending_verification_nodes,
        "planner_usable_hypothesis_summary": planner_usable_hypothesis_summary,
        "detector_poi_followthrough": poi_followthrough,
        "reachable_targets": world_view["reachable_targets"],
        "reachable_targets_split": world_view["reachable_targets_split"],
        "frontier_targets": world_view["frontier_targets"],
        "blocked_targets": world_view["blocked_targets"],
        "promising_pois": candidate_seed_sets["promising_pois"],
        "approachable_pois": candidate_seed_sets["approachable_pois"],
        "trigger_candidates": candidate_seed_sets["trigger_candidates"],
        "recovery_candidates": candidate_seed_sets["recovery_candidates"],
        "local_context": local_context_view,
        "prior_failure_success_context": tactical_memory_view["tactical_context"],
        "topology": world_view["topology"],
        "current_area_id": current_area_id,
        "available_action_families": sorted(available_action_families),
        "control_mode": control_mode,
        "avatar_runtime_status": avatar_runtime_status,
        "versions": versions,
        "precedence": {
            "blackboard_facts": 0,
            "tactical_memory": 1,
            "durable_priors": 2,
        },
    }
