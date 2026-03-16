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
    compact_plan_memory = [_compact_history_row(row) for row in plan_memory]

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

    promising_pois = sorted(list(local_pois) + [row for row in reachable if row not in local_pois], key=lambda row: (-float(row.get("utility", 0.0)), -float(row.get("confidence", 0.0)), row.get("entity_id", "")))[:12]
    trigger_candidates = sorted([row for row in reachable if str(row.get("entity_id")) in trigger_support], key=lambda row: (-len(trigger_support.get(str(row.get("entity_id")), [])), -float(row.get("utility", 0.0)), row.get("entity_id", "")))
    recovery_candidates = sorted([row for row in blocked_rows if bool(row.get("reachable_later")) or float(row.get("distance_score", 0.0)) > 0.0], key=lambda row: (-float(row.get("distance_score", 0.0)), -float(row.get("motion_score", 0.0)), row.get("entity_id", "")))
    candidate_seed_sets = {
        "promising_pois": promising_pois,
        "trigger_candidates": trigger_candidates,
        "recovery_candidates": recovery_candidates,
        "frontier_targets": frontier,
        "blocked_targets": blocked_rows,
        "reachable_targets": world_view["reachable_targets"],
    }

    available_action_families = {"move"}
    targets_for_actions = reachable + local_pois
    if any(float(target.get("interact_effect_score", 0.0)) > 0.0 or int(target.get("interact_attempts", 0) or 0) > 0 for target in targets_for_actions):
        available_action_families.add("interact")
    if any(float(target.get("click_effect_score", 0.0)) > 0.0 or int(target.get("click_attempts", 0) or 0) > 0 for target in targets_for_actions):
        available_action_families.add("click_at")

    return {
        "world_view": world_view,
        "tactical_memory_view": tactical_memory_view,
        "durable_prior_view": durable_prior_view,
        "candidate_seed_sets": candidate_seed_sets,
        "local_context_view": local_context_view,
        "support_view": support_view,
        "mechanic_graph_view": mechanic_graph_view,
        "reachable_targets": world_view["reachable_targets"],
        "reachable_targets_split": world_view["reachable_targets_split"],
        "frontier_targets": world_view["frontier_targets"],
        "blocked_targets": world_view["blocked_targets"],
        "promising_pois": candidate_seed_sets["promising_pois"],
        "trigger_candidates": candidate_seed_sets["trigger_candidates"],
        "recovery_candidates": candidate_seed_sets["recovery_candidates"],
        "local_context": local_context_view,
        "prior_failure_success_context": tactical_memory_view["tactical_context"],
        "topology": world_view["topology"],
        "current_area_id": current_area_id,
        "available_action_families": sorted(available_action_families),
        "versions": versions,
        "precedence": {
            "blackboard_facts": 0,
            "tactical_memory": 1,
            "durable_priors": 2,
        },
    }
