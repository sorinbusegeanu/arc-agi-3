from __future__ import annotations

from v3_1.world.consequences import normalized_consequence_action_key
from v3_1.world.queries import area_local_pois, frontier_candidates, reachable_targets, unreachable_targets


def normalized_target_key(entity_id: str | None, area_id: str | None = None, *, target_class: str | None = None) -> str:
    return f"target:{target_class or 'unknown'}:{area_id or 'none'}:{entity_id or 'none'}"


def normalized_route_signature(*, candidate_class: str, area_id: str | None, target_entity_id: str | None = None, centroid: list | tuple | None = None, action_hint: str | None = None) -> str:
    if action_hint:
        return f"route:{candidate_class}:{area_id or 'none'}:{action_hint}"
    if centroid and len(centroid) >= 2:
        return f"route:{candidate_class}:{area_id or 'none'}:{int(centroid[0])}:{int(centroid[1])}:{target_entity_id or 'none'}"
    return f"route:{candidate_class}:{area_id or 'none'}:{target_entity_id or 'none'}"


def normalized_trigger_zone_key(*, trigger_zone_id: str | None = None, entity_id: str | None = None, area_id: str | None = None) -> str:
    return f"trigger:{trigger_zone_id or 'none'}:{area_id or 'none'}:{entity_id or 'none'}"


def _row_provenance(*, source_section: str, refs: list[str] | None = None, derived: bool = False) -> dict:
    return {
        "source_section": source_section,
        "supporting_refs": list(dict.fromkeys(str(ref) for ref in list(refs or []) if ref)),
        "derived": bool(derived),
    }


def _local_context(*, current_area_id, blackboard_snapshot: dict, plan_memory: list[dict]) -> dict:
    area_entities = list(blackboard_snapshot.get("indexes", {}).get("entities_by_area", {}).get(str(current_area_id), [])) if current_area_id is not None else []
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
        "area_entity_ids": area_entities,
        "recent_target_entity_ids": recent_target_ids,
        "recent_decisions": recent_decisions,
    }


def _localized_failure_success_context(plan_memory: list[dict], *, current_area_id: str | None) -> dict:
    by_area: dict[str, dict] = {}
    by_zone: dict[str, dict] = {}
    for row in plan_memory:
        decision = dict(row.get("decision", {}))
        outcome = dict(row.get("outcome", {}))
        selected = dict(decision.get("selected_action", {})) if isinstance(decision.get("selected_action"), dict) else {}
        area_id = str(selected.get("target_area_id") or current_area_id or "global")
        zone_key = f"{area_id}:{selected.get('target_entity_id') or selected.get('target') or decision.get('selected_candidate_id') or 'none'}"
        success = bool(outcome.get("success"))
        area_row = by_area.setdefault(area_id, {"successes": 0, "failures": 0, "candidate_ids": []})
        zone_row = by_zone.setdefault(zone_key, {"successes": 0, "failures": 0, "candidate_ids": []})
        area_row["successes" if success else "failures"] += 1
        zone_row["successes" if success else "failures"] += 1
        candidate_id = decision.get("selected_candidate_id")
        if candidate_id:
            area_row["candidate_ids"].append(str(candidate_id))
            zone_row["candidate_ids"].append(str(candidate_id))
    return {"by_area": by_area, "by_zone": by_zone}


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
        candidate["frontier_novelty"] = novelty
        candidate["expected_information_gain"] = min(1.0, novelty + (0.35 * float(candidate.get("motion_score", 0.0))) + (0.2 * float(candidate.get("utility", 0.0))))
        candidate["frontier_route_cost"] = route_cost
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            -float(item.get("expected_information_gain", 0.0)),
            -float(item.get("frontier_novelty", 0.0)),
            float(item.get("frontier_route_cost", 0.0)),
            str(item.get("entity_id", "")),
        )
    )
    return ranked


def _durable_prior_merge_layer(durable_priors: dict, *, reachable: list[dict], local_pois: list[dict], trigger_support: dict[str, list[dict]], current_area_id: str | None) -> dict:
    persistent_candidate_outcomes = dict(durable_priors.get("candidate_outcomes", {}))
    persistent_poi_patterns = dict(durable_priors.get("poi_patterns", {}))
    persistent_trigger_patterns = dict(durable_priors.get("trigger_patterns", {}))
    persistent_recovery_patterns = dict(durable_priors.get("recovery_patterns", {}))
    per_target: dict[str, dict] = {}
    for row in reachable:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            per_target[entity_id] = {
                "candidate_outcomes": dict(persistent_candidate_outcomes.get(str(row.get("candidate_class") or "target"), {})),
                "poi_pattern": dict(persistent_poi_patterns.get(str(row.get("signature") or entity_id), {})),
            }
    per_poi_class: dict[str, dict] = {}
    for poi in local_pois:
        poi_class = str(poi.get("kind") or poi.get("poi_class") or "unknown")
        entry = per_poi_class.setdefault(poi_class, {"count": 0, "prior": {}})
        entry["count"] += 1
        if not entry["prior"]:
            entry["prior"] = dict(persistent_poi_patterns.get(str(poi.get("signature") or poi_class), {}))
    per_trigger_type: dict[str, dict] = {}
    for entity_id, rows in trigger_support.items():
        trigger_key = normalized_trigger_zone_key(entity_id=entity_id, area_id=current_area_id)
        per_trigger_type[trigger_key] = dict(persistent_trigger_patterns.get(entity_id, {}))
        per_trigger_type[trigger_key]["support_count"] = len(rows)
    return {
        "per_target": per_target,
        "per_poi_class": per_poi_class,
        "per_trigger_type": per_trigger_type,
        "per_route_pattern": {str(key): dict(value) for key, value in persistent_recovery_patterns.items()},
    }


def build_belief(blackboard_snapshot: dict, memory_snapshot: dict) -> dict:
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
    if isinstance(raw_plan_memory, dict):
        plan_memory = list(raw_plan_memory.get("history", []))
    else:
        plan_memory = list(raw_plan_memory)

    current_area_id = None
    if plan_memory:
        last_decision = dict(plan_memory[-1].get("decision", {}))
        selected_candidate_id = last_decision.get("selected_candidate_id")
        if selected_candidate_id and selected_candidate_id in blackboard_snapshot.get("entities", {}):
            current_area_id = blackboard_snapshot["entities"][selected_candidate_id].get("area_id")
    if current_area_id is None and indexes.get("entities_by_area"):
        current_area_id = next(iter(indexes["entities_by_area"]))

    blackboard_version = str(blackboard_snapshot.get("blackboard_version", "bb:unknown"))
    memory_version = str(memory_snapshot.get("memory_version", "mem:unknown"))
    durable_prior_version = str(memory_snapshot.get("durable_checkpoint_id") or durable_priors.get("durable_prior_version") or "durable:none")

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

    reachable_split = {
        "directly_reachable_now": [],
        "reachable_with_route": [],
        "reachable_high_risk": [],
    }
    for row in reachable:
        enriched = dict(row)
        area_id = str(enriched.get("area_id") or current_area_id or "none")
        entity_id = str(enriched.get("entity_id") or "")
        enriched["target_key"] = normalized_target_key(entity_id, area_id, target_class=str(enriched.get("kind") or "entity"))
        enriched["freshness"] = {
            "blackboard_version": blackboard_version,
            "memory_version": memory_version,
            "durable_prior_version": durable_prior_version,
        }
        enriched["contradiction_markers"] = {
            "stale_target": str(enriched.get("lifecycle_state") or "") == "stale",
            "stale_trigger_support": bool(entity_id) and not bool(trigger_support.get(entity_id)),
            "topology_invalidation": not bool(enriched.get("reachable_now") or enriched.get("reachable_later")),
            "evidence_decay": int(enriched.get("observations", 0) or 0) <= 1,
        }
        enriched["provenance"] = _row_provenance(source_section="reachable_targets", refs=list(enriched.get("evidence_refs", [])), derived=False)
        reachable_split[_reachable_bucket(enriched)].append(enriched)

    blocked_rows = []
    for row in blocked:
        enriched = dict(row)
        area_id = str(enriched.get("area_id") or current_area_id or "none")
        entity_id = str(enriched.get("entity_id") or "")
        enriched["target_key"] = normalized_target_key(entity_id, area_id, target_class=str(enriched.get("kind") or "entity"))
        enriched["freshness"] = {
            "blackboard_version": blackboard_version,
            "memory_version": memory_version,
            "durable_prior_version": durable_prior_version,
        }
        enriched["contradiction_markers"] = {
            "stale_target": str(enriched.get("lifecycle_state") or "") == "stale",
            "stale_trigger_support": bool(entity_id) and not bool(trigger_support.get(entity_id)),
            "topology_invalidation": True,
            "evidence_decay": int(enriched.get("observations", 0) or 0) <= 1,
        }
        enriched["provenance"] = _row_provenance(source_section="blocked_targets", refs=list(enriched.get("evidence_refs", [])), derived=True)
        blocked_rows.append(enriched)

    promising_pois = sorted(
        list(local_pois) + [row for row in reachable if row not in local_pois],
        key=lambda row: (-float(row.get("utility", 0.0)), -float(row.get("confidence", 0.0)), row.get("entity_id", "")),
    )[:12]
    trigger_candidates = sorted(
        [row for row in reachable if str(row.get("entity_id")) in trigger_support],
        key=lambda row: (-len(trigger_support.get(str(row.get("entity_id")), [])), -float(row.get("utility", 0.0)), row.get("entity_id", "")),
    )
    recovery_candidates = sorted(
        [row for row in blocked_rows if bool(row.get("reachable_later")) or float(row.get("distance_score", 0.0)) > 0.0],
        key=lambda row: (-float(row.get("distance_score", 0.0)), -float(row.get("motion_score", 0.0)), row.get("entity_id", "")),
    )

    failed_candidates: dict[str, int] = {}
    for row in plan_memory:
        outcome = dict(row.get("outcome", {}))
        candidate_id = outcome.get("candidate_id")
        if candidate_id and not bool(outcome.get("success")):
            failed_candidates[str(candidate_id)] = failed_candidates.get(str(candidate_id), 0) + 1
    local_context = _local_context(current_area_id=current_area_id, blackboard_snapshot=blackboard_snapshot, plan_memory=plan_memory)
    localized_context = _localized_failure_success_context(plan_memory, current_area_id=current_area_id)
    prior_failure_success_context = {
        "candidate_failures": failed_candidates,
        "by_area": dict(localized_context.get("by_area", {})),
        "by_zone": dict(localized_context.get("by_zone", {})),
        "recent_target_entity_ids": list(local_context.get("recent_target_entity_ids", [])),
    }
    durable_prior_merge = _durable_prior_merge_layer(
        durable_priors,
        reachable=reachable,
        local_pois=local_pois,
        trigger_support=trigger_support,
        current_area_id=str(current_area_id) if current_area_id is not None else None,
    )
    available_action_families = {"move"}
    if any(float(target.get("interact_effect_score", 0.0)) > 0.0 or int(target.get("interact_attempts", 0) or 0) > 0 for target in reachable + local_pois):
        available_action_families.add("interact")
    if any(float(target.get("click_effect_score", 0.0)) > 0.0 or int(target.get("click_attempts", 0) or 0) > 0 for target in reachable + local_pois):
        available_action_families.add("click_at")

    return {
        "current_area_id": current_area_id,
        "reachable_targets": list(reachable_split["directly_reachable_now"]) + list(reachable_split["reachable_with_route"]) + list(reachable_split["reachable_high_risk"]),
        "reachable_targets_split": reachable_split,
        "frontier_targets": frontier,
        "blocked_targets": blocked_rows,
        "local_pois": local_pois,
        "promising_pois": promising_pois,
        "trigger_candidates": trigger_candidates,
        "recovery_candidates": recovery_candidates,
        "local_context": local_context,
        "localized_context": localized_context,
        "prior_failure_success_context": prior_failure_success_context,
        "durable_prior_merge": durable_prior_merge,
        "cooldowns": cooldowns,
        "exhausted": sorted(exhausted),
        "exhaustion_map": exhaustion_map,
        "exhausted_keys": sorted(exhausted_keys),
        "retries": retries,
        "failed_candidates": failed_candidates,
        "topology": {
            "nodes": topology_nodes,
            "edges": topology_edges,
            "reachable_node_count": len([node for node in topology_nodes.values() if node.get("status") != "blocked"]),
        },
        "consequence_support": consequence_support,
        "trigger_support": trigger_support,
        "available_action_families": sorted(available_action_families),
        "versions": {
            "blackboard_version": blackboard_version,
            "memory_version": memory_version,
            "durable_prior_version": durable_prior_version,
        },
        "indexes": indexes,
        "plan_memory": plan_memory,
        "durable_priors": durable_priors,
        "persistent_skill_priors": dict(durable_priors.get("skill_stats", {})),
        "persistent_candidate_outcomes": dict(durable_priors.get("candidate_outcomes", {})),
        "persistent_failure_patterns": dict(durable_priors.get("failure_patterns", {})),
        "persistent_recovery_patterns": dict(durable_priors.get("recovery_patterns", {})),
        "persistent_poi_patterns": dict(durable_priors.get("poi_patterns", {})),
        "persistent_trigger_patterns": dict(durable_priors.get("trigger_patterns", {})),
        "persistent_consequence_patterns": dict(durable_priors.get("consequence_patterns", {})),
    }
