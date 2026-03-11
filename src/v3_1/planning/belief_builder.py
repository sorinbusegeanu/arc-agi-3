from __future__ import annotations

from v3_1.world.queries import area_local_pois, frontier_candidates, reachable_targets, unreachable_targets


def build_belief(blackboard_snapshot: dict, memory_snapshot: dict) -> dict:
    indexes = dict(blackboard_snapshot.get("indexes", {}))
    topology_nodes = dict(blackboard_snapshot.get("topology_nodes", {}))
    topology_edges = dict(blackboard_snapshot.get("topology_edges", {}))
    retries = dict(memory_snapshot.get("retries", {}))
    cooldowns = dict(memory_snapshot.get("cooldowns", {}))
    exhausted = set(memory_snapshot.get("exhausted", []))
    raw_plan_memory = memory_snapshot.get("plan_memory", {})
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

    reachable = reachable_targets(blackboard_snapshot)
    frontier = frontier_candidates(blackboard_snapshot)
    blocked = unreachable_targets(blackboard_snapshot)
    local_pois = area_local_pois(blackboard_snapshot, current_area_id) if current_area_id is not None else []

    consequence_support: dict[str, list[dict]] = {}
    for consequence in blackboard_snapshot.get("consequences", {}).values():
        action_text = str(consequence.get("action"))
        consequence_support.setdefault(action_text, []).append(consequence)

    trigger_support: dict[str, list[dict]] = {}
    for trigger in blackboard_snapshot.get("trigger_zones", {}).values():
        entity_id = trigger.get("entity_id")
        if entity_id:
            trigger_support.setdefault(str(entity_id), []).append(trigger)

    failed_candidates: dict[str, int] = {}
    for row in plan_memory:
        outcome = dict(row.get("outcome", {}))
        candidate_id = outcome.get("candidate_id")
        if candidate_id and not bool(outcome.get("success")):
            failed_candidates[str(candidate_id)] = failed_candidates.get(str(candidate_id), 0) + 1

    return {
        "current_area_id": current_area_id,
        "reachable_targets": reachable,
        "frontier_targets": frontier,
        "blocked_targets": blocked,
        "local_pois": local_pois,
        "cooldowns": cooldowns,
        "exhausted": exhausted,
        "retries": retries,
        "failed_candidates": failed_candidates,
        "topology": {
            "nodes": topology_nodes,
            "edges": topology_edges,
            "reachable_node_count": len([node for node in topology_nodes.values() if node.get("status") != "blocked"]),
        },
        "consequence_support": consequence_support,
        "trigger_support": trigger_support,
        "indexes": indexes,
        "plan_memory": plan_memory,
    }
