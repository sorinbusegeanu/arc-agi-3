from __future__ import annotations


def compute_route_features(blackboard_state: dict, candidates: list[dict]) -> dict[str, dict]:
    topology_nodes = dict(blackboard_state.get("topology_nodes", {}))
    topology_edges = dict(blackboard_state.get("topology_edges", {}))
    reachable_cells = {tuple(node.get("cell", ())) for node in topology_nodes.values() if node.get("status") != "blocked"}
    edge_risk = {
        edge_id: (
            float(edge.get("blocked_count", 0)) + (0.5 * float(edge.get("uncertain_count", 0)))
        ) / float(max(1, float(edge.get("success_count", 0)) + float(edge.get("blocked_count", 0)) + float(edge.get("uncertain_count", 0))))
        for edge_id, edge in topology_edges.items()
    }
    mean_risk = sum(edge_risk.values()) / float(max(1, len(edge_risk)))

    features: dict[str, dict] = {}
    for candidate in candidates:
        centroid = candidate.get("action", {}).get("centroid")
        target_cell = None
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            target_cell = (int(float(centroid[0])), int(float(centroid[1])))
        reachable_now = target_cell in reachable_cells if target_cell is not None else bool(candidate.get("reachable_now"))
        cost = 0.0 if reachable_now else (1.0 if candidate.get("reachable_later") else 2.5)
        progress_potential = 0.8 if reachable_now else (0.45 if candidate.get("reachable_later") else 0.1)
        risk = mean_risk + (0.35 if candidate.get("candidate_class") == "recovery_move" else 0.0)
        if candidate.get("candidate_class") == "trigger_probe":
            progress_potential += 0.15
        features[candidate["candidate_id"]] = {
            "reachable_now": reachable_now,
            "cost": cost,
            "risk": risk,
            "progress_potential": progress_potential,
            "target_cell": list(target_cell) if target_cell is not None else None,
        }
    return features
