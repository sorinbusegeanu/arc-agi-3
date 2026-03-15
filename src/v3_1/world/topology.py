from __future__ import annotations


def build_topology_edges(episodes: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    for episode in episodes:
        previous = None
        for step in episode.get("steps", []):
            current = step.get("avatar_cell")
            if current is None:
                previous = None
                continue
            node_id = f"cell:{current[0]}:{current[1]}"
            nodes.setdefault(
                node_id,
                {"node_id": node_id, "cell": current, "visits": 0, "blocked_visits": 0, "uncertain_visits": 0},
            )
            nodes[node_id]["visits"] += 1
            if step.get("blocked"):
                nodes[node_id]["blocked_visits"] += 1
            if step.get("uncertain"):
                nodes[node_id]["uncertain_visits"] += 1
            if previous is not None:
                transition_type = step.get("transition_type", "move")
                action_key = normalized_topology_action_key(step)
                edge_id = f"{previous}->{node_id}:{action_key}:{transition_type}"
                edges.setdefault(
                    edge_id,
                    {
                        "edge_id": edge_id,
                        "src": previous,
                        "dst": node_id,
                        "action_key": action_key,
                        "transition_type": transition_type,
                        "success_count": 0,
                        "blocked_count": 0,
                        "uncertain_count": 0,
                        "evidence_refs": [],
                    },
                )
                if step.get("blocked"):
                    edges[edge_id]["blocked_count"] += 1
                elif step.get("uncertain"):
                    edges[edge_id]["uncertain_count"] += 1
                else:
                    edges[edge_id]["success_count"] += 1
                for evidence_ref in list(step.get("evidence_refs", [])):
                    edges[edge_id]["evidence_refs"].append(str(evidence_ref))
                if step.get("evidence_ref"):
                    edges[edge_id]["evidence_refs"].append(str(step["evidence_ref"]))
            previous = node_id
    return nodes, edges


def normalized_topology_action_key(step: dict) -> str:
    action_family = str(step.get("action_family") or "").strip().lower()
    if action_family and action_family != "unknown":
        return action_family
    action_name = str(step.get("action_name") or "").strip().lower()
    if action_name:
        return action_name
    return stable_action_key(step.get("action"))


def stable_action_key(action: object) -> str:
    if isinstance(action, dict):
        return "|".join(f"{key}={action[key]}" for key in sorted(action))
    return str(action)


def transition_type_for_edge(payload: dict) -> str:
    if payload.get("blocked_count", 0) > payload.get("success_count", 0):
        return "blocked"
    if payload.get("uncertain_count", 0) > 0 and payload.get("success_count", 0) == 0:
        return "uncertain"
    return str(payload.get("transition_type", "move"))


def merge_topology(existing_nodes: dict[str, dict], existing_edges: dict[str, dict], nodes: dict[str, dict], edges: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    merged_nodes = {node_id: dict(row) for node_id, row in existing_nodes.items()}
    merged_edges = {edge_id: dict(row) for edge_id, row in existing_edges.items()}

    for node_id, payload in nodes.items():
        prior = merged_nodes.get(node_id, {})
        row = dict(prior)
        row.update(payload)
        row["visits"] = int(prior.get("visits", 0)) + int(payload.get("visits", 0))
        row["blocked_visits"] = int(prior.get("blocked_visits", 0)) + int(payload.get("blocked_visits", 0))
        row["uncertain_visits"] = int(prior.get("uncertain_visits", 0)) + int(payload.get("uncertain_visits", 0))
        row["status"] = "blocked" if row["blocked_visits"] > row["visits"] // 2 and row["visits"] > 0 else "confirmed"
        merged_nodes[node_id] = row

    for edge_id, payload in edges.items():
        prior = merged_edges.get(edge_id, {})
        row = dict(prior)
        row.update(payload)
        row["action_key"] = row.get("action_key") or normalized_topology_action_key(row)
        row["transition_type"] = row.get("transition_type", "move")
        row["success_count"] = int(prior.get("success_count", 0)) + int(payload.get("success_count", 0))
        row["blocked_count"] = int(prior.get("blocked_count", 0)) + int(payload.get("blocked_count", 0))
        row["uncertain_count"] = int(prior.get("uncertain_count", 0)) + int(payload.get("uncertain_count", 0))
        row["evidence_refs"] = sorted(set(prior.get("evidence_refs", [])) | set(payload.get("evidence_refs", [])))[-32:]
        row["transition_type"] = transition_type_for_edge(row)
        opposite_id = f"{row['dst']}->{row['src']}:{row['action_key']}:{row['transition_type']}"
        opposite = merged_edges.get(opposite_id)
        if opposite is not None:
            row["bidirectional_consistency"] = min(
                row["success_count"],
                opposite.get("success_count", 0),
            ) / float(max(1, max(row["success_count"], opposite.get("success_count", 0))))
        else:
            row["bidirectional_consistency"] = 0.0
        merged_edges[edge_id] = row
    return merged_nodes, merged_edges
