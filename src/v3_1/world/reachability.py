from __future__ import annotations

from collections import deque


def _cell_for_entity(entity: dict) -> tuple[int, int] | None:
    centroid = entity.get("centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        return None
    return int(float(centroid[0])), int(float(centroid[1]))


def _graph_adjacency(topology_edges: dict[str, dict]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for edge in topology_edges.values():
        if edge.get("transition_type") == "blocked":
            continue
        graph.setdefault(str(edge["src"]), []).append(str(edge["dst"]))
    return graph


def _distance_from_frontier(graph: dict[str, list[str]], topology_nodes: dict[str, dict], start_nodes: list[str]) -> dict[str, int]:
    distances = {node_id: 0 for node_id in start_nodes}
    queue = deque(start_nodes)
    while queue:
        node_id = queue.popleft()
        for neighbor in graph.get(node_id, []):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node_id] + 1
            queue.append(neighbor)
    return distances


def reachable_entities(entities: dict[str, dict], topology_nodes: dict[str, dict], topology_edges: dict[str, dict] | None = None) -> dict[str, dict]:
    topology_edges = topology_edges or {}
    visited_now = {tuple(node.get("cell", ())) for node in topology_nodes.values() if node.get("status") != "blocked"}
    frontier_nodes = [node_id for node_id, node in topology_nodes.items() if int(node.get("uncertain_visits", 0)) > 0 or int(node.get("blocked_visits", 0)) > 0]
    graph = _graph_adjacency(topology_edges)
    distances = _distance_from_frontier(graph, topology_nodes, frontier_nodes)
    out: dict[str, dict] = {}
    for entity_id, entity in entities.items():
        cell = _cell_for_entity(entity)
        reachable_now = cell in visited_now if cell is not None else False
        nearest_node_id = None
        nearest_distance = 9999
        for node_id, node in topology_nodes.items():
            node_cell = tuple(node.get("cell", ()))
            if cell is None or len(node_cell) != 2:
                continue
            distance = abs(node_cell[0] - cell[0]) + abs(node_cell[1] - cell[1])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node_id = node_id
        reachable_later = bool(nearest_node_id is not None and nearest_node_id in distances) or (nearest_distance <= 3 if nearest_node_id is not None else False)
        access_profile = "reachable_now" if reachable_now else ("frontier_adjacent" if reachable_later else "unknown")
        payload = dict(entity)
        payload["reachable_now"] = reachable_now
        payload["reachable_later"] = bool(reachable_later and not reachable_now)
        payload["access_profile"] = {
            "nearest_node_id": nearest_node_id,
            "nearest_node_distance": None if nearest_node_id is None else nearest_distance,
            "profile": access_profile,
        }
        out[entity_id] = payload
    return out
