from __future__ import annotations

from collections import deque


def _cell_for_entity(entity: dict) -> tuple[int, int] | None:
    centroid = entity.get("centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        return None
    return int(float(centroid[0])), int(float(centroid[1]))


def _bbox_for_entity(entity: dict) -> tuple[int, int, int, int] | None:
    bbox = dict(entity.get("bbox", {}) or {})
    try:
        return int(bbox["x1"]), int(bbox["x2"]), int(bbox["y1"]), int(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_distance_to_cell(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> int:
    x1, x2, y1, y2 = bbox
    cx, cy = int(cell[0]), int(cell[1])
    dx = 0 if x1 <= cx <= x2 else min(abs(cx - x1), abs(cx - x2))
    dy = 0 if y1 <= cy <= y2 else min(abs(cy - y1), abs(cy - y2))
    return int(dx + dy)


def _detector_backed_poi(entity: dict) -> bool:
    provenance = set(str(value) for value in list(entity.get("poi_source_provenance", []) or []))
    return entity.get("kind") == "poi" and bool(entity.get("planner_targetable", True)) and "detector" in provenance


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
    frontier_cells = [
        tuple(node.get("cell", ()))
        for node_id, node in topology_nodes.items()
        if node_id in frontier_nodes and isinstance(node.get("cell"), (list, tuple)) and len(node.get("cell")) == 2
    ]
    out: dict[str, dict] = {}
    for entity_id, entity in entities.items():
        cell = _cell_for_entity(entity)
        bbox = _bbox_for_entity(entity)
        reachable_now = cell in visited_now if cell is not None else False
        bbox_edge_distance_now = None
        if bbox is not None and visited_now:
            bbox_edge_distance_now = min(_bbox_distance_to_cell(bbox, visited_cell) for visited_cell in visited_now)
            if bbox_edge_distance_now == 0:
                reachable_now = True
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
        nearest_bbox_distance = None
        nearest_frontier_bbox_distance = None
        frontier_adjacent = False
        if bbox is not None:
            topology_cells = [
                tuple(node.get("cell", ()))
                for node in topology_nodes.values()
                if isinstance(node.get("cell"), (list, tuple)) and len(node.get("cell")) == 2
            ]
            if topology_cells:
                nearest_bbox_distance = min(_bbox_distance_to_cell(bbox, node_cell) for node_cell in topology_cells)
            if frontier_cells:
                nearest_frontier_bbox_distance = min(_bbox_distance_to_cell(bbox, node_cell) for node_cell in frontier_cells)
                frontier_adjacent = nearest_frontier_bbox_distance <= 2
        reachable_later = bool(nearest_node_id is not None and nearest_node_id in distances) or (nearest_distance <= 3 if nearest_node_id is not None else False)
        if not reachable_later and nearest_bbox_distance is not None:
            reachable_later = nearest_bbox_distance <= 2
        approachable_now = False
        approachable_later = False
        if _detector_backed_poi(entity):
            perimeter_approachable = nearest_bbox_distance is not None and nearest_bbox_distance <= 2
            same_area_soft = nearest_bbox_distance is not None and nearest_bbox_distance <= 4
            frontier_approachable = frontier_adjacent
            if not reachable_now and not reachable_later:
                approachable_now = bool(perimeter_approachable or frontier_approachable)
                approachable_later = bool(same_area_soft or (nearest_frontier_bbox_distance is not None and nearest_frontier_bbox_distance <= 4))
        access_profile = "reachable_now" if reachable_now else ("frontier_adjacent" if reachable_later or frontier_adjacent else "unknown")
        reachable_status = "reachable_now" if reachable_now else ("reachable_later" if reachable_later else "blocked")
        approachable_status = "approachable_now" if approachable_now else ("approachable_later" if approachable_later else ("reachable" if reachable_now or reachable_later else "not_approachable"))
        payload = dict(entity)
        payload["reachable_now"] = reachable_now
        payload["reachable_later"] = bool(reachable_later and not reachable_now)
        payload["reachable_status"] = reachable_status
        payload["approachable_now"] = approachable_now
        payload["approachable_later"] = approachable_later
        payload["approachable_status"] = approachable_status
        payload["access_profile"] = {
            "nearest_node_id": nearest_node_id,
            "nearest_node_distance": None if nearest_node_id is None else nearest_distance,
            "nearest_bbox_distance": nearest_bbox_distance,
            "bbox_edge_distance_now": bbox_edge_distance_now,
            "nearest_frontier_bbox_distance": nearest_frontier_bbox_distance,
            "frontier_adjacent": frontier_adjacent,
            "profile": access_profile,
        }
        out[entity_id] = payload
    return out
