from __future__ import annotations


def build_indexes(state: dict) -> dict:
    entities_by_area: dict[str, list[str]] = {}
    pois_by_area: dict[str, list[str]] = {}
    pois_by_type: dict[str, list[str]] = {}
    reachable_targets: list[str] = []
    blocked_targets: list[str] = []
    frontier_candidates: list[str] = []
    consequence_by_action: dict[str, list[str]] = {}
    evidence_index: dict[str, list[str]] = {}
    topology_lookup = {
        "node_ids_by_cell": {},
        "out_edges_by_src": {},
    }

    for area_id, area in state.get("areas", {}).items():
        entities_by_area.setdefault(str(area_id), [])
        pois_by_area.setdefault(str(area_id), [])
        if area.get("topology_cells"):
            for cell in area["topology_cells"]:
                topology_lookup["node_ids_by_cell"][f"{cell[0]}:{cell[1]}"] = f"cell:{cell[0]}:{cell[1]}"

    for entity_id, entity in state.get("entities", {}).items():
        area_id = str(entity.get("area_id") or "global")
        entities_by_area.setdefault(area_id, []).append(entity_id)
        if entity.get("kind") == "poi":
            pois_by_area.setdefault(area_id, []).append(entity_id)
            poi_type = str(entity.get("canonical_descriptor", {}).get("kind") or entity.get("kind"))
            pois_by_type.setdefault(poi_type, []).append(entity_id)
        if entity.get("reachable_now"):
            reachable_targets.append(entity_id)
        elif entity.get("reachable_later"):
            frontier_candidates.append(entity_id)
        else:
            blocked_targets.append(entity_id)
        for evidence_ref in entity.get("evidence_refs", []):
            evidence_index.setdefault(str(evidence_ref), []).append(entity_id)

    for consequence_id, consequence in state.get("consequences", {}).items():
        action_key = str(consequence.get("action"))
        consequence_by_action.setdefault(action_key, []).append(consequence_id)
        for evidence_ref in consequence.get("evidence_refs", []):
            evidence_index.setdefault(str(evidence_ref), []).append(consequence_id)

    for edge_id, edge in state.get("topology_edges", {}).items():
        topology_lookup["out_edges_by_src"].setdefault(str(edge["src"]), []).append(edge_id)

    return {
        "entities_by_area": {key: sorted(value) for key, value in entities_by_area.items()},
        "pois_by_area": {key: sorted(value) for key, value in pois_by_area.items()},
        "pois_by_type": {key: sorted(value) for key, value in pois_by_type.items()},
        "reachable_targets": sorted(reachable_targets),
        "blocked_targets": sorted(blocked_targets),
        "frontier_candidates": sorted(frontier_candidates),
        "consequence_by_action": {key: sorted(value) for key, value in consequence_by_action.items()},
        "evidence_index": {key: sorted(value) for key, value in evidence_index.items()},
        "topology_lookup": topology_lookup,
        "entity_count": len(state.get("entities", {})),
        "area_count": len(state.get("areas", {})),
        "topology_node_count": len(state.get("topology_nodes", {})),
    }
