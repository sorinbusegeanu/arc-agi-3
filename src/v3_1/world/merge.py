from __future__ import annotations

from v3_1.world.areas import merge_areas
from v3_1.world.consequences import extract_consequence_records, merge_consequences
from v3_1.world.entities import merge_entities
from v3_1.world.indexes import build_indexes
from v3_1.world.reachability import reachable_entities
from v3_1.world.topology import merge_topology
from v3_1.world.trigger_zones import merge_trigger_zones, propose_trigger_zones


def _merge_area_topology_metadata(areas: dict[str, dict], topology_nodes: dict[str, dict]) -> dict[str, dict]:
    updated = {area_id: dict(area) for area_id, area in areas.items()}
    for area_id, area in updated.items():
        cells = [tuple(cell) for cell in area.get("topology_cells", [])]
        existing_cells = set(cells)
        for node in topology_nodes.values():
            if node.get("area_id") == area_id and tuple(node.get("cell", ())) not in existing_cells:
                cells.append(tuple(node["cell"]))
        area["topology_cells"] = [list(cell) for cell in sorted(set(cells))]
    return updated


def _consequence_transport_complete(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    action_name = str(row.get("action_name") or "").strip()
    action_family = str(row.get("action_family") or "").strip()
    evidence_refs = list(row.get("evidence_refs", []))
    return bool(action_name and action_family and evidence_refs)


def apply_delta(state: dict, delta: dict) -> tuple[dict, bool]:
    next_state = dict(state)
    merged_areas = merge_areas(state.get("areas", {}), delta.get("areas", ()))
    merged_entities = merge_entities(state.get("entities", {}), delta.get("entities", ()))
    prepopulated_consequences = list(delta.get("consequences", ()))
    raw_consequences = (
        prepopulated_consequences
        if prepopulated_consequences and all(_consequence_transport_complete(row) for row in prepopulated_consequences)
        else extract_consequence_records(delta)
    )
    merged_consequences = merge_consequences(state.get("consequences", {}), raw_consequences)
    topology_nodes, topology_edges = merge_topology(
        state.get("topology_nodes", {}),
        state.get("topology_edges", {}),
        {row["node_id"]: row for row in delta.get("topology_nodes", ())},
        {row["edge_id"]: row for row in delta.get("topology_edges", ())},
    )
    merged_entities = reachable_entities(merged_entities, topology_nodes, topology_edges)
    proposed_trigger_zones = list(delta.get("trigger_zones", ())) + propose_trigger_zones(
        entities=merged_entities,
        consequences=merged_consequences,
    )
    merged_trigger_zones = merge_trigger_zones(state.get("trigger_zones", {}), proposed_trigger_zones)
    merged_areas = _merge_area_topology_metadata(merged_areas, topology_nodes)

    next_state["areas"] = merged_areas
    next_state["entities"] = merged_entities
    next_state["consequences"] = merged_consequences
    next_state["trigger_zones"] = merged_trigger_zones
    next_state["topology_nodes"] = topology_nodes
    next_state["topology_edges"] = topology_edges
    next_state["indexes"] = build_indexes(next_state)
    material_change = bool(
        delta.get("material_change", True)
        or delta.get("entities")
        or delta.get("areas")
        or raw_consequences
        or delta.get("topology_edges")
    )
    return next_state, material_change
