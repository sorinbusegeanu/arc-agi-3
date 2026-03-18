from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.contracts.snapshots import BlackboardSnapshot
from v3_1.contracts.versions import next_blackboard_version
from v3_1.utils.ids import make_handle
from v3_1.world.merge import apply_delta


def _empty_blackboard_state() -> dict:
    return {
        "areas": {},
        "entities": {},
        "consequences": {},
        "trigger_zones": {},
        "topology_nodes": {},
        "topology_edges": {},
        "observed_entities": {},
        "hypothesized_entities": {},
        "observed_consequences": {},
        "hypothesized_consequences": {},
        "observed_trigger_zones": {},
        "hypothesized_trigger_zones": {},
        "observed_topology": {"nodes": {}, "edges": {}},
        "hypothesized_topology": {"nodes": {}, "edges": {}},
        "indexes": {},
        "split_indexes": {"observed": {}, "hypothesized": {}},
    }


def is_compatibility_snapshot(snapshot) -> bool:
    if isinstance(snapshot, BlackboardSnapshot):
        return bool(dict(getattr(snapshot, "state", {}) or {}).get("compatibility_only"))
    if isinstance(snapshot, dict):
        return bool(snapshot.get("compatibility_only"))
    return False


def export_strict_snapshot(snapshot) -> dict:
    state = {}
    blackboard_version = None
    created_round_id = None
    created_pass_id = None
    material_change = None
    snapshot_handle = None
    if isinstance(snapshot, BlackboardSnapshot):
        state = dict(getattr(snapshot, "state", {}) or {})
        blackboard_version = getattr(snapshot, "blackboard_version", None)
        created_round_id = getattr(snapshot, "created_round_id", None)
        created_pass_id = getattr(snapshot, "created_pass_id", None)
        material_change = getattr(snapshot, "material_change", None)
        snapshot_handle = getattr(snapshot, "snapshot_handle", None)
    elif isinstance(snapshot, dict):
        state = dict(snapshot.get("state", snapshot) or {})
        blackboard_version = snapshot.get("blackboard_version")
        created_round_id = snapshot.get("created_round_id")
        created_pass_id = snapshot.get("created_pass_id")
        material_change = snapshot.get("material_change")
        snapshot_handle = snapshot.get("snapshot_handle")
    observed_topology = dict(state.get("observed_topology", {}) or {})
    hypothesized_topology = dict(state.get("hypothesized_topology", {}) or {})
    payload = {
        "snapshot_kind": "strict_split_world_export",
        "compatibility_only": False,
        "default_truth_surface": "strict_split_native",
        "index_contract_mode": "strict_split_native",
        "blackboard_version": blackboard_version,
        "created_round_id": created_round_id,
        "created_pass_id": created_pass_id,
        "material_change": material_change,
        "snapshot_handle": snapshot_handle,
        "areas": dict(state.get("areas", {})),
        "observed_entities": dict(state.get("observed_entities", {})),
        "hypothesized_entities": dict(state.get("hypothesized_entities", {})),
        "observed_consequences": dict(state.get("observed_consequences", {})),
        "hypothesized_consequences": dict(state.get("hypothesized_consequences", {})),
        "observed_trigger_zones": dict(state.get("observed_trigger_zones", {})),
        "hypothesized_trigger_zones": dict(state.get("hypothesized_trigger_zones", {})),
        "observed_topology": observed_topology,
        "hypothesized_topology": hypothesized_topology,
        "split_indexes": dict(state.get("split_indexes", {})),
        "combined_views": {
            "compatibility_only": True,
            "entities": dict(state.get("entities", {})),
            "consequences": dict(state.get("consequences", {})),
            "trigger_zones": dict(state.get("trigger_zones", {})),
            "topology_nodes": dict(state.get("topology_nodes", {})),
            "topology_edges": dict(state.get("topology_edges", {})),
            "indexes": dict(state.get("indexes", {})),
        },
        "summary": {
            "observed_entity_count": len(dict(state.get("observed_entities", {}))),
            "hypothesized_entity_count": len(dict(state.get("hypothesized_entities", {}))),
            "observed_consequence_count": len(dict(state.get("observed_consequences", {}))),
            "hypothesized_consequence_count": len(dict(state.get("hypothesized_consequences", {}))),
            "observed_trigger_count": len(dict(state.get("observed_trigger_zones", {}))),
            "hypothesized_trigger_count": len(dict(state.get("hypothesized_trigger_zones", {}))),
            "observed_topology_node_count": len(dict(observed_topology.get("nodes", {}))),
            "hypothesized_topology_node_count": len(dict(hypothesized_topology.get("nodes", {}))),
        },
    }
    return payload


def _build_strict_indexes(*, areas: dict, entities: dict, consequences: dict, trigger_zones: dict, topology: dict) -> dict:
    entities_by_area: dict[str, list[dict]] = {}
    pois_by_area: dict[str, list[dict]] = {}
    pois_by_type: dict[str, list[dict]] = {}
    reachable_targets: list[dict] = []
    blocked_targets: list[dict] = []
    frontier_candidates: list[dict] = []
    consequence_by_action: dict[str, list[dict]] = {}
    evidence_index: dict[str, list[dict]] = {}
    topology_lookup = {
        "node_ids_by_cell": {},
        "out_edges_by_src": {},
    }
    for area_id, area in dict(areas).items():
        entities_by_area.setdefault(str(area_id), [])
        pois_by_area.setdefault(str(area_id), [])
        for cell in list(area.get("topology_cells", []) or []):
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                topology_lookup["node_ids_by_cell"][f"{int(cell[0])}:{int(cell[1])}"] = f"cell:{int(cell[0])}:{int(cell[1])}"
    for entity_id, entity in dict(entities).items():
        area_id = str(entity.get("area_id") or "global")
        row = {"entity_id": str(entity_id), "area_id": area_id, "evidence_tier": str(entity.get("evidence_tier") or "hypothesized")}
        entities_by_area.setdefault(area_id, []).append(row)
        if entity.get("kind") == "poi":
            pois_by_area.setdefault(area_id, []).append(row)
            poi_type = str(entity.get("canonical_descriptor", {}).get("kind") or entity.get("kind") or "unknown")
            pois_by_type.setdefault(poi_type, []).append(row)
        if entity.get("reachable_now"):
            reachable_targets.append(row)
        elif entity.get("reachable_later"):
            frontier_candidates.append(row)
        else:
            blocked_targets.append(row)
        for evidence_ref in list(entity.get("evidence_refs", []) or []):
            evidence_index.setdefault(str(evidence_ref), []).append({"row_id": str(entity_id), "evidence_tier": row["evidence_tier"]})
    for consequence_id, consequence in dict(consequences).items():
        action_key = str(consequence.get("action_name") or consequence.get("action_key") or "unknown")
        consequence_row = {
            "consequence_id": str(consequence_id),
            "action_key": action_key,
            "evidence_tier": str(consequence.get("evidence_tier") or "hypothesized"),
        }
        consequence_by_action.setdefault(action_key, []).append(consequence_row)
        for evidence_ref in list(consequence.get("evidence_refs", []) or []):
            evidence_index.setdefault(str(evidence_ref), []).append({"row_id": str(consequence_id), "evidence_tier": consequence_row["evidence_tier"]})
    for trigger_id, trigger in dict(trigger_zones).items():
        for evidence_ref in list(trigger.get("evidence_refs", []) or []):
            evidence_index.setdefault(str(evidence_ref), []).append(
                {"row_id": str(trigger_id), "evidence_tier": str(trigger.get("evidence_tier") or "hypothesized")}
            )
    for edge_id, edge in dict(topology.get("edges", {})).items():
        topology_lookup["out_edges_by_src"].setdefault(str(edge.get("src")), []).append(str(edge_id))
        for evidence_ref in list(edge.get("evidence_refs", []) or []):
            evidence_index.setdefault(str(evidence_ref), []).append(
                {"row_id": str(edge_id), "evidence_tier": str(edge.get("evidence_tier") or "hypothesized")}
            )
    return {
        "entities_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in entities_by_area.items()},
        "pois_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_area.items()},
        "pois_by_type_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_type.items()},
        "reachable_targets_rows": sorted(reachable_targets, key=lambda row: row["entity_id"]),
        "blocked_targets_rows": sorted(blocked_targets, key=lambda row: row["entity_id"]),
        "frontier_candidates_rows": sorted(frontier_candidates, key=lambda row: row["entity_id"]),
        "consequence_by_action_rows": {key: sorted(value, key=lambda row: row["consequence_id"]) for key, value in consequence_by_action.items()},
        "evidence_index_rows": {key: sorted(value, key=lambda row: row["row_id"]) for key, value in evidence_index.items()},
        "topology_lookup": topology_lookup,
        "entity_count": len(dict(entities)),
        "area_count": len(dict(areas)),
        "topology_node_count": len(dict(topology.get("nodes", {}))),
        "trigger_count": len(dict(trigger_zones)),
        "consequence_count": len(dict(consequences)),
    }


@dataclass
class BlackboardState:
    session_id: str
    game_id: str
    max_consequences: int = 100
    revision: int = 0
    state: dict = field(default_factory=_empty_blackboard_state)

    def observed_view(self) -> dict:
        return {
            "areas": dict(self.state.get("areas", {})),
            "entities": dict(self.state.get("observed_entities", {})),
            "consequences": dict(self.state.get("observed_consequences", {})),
            "trigger_zones": dict(self.state.get("observed_trigger_zones", {})),
            "topology_nodes": dict(self.state.get("observed_topology", {}).get("nodes", {})),
            "topology_edges": dict(self.state.get("observed_topology", {}).get("edges", {})),
            "indexes": dict(self.state.get("indexes", {})),
        }

    def hypothesized_view(self) -> dict:
        return {
            "areas": dict(self.state.get("areas", {})),
            "entities": dict(self.state.get("hypothesized_entities", {})),
            "consequences": dict(self.state.get("hypothesized_consequences", {})),
            "trigger_zones": dict(self.state.get("hypothesized_trigger_zones", {})),
            "topology_nodes": dict(self.state.get("hypothesized_topology", {}).get("nodes", {})),
            "topology_edges": dict(self.state.get("hypothesized_topology", {}).get("edges", {})),
            "indexes": dict(self.state.get("indexes", {})),
        }

    def combined_view(self) -> dict:
        return {
            "areas": dict(self.state.get("areas", {})),
            "entities": dict(self.state.get("entities", {})),
            "consequences": dict(self.state.get("consequences", {})),
            "trigger_zones": dict(self.state.get("trigger_zones", {})),
            "topology_nodes": dict(self.state.get("topology_nodes", {})),
            "topology_edges": dict(self.state.get("topology_edges", {})),
            "indexes": dict(self.state.get("indexes", {})),
        }

    def snapshot_observed(self) -> dict:
        return {
            "snapshot_kind": "observed_only",
            "compatibility_only": False,
            "areas": dict(self.state.get("areas", {})),
            "observed_entities": dict(self.state.get("observed_entities", {})),
            "observed_consequences": dict(self.state.get("observed_consequences", {})),
            "observed_trigger_zones": dict(self.state.get("observed_trigger_zones", {})),
            "observed_topology": dict(self.state.get("observed_topology", {})),
            "split_indexes": dict(self.state.get("split_indexes", {}).get("observed", {})),
        }

    def snapshot_hypothesized(self) -> dict:
        return {
            "snapshot_kind": "hypothesized_only",
            "compatibility_only": False,
            "areas": dict(self.state.get("areas", {})),
            "hypothesized_entities": dict(self.state.get("hypothesized_entities", {})),
            "hypothesized_consequences": dict(self.state.get("hypothesized_consequences", {})),
            "hypothesized_trigger_zones": dict(self.state.get("hypothesized_trigger_zones", {})),
            "hypothesized_topology": dict(self.state.get("hypothesized_topology", {})),
            "split_indexes": dict(self.state.get("split_indexes", {}).get("hypothesized", {})),
        }

    def snapshot_strict(self) -> dict:
        strict_indexes = {
            "observed": _build_strict_indexes(
                areas=dict(self.state.get("areas", {})),
                entities=dict(self.state.get("observed_entities", {})),
                consequences=dict(self.state.get("observed_consequences", {})),
                trigger_zones=dict(self.state.get("observed_trigger_zones", {})),
                topology=dict(self.state.get("observed_topology", {})),
            ),
            "hypothesized": _build_strict_indexes(
                areas=dict(self.state.get("areas", {})),
                entities=dict(self.state.get("hypothesized_entities", {})),
                consequences=dict(self.state.get("hypothesized_consequences", {})),
                trigger_zones=dict(self.state.get("hypothesized_trigger_zones", {})),
                topology=dict(self.state.get("hypothesized_topology", {})),
            ),
        }
        return {
            "snapshot_kind": "strict_split_world",
            "compatibility_only": False,
            "index_contract_mode": "strict_split_native",
            "areas": dict(self.state.get("areas", {})),
            "observed_entities": dict(self.state.get("observed_entities", {})),
            "hypothesized_entities": dict(self.state.get("hypothesized_entities", {})),
            "observed_consequences": dict(self.state.get("observed_consequences", {})),
            "hypothesized_consequences": dict(self.state.get("hypothesized_consequences", {})),
            "observed_trigger_zones": dict(self.state.get("observed_trigger_zones", {})),
            "hypothesized_trigger_zones": dict(self.state.get("hypothesized_trigger_zones", {})),
            "observed_topology": dict(self.state.get("observed_topology", {})),
            "hypothesized_topology": dict(self.state.get("hypothesized_topology", {})),
            "split_indexes": strict_indexes,
        }

    def snapshot(self, *, round_id: int, pass_id: int, material_change: bool) -> BlackboardSnapshot:
        version = next_blackboard_version(self.session_id, round_id, self.revision)
        compatibility_state = dict(self.state)
        compatibility_state["compatibility_only"] = True
        payload = {"version": version, "revision": self.revision, "state": compatibility_state}
        return BlackboardSnapshot(
            snapshot_handle=make_handle("snapshot:blackboard", payload),
            blackboard_version=version,
            created_round_id=round_id,
            created_pass_id=pass_id,
            material_change=material_change,
            state=compatibility_state,
            indexes=dict(self.state.get("indexes", {})),
        )

    def merge(self, *, round_id: int, pass_id: int, deltas: list[dict]) -> BlackboardSnapshot:
        material_change = False
        next_state = self.state
        for delta in deltas:
            next_state, changed = apply_delta(next_state, delta, max_consequences=self.max_consequences)
            material_change = material_change or changed
        self.revision += 1
        self.state = next_state
        return self.snapshot(round_id=round_id, pass_id=pass_id, material_change=material_change)
