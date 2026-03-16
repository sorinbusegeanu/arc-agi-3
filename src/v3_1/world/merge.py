from __future__ import annotations

from v3_1.world.areas import merge_areas
from v3_1.world.consequences import extract_consequence_records, merge_consequences, normalized_consequence_action_key
from v3_1.world.entities import merge_entities
from v3_1.world.indexes import build_indexes
from v3_1.world.reachability import reachable_entities
from v3_1.world.topology import merge_topology
from v3_1.world.trigger_zones import merge_trigger_zones, propose_trigger_zones


def _ensure_state(state: dict) -> dict:
    payload = dict(state)
    payload.setdefault("areas", {})
    payload.setdefault("entities", {})
    payload.setdefault("consequences", {})
    payload.setdefault("trigger_zones", {})
    payload.setdefault("topology_nodes", {})
    payload.setdefault("topology_edges", {})
    payload.setdefault("observed_entities", {})
    payload.setdefault("hypothesized_entities", {})
    payload.setdefault("observed_consequences", {})
    payload.setdefault("hypothesized_consequences", {})
    payload.setdefault("observed_trigger_zones", {})
    payload.setdefault("hypothesized_trigger_zones", {})
    payload.setdefault("observed_topology", {"nodes": {}, "edges": {}})
    payload.setdefault("hypothesized_topology", {"nodes": {}, "edges": {}})
    payload.setdefault("indexes", {})
    payload.setdefault("split_indexes", {"observed": {}, "hypothesized": {}})
    return payload


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


def _has_direct_fields(row: dict, required_fields: set[str]) -> bool:
    for field in required_fields:
        value = row.get(field)
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return False
    return True


def _base_observed_checks(kind: str, row: dict, *, required_fields: set[str], allowed_objectives: set[str] | None = None) -> tuple[bool, str]:
    if not bool(row.get("factual_observation")):
        return False, "missing_factual_observation"
    if not bool(row.get("direct_evidence_present")):
        return False, "missing_direct_evidence"
    if bool(row.get("contradiction_flag")):
        return False, "contradiction_flag_present"
    if not _has_direct_fields(row, required_fields):
        return False, "missing_required_direct_fields"
    if allowed_objectives is not None and str(row.get("analysis_objective") or "") not in allowed_objectives:
        return False, "analysis_objective_not_admissible"
    if str(row.get("source_stage") or "analysis") not in {"analysis"}:
        return False, "source_stage_not_admissible"
    return True, "base_checks_passed"


def _validate_observed_area(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("areas", row, required_fields={"area_id", "area_signature", "state_hash"})
    if not ok:
        return ok, reason
    return (bool(row.get("width")) and bool(row.get("height"))), ("area_dimensions_present" if bool(row.get("width")) and bool(row.get("height")) else "missing_area_dimensions")


def _validate_observed_entity(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("entities", row, required_fields={"entity_id"}, allowed_objectives={"discovery", "terminal_attribution"})
    if not ok:
        return ok, reason
    evidence_fields = set(str(field) for field in list(row.get("direct_evidence_fields", []) or []))
    valid = bool({"bbox", "centroid"} & evidence_fields or row.get("bbox") or row.get("centroid"))
    return valid, ("entity_localization_present" if valid else "missing_entity_localization")


def _validate_observed_consequence(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("consequences", row, required_fields={"consequence_id", "action_name"}, allowed_objectives={"terminal_attribution"})
    if not ok:
        return ok, reason
    valid = bool(row.get("reward") not in {None, ""} or row.get("done") is not None or int(row.get("local_change_area", 0) or 0) > 0)
    return valid, ("consequence_effect_present" if valid else "missing_consequence_effect")


def _validate_observed_trigger_zone(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("trigger_zones", row, required_fields={"trigger_id"}, allowed_objectives={"broad_trigger_suspicion", "route_progress_attribution", "terminal_attribution"})
    if not ok:
        return ok, reason
    evidence_fields = set(str(field) for field in list(row.get("direct_evidence_fields", []) or []))
    valid = bool({"effect_region", "supporting_steps", "zone_bbox"} & evidence_fields or row.get("zone_bbox") or row.get("supporting_steps"))
    return valid, ("trigger_zone_local_support_present" if valid else "missing_trigger_zone_local_support")


def _validate_observed_topology_node(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("topology_nodes", row, required_fields={"node_id"}, allowed_objectives={"discovery", "route_progress_attribution"})
    if not ok:
        return ok, reason
    return (bool(row.get("cell"))), ("topology_node_cell_present" if bool(row.get("cell")) else "missing_topology_node_cell")


def _validate_observed_topology_edge(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("topology_edges", row, required_fields={"edge_id"}, allowed_objectives={"discovery", "route_progress_attribution"})
    if not ok:
        return ok, reason
    valid = bool(row.get("src") and row.get("dst") and row.get("action_key"))
    return valid, ("topology_edge_transition_present" if valid else "missing_topology_edge_transition")


OBSERVED_VALIDATORS = {
    "areas": ("validate_observed_area", _validate_observed_area),
    "entities": ("validate_observed_entity", _validate_observed_entity),
    "consequences": ("validate_observed_consequence", _validate_observed_consequence),
    "trigger_zones": ("validate_observed_trigger_zone", _validate_observed_trigger_zone),
    "topology_nodes": ("validate_observed_topology_node", _validate_observed_topology_node),
    "topology_edges": ("validate_observed_topology_edge", _validate_observed_topology_edge),
}


def _row_key(kind: str, row: dict) -> str:
    if kind == "entities":
        return str(row.get("entity_id") or row.get("poi_id") or row.get("signature") or "")
    if kind == "consequences":
        return str(row.get("consequence_id") or "")
    if kind == "trigger_zones":
        return str(row.get("trigger_id") or row.get("trigger_zone_id") or "")
    if kind == "topology_nodes":
        return str(row.get("node_id") or "")
    if kind == "topology_edges":
        return str(row.get("edge_id") or "")
    if kind == "areas":
        return str(row.get("area_id") or "")
    return ""


def _classify_row(kind: str, row: dict, delta: dict) -> dict:
    payload = dict(row)
    inference_method = str(payload.get("inference_method") or "").strip().lower()
    validator_name, validator = OBSERVED_VALIDATORS.get(kind, ("validate_observed_default_reject", lambda _row: (False, "row_kind_not_observed_eligible")))
    validator_ok, validator_result = validator(payload)
    if validator_ok:
        evidence_tier = "observed"
        observed_admission_reason = "direct_fact_allowlisted"
    else:
        evidence_tier = "hypothesized"
        if inference_method == "direct_observation":
            observed_admission_reason = "rejected_direct_observation_not_sufficient"
        else:
            observed_admission_reason = f"rejected_{validator_result}"
    payload["evidence_tier"] = evidence_tier
    payload["observed_admission_reason"] = observed_admission_reason
    payload["observed_validator_name"] = validator_name
    payload["observed_validator_result"] = validator_result
    payload["source_stage"] = str(payload.get("source_stage") or "analysis")
    payload["source_pass_id"] = int(payload.get("source_pass_id", delta.get("pass_id", 0)) or 0)
    payload["source_episode_id"] = str(payload.get("source_episode_id") or delta.get("episode_id") or "")
    payload["inference_method"] = str(payload.get("inference_method") or ("direct_observation" if evidence_tier == "observed" else f"{kind}_inference"))
    payload["confidence"] = float(payload.get("confidence", 1.0 if evidence_tier == "observed" else 0.5) or 0.0)
    return payload


def _combine_rows(*stores: dict[str, dict]) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for store in stores:
        for row_id, row in store.items():
            combined[str(row_id)] = dict(row)
    return combined


def _apply_step_target_effects(entities: dict[str, dict], delta: dict) -> dict[str, dict]:
    updated = {entity_id: dict(row) for entity_id, row in dict(entities or {}).items()}
    metadata = dict(delta.get("metadata", {})) if isinstance(delta.get("metadata"), dict) else {}
    step_rows = list(metadata.get("step_rows", []) or [])
    grouped: dict[str, dict[str, int]] = {}
    for row in step_rows:
        if not isinstance(row, dict):
            continue
        target_id = row.get("target_entity_id")
        if not target_id or str(target_id) not in updated:
            continue
        family = str(row.get("action_family") or "unknown").strip().lower()
        changed_cells = int(row.get("changed_cells", 0) or 0)
        stats = grouped.setdefault(
            str(target_id),
            {
                "movement_attempts": 0,
                "interact_attempts": 0,
                "click_attempts": 0,
                "movement_effect_sum": 0,
                "interact_effect_sum": 0,
                "click_effect_sum": 0,
            },
        )
        if family == "move":
            stats["movement_attempts"] += 1
            stats["movement_effect_sum"] += changed_cells
        elif family == "interact":
            stats["interact_attempts"] += 1
            stats["interact_effect_sum"] += changed_cells
        elif family == "click_at":
            stats["click_attempts"] += 1
            stats["click_effect_sum"] += changed_cells
    for target_id, stats in grouped.items():
        entity = updated.get(target_id)
        if not entity:
            continue
        movement_attempts = int(stats["movement_attempts"])
        interact_attempts = int(stats["interact_attempts"])
        click_attempts = int(stats["click_attempts"])
        movement_effect_sum = int(stats["movement_effect_sum"])
        interact_effect_sum = int(stats["interact_effect_sum"])
        click_effect_sum = int(stats["click_effect_sum"])
        movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts) / 50.0) if movement_attempts > 0 else 0.0)
        interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts) / 50.0) if interact_attempts > 0 else 0.0)
        click_effect_score = min(1.0, (click_effect_sum / float(click_attempts) / 50.0) if click_attempts > 0 else 0.0)
        if interact_attempts > 0:
            candidate_effect_mode = "interact"
            candidate_effect_score = interact_effect_score
        elif click_attempts > 0:
            candidate_effect_mode = "click_at"
            candidate_effect_score = click_effect_score
        else:
            candidate_effect_mode = "move"
            candidate_effect_score = movement_effect_score
        entity["movement_attempts"] = int(entity.get("movement_attempts", 0) or 0) + movement_attempts
        entity["interact_attempts"] = int(entity.get("interact_attempts", 0) or 0) + interact_attempts
        entity["click_attempts"] = int(entity.get("click_attempts", 0) or 0) + click_attempts
        entity["movement_effect_sum"] = int(entity.get("movement_effect_sum", 0) or 0) + movement_effect_sum
        entity["interact_effect_sum"] = int(entity.get("interact_effect_sum", 0) or 0) + interact_effect_sum
        entity["click_effect_sum"] = int(entity.get("click_effect_sum", 0) or 0) + click_effect_sum
        entity["movement_effect_score"] = max(float(entity.get("movement_effect_score", 0.0) or 0.0), movement_effect_score)
        entity["interact_effect_score"] = max(float(entity.get("interact_effect_score", 0.0) or 0.0), interact_effect_score)
        entity["click_effect_score"] = max(float(entity.get("click_effect_score", 0.0) or 0.0), click_effect_score)
        if float(candidate_effect_score) >= float(entity.get("candidate_effect_score", 0.0) or 0.0):
            entity["candidate_effect_mode"] = candidate_effect_mode
        entity["candidate_effect_score"] = max(float(entity.get("candidate_effect_score", 0.0) or 0.0), float(candidate_effect_score))
    return updated


def _merge_topology_store(existing: dict[str, dict], incoming_nodes: list[dict], incoming_edges: list[dict]) -> dict[str, dict]:
    nodes, edges = merge_topology(
        dict(existing.get("nodes", {})),
        dict(existing.get("edges", {})),
        {row["node_id"]: row for row in incoming_nodes if row.get("node_id")},
        {row["edge_id"]: row for row in incoming_edges if row.get("edge_id")},
    )
    return {"nodes": nodes, "edges": edges}


def _consequence_sort_key(row: dict) -> tuple:
    return (
        int(row.get("last_seen_round", row.get("round_id", row.get("source_round_id", 0))) or 0),
        int(row.get("step_idx", -1) or -1),
        str(row.get("consequence_id") or ""),
    )


def _prune_consequence_store(store: dict[str, dict], *, limit: int) -> dict[str, dict]:
    if limit <= 0 or len(store) <= limit:
        return {str(row_id): dict(row) for row_id, row in store.items()}
    ordered = sorted((dict(row) for row in store.values()), key=_consequence_sort_key, reverse=True)
    kept = ordered[:limit]
    return {str(row.get("consequence_id")): row for row in kept if row.get("consequence_id")}


def _enrich_indexes_with_evidence_tiers(next_state: dict) -> dict:
    combined_state = {
        "areas": dict(next_state.get("areas", {})),
        "entities": dict(next_state.get("entities", {})),
        "consequences": dict(next_state.get("consequences", {})),
        "trigger_zones": dict(next_state.get("trigger_zones", {})),
        "topology_nodes": dict(next_state.get("topology_nodes", {})),
        "topology_edges": dict(next_state.get("topology_edges", {})),
    }
    indexes = build_indexes(combined_state)

    def entity_row(entity_id: str) -> dict:
        entity = dict(next_state.get("entities", {}).get(entity_id, {}))
        return {
            "entity_id": entity_id,
            "area_id": entity.get("area_id"),
            "evidence_tier": entity.get("evidence_tier", "hypothesized"),
        }

    def consequence_row(consequence_id: str) -> dict:
        consequence = dict(next_state.get("consequences", {}).get(consequence_id, {}))
        return {
            "consequence_id": consequence_id,
            "action_key": normalized_consequence_action_key(consequence),
            "evidence_tier": consequence.get("evidence_tier", "hypothesized"),
        }

    indexes["entities_by_area_rows"] = {
        area_id: [entity_row(entity_id) for entity_id in entity_ids]
        for area_id, entity_ids in dict(indexes.get("entities_by_area", {})).items()
    }
    indexes["pois_by_area_rows"] = {
        area_id: [entity_row(entity_id) for entity_id in entity_ids]
        for area_id, entity_ids in dict(indexes.get("pois_by_area", {})).items()
    }
    indexes["reachable_targets_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("reachable_targets", []))]
    indexes["blocked_targets_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("blocked_targets", []))]
    indexes["frontier_candidates_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("frontier_candidates", []))]
    indexes["consequence_by_action_rows"] = {
        action_key: [consequence_row(consequence_id) for consequence_id in consequence_ids]
        for action_key, consequence_ids in dict(indexes.get("consequence_by_action", {})).items()
    }
    indexes["evidence_index_rows"] = {
        evidence_ref: [
            {
                "row_id": row_id,
                "evidence_tier": (
                    next_state.get("entities", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("consequences", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("trigger_zones", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("topology_nodes", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("topology_edges", {}).get(row_id, {}).get("evidence_tier")
                    or "hypothesized"
                ),
            }
            for row_id in row_ids
        ]
        for evidence_ref, row_ids in dict(indexes.get("evidence_index", {})).items()
    }
    return indexes


def _build_strict_split_indexes(*, areas: dict, entities: dict, consequences: dict, trigger_zones: dict, topology_nodes: dict, topology_edges: dict) -> dict:
    entities_by_area_rows: dict[str, list[dict]] = {}
    pois_by_area_rows: dict[str, list[dict]] = {}
    pois_by_type_rows: dict[str, list[dict]] = {}
    reachable_targets_rows: list[dict] = []
    blocked_targets_rows: list[dict] = []
    frontier_candidates_rows: list[dict] = []
    consequence_by_action_rows: dict[str, list[dict]] = {}
    evidence_index_rows: dict[str, list[dict]] = {}
    topology_lookup = {"node_ids_by_cell": {}, "out_edges_by_src": {}}
    for area_id, area in dict(areas).items():
        entities_by_area_rows.setdefault(str(area_id), [])
        pois_by_area_rows.setdefault(str(area_id), [])
        for cell in list(area.get("topology_cells", []) or []):
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                topology_lookup["node_ids_by_cell"][f"{int(cell[0])}:{int(cell[1])}"] = f"cell:{int(cell[0])}:{int(cell[1])}"
    for entity_id, entity in dict(entities).items():
        row = {"entity_id": str(entity_id), "area_id": str(entity.get("area_id") or "global"), "evidence_tier": str(entity.get("evidence_tier") or "hypothesized")}
        entities_by_area_rows.setdefault(row["area_id"], []).append(row)
        if entity.get("kind") == "poi":
            pois_by_area_rows.setdefault(row["area_id"], []).append(row)
            poi_type = str(entity.get("canonical_descriptor", {}).get("kind") or entity.get("kind") or "unknown")
            pois_by_type_rows.setdefault(poi_type, []).append(row)
        if entity.get("reachable_now"):
            reachable_targets_rows.append(row)
        elif entity.get("reachable_later"):
            frontier_candidates_rows.append(row)
        else:
            blocked_targets_rows.append(row)
        for evidence_ref in list(entity.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(entity_id), "evidence_tier": row["evidence_tier"]})
    for consequence_id, consequence in dict(consequences).items():
        action_key = normalized_consequence_action_key(consequence)
        row = {"consequence_id": str(consequence_id), "action_key": action_key, "evidence_tier": str(consequence.get("evidence_tier") or "hypothesized")}
        consequence_by_action_rows.setdefault(action_key, []).append(row)
        for evidence_ref in list(consequence.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(consequence_id), "evidence_tier": row["evidence_tier"]})
    for trigger_id, trigger in dict(trigger_zones).items():
        for evidence_ref in list(trigger.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(trigger_id), "evidence_tier": str(trigger.get("evidence_tier") or "hypothesized")})
    for edge_id, edge in dict(topology_edges).items():
        topology_lookup["out_edges_by_src"].setdefault(str(edge.get("src")), []).append(str(edge_id))
        for evidence_ref in list(edge.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(edge_id), "evidence_tier": str(edge.get("evidence_tier") or "hypothesized")})
    return {
        "entities_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in entities_by_area_rows.items()},
        "pois_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_area_rows.items()},
        "pois_by_type_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_type_rows.items()},
        "reachable_targets_rows": sorted(reachable_targets_rows, key=lambda row: row["entity_id"]),
        "blocked_targets_rows": sorted(blocked_targets_rows, key=lambda row: row["entity_id"]),
        "frontier_candidates_rows": sorted(frontier_candidates_rows, key=lambda row: row["entity_id"]),
        "consequence_by_action_rows": {key: sorted(value, key=lambda row: row["consequence_id"]) for key, value in consequence_by_action_rows.items()},
        "evidence_index_rows": {key: sorted(value, key=lambda row: row["row_id"]) for key, value in evidence_index_rows.items()},
        "topology_lookup": topology_lookup,
        "entity_count": len(dict(entities)),
        "area_count": len(dict(areas)),
        "topology_node_count": len(dict(topology_nodes)),
        "trigger_count": len(dict(trigger_zones)),
        "consequence_count": len(dict(consequences)),
    }


def _split_index_state(*, areas: dict, entities: dict, consequences: dict, trigger_zones: dict, topology_nodes: dict, topology_edges: dict) -> dict:
    return _build_strict_split_indexes(
        areas=dict(areas),
        entities=dict(entities),
        consequences=dict(consequences),
        trigger_zones=dict(trigger_zones),
        topology_nodes=dict(topology_nodes),
        topology_edges=dict(topology_edges),
    )


def apply_delta(state: dict, delta: dict, *, max_consequences: int = 100) -> tuple[dict, bool]:
    state = _ensure_state(state)
    next_state = dict(state)
    classified_areas = [_classify_row("areas", row, delta) for row in list(delta.get("areas", ()) or [])]
    classified_entities = [_classify_row("entities", row, delta) for row in list(delta.get("entities", ()) or [])]
    prepopulated_consequences = list(delta.get("consequences", ()))
    raw_consequences = (
        prepopulated_consequences
        if prepopulated_consequences and all(_consequence_transport_complete(row) for row in prepopulated_consequences)
        else extract_consequence_records(delta)
    )
    classified_consequences = [_classify_row("consequences", row, delta) for row in raw_consequences]
    classified_trigger_zones = [_classify_row("trigger_zones", row, delta) for row in list(delta.get("trigger_zones", ()) or [])]
    classified_topology_nodes = [_classify_row("topology_nodes", row, delta) for row in list(delta.get("topology_nodes", ()) or [])]
    classified_topology_edges = [_classify_row("topology_edges", row, delta) for row in list(delta.get("topology_edges", ()) or [])]

    merged_areas = merge_areas(state.get("areas", {}), classified_areas)
    observed_entities = merge_entities(
        state.get("observed_entities", {}),
        [row for row in classified_entities if row.get("evidence_tier") == "observed"],
    )
    hypothesized_entities = merge_entities(
        state.get("hypothesized_entities", {}),
        [row for row in classified_entities if row.get("evidence_tier") != "observed"],
    )
    observed_consequences = merge_consequences(
        state.get("observed_consequences", {}),
        [row for row in classified_consequences if row.get("evidence_tier") == "observed"],
    )
    hypothesized_consequences = merge_consequences(
        state.get("hypothesized_consequences", {}),
        [row for row in classified_consequences if row.get("evidence_tier") != "observed"],
    )
    observed_consequences = _prune_consequence_store(observed_consequences, limit=max_consequences)
    hypothesized_consequences = _prune_consequence_store(hypothesized_consequences, limit=max_consequences)
    observed_topology = _merge_topology_store(
        dict(state.get("observed_topology", {})),
        [row for row in classified_topology_nodes if row.get("evidence_tier") == "observed"],
        [row for row in classified_topology_edges if row.get("evidence_tier") == "observed"],
    )
    hypothesized_topology = _merge_topology_store(
        dict(state.get("hypothesized_topology", {})),
        [row for row in classified_topology_nodes if row.get("evidence_tier") != "observed"],
        [row for row in classified_topology_edges if row.get("evidence_tier") != "observed"],
    )

    merged_entities = _combine_rows(hypothesized_entities, observed_entities)
    merged_consequences = _combine_rows(hypothesized_consequences, observed_consequences)
    merged_consequences = _prune_consequence_store(merged_consequences, limit=max_consequences)
    merged_ids = set(merged_consequences.keys())
    observed_consequences = {row_id: row for row_id, row in observed_consequences.items() if row_id in merged_ids}
    hypothesized_consequences = {row_id: row for row_id, row in hypothesized_consequences.items() if row_id in merged_ids}
    topology_nodes = _combine_rows(
        dict(hypothesized_topology.get("nodes", {})),
        dict(observed_topology.get("nodes", {})),
    )
    topology_edges = _combine_rows(
        dict(hypothesized_topology.get("edges", {})),
        dict(observed_topology.get("edges", {})),
    )
    merged_entities = reachable_entities(merged_entities, topology_nodes, topology_edges)
    merged_entities = _apply_step_target_effects(merged_entities, delta)
    proposed_trigger_zones = classified_trigger_zones + [
        _classify_row("trigger_zones", row, delta)
        for row in propose_trigger_zones(
            entities=merged_entities,
            consequences=merged_consequences,
        )
    ]
    observed_trigger_zones = merge_trigger_zones(
        state.get("observed_trigger_zones", {}),
        [row for row in proposed_trigger_zones if row.get("evidence_tier") == "observed"],
    )
    hypothesized_trigger_zones = merge_trigger_zones(
        state.get("hypothesized_trigger_zones", {}),
        [row for row in proposed_trigger_zones if row.get("evidence_tier") != "observed"],
    )
    merged_trigger_zones = _combine_rows(hypothesized_trigger_zones, observed_trigger_zones)
    merged_areas = _merge_area_topology_metadata(merged_areas, topology_nodes)

    next_state["areas"] = merged_areas
    next_state["observed_entities"] = observed_entities
    next_state["hypothesized_entities"] = hypothesized_entities
    next_state["entities"] = merged_entities
    next_state["observed_consequences"] = observed_consequences
    next_state["hypothesized_consequences"] = hypothesized_consequences
    next_state["consequences"] = merged_consequences
    next_state["observed_trigger_zones"] = observed_trigger_zones
    next_state["hypothesized_trigger_zones"] = hypothesized_trigger_zones
    next_state["trigger_zones"] = merged_trigger_zones
    next_state["observed_topology"] = observed_topology
    next_state["hypothesized_topology"] = hypothesized_topology
    next_state["topology_nodes"] = topology_nodes
    next_state["topology_edges"] = topology_edges
    next_state["indexes"] = _enrich_indexes_with_evidence_tiers(next_state)
    next_state["split_indexes"] = {
        "observed": _split_index_state(
            areas=merged_areas,
            entities=observed_entities,
            consequences=observed_consequences,
            trigger_zones=observed_trigger_zones,
            topology_nodes=dict(observed_topology.get("nodes", {})),
            topology_edges=dict(observed_topology.get("edges", {})),
        ),
        "hypothesized": _split_index_state(
            areas=merged_areas,
            entities=hypothesized_entities,
            consequences=hypothesized_consequences,
            trigger_zones=hypothesized_trigger_zones,
            topology_nodes=dict(hypothesized_topology.get("nodes", {})),
            topology_edges=dict(hypothesized_topology.get("edges", {})),
        ),
    }
    material_change = bool(
        delta.get("material_change", True)
        or delta.get("entities")
        or delta.get("areas")
        or raw_consequences
        or delta.get("topology_edges")
    )
    return next_state, material_change
