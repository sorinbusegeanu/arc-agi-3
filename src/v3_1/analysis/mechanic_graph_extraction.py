from __future__ import annotations

from v3_1.analysis.consequences import (
    delayed_change_evidence,
    remote_region_change_evidence,
    repeated_effect_region_support,
    repeated_contact_to_change_support,
    trigger_contact_evidence,
)
from v3_1.contracts.messages import AnalyzedEpisode, MechanicGraphDelta, RawEpisode
from v3_1.utils.ids import stable_digest


def _experiment_result(raw_episode: RawEpisode) -> dict:
    metadata = dict(raw_episode.metadata or {})
    if isinstance(metadata.get("experiment_result"), dict):
        return dict(metadata.get("experiment_result") or {})
    for step in reversed(list(raw_episode.steps or ())):
        info = dict(step.info or {})
        if isinstance(info.get("experiment_result"), dict):
            return dict(info.get("experiment_result") or {})
    return {}


def _infer_node_kind(row: dict) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            row.get("poi_class"),
            row.get("kind"),
            row.get("canonical_descriptor", {}).get("kind") if isinstance(row.get("canonical_descriptor"), dict) else "",
            row.get("target_label"),
        )
    ).lower()
    if "exit" in text:
        return "exit"
    if "gate" in text or "door" in text:
        return "gate"
    if "panel" in text or "switch" in text:
        return "panel"
    if "trigger" in text or "button" in text:
        return "trigger"
    return "poi"


def _avatar_like_poi(row: dict) -> bool:
    type_hints = [str(value or "").lower() for value in list(row.get("type_hints", []) or [])]
    poi_class = str(row.get("poi_class") or "").lower()
    canonical_kind = ""
    canonical_descriptor = row.get("canonical_descriptor")
    if isinstance(canonical_descriptor, dict):
        canonical_kind = str(canonical_descriptor.get("kind") or "").lower()
    kind = str(row.get("kind") or "").lower()
    text = " ".join(type_hints + [poi_class, canonical_kind, kind]).lower()
    return any(
        marker in text
        for marker in (
            "candidate_avatar",
            "avatar",
            "mobile_candidate",
        )
    )


def _bbox_dict(row: dict) -> dict[str, int]:
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        return {
            "x1": int(bbox.get("x1", 0) or 0),
            "y1": int(bbox.get("y1", 0) or 0),
            "x2": int(bbox.get("x2", 0) or 0),
            "y2": int(bbox.get("y2", 0) or 0),
        }
    if isinstance(bbox, list) and len(bbox) == 4:
        return {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])}
    centroid = list(row.get("centroid", [0, 0]) or [0, 0])
    x = int(float(centroid[0])) if len(centroid) >= 1 else 0
    y = int(float(centroid[1])) if len(centroid) >= 2 else 0
    return {"x1": x, "y1": y, "x2": x, "y2": y}


def _bbox_area(bbox: dict[str, int]) -> int:
    return max(0, (int(bbox["x2"]) - int(bbox["x1"]) + 1) * (int(bbox["y2"]) - int(bbox["y1"]) + 1))


def _bbox_centroid(bbox: dict[str, int]) -> list[float]:
    return [((bbox["x1"] + bbox["x2"]) / 2.0), ((bbox["y1"] + bbox["y2"]) / 2.0)]


def _heuristic_node_roles(pois: list[dict], *, width: int, height: int) -> dict[str, str]:
    if not pois or width <= 0 or height <= 0:
        return {}
    rows = []
    for poi in list(pois or []):
        if _avatar_like_poi(poi):
            continue
        bbox = _bbox_dict(poi)
        centroid = _bbox_centroid(bbox)
        rows.append(
            {
                "poi": dict(poi),
                "bbox": bbox,
                "centroid": centroid,
                "bbox_area": _bbox_area(bbox),
                "w": max(1, bbox["x2"] - bbox["x1"] + 1),
                "h": max(1, bbox["y2"] - bbox["y1"] + 1),
            }
        )
    roles: dict[str, str] = {}
    center_x = width / 2.0

    top_candidates = sorted(
        rows,
        key=lambda row: (
            row["centroid"][1],
            abs(row["centroid"][0] - center_x),
            row["bbox_area"],
        ),
    )
    if top_candidates:
        exit_row = top_candidates[0]
        roles[str(exit_row["poi"].get("entity_id") or exit_row["poi"].get("poi_id") or "")] = "exit"

    bottom_small = [
        row for row in rows
        if row["centroid"][1] >= (height * 0.72)
        and row["bbox_area"] <= 9
    ]
    for row in sorted(bottom_small, key=lambda value: (value["centroid"][1], value["bbox_area"], abs(value["centroid"][0] - center_x))):
        poi_id = str(row["poi"].get("entity_id") or row["poi"].get("poi_id") or "")
        if poi_id and poi_id not in roles:
            roles[poi_id] = "trigger"
    large_mid = [
        row for row in rows
        if row["bbox_area"] >= 18
        and row["centroid"][1] >= (height * 0.25)
        and row["centroid"][1] <= (height * 0.85)
    ]
    for row in sorted(large_mid, key=lambda value: (-value["bbox_area"], abs(value["centroid"][0] - center_x))):
        poi_id = str(row["poi"].get("entity_id") or row["poi"].get("poi_id") or "")
        if poi_id and poi_id not in roles:
            roles[poi_id] = "gate"
            break
    medium_mid = [
        row for row in rows
        if 2 <= row["bbox_area"] <= 12
        and row["centroid"][1] >= (height * 0.2)
        and row["centroid"][1] <= (height * 0.7)
    ]
    for row in sorted(medium_mid, key=lambda value: (abs(value["centroid"][0] - center_x), value["bbox_area"])):
        poi_id = str(row["poi"].get("entity_id") or row["poi"].get("poi_id") or "")
        if poi_id and poi_id not in roles:
            roles[poi_id] = "panel"
            break
    return roles


def _node_from_poi(poi: dict, *, round_id: int, episode_id: str) -> dict:
    node_kind = str(poi.get("node_kind") or _infer_node_kind(poi))
    node_id = f"mg:{node_kind}:{poi.get('poi_id') or poi.get('entity_id') or stable_digest(poi.get('signature') or poi)}"
    evidence_tier = str(poi.get("evidence_tier") or "")
    if evidence_tier not in {"observed", "hypothesized"}:
        evidence_tier = (
            "observed"
            if bool(poi.get("factual_observation")) and bool(poi.get("direct_evidence_present"))
            else ("observed" if float(poi.get("confidence", 0.0) or 0.0) >= 0.5 else "hypothesized")
        )
    support_count = max(1, int(poi.get("observations", 1) or 1))
    entity_ref = str(poi.get("entity_id") or "")
    stable_entity_hint = str(poi.get("stable_entity_id_hint") or "")
    object_ref = entity_ref
    if not object_ref or object_ref.startswith("object:"):
        object_ref = stable_entity_hint or object_ref
    if not object_ref:
        object_ref = str(poi.get("poi_id") or poi.get("signature") or "")
    return {
        "node_id": node_id,
        "semantic_key": node_id,
        "node_kind": node_kind,
        "evidence_tier": evidence_tier,
        "confidence": float(poi.get("confidence", 0.0) or 0.0),
        "source_episode_ids": [str(episode_id)],
        "source_round_ids": [int(round_id)],
        "support_count": support_count,
        "contradiction_count": 0,
        "first_seen_round": int(round_id),
        "last_seen_round": int(round_id),
        "object_ref": object_ref,
        "pattern_id": str(poi.get("pattern_id") or ""),
        "source_entity_id": entity_ref or None,
        "identity_confidence": float(poi.get("identity_confidence", 0.0) or 0.0),
        "identity_status": str(poi.get("identity_status") or "unknown"),
        "identity_history": tuple(
            {
                "matched_prior_id": poi.get("matched_prior_id"),
                "candidate_prior_ids": list(poi.get("candidate_prior_ids", []) or []),
                "round_id": int(round_id),
            }
            for _ in [0]
        ),
        "object_backed": True,
        "synthetic_region_only": False,
        "support_round_count": 1,
        "observed_support_count": support_count if evidence_tier == "observed" else 0,
        "exit_link_support_count": 0,
        "counterfactual_support_count": 0,
        "metadata": {
            "area_id": poi.get("area_id"),
            "centroid": poi.get("centroid"),
            "descriptor": dict(poi.get("pattern_descriptor", {}) or {}),
            "bbox": _bbox_dict(poi),
            "identity_confidence": float(poi.get("identity_confidence", 0.0) or 0.0),
            "identity_status": str(poi.get("identity_status") or "unknown"),
        },
    }


def _effect_region_node(effect_row: dict, *, round_id: int, episode_id: str) -> dict:
    bbox = list(effect_row.get("bbox", []))
    node_id = f"mg:effect_region:{stable_digest((bbox, round_id, effect_row.get('step_idx')))}"
    return {
        "node_id": node_id,
        "semantic_key": f"effect_region:{bbox}",
        "node_kind": "effect_region",
        "evidence_tier": "observed",
        "confidence": min(1.0, 0.4 + (0.02 * int(effect_row.get("changed_cells", 0) or 0))),
        "source_episode_ids": [str(episode_id)],
        "source_round_ids": [int(round_id)],
        "support_count": 1,
        "contradiction_count": 0,
        "first_seen_round": int(round_id),
        "last_seen_round": int(round_id),
        "metadata": {"bbox": bbox, "step_idx": int(effect_row.get("step_idx", 0) or 0)},
    }


def _trigger_zone_node(zone: dict, *, round_id: int, episode_id: str) -> dict:
    zone_id = str(zone.get("trigger_zone_id") or zone.get("trigger_id") or stable_digest(zone))
    bbox = _bbox_dict(zone)
    evidence_tier = str(zone.get("evidence_tier") or "")
    if evidence_tier not in {"observed", "hypothesized"}:
        evidence_tier = "observed" if bool(zone.get("factual_observation")) and bool(zone.get("direct_evidence_present")) else "hypothesized"
    support_count = max(1, int(zone.get("support_count", zone.get("observations", 1)) or 1))
    entity_id = str(zone.get("entity_id") or "")
    return {
        "node_id": f"mg:trigger:{zone_id}",
        "semantic_key": f"trigger:{zone_id}",
        "node_kind": "trigger",
        "evidence_tier": evidence_tier,
        "confidence": float(zone.get("confidence", 0.0) or 0.0),
        "source_episode_ids": [str(episode_id)],
        "source_round_ids": [int(round_id)],
        "support_count": support_count,
        "contradiction_count": int(zone.get("contradiction_count", 0) or 0),
        "first_seen_round": int(round_id),
        "last_seen_round": int(round_id),
        "object_ref": entity_id or zone_id,
        "source_entity_id": entity_id or None,
        "identity_confidence": float(zone.get("identity_confidence", 0.0) or 0.0),
        "identity_status": str(zone.get("identity_status") or ("match_existing" if entity_id else "unknown")),
        "identity_history": tuple(),
        "object_backed": bool(entity_id),
        "synthetic_region_only": not bool(entity_id),
        "support_round_count": 1,
        "observed_support_count": support_count if evidence_tier == "observed" else 0,
        "exit_link_support_count": 0,
        "counterfactual_support_count": 0,
        "metadata": {
            "centroid": list(zone.get("centroid", _bbox_centroid(bbox)) or _bbox_centroid(bbox)),
            "bbox": bbox,
            "trigger_zone_id": zone_id,
            "trigger_kind": str(zone.get("trigger_kind") or ""),
            "region_backed": bool(zone.get("region_backed", False)),
            "trigger_evidence_class": str(zone.get("trigger_evidence_class") or ("object_backed" if entity_id else "region_suspicion")),
        },
    }


def _avatar_confident(row: dict, *, threshold: float = 0.6) -> bool:
    telemetry = dict(row.get("telemetry", {}) or {})
    return bool(float(telemetry.get("avatar_confidence", 0.0) or 0.0) >= threshold and not bool(telemetry.get("avatar_ambiguous", False)))


def _heuristic_trigger_chainworthy(node: dict) -> bool:
    return bool(node.get("object_backed", False))


def extract_mechanic_graph_delta(
    raw_episode: RawEpisode,
    analyzed_episode: AnalyzedEpisode,
    current_blackboard_snapshot: dict | None = None,
    current_mechanic_graph_snapshot: dict | None = None,
    hypothesis_config: object | None = None,
    llm_adapter: object | None = None,
    hypothesis_registry_snapshot: dict | None = None,
) -> MechanicGraphDelta:
    del current_mechanic_graph_snapshot, hypothesis_config, llm_adapter, hypothesis_registry_snapshot
    summary = dict(analyzed_episode.summary or {})
    step_rows = list(summary.get("step_rows", []) or [])
    pois = [dict(row) for row in list(analyzed_episode.points_of_interest or [])]
    entity_rows: list[dict] = []
    trigger_zone_rows: list[dict] = []
    for delta in list(analyzed_episode.blackboard_deltas or []):
        entity_rows.extend(dict(row) for row in list(getattr(delta, "entities", ()) or ()))
        trigger_zone_rows.extend(dict(row) for row in list(getattr(delta, "trigger_zones", ()) or ()))
    if entity_rows:
        entity_by_id: dict[str, dict] = {}
        for row in entity_rows:
            entity_id = str(row.get("entity_id") or row.get("poi_id") or "")
            if not entity_id:
                continue
            prior = entity_by_id.get(entity_id)
            if prior is None or float(row.get("confidence", 0.0) or 0.0) >= float(prior.get("confidence", 0.0) or 0.0):
                entity_by_id[entity_id] = row
        pois_by_id = {str(row.get("entity_id") or row.get("poi_id") or ""): dict(row) for row in pois}
        merged_rows: list[dict] = []
        for entity_id, row in entity_by_id.items():
            merged_rows.append({**pois_by_id.get(entity_id, {}), **row})
        for entity_id, row in pois_by_id.items():
            if entity_id and entity_id not in entity_by_id:
                merged_rows.append(dict(row))
        pois = merged_rows
    width = int(summary.get("width", 64) or 64)
    height = int(summary.get("height", 64) or 64)
    heuristic_roles = _heuristic_node_roles(pois, width=width, height=height)
    pois = [{**poi, "node_kind": heuristic_roles.get(str(poi.get("entity_id") or poi.get("poi_id") or "")) or _infer_node_kind(poi)} for poi in pois]
    nodes = [_node_from_poi(poi, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id) for poi in pois]
    nodes.extend(_trigger_zone_node(row, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id) for row in trigger_zone_rows)
    nodes_by_object = {str(node.get("object_ref") or node.get("node_id")): node for node in nodes}
    edges: list[dict] = []

    avatar_confident_rows = [row for row in list(step_rows or []) if _avatar_confident(dict(row))]
    for effect_row in remote_region_change_evidence(step_rows):
        effect_node = _effect_region_node(effect_row, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id)
        nodes.append(effect_node)
        contact_rows = [row for row in trigger_contact_evidence(avatar_confident_rows) if int(row.get("step_idx", -1)) <= int(effect_row.get("step_idx", 0))]
        if contact_rows:
            contact = contact_rows[-1]
            src_node = nodes_by_object.get(str(contact.get("target_entity_id") or ""))
            if src_node:
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((src_node['node_id'], 'changes', effect_node['node_id']))}",
                        "src_node_id": src_node["node_id"],
                        "edge_kind": "changes",
                        "dst_node_id": effect_node["node_id"],
                        "condition_key": f"step:{int(contact.get('step_idx', 0) or 0)}",
                        "evidence_tier": "observed",
                        "confidence": min(1.0, 0.45 + (0.02 * int(effect_row.get("changed_cells", 0) or 0))),
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 1,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": True,
                        "metadata": {"evidence_refs": [effect_row.get("evidence_ref")]},
                    }
                )

    for effect_row in repeated_effect_region_support(step_rows):
        effect_node = _effect_region_node(effect_row, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id)
        nodes.append(effect_node)

    by_pattern: dict[str, list[dict]] = {}
    for node in list(nodes):
        pattern_id = str(node.get("pattern_id") or "")
        if pattern_id:
            symbol_node_id = f"mg:symbol_state:{pattern_id}"
            nodes.append(
                {
                    "node_id": symbol_node_id,
                    "semantic_key": f"symbol:{pattern_id}",
                    "node_kind": "symbol_state",
                    "evidence_tier": node.get("evidence_tier", "hypothesized"),
                    "confidence": float(node.get("confidence", 0.0) or 0.0),
                    "source_episode_ids": [raw_episode.episode_id],
                    "source_round_ids": [raw_episode.round_id],
                    "support_count": 1,
                    "contradiction_count": 0,
                    "first_seen_round": raw_episode.round_id,
                    "last_seen_round": raw_episode.round_id,
                    "pattern_id": pattern_id,
                    "metadata": dict(node.get("metadata", {})),
                }
            )
            edges.append(
                {
                    "edge_id": f"mg_edge:{stable_digest((node['node_id'], 'displays', symbol_node_id))}",
                    "src_node_id": node["node_id"],
                    "edge_kind": "displays",
                    "dst_node_id": symbol_node_id,
                    "condition_key": pattern_id,
                    "evidence_tier": node.get("evidence_tier", "hypothesized"),
                    "confidence": float(node.get("confidence", 0.0) or 0.0),
                    "source_episode_ids": [raw_episode.episode_id],
                    "source_round_ids": [raw_episode.round_id],
                    "support_count": 1,
                    "contradiction_count": 0,
                    "first_seen_round": raw_episode.round_id,
                    "last_seen_round": raw_episode.round_id,
                    "direct_support_present": str(node.get("evidence_tier") or "") == "observed",
                }
            )
            by_pattern.setdefault(pattern_id, []).append(node)

    for pattern_id, rows in by_pattern.items():
        if len(rows) < 2:
            continue
        left, right = rows[0], rows[1]
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((left['node_id'], 'matches', right['node_id'], pattern_id))}",
                "src_node_id": left["node_id"],
                "edge_kind": "matches",
                "dst_node_id": right["node_id"],
                "condition_key": pattern_id,
                "evidence_tier": "hypothesized",
                "confidence": 0.55,
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": len(rows),
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
            }
        )

    support_rows = repeated_contact_to_change_support(avatar_confident_rows)
    exits = [node for node in nodes if str(node.get("node_kind") or "") == "exit"]
    gates = [node for node in nodes if str(node.get("node_kind") or "") == "gate"]
    panels = [node for node in nodes if str(node.get("node_kind") or "") == "panel"]
    triggers = [node for node in nodes if str(node.get("node_kind") or "") == "trigger"]
    for support_row in support_rows:
        trigger_node = nodes_by_object.get(str(support_row.get("target_entity_id") or ""))
        if trigger_node is None or not exits:
            continue
        exit_node = exits[0]
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'requires', exit_node['node_id']))}",
                "src_node_id": trigger_node["node_id"],
                "edge_kind": "requires",
                "dst_node_id": exit_node["node_id"],
                "condition_key": "requires_before_exit",
                "evidence_tier": "hypothesized",
                "confidence": min(0.8, 0.35 + (0.08 * int(support_row.get("changed_support_count", 0) or 0))),
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": int(support_row.get("support_count", 0) or 0),
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
                "metadata": {"evidence_refs": list(support_row.get("evidence_refs", []))},
            }
        )

    if triggers and exits:
        target_exit = exits[0]
        for trigger_node in triggers:
            if not _heuristic_trigger_chainworthy(trigger_node):
                continue
            if gates:
                primary_gate = sorted(
                    gates,
                    key=lambda row: (
                        abs(float(dict(row.get("metadata", {})).get("centroid", [0, 0])[0]) - float(dict(trigger_node.get("metadata", {})).get("centroid", [0, 0])[0])),
                        -int(row.get("support_count", 0) or 0),
                    ),
                )[0]
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'requires', primary_gate['node_id']))}",
                        "src_node_id": trigger_node["node_id"],
                        "edge_kind": "requires",
                        "dst_node_id": primary_gate["node_id"],
                        "condition_key": "heuristic_trigger_gate",
                        "evidence_tier": "hypothesized",
                        "confidence": 0.62,
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 2,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": False,
                        "counterfactual_support_count": 1 if bool(trigger_node.get("object_backed", False) or int(trigger_node.get("observed_support_count", 0) or 0) > 0) else 0,
                    }
                )
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((primary_gate['node_id'], 'controls_access', target_exit['node_id']))}",
                        "src_node_id": primary_gate["node_id"],
                        "edge_kind": "controls_access",
                        "dst_node_id": target_exit["node_id"],
                        "condition_key": "heuristic_gate_exit",
                        "evidence_tier": "hypothesized",
                        "confidence": 0.64,
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 2,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": False,
                        "directed_outcome_support_count": 1,
                    }
                )
            else:
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'requires', target_exit['node_id']))}",
                        "src_node_id": trigger_node["node_id"],
                        "edge_kind": "requires",
                        "dst_node_id": target_exit["node_id"],
                        "condition_key": "heuristic_trigger_exit",
                        "evidence_tier": "hypothesized",
                        "confidence": 0.58,
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 2,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": False,
                        "counterfactual_support_count": 1 if bool(trigger_node.get("object_backed", False) or int(trigger_node.get("observed_support_count", 0) or 0) > 0) else 0,
                    }
                )
        if panels and gates:
            for panel in panels[:2]:
                gate = sorted(
                    gates,
                    key=lambda row: abs(float(dict(row.get("metadata", {})).get("centroid", [0, 0])[0]) - float(dict(panel.get("metadata", {})).get("centroid", [0, 0])[0])),
                )[0]
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((panel['node_id'], 'matches', gate['node_id']))}",
                        "src_node_id": panel["node_id"],
                        "edge_kind": "matches",
                        "dst_node_id": gate["node_id"],
                        "condition_key": "heuristic_panel_gate_match",
                        "evidence_tier": "hypothesized",
                        "confidence": 0.57,
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 2,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": False,
                    }
                )

    for delayed in delayed_change_evidence(avatar_confident_rows):
        trigger_node = nodes_by_object.get(str(delayed.get("target_entity_id") or ""))
        if trigger_node is None:
            continue
        effect_node = next((node for node in nodes if str(node.get("node_kind") or "") == "effect_region" and int(dict(node.get("metadata", {})).get("step_idx", -1)) == int(delayed.get("effect_step_idx", -1))), None)
        if effect_node is None:
            continue
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'causes_remote_change', effect_node['node_id']))}",
                "src_node_id": trigger_node["node_id"],
                "edge_kind": "causes_remote_change",
                "dst_node_id": effect_node["node_id"],
                "condition_key": f"delay:{delayed.get('effect_step_idx')}",
                "evidence_tier": "hypothesized",
                "confidence": 0.5,
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": 1,
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
            }
        )

    current_entities = dict(dict(current_blackboard_snapshot or {}).get("entities", {}) or {})
    for step in list(step_rows or []):
        target_entity_id = str(step.get("target_entity_id") or "")
        if not target_entity_id:
            continue
        target_row = dict(current_entities.get(target_entity_id, {}) or {})
        provenance = {str(value) for value in list(target_row.get("poi_source_provenance", []) or []) if value}
        if str(target_row.get("kind") or "") != "poi" or "detector" not in provenance:
            continue
        poi_node = nodes_by_object.get(target_entity_id)
        if poi_node is None:
            continue
        changed_cells = int(step.get("changed_cells", 0) or 0)
        if changed_cells > 0:
            effect_node = _effect_region_node(dict(step.get("telemetry", {}) or {}).get("effect_region", {}) or {"step_idx": int(step.get("step_idx", 0) or 0), "changed_cells": changed_cells, "bbox": dict(dict(step.get("telemetry", {}) or {}).get("effect_region", {}) or {}).get("bbox", [])}, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id)
            nodes.append(effect_node)
            edges.append(
                {
                    "edge_id": f"mg_edge:{stable_digest((poi_node['node_id'], 'causes_remote_change', effect_node['node_id'], 'poi_visit'))}",
                    "src_node_id": poi_node["node_id"],
                    "edge_kind": "causes_remote_change",
                    "dst_node_id": effect_node["node_id"],
                    "condition_key": f"poi_visit:{target_entity_id}",
                    "evidence_tier": "hypothesized",
                    "confidence": 0.56,
                    "source_episode_ids": [raw_episode.episode_id],
                    "source_round_ids": [raw_episode.round_id],
                    "support_count": 1,
                    "contradiction_count": 0,
                    "first_seen_round": raw_episode.round_id,
                    "last_seen_round": raw_episode.round_id,
                    "direct_support_present": False,
                    "poi_visit_support_count": 1,
                    "post_visit_remote_change_count": 1,
                    "directed_outcome_support_count": 1 if bool(step.get("trigger_contact_based_on_confident_avatar")) else 0,
                    "lag_consistency_score": 1.0,
                    "support_consistency_score": 0.6,
                }
            )
        exit_failed_without_new_support = bool(step.get("exit_attempt_failed_without_new_support", False))
        if str(step.get("termination_reason") or "") in {"done", "blocked", "exit_failure"} and not exit_failed_without_new_support:
            exit_node = next((node for node in nodes if str(node.get("node_kind") or "") == "exit"), None)
            if exit_node is not None:
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((poi_node['node_id'], 'requires', exit_node['node_id'], 'poi_visit_exit'))}",
                        "src_node_id": poi_node["node_id"],
                        "edge_kind": "requires",
                        "dst_node_id": exit_node["node_id"],
                        "condition_key": f"poi_exit:{target_entity_id}",
                        "evidence_tier": "hypothesized",
                        "confidence": 0.52,
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 1,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": False,
                        "poi_visit_support_count": 1,
                        "post_visit_exit_effect_count": 1,
                        "directed_outcome_support_count": 1 if bool(step.get("new_support_since_previous_exit_attempt", False)) else 0,
                        "exit_attempt_support_count": 1,
                        "support_consistency_score": 0.55,
                    }
                )

    delta = MechanicGraphDelta(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        delta_id=f"mechanic_graph_delta:{raw_episode.episode_id}:{stable_digest((nodes, edges))}",
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )
    return delta
