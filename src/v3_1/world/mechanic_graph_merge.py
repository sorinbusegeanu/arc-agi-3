from __future__ import annotations

from typing import Any

from v3_1.utils.ids import stable_digest
from v3_1.world.mechanic_graph import build_mechanic_graph_indexes


def _unique_strings(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in list(values or []) if value))


def _stable_node_id(row: dict) -> str:
    return str(row.get("node_id") or f"mechanic_node:{stable_digest(row.get('semantic_key') or row)}")


def _stable_edge_id(row: dict) -> str:
    src = str(row.get("src_node_id") or "")
    dst = str(row.get("dst_node_id") or "")
    edge_kind = str(row.get("edge_kind") or "")
    condition_key = str(row.get("condition_key") or "")
    return str(row.get("edge_id") or f"mechanic_edge:{stable_digest((src, edge_kind, dst, condition_key))}")


def _merge_tier(existing: str, incoming: str, *, direct_support_present: bool) -> str:
    existing_norm = str(existing or "hypothesized")
    incoming_norm = str(incoming or "hypothesized")
    if existing_norm == "observed":
        return "observed"
    if incoming_norm == "observed" and direct_support_present:
        return "observed"
    return incoming_norm if incoming_norm else existing_norm


def _merge_node(existing: dict | None, incoming: dict, *, round_id: int) -> dict:
    existing = dict(existing or {})
    incoming = dict(incoming)
    support_increment = max(1, int(incoming.get("support_count", 1) or 1))
    contradiction_increment = int(incoming.get("contradiction_count", 0) or 0)
    direct_support_present = bool(incoming.get("direct_support_present") or str(incoming.get("evidence_tier") or "") == "observed")
    merged = {
        **existing,
        **incoming,
        "node_id": _stable_node_id(incoming),
        "evidence_tier": _merge_tier(existing.get("evidence_tier", "hypothesized"), incoming.get("evidence_tier", "hypothesized"), direct_support_present=direct_support_present),
        "confidence": max(float(existing.get("confidence", 0.0) or 0.0), float(incoming.get("confidence", 0.0) or 0.0)),
        "source_episode_ids": _unique_strings(list(existing.get("source_episode_ids", ())) + list(incoming.get("source_episode_ids", ()))),
        "source_round_ids": tuple(sorted({int(value) for value in list(existing.get("source_round_ids", ())) + list(incoming.get("source_round_ids", ())) if value is not None})),
        "support_count": int(existing.get("support_count", 0) or 0) + support_increment,
        "contradiction_count": int(existing.get("contradiction_count", 0) or 0) + contradiction_increment,
        "first_seen_round": min(int(existing.get("first_seen_round", round_id) or round_id), int(incoming.get("first_seen_round", round_id) or round_id)),
        "last_seen_round": max(int(existing.get("last_seen_round", 0) or 0), int(incoming.get("last_seen_round", round_id) or round_id), int(round_id)),
        "object_backed": bool(existing.get("object_backed", False) or incoming.get("object_backed", False)),
        "synthetic_region_only": bool(existing.get("synthetic_region_only", False) and incoming.get("synthetic_region_only", False)) if existing else bool(incoming.get("synthetic_region_only", False)),
        "observed_support_count": int(existing.get("observed_support_count", 0) or 0) + (support_increment if str(incoming.get("evidence_tier") or "") == "observed" else 0),
        "exit_link_support_count": int(existing.get("exit_link_support_count", 0) or 0) + int(incoming.get("exit_link_support_count", 0) or 0),
        "counterfactual_support_count": int(existing.get("counterfactual_support_count", 0) or 0) + int(incoming.get("counterfactual_support_count", 0) or 0),
    }
    merged["support_round_count"] = len({int(value) for value in list(merged.get("source_round_ids", ())) if value is not None})
    return merged


def _merge_edge(existing: dict | None, incoming: dict, *, round_id: int) -> dict:
    existing = dict(existing or {})
    incoming = dict(incoming)
    support_increment = max(1, int(incoming.get("support_count", 1) or 1))
    contradiction_increment = int(incoming.get("contradiction_count", 0) or 0)
    incoming_tier = str(incoming.get("evidence_tier") or "hypothesized")
    direct_support_present = bool(incoming.get("direct_support_present") or incoming_tier == "observed")
    existing_tier = str(existing.get("evidence_tier") or "hypothesized")
    effective_tier = _merge_tier(existing_tier, incoming_tier, direct_support_present=direct_support_present)
    if existing_tier != "observed" and incoming_tier == "observed" and not direct_support_present:
        effective_tier = existing_tier
    observed_support = int(existing.get("observed_support_count", 0) or 0) + (support_increment if incoming_tier == "observed" and direct_support_present else 0)
    hypothesized_support = int(existing.get("hypothesized_support_count", 0) or 0) + (support_increment if incoming_tier != "observed" or not direct_support_present else 0)
    confidence = max(float(existing.get("confidence", 0.0) or 0.0), float(incoming.get("confidence", 0.0) or 0.0))
    if contradiction_increment > 0:
        confidence = max(0.0, confidence - min(0.5, 0.08 * contradiction_increment))
    elif support_increment > 0:
        confidence = min(1.0, confidence + min(0.25, 0.03 * support_increment))
    edge_kind = str(incoming.get("edge_kind") or existing.get("edge_kind") or "")
    support_consistency_score = min(1.0, float(existing.get("support_consistency_score", 0.0) or 0.0) + (0.2 if support_increment > 0 else 0.0))
    lag_consistency_score = max(float(existing.get("lag_consistency_score", 1.0) or 1.0), float(incoming.get("lag_consistency_score", 1.0) or 1.0))
    counterfactual_support_count = int(existing.get("counterfactual_support_count", 0) or 0) + int(incoming.get("counterfactual_support_count", 0) or 0)
    directed_outcome_support_count = int(existing.get("directed_outcome_support_count", 0) or 0) + int(incoming.get("directed_outcome_support_count", 0) or 0)
    poi_visit_support_count = int(existing.get("poi_visit_support_count", 0) or 0) + int(incoming.get("poi_visit_support_count", 0) or 0)
    post_visit_remote_change_count = int(existing.get("post_visit_remote_change_count", 0) or 0) + int(incoming.get("post_visit_remote_change_count", 0) or 0)
    post_visit_panel_match_count = int(existing.get("post_visit_panel_match_count", 0) or 0) + int(incoming.get("post_visit_panel_match_count", 0) or 0)
    post_visit_exit_effect_count = int(existing.get("post_visit_exit_effect_count", 0) or 0) + int(incoming.get("post_visit_exit_effect_count", 0) or 0)
    if edge_kind == "matches" and support_increment < 2:
        confidence = min(confidence, 0.55)
    if edge_kind == "requires" and counterfactual_support_count <= 0 and support_increment < 2:
        confidence = min(confidence, 0.5)
    if edge_kind == "controls_access" and directed_outcome_support_count <= 0:
        confidence = min(confidence, 0.55)
    if edge_kind == "causes_remote_change" and lag_consistency_score < 0.5:
        confidence = min(confidence, 0.5)
    return {
        **existing,
        **incoming,
        "edge_id": _stable_edge_id(incoming),
        "evidence_tier": effective_tier,
        "confidence": confidence,
        "source_episode_ids": _unique_strings(list(existing.get("source_episode_ids", ())) + list(incoming.get("source_episode_ids", ()))),
        "source_round_ids": tuple(sorted({int(value) for value in list(existing.get("source_round_ids", ())) + list(incoming.get("source_round_ids", ())) if value is not None})),
        "support_count": int(existing.get("support_count", 0) or 0) + support_increment,
        "contradiction_count": int(existing.get("contradiction_count", 0) or 0) + contradiction_increment,
        "first_seen_round": min(int(existing.get("first_seen_round", round_id) or round_id), int(incoming.get("first_seen_round", round_id) or round_id)),
        "last_seen_round": max(int(existing.get("last_seen_round", 0) or 0), int(incoming.get("last_seen_round", round_id) or round_id), int(round_id)),
        "observed_support_count": observed_support,
        "hypothesized_support_count": hypothesized_support,
        "origin_provenance": str(existing.get("origin_provenance") or incoming.get("origin_provenance") or incoming.get("provenance") or ""),
        "supporting_hypothesis_ids": list(dict.fromkeys([*list(existing.get("supporting_hypothesis_ids", []) or []), *list(incoming.get("supporting_hypothesis_ids", []) or [])])),
        "validated_from_hypothesis_id": existing.get("validated_from_hypothesis_id") or incoming.get("validated_from_hypothesis_id"),
        "validation_round_ids": sorted({int(value) for value in [*list(existing.get("validation_round_ids", []) or []), *list(incoming.get("validation_round_ids", []) or [])] if value is not None}),
        "support_consistency_score": support_consistency_score,
        "lag_consistency_score": lag_consistency_score,
        "counterfactual_support_count": counterfactual_support_count,
        "directed_outcome_support_count": directed_outcome_support_count,
        "poi_visit_support_count": poi_visit_support_count,
        "post_visit_remote_change_count": post_visit_remote_change_count,
        "post_visit_panel_match_count": post_visit_panel_match_count,
        "post_visit_exit_effect_count": post_visit_exit_effect_count,
    }


def _matching_hypothesis_ids(edge: dict, registry_snapshot: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    registry = dict(registry_snapshot or {})
    deterministic = []
    llm = []
    signature = (
        str(edge.get("src_node_id") or ""),
        str(edge.get("edge_kind") or ""),
        str(edge.get("dst_node_id") or ""),
    )
    for proposal_id, row in dict(registry.get("deterministic_proposals", {})).items():
        if (
            str(row.get("src_node_id") or "") == signature[0]
            and str(row.get("edge_kind", row.get("path_kind", "")) or "") == signature[1]
            and str(row.get("dst_node_id") or "") == signature[2]
        ):
            deterministic.append(str(proposal_id))
    for proposal_id, row in dict(registry.get("llm_proposals", {})).items():
        if (
            str(row.get("src_node_id") or "") == signature[0]
            and str(row.get("edge_kind", row.get("path_kind", "")) or "") == signature[1]
            and str(row.get("dst_node_id") or "") == signature[2]
        ):
            llm.append(str(proposal_id))
    return deterministic, llm


def _eligible_exit_link_feedback(src: dict, edge: dict) -> bool:
    if not src:
        return False
    if str(edge.get("evidence_tier") or "") == "observed":
        return True
    if bool(src.get("object_backed", False)):
        return True
    if int(src.get("observed_support_count", 0) or 0) > 0:
        return True
    if not bool(src.get("synthetic_region_only", False)) and float(edge.get("confidence", 0.0) or 0.0) >= 0.75:
        return True
    return False


def merge_mechanic_graph_delta(state: dict[str, Any], delta: dict[str, Any], registry_snapshot: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(state or {})
    nodes_by_id = {str(node_id): dict(row) for node_id, row in dict(payload.get("nodes_by_id", {})).items()}
    edges_by_id = {str(edge_id): dict(row) for edge_id, row in dict(payload.get("edges_by_id", {})).items()}
    round_id = int(delta.get("round_id", 0) or 0)
    node_count_before = len(nodes_by_id)
    edge_count_before = len(edges_by_id)
    observed_edge_count_before = sum(1 for row in edges_by_id.values() if str(row.get("evidence_tier") or "") == "observed")
    feedback = {
        "supported_proposal_ids": [],
        "contradicted_proposal_ids": [],
        "validated_proposal_ids": [],
    }

    for row in list(delta.get("nodes", ()) or []):
        semantic_key = str(row.get("semantic_key") or row.get("node_id") or "")
        existing_key = next((node_id for node_id, node in nodes_by_id.items() if str(node.get("semantic_key") or node_id) == semantic_key), None)
        merged = _merge_node(nodes_by_id.get(existing_key), row, round_id=round_id)
        nodes_by_id[str(merged["node_id"])] = merged
        if existing_key and existing_key != str(merged["node_id"]):
            nodes_by_id.pop(existing_key, None)

    for row in list(delta.get("edges", ()) or []):
        condition_key = str(row.get("condition_key") or "")
        existing_key = next(
            (
                edge_id
                for edge_id, edge in edges_by_id.items()
                if (
                    str(edge.get("src_node_id") or "") == str(row.get("src_node_id") or "")
                    and str(edge.get("edge_kind") or "") == str(row.get("edge_kind") or "")
                    and str(edge.get("dst_node_id") or "") == str(row.get("dst_node_id") or "")
                    and str(edge.get("condition_key") or "") == condition_key
                )
            ),
            None,
        )
        merged = _merge_edge(edges_by_id.get(existing_key), row, round_id=round_id)
        deterministic_ids, llm_ids = _matching_hypothesis_ids(merged, registry_snapshot)
        matched_ids = [*deterministic_ids, *llm_ids]
        if matched_ids:
            merged["supporting_hypothesis_ids"] = list(dict.fromkeys([*list(merged.get("supporting_hypothesis_ids", []) or []), *matched_ids]))
            if str(merged.get("evidence_tier") or "") == "observed":
                feedback["supported_proposal_ids"].extend(matched_ids)
            if int(merged.get("contradiction_count", 0) or 0) > 0 or str(merged.get("edge_kind") or "") == "contradicts":
                feedback["contradicted_proposal_ids"].extend(matched_ids)
            if bool(merged.get("direct_support_present")) and float(merged.get("confidence", 0.0) or 0.0) >= 0.8:
                merged["validated_from_hypothesis_id"] = merged.get("validated_from_hypothesis_id") or (matched_ids[0] if matched_ids else None)
                merged["validation_round_ids"] = sorted({*list(merged.get("validation_round_ids", []) or []), round_id})
                feedback["validated_proposal_ids"].extend(matched_ids)
        edges_by_id[str(merged["edge_id"])] = merged
        if existing_key and existing_key != str(merged["edge_id"]):
            edges_by_id.pop(existing_key, None)

    for edge in edges_by_id.values():
        src_id = str(edge.get("src_node_id") or "")
        dst_id = str(edge.get("dst_node_id") or "")
        src = dict(nodes_by_id.get(src_id, {}))
        dst = dict(nodes_by_id.get(dst_id, {}))
        if (
            str(edge.get("edge_kind") or "") in {"requires", "controls_access"}
            and str(dst.get("node_kind") or "") == "exit"
            and src
            and _eligible_exit_link_feedback(src, edge)
        ):
            src["exit_link_support_count"] = int(src.get("exit_link_support_count", 0) or 0) + int(edge.get("support_count", 0) or 0)
            src["counterfactual_support_count"] = max(int(src.get("counterfactual_support_count", 0) or 0), int(edge.get("counterfactual_support_count", 0) or 0))
            nodes_by_id[src_id] = src

    indexes = build_mechanic_graph_indexes(nodes_by_id, edges_by_id)
    next_state = {
        **payload,
        "nodes_by_id": nodes_by_id,
        "edges_by_id": edges_by_id,
        **indexes,
    }
    observed_edge_count_after = sum(1 for row in edges_by_id.values() if str(row.get("evidence_tier") or "") == "observed")
    return next_state, {
        "node_count_added": max(0, len(nodes_by_id) - node_count_before),
        "edge_count_added": max(0, len(edges_by_id) - edge_count_before),
        "observed_edge_count_added": max(0, observed_edge_count_after - observed_edge_count_before),
        "hypothesized_edge_count_added": max(0, (len(edges_by_id) - observed_edge_count_after) - (edge_count_before - observed_edge_count_before)),
        "registry_feedback": {
            "supported_proposal_ids": sorted(set(feedback["supported_proposal_ids"])),
            "contradicted_proposal_ids": sorted(set(feedback["contradicted_proposal_ids"])),
            "validated_proposal_ids": sorted(set(feedback["validated_proposal_ids"])),
        },
    }
