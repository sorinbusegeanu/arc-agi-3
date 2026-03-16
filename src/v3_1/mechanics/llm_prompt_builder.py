from __future__ import annotations

from collections import defaultdict
from typing import Any

from v3_1.mechanics.hypothesis_types import HypothesisBundle
from v3_1.mechanics.llm_schema import LLMHypothesisInput

DEFAULT_MAX_NODES = 16
DEFAULT_MAX_EDGES = 24
DEFAULT_MAX_PATHS = 5
DEFAULT_MAX_CONTRADICTIONS = 5
DEFAULT_MAX_EXIT_ATTEMPTS = 5
DEFAULT_MAX_PATTERN_RELATIONS = 5
DEFAULT_MAX_ALLOWED_NODE_IDS = 24
DEFAULT_MAX_K_HOPS = 2

ALLOWED_EDGE_KINDS = (
    "changes",
    "displays",
    "matches",
    "controls_access",
    "opens",
    "requires",
    "causes_remote_change",
    "enables_exit",
    "contradicts",
)
ALLOWED_PATH_KINDS = (
    "trigger_then_exit",
    "unlock_then_exit",
    "panel_gate_exit",
    "state_sync_probe",
    "prerequisite_chain",
)


def _graph_state(snapshot: dict | None) -> dict[str, Any]:
    return dict((snapshot or {}).get("state", snapshot or {}))


def _node_rows(snapshot: dict | None) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in dict(_graph_state(snapshot).get("nodes_by_id", {})).items()}


def _edge_rows(snapshot: dict | None) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in dict(_graph_state(snapshot).get("edges_by_id", {})).items()}


def _adjacency(snapshot: dict | None) -> dict[str, list[str]]:
    graph_state = _graph_state(snapshot)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for node_id, edge_ids in dict(graph_state.get("adjacency_out", {})).items():
        adjacency[str(node_id)].extend(str(edge_id) for edge_id in list(edge_ids or []))
    for edge_id, edge in _edge_rows(snapshot).items():
        src = str(edge.get("src_node_id") or "")
        dst = str(edge.get("dst_node_id") or "")
        if src and edge_id not in adjacency[src]:
            adjacency[src].append(edge_id)
        if dst and edge_id not in adjacency[dst]:
            adjacency[dst].append(edge_id)
    return adjacency


def _compact_node_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(row.get("node_id") or ""),
        "node_kind": str(row.get("node_kind") or ""),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "evidence_tier": str(row.get("evidence_tier") or "hypothesized"),
        "support_count": int(row.get("support_count", 0) or 0),
        "contradiction_count": int(row.get("contradiction_count", 0) or 0),
        "last_updated_round": row.get("last_updated_round"),
        "source_provenance": str(row.get("source_provenance") or ""),
    }


def _compact_edge_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "src_node_id": str(row.get("src_node_id") or ""),
        "edge_kind": str(row.get("edge_kind") or ""),
        "dst_node_id": str(row.get("dst_node_id") or ""),
        "support_count": int(row.get("support_count", 0) or 0),
        "contradiction_count": int(row.get("contradiction_count", 0) or 0),
        "evidence_tier": str(row.get("evidence_tier") or "hypothesized"),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "last_updated_round": row.get("last_updated_round"),
    }


def _proposal_path_rows(bundle: HypothesisBundle) -> list[dict[str, Any]]:
    rows = []
    for proposal in list(getattr(bundle, "path_proposals", ()) or []):
        rows.append(
            {
                "path_id": str(getattr(proposal, "proposal_id", "") or ""),
                "ordered_node_ids": list(getattr(proposal, "metadata", {}).get("node_ids", ()) or getattr(proposal, "metadata", {}).get("ordered_node_ids", ()) or ()),
                "ordered_edge_kinds": list(getattr(proposal, "edge_kinds", ()) or ()),
                "support_count": len(list(getattr(proposal, "support_refs", ()) or [])),
                "contradiction_count": len(list(getattr(proposal, "contradiction_refs", ()) or [])),
                "confidence": float(getattr(proposal, "confidence", 0.0) or 0.0),
                "src_node_id": str(getattr(proposal, "src_node_id", "") or ""),
                "dst_node_id": str(getattr(proposal, "dst_node_id", "") or ""),
                "path_kind": str(getattr(proposal, "path_kind", "") or ""),
                "round_id": int(getattr(proposal, "round_id", 0) or 0),
                "provenance": str(getattr(proposal, "provenance", "") or ""),
            }
        )
    return rows


def _proposal_edge_rows(bundle: HypothesisBundle) -> list[dict[str, Any]]:
    rows = []
    for proposal in list(getattr(bundle, "edge_proposals", ()) or []):
        rows.append(
            {
                "edge_id": str(getattr(proposal, "proposal_id", "") or ""),
                "src_node_id": str(getattr(proposal, "src_node_id", "") or ""),
                "edge_kind": str(getattr(proposal, "edge_kind", "") or ""),
                "dst_node_id": str(getattr(proposal, "dst_node_id", "") or ""),
                "support_count": len(list(getattr(proposal, "support_refs", ()) or [])),
                "contradiction_count": len(list(getattr(proposal, "contradiction_refs", ()) or [])),
                "confidence": float(getattr(proposal, "confidence", 0.0) or 0.0),
                "evidence_tier": "observed" if any(str(getattr(ref, "evidence_tier", "")) == "observed" for ref in list(getattr(proposal, "support_refs", ()) or [])) else "hypothesized",
                "last_updated_round": int(getattr(proposal, "round_id", 0) or 0),
            }
        )
    return rows


def _compact_path_row(row: dict[str, Any]) -> dict[str, Any]:
    ordered_node_ids = [str(node_id) for node_id in list(row.get("ordered_node_ids", []) or []) if node_id]
    if not ordered_node_ids:
        ordered_node_ids = [str(row.get("src_node_id") or ""), str(row.get("dst_node_id") or "")]
        ordered_node_ids = [node_id for node_id in ordered_node_ids if node_id]
    return {
        "path_id": str(row.get("path_id") or ""),
        "ordered_node_ids": ordered_node_ids,
        "ordered_edge_kinds": [str(kind) for kind in list(row.get("ordered_edge_kinds", []) or []) if kind],
        "support_count": int(row.get("support_count", 0) or 0),
        "contradiction_count": int(row.get("contradiction_count", 0) or 0),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "path_kind": str(row.get("path_kind") or ""),
    }


def _compact_contradiction_row(row: dict[str, Any]) -> dict[str, Any]:
    affected_ids = list(row.get("affected_ids", []) or [])
    if not affected_ids:
        affected_ids = list(row.get("affected_node_ids", []) or [])
    if not affected_ids:
        affected_ids = list(row.get("affected_path_ids", []) or [])
    if not affected_ids:
        for key in ("node_id", "path_id", "proposal_id"):
            value = row.get(key)
            if value:
                affected_ids.append(str(value))
    return {
        "contradiction_id": str(row.get("contradiction_id") or row.get("id") or ""),
        "affected_ids": [str(value) for value in affected_ids if value],
        "contradiction_type": str(row.get("contradiction_type") or row.get("type") or "unknown"),
        "count": int(row.get("count", row.get("contradiction_count", 0)) or 0),
        "recency": row.get("recency", row.get("last_seen_round")),
    }


def _compact_exit_attempt_row(row: dict[str, Any]) -> dict[str, Any]:
    exit_id = str(row.get("exit_id") or row.get("node_id") or "")
    if not exit_id:
        return {}
    return {
        "exit_id": exit_id,
        "prerequisite_trigger_ids": [str(value) for value in list(row.get("prerequisite_trigger_ids", row.get("trigger_ids_touched", [])) or []) if value],
        "success": bool(row.get("success", False)),
        "failure": bool(row.get("failure", not bool(row.get("success", False)))),
        "count": int(row.get("count", row.get("attempt_count", 0)) or 0),
        "last_seen_round": row.get("last_seen_round", row.get("round_id")),
    }


def _compact_pattern_relation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": str(row.get("relation_id") or row.get("id") or ""),
        "src_node_id": str(row.get("src_node_id") or ""),
        "relation_kind": str(row.get("relation_kind") or row.get("edge_kind") or ""),
        "dst_node_id": str(row.get("dst_node_id") or ""),
        "support_count": int(row.get("support_count", row.get("match_edge_count", 0)) or 0),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
    }


def _query_target_kind(node_row: dict[str, Any]) -> str:
    return str(node_row.get("node_kind") or "unknown")


def _select_query_target(
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    deterministic_paths: list[dict[str, Any]],
    deterministic_edges: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    prompt_mode: str,
    query_target: dict[str, Any] | str | None,
) -> dict[str, Any]:
    if isinstance(query_target, dict) and query_target.get("node_id"):
        return {"node_id": str(query_target.get("node_id")), "target_kind": str(query_target.get("target_kind") or query_target.get("node_kind") or "")}
    if isinstance(query_target, str) and query_target:
        row = dict(nodes_by_id.get(str(query_target), {}))
        return {"node_id": str(query_target), "target_kind": _query_target_kind(row) if row else "unknown"}
    if prompt_mode == "resolve_contradiction":
        for row in contradictions:
            for affected_id in list(row.get("affected_ids", []) or []):
                if str(affected_id) in nodes_by_id:
                    return {"node_id": str(affected_id), "target_kind": _query_target_kind(nodes_by_id[str(affected_id)])}
    if prompt_mode == "suggest_experiment":
        ranked = sorted(deterministic_paths, key=lambda row: (-int(row.get("contradiction_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0)))
        for row in ranked:
            for node_id in list(row.get("ordered_node_ids", []) or []):
                if str(node_id) in nodes_by_id:
                    return {"node_id": str(node_id), "target_kind": _query_target_kind(nodes_by_id[str(node_id)])}
    ranked_edges = sorted(
        [dict(row) for row in list(deterministic_edges or [])],
        key=lambda row: (-float(row.get("confidence", 0.0) or 0.0), -int(row.get("support_count", 0) or 0)),
    )
    for row in ranked_edges:
        for node_id in (str(row.get("src_node_id") or ""), str(row.get("dst_node_id") or "")):
            if node_id in nodes_by_id:
                return {"node_id": node_id, "target_kind": _query_target_kind(nodes_by_id[node_id])}
    exits = [row for row in nodes_by_id.values() if str(row.get("node_kind") or "") == "exit"]
    if exits:
        exits.sort(key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0)))
        return {"node_id": str(exits[0].get("node_id") or ""), "target_kind": "exit"}
    first_node_id = next(iter(nodes_by_id.keys()), "")
    return {"node_id": str(first_node_id), "target_kind": _query_target_kind(nodes_by_id.get(first_node_id, {})) if first_node_id else "unknown"}


def _node_relevance(
    *,
    node_id: str,
    node_row: dict[str, Any],
    query_target_id: str,
    deterministic_paths: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    connected_to_target: set[str],
) -> float:
    score = 0.0
    if node_id == query_target_id:
        score += 10.0
    if node_id in connected_to_target:
        score += 4.0
    for row in deterministic_paths:
        if node_id in set(row.get("ordered_node_ids", []) or []):
            score += 5.0
            if str(row.get("provenance") or "") == "directed_outcome":
                score += 2.0
    for row in contradictions:
        if node_id in set(row.get("affected_ids", []) or []):
            score += 4.0 * float(row.get("count", 1) or 1)
    score += min(3.0, float(node_row.get("confidence", 0.0) or 0.0) * 2.0)
    score += min(3.0, int(node_row.get("support_count", 0) or 0) * 0.5)
    if str(node_row.get("source_provenance") or "") == "directed_outcome":
        score += 2.0
    if node_row.get("last_updated_round") is not None:
        score += min(2.0, float(node_row.get("last_updated_round", 0) or 0) * 0.1)
    return score


def _edge_relevance(
    *,
    edge_row: dict[str, Any],
    selected_node_ids: set[str],
    query_target_id: str,
    contradictions: list[dict[str, Any]],
) -> float:
    src_node_id = str(edge_row.get("src_node_id") or "")
    dst_node_id = str(edge_row.get("dst_node_id") or "")
    score = 0.0
    if src_node_id in selected_node_ids and dst_node_id in selected_node_ids:
        score += 5.0
    if query_target_id in {src_node_id, dst_node_id}:
        score += 4.0
    score += min(3.0, float(edge_row.get("confidence", 0.0) or 0.0) * 2.0)
    score += min(3.0, int(edge_row.get("support_count", 0) or 0) * 0.5)
    score += min(3.0, int(edge_row.get("contradiction_count", 0) or 0) * 0.5)
    for row in contradictions:
        if src_node_id in set(row.get("affected_ids", []) or []) or dst_node_id in set(row.get("affected_ids", []) or []):
            score += 2.0
    if str(edge_row.get("evidence_tier") or "") == "observed":
        score += 1.0
    if edge_row.get("last_updated_round") is not None:
        score += min(2.0, float(edge_row.get("last_updated_round", 0) or 0) * 0.1)
    return score


def _collect_k_hop_nodes(*, query_target_id: str, adjacency: dict[str, list[str]], edges_by_id: dict[str, dict[str, Any]], max_hops: int) -> set[str]:
    if not query_target_id:
        return set()
    frontier = {str(query_target_id)}
    visited = {str(query_target_id)}
    for _ in range(max(1, int(max_hops))):
        next_frontier: set[str] = set()
        for node_id in frontier:
            for edge_id in adjacency.get(str(node_id), []):
                edge = dict(edges_by_id.get(str(edge_id), {}))
                src = str(edge.get("src_node_id") or "")
                dst = str(edge.get("dst_node_id") or "")
                for neighbor_id in (src, dst):
                    if neighbor_id and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _trim_ranked(rows: list[tuple[float, dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: (-float(item[0]), str(item[1])))
    return [dict(row) for _, row in ranked[: max(0, int(limit))]]


def build_focused_llm_payload(
    *,
    mechanic_graph_snapshot: dict,
    deterministic_hypothesis_bundle: HypothesisBundle,
    exit_attempt_summary: list[dict],
    contradictions: list[dict],
    query_target: dict[str, Any] | str | None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_paths: int = DEFAULT_MAX_PATHS,
    max_contradictions: int = DEFAULT_MAX_CONTRADICTIONS,
    max_exit_attempts: int = DEFAULT_MAX_EXIT_ATTEMPTS,
    max_pattern_relations: int = DEFAULT_MAX_PATTERN_RELATIONS,
    prompt_mode: str = "hypothesis_for_exit",
    max_allowed_node_ids: int = DEFAULT_MAX_ALLOWED_NODE_IDS,
    pattern_relation_summary: list[dict] | None = None,
) -> dict[str, Any]:
    nodes_by_id = _node_rows(mechanic_graph_snapshot)
    edges_by_id = _edge_rows(mechanic_graph_snapshot)
    adjacency = _adjacency(mechanic_graph_snapshot)
    deterministic_paths = [_compact_path_row(row) for row in _proposal_path_rows(deterministic_hypothesis_bundle)]
    deterministic_edges = [_compact_edge_row(row) for row in _proposal_edge_rows(deterministic_hypothesis_bundle)]
    compact_contradictions = [row for row in (_compact_contradiction_row(row) for row in list(contradictions or [])) if int(row.get("count", 0) or 0) > 0 or row.get("affected_ids")]
    compact_exit_attempts = [row for row in (_compact_exit_attempt_row(row) for row in list(exit_attempt_summary or [])) if row]
    compact_pattern_relations = [_compact_pattern_relation_row(row) for row in list(pattern_relation_summary or [])]
    selected_target = _select_query_target(
        nodes_by_id=nodes_by_id,
        deterministic_paths=deterministic_paths,
        deterministic_edges=deterministic_edges,
        contradictions=compact_contradictions,
        prompt_mode=str(prompt_mode),
        query_target=query_target,
    )
    query_target_id = str(selected_target.get("node_id") or "")
    neighborhood_node_ids = _collect_k_hop_nodes(
        query_target_id=query_target_id,
        adjacency=adjacency,
        edges_by_id=edges_by_id,
        max_hops=DEFAULT_MAX_K_HOPS,
    )
    connected_to_target = set()
    for edge_id in adjacency.get(query_target_id, []):
        edge = dict(edges_by_id.get(str(edge_id), {}))
        connected_to_target.add(str(edge.get("src_node_id") or ""))
        connected_to_target.add(str(edge.get("dst_node_id") or ""))
    node_candidates: set[str] = set(neighborhood_node_ids)
    for row in deterministic_edges[: max(0, int(max_edges) * 2)]:
        node_candidates.add(str(row.get("src_node_id") or ""))
        node_candidates.add(str(row.get("dst_node_id") or ""))
    for row in deterministic_paths[: max(0, int(max_paths) * 2)]:
        node_candidates.update(str(node_id) for node_id in list(row.get("ordered_node_ids", []) or []))
    for row in compact_contradictions[: max(0, int(max_contradictions) * 2)]:
        node_candidates.update(str(node_id) for node_id in list(row.get("affected_ids", []) or []))
    node_rows = _trim_ranked(
        [
            (
                _node_relevance(
                    node_id=node_id,
                    node_row=nodes_by_id.get(node_id, {}),
                    query_target_id=query_target_id,
                    deterministic_paths=deterministic_paths,
                    contradictions=compact_contradictions,
                    connected_to_target=connected_to_target,
                ),
                _compact_node_row(nodes_by_id.get(node_id, {})),
            )
            for node_id in node_candidates
            if node_id in nodes_by_id
        ],
        max_nodes,
    )
    selected_node_ids = {str(row.get("node_id") or "") for row in node_rows if row.get("node_id")}
    edge_rows = _trim_ranked(
        [
            (
                _edge_relevance(
                    edge_row=edge_row,
                    selected_node_ids=selected_node_ids,
                    query_target_id=query_target_id,
                    contradictions=compact_contradictions,
                ),
                _compact_edge_row(edge_row),
            )
            for edge_row in edges_by_id.values()
            if str(edge_row.get("src_node_id") or "") in selected_node_ids or str(edge_row.get("dst_node_id") or "") in selected_node_ids
        ]
        + [
            (
                _edge_relevance(
                    edge_row=edge_row,
                    selected_node_ids=selected_node_ids,
                    query_target_id=query_target_id,
                    contradictions=compact_contradictions,
                ) + 1.5,
                edge_row,
            )
            for edge_row in deterministic_edges
            if str(edge_row.get("src_node_id") or "") in selected_node_ids or str(edge_row.get("dst_node_id") or "") in selected_node_ids
        ],
        max_edges,
    )
    selected_node_ids.update(str(row.get("src_node_id") or "") for row in edge_rows if row.get("src_node_id"))
    selected_node_ids.update(str(row.get("dst_node_id") or "") for row in edge_rows if row.get("dst_node_id"))
    path_rows = _trim_ranked(
        [
            (
                float(row.get("confidence", 0.0) or 0.0) * 4.0
                + int(row.get("support_count", 0) or 0)
                + (5.0 if query_target_id in set(row.get("ordered_node_ids", []) or []) else 0.0)
                + (2.0 if str(row.get("path_kind") or "") in {"unlock_then_exit", "trigger_then_exit", "panel_gate_exit"} else 0.0),
                row,
            )
            for row in deterministic_paths
            if set(str(node_id) for node_id in list(row.get("ordered_node_ids", []) or [])) & selected_node_ids
        ],
        max_paths,
    )
    contradiction_rows = _trim_ranked(
        [
            (
                float(row.get("count", 1) or 1) * 2.0
                + (3.0 if query_target_id in set(row.get("affected_ids", []) or []) else 0.0),
                row,
            )
            for row in compact_contradictions
            if not row.get("affected_ids") or set(row.get("affected_ids", []) or []) & selected_node_ids
        ],
        max_contradictions,
    )
    exit_attempt_rows = _trim_ranked(
        [
            (
                float(row.get("count", 0) or 0)
                + (3.0 if str(row.get("exit_id") or "") == query_target_id else 0.0)
                + (2.0 if bool(row.get("failure", False)) else 0.0),
                row,
            )
            for row in compact_exit_attempts
        ],
        max_exit_attempts,
    )
    pattern_rows = _trim_ranked(
        [
            (
                float(row.get("support_count", 0) or 0) + float(row.get("confidence", 0.0) or 0.0) * 2.0,
                row,
            )
            for row in compact_pattern_relations
            if {str(row.get("src_node_id") or ""), str(row.get("dst_node_id") or "")} & selected_node_ids
        ],
        max_pattern_relations,
    )
    if str(prompt_mode) == "suggest_experiment":
        path_rows = path_rows[: min(3, len(path_rows))]
        contradiction_rows = contradiction_rows[: min(3, len(contradiction_rows))]
    referenced_node_ids = set(selected_node_ids)
    for row in path_rows:
        referenced_node_ids.update(str(node_id) for node_id in list(row.get("ordered_node_ids", []) or []))
    for row in contradiction_rows:
        referenced_node_ids.update(str(node_id) for node_id in list(row.get("affected_ids", []) or []))
    for row in exit_attempt_rows:
        referenced_node_ids.update(str(node_id) for node_id in list(row.get("prerequisite_trigger_ids", []) or []))
        if row.get("exit_id"):
            referenced_node_ids.add(str(row.get("exit_id")))
    for row in pattern_rows:
        if row.get("src_node_id"):
            referenced_node_ids.add(str(row.get("src_node_id")))
        if row.get("dst_node_id"):
            referenced_node_ids.add(str(row.get("dst_node_id")))
    allowed_node_ids = sorted(node_id for node_id in referenced_node_ids if node_id)[: max(0, int(max_allowed_node_ids))]
    query_target_kind = str(selected_target.get("target_kind") or "unknown")
    if str(prompt_mode) == "hypothesis_for_exit" and query_target_kind != "exit":
        prompt_mode = "suggest_experiment"
    prompt_instructions = {
        "hypothesis_for_exit": "Focus on one exit-centered local dependency subgraph. Propose only JSON hypotheses for the targeted exit path.",
        "resolve_contradiction": "Focus on one contradicted local relation or path. Resolve the contradiction using only the provided local subgraph.",
        "suggest_experiment": "Focus on only the competing local hypotheses and suggest one discriminating experiment in JSON form.",
    }
    return {
        "system_instruction": (
            "Output exactly one JSON object. "
            "Do not output prose before or after the JSON. "
            "Do not use markdown fences. "
            "Do not reveal chain-of-thought. "
            "Use only allowed node ids. "
            "Use only allowed edge kinds. "
            "Use only allowed path kinds. "
            + prompt_instructions.get(str(prompt_mode), prompt_instructions["hypothesis_for_exit"])
        ),
        "prompt_mode": str(prompt_mode),
        "query_target": {"node_id": query_target_id, "target_kind": query_target_kind},
        "graph_nodes": tuple(node_rows),
        "graph_edges": tuple(edge_rows),
        "top_deterministic_edges": tuple(sorted(deterministic_edges, key=lambda row: (-float(row.get("confidence", 0.0) or 0.0), -int(row.get("support_count", 0) or 0), str(row.get("edge_id") or "")))[:max(0, int(max_paths))]),
        "top_deterministic_paths": tuple(path_rows),
        "open_questions": (
            f"Which local dependency around {query_target_id or 'the target'} best explains the current blocker?",
        ),
        "contradictions": tuple(contradiction_rows),
        "exit_attempts": tuple(exit_attempt_rows),
        "pattern_relations": tuple(pattern_rows),
        "allowed_node_ids": tuple(allowed_node_ids),
        "allowed_edge_kinds": ALLOWED_EDGE_KINDS,
        "allowed_path_kinds": ALLOWED_PATH_KINDS,
        "payload_section_counts": {
            "graph_nodes": len(node_rows),
            "graph_edges": len(edge_rows),
            "top_deterministic_paths": len(path_rows),
            "contradictions": len(contradiction_rows),
            "exit_attempts": len(exit_attempt_rows),
            "pattern_relations": len(pattern_rows),
            "allowed_node_ids": len(allowed_node_ids),
        },
    }


def build_llm_hypothesis_input(
    mechanic_graph_snapshot: dict,
    deterministic_hypothesis_bundle: HypothesisBundle,
    recent_episode_summaries: list[dict],
    unresolved_contradictions: list[dict],
    exit_attempt_summary: list[dict],
    pattern_relation_summary: list[dict],
) -> LLMHypothesisInput:
    del recent_episode_summaries
    focused = build_focused_llm_payload(
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        deterministic_hypothesis_bundle=deterministic_hypothesis_bundle,
        exit_attempt_summary=exit_attempt_summary,
        contradictions=unresolved_contradictions,
        query_target=None,
        pattern_relation_summary=pattern_relation_summary,
    )
    return LLMHypothesisInput(
        system_instruction=str(focused.get("system_instruction") or ""),
        graph_nodes=tuple(dict(row) for row in list(focused.get("graph_nodes", ()) or ())),
        graph_edges=tuple(dict(row) for row in list(focused.get("graph_edges", ()) or ())),
        top_deterministic_edges=tuple(dict(row) for row in list(focused.get("top_deterministic_edges", ()) or ())),
        top_deterministic_paths=tuple(dict(row) for row in list(focused.get("top_deterministic_paths", ()) or ())),
        open_questions=tuple(str(row) for row in list(focused.get("open_questions", ()) or ())),
        contradictions=tuple(dict(row) for row in list(focused.get("contradictions", ()) or ())),
        exit_attempts=tuple(dict(row) for row in list(focused.get("exit_attempts", ()) or ())),
        pattern_relations=tuple(dict(row) for row in list(focused.get("pattern_relations", ()) or ())),
        allowed_node_ids=tuple(str(row) for row in list(focused.get("allowed_node_ids", ()) or ())),
        allowed_edge_kinds=tuple(str(row) for row in list(focused.get("allowed_edge_kinds", ()) or ())),
        allowed_path_kinds=tuple(str(row) for row in list(focused.get("allowed_path_kinds", ()) or ())),
    )
