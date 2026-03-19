from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNeighbor:
    node_id: str
    edge_id: str
    edge_kind: str
    neighbor_node_id: str
    confidence: float
    evidence_tier: str


@dataclass(frozen=True)
class GraphPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    support_strength: float
    contradiction_count: int
    hypothesis_only: bool
    ordered_step_plan: tuple[dict, ...] = ()
    verification_nodes: tuple[str, ...] = ()
    counterfactual_strength: float = 0.0
    execution_feasibility_score: float = 0.0
    first_step_executability_score: float = 0.0
    evidence_diversity_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphQueryResult:
    nodes: tuple[dict, ...] = ()
    edges: tuple[dict, ...] = ()
    paths: tuple[GraphPath, ...] = ()


def _nodes(snapshot: dict) -> dict[str, dict]:
    return dict(dict(snapshot or {}).get("nodes_by_id", {}))


def _edges(snapshot: dict) -> dict[str, dict]:
    return dict(dict(snapshot or {}).get("edges_by_id", {}))


def neighbors(snapshot: dict, node_id: str, edge_kind: str | None = None) -> tuple[GraphNeighbor, ...]:
    edges = _edges(snapshot)
    rows = []
    for edge_id in list(dict(snapshot or {}).get("adjacency_out", {}).get(str(node_id), []) or []):
        edge = dict(edges.get(edge_id, {}))
        if edge_kind is not None and str(edge.get("edge_kind") or "") != str(edge_kind):
            continue
        rows.append(
            GraphNeighbor(
                node_id=str(node_id),
                edge_id=str(edge_id),
                edge_kind=str(edge.get("edge_kind") or ""),
                neighbor_node_id=str(edge.get("dst_node_id") or ""),
                confidence=float(edge.get("confidence", 0.0) or 0.0),
                evidence_tier=str(edge.get("evidence_tier") or "hypothesized"),
            )
        )
    return tuple(rows)


def find_nodes_by_kind(snapshot: dict, node_kind: str) -> GraphQueryResult:
    rows = [dict(row) for row in _nodes(snapshot).values() if str(row.get("node_kind") or "") == str(node_kind)]
    return GraphQueryResult(nodes=tuple(rows))


def find_edges_by_kind(snapshot: dict, edge_kind: str) -> GraphQueryResult:
    rows = [dict(row) for row in _edges(snapshot).values() if str(row.get("edge_kind") or "") == str(edge_kind)]
    return GraphQueryResult(edges=tuple(rows))


def _path_strength(snapshot: dict, edge_ids: list[str]) -> tuple[float, int, bool]:
    edges = _edges(snapshot)
    if not edge_ids:
        return 0.0, 0, True
    confidence = 1.0
    contradiction_count = 0
    hypothesis_only = True
    for edge_id in edge_ids:
        edge = dict(edges.get(edge_id, {}))
        confidence *= max(0.0, float(edge.get("confidence", 0.0) or 0.0) or 0.01)
        contradiction_count += int(edge.get("contradiction_count", 0) or 0)
        if str(edge.get("evidence_tier") or "") == "observed":
            hypothesis_only = False
    return confidence, contradiction_count, hypothesis_only


def _path_annotations(snapshot: dict, node_ids: list[str], edge_ids: list[str]) -> tuple[tuple[dict, ...], tuple[str, ...], float, float]:
    edges = _edges(snapshot)
    nodes = _nodes(snapshot)
    verification_nodes = []
    counterfactual_strength = 0.0
    directed_support = 0.0
    step_plan = []
    for node_id in list(node_ids or []):
        node = dict(nodes.get(str(node_id), {}))
        node_kind = str(node.get("node_kind") or "poi")
        step_kind = "go_to_trigger" if node_kind == "trigger" else "verify_panel" if node_kind in {"panel", "symbol_state"} else "verify_gate" if node_kind == "gate" else "attempt_exit" if node_kind == "exit" else "reobserve_region"
        if step_kind in {"verify_panel", "verify_gate"}:
            verification_nodes.append(str(node_id))
        step_plan.append({"target_node_id": str(node_id), "step_kind": step_kind})
    for edge_id in list(edge_ids or []):
        edge = dict(edges.get(str(edge_id), {}))
        counterfactual_strength += float(edge.get("counterfactual_support_count", 0) or 0)
        directed_support += float(edge.get("directed_outcome_support_count", 0) or 0)
    execution_feasibility_score = min(1.0, (0.4 if step_plan else 0.0) + (0.1 * len(step_plan)) + (0.08 * directed_support) - (0.06 * max(0, len(step_plan) - 4)))
    return tuple(step_plan), tuple(dict.fromkeys(verification_nodes)), min(1.0, counterfactual_strength), max(0.0, execution_feasibility_score)


def _path_quality(snapshot: dict, node_ids: list[str], edge_ids: list[str]) -> tuple[float, float]:
    nodes = _nodes(snapshot)
    edges = _edges(snapshot)
    if not node_ids:
        return 0.0, 0.0
    first = dict(nodes.get(str(node_ids[0]), {}))
    first_step = 0.2
    if bool(first.get("object_backed", False)):
        first_step += 0.25
    if int(first.get("observed_support_count", 0) or 0) > 0:
        first_step += 0.2
    if int(first.get("support_round_count", 0) or 0) > 1:
        first_step += 0.15
    if int(first.get("exit_link_support_count", 0) or 0) > 0:
        first_step += 0.1
    if int(first.get("counterfactual_support_count", 0) or 0) > 0:
        first_step += 0.05
    if bool(first.get("synthetic_region_only", False)):
        first_step -= 0.25
    edge_kinds = {str(dict(edges.get(edge_id, {})).get("edge_kind") or "") for edge_id in edge_ids}
    evidence_diversity = min(0.4, 0.12 * len(edge_kinds))
    evidence_diversity += 0.2 if int(first.get("counterfactual_support_count", 0) or 0) > 0 else 0.0
    evidence_diversity += 0.2 if int(first.get("exit_link_support_count", 0) or 0) > 0 else 0.0
    evidence_diversity += 0.2 if int(first.get("observed_support_count", 0) or 0) > 0 else 0.0
    return max(0.0, min(1.0, first_step)), max(0.0, min(1.0, evidence_diversity))


def find_dependency_paths(snapshot: dict, start_node_id: str, max_hops: int = 4) -> GraphQueryResult:
    queue = [(str(start_node_id), [str(start_node_id)], [])]
    paths: list[GraphPath] = []
    while queue:
        node_id, node_path, edge_path = queue.pop(0)
        if len(edge_path) >= int(max_hops):
            continue
        for neighbor in neighbors(snapshot, node_id):
            if neighbor.neighbor_node_id in node_path:
                continue
            next_nodes = [*node_path, neighbor.neighbor_node_id]
            next_edges = [*edge_path, neighbor.edge_id]
            strength, contradiction_count, hypothesis_only = _path_strength(snapshot, next_edges)
            paths.append(
                GraphPath(
                    node_ids=tuple(next_nodes),
                    edge_ids=tuple(next_edges),
                    support_strength=float(strength),
                    contradiction_count=int(contradiction_count),
                    hypothesis_only=bool(hypothesis_only),
                    ordered_step_plan=_path_annotations(snapshot, next_nodes, next_edges)[0],
                    verification_nodes=_path_annotations(snapshot, next_nodes, next_edges)[1],
                    counterfactual_strength=_path_annotations(snapshot, next_nodes, next_edges)[2],
                    execution_feasibility_score=_path_annotations(snapshot, next_nodes, next_edges)[3],
                    first_step_executability_score=_path_quality(snapshot, next_nodes, next_edges)[0],
                    evidence_diversity_score=_path_quality(snapshot, next_nodes, next_edges)[1],
                )
            )
            queue.append((neighbor.neighbor_node_id, next_nodes, next_edges))
    return GraphQueryResult(paths=tuple(paths))


def find_exit_prerequisite_paths(snapshot: dict, exit_node_id: str, max_hops: int = 4) -> GraphQueryResult:
    paths = [
        path for path in find_dependency_paths(snapshot, str(exit_node_id), max_hops=max_hops).paths
        if len(path.node_ids) > 1
    ]
    return GraphQueryResult(paths=tuple(paths))


def find_trigger_to_exit_paths(snapshot: dict, trigger_node_id: str, max_hops: int = 4) -> GraphQueryResult:
    nodes = _nodes(snapshot)
    paths = []
    for path in find_dependency_paths(snapshot, str(trigger_node_id), max_hops=max_hops).paths:
        if str(nodes.get(path.node_ids[-1], {}).get("node_kind") or "") == "exit":
            paths.append(path)
    return GraphQueryResult(paths=tuple(paths))


def find_match_relations_for_panel(snapshot: dict, panel_node_id: str) -> GraphQueryResult:
    edges = [dict(edge) for edge in _edges(snapshot).values() if str(edge.get("src_node_id") or "") == str(panel_node_id) and str(edge.get("edge_kind") or "") == "matches"]
    return GraphQueryResult(edges=tuple(edges))


def best_supported_paths_to_exit(snapshot: dict, *, max_hops: int = 4) -> GraphQueryResult:
    nodes = _nodes(snapshot)
    paths = []
    for node_id, node in nodes.items():
        if str(node.get("node_kind") or "") != "trigger":
            continue
        paths.extend(find_trigger_to_exit_paths(snapshot, str(node_id), max_hops=max_hops).paths)
    ranked = sorted(paths, key=lambda row: (row.hypothesis_only, row.contradiction_count, -row.first_step_executability_score, -row.execution_feasibility_score, -row.evidence_diversity_score, -row.counterfactual_strength, -row.support_strength))
    return GraphQueryResult(paths=tuple(ranked[:10]))


def compute_exit_readiness_score(
    exit_node_id: str,
    mechanic_graph_snapshot: dict,
    hypothesis_registry_snapshot: dict | None,
    recent_outcomes: list[dict] | None,
) -> dict:
    snapshot = dict(mechanic_graph_snapshot or {})
    registry_snapshot = dict(hypothesis_registry_snapshot or {})
    recent_outcomes = [dict(row or {}) for row in list(recent_outcomes or [])]
    edges = _edges(snapshot)
    relevant_edges = [
        dict(edge)
        for edge in edges.values()
        if str(edge.get("dst_node_id") or "") == str(exit_node_id)
        or str(edge.get("src_node_id") or "") == str(exit_node_id)
    ]
    prerequisite_paths = list(find_exit_prerequisite_paths(snapshot, str(exit_node_id), max_hops=4).paths)
    path_edge_ids = {str(edge_id) for path in prerequisite_paths for edge_id in list(path.edge_ids or ())}
    relevant_edges.extend(dict(edges.get(edge_id, {})) for edge_id in path_edge_ids if edge_id in edges)
    deduped_edges = {}
    for edge in relevant_edges:
        edge_id = str(edge.get("edge_id") or _stable_edge_id_for_readiness(edge))
        deduped_edges[edge_id] = edge
    relevant_edges = list(deduped_edges.values())

    verified_trigger_contact = any(
        bool(row.get("step_success"))
        and str(row.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact", "retry_trigger"}
        for row in recent_outcomes
    ) or any(
        int(edge.get("poi_visit_support_count", 0) or 0) > 0
        or int(edge.get("directed_outcome_support_count", 0) or 0) > 0
        for edge in relevant_edges
        if str(edge.get("edge_kind") or "") in {"requires", "changes", "causes_remote_change"}
    )
    remote_change_support = any(
        bool(dict(row.get("mechanic_graph_evidence", {}) or {}).get("remote_region_changed"))
        or str(row.get("step_kind") or "") == "reobserve_remote_change" and bool(row.get("step_success"))
        for row in recent_outcomes
    ) or any(
        int(edge.get("post_visit_remote_change_count", 0) or 0) > 0
        or str(edge.get("edge_kind") or "") == "causes_remote_change" and int(edge.get("observed_support_count", 0) or 0) > 0
        for edge in relevant_edges
    )
    panel_or_gate_confirmation = any(
        str(row.get("step_kind") or "") in {"verify_panel", "verify_gate", "verify_panel_state", "verify_gate_match"}
        and bool(row.get("step_success"))
        for row in recent_outcomes
    ) or any(
        int(edge.get("post_visit_panel_match_count", 0) or 0) > 0
        or str(edge.get("edge_kind") or "") in {"matches", "controls_access"}
        and int(edge.get("observed_support_count", 0) or 0) > 0
        for edge in relevant_edges
    )
    contradiction_burden = sum(int(edge.get("contradiction_count", 0) or 0) for edge in relevant_edges)
    planner_usable_hypothesis_count = sum(
        1
        for state in dict(registry_snapshot.get("planner_usable_state", {}) or {}).values()
        if str(state or "") == "planner_usable"
    )

    exit_attempt_rows = [
        row for row in recent_outcomes
        if str(row.get("step_kind") or "") == "attempt_exit"
        or bool(row.get("exit_attempt_failed"))
        or str(row.get("candidate_class") or "") == "unlock_then_exit"
    ]
    last_failed_exit = dict(exit_attempt_rows[-1]) if exit_attempt_rows and bool(exit_attempt_rows[-1].get("exit_attempt_failed")) else {}
    last_failed_round = int(last_failed_exit.get("round_id", -1) or -1)
    new_support_since_last_failure = False
    if last_failed_round >= 0:
        new_support_since_last_failure = any(
            int(row.get("round_id", -1) or -1) > last_failed_round
            and (
                bool(row.get("step_success")) and str(row.get("step_kind") or "") in {"go_to_trigger", "verify_trigger_contact", "retry_trigger", "verify_panel", "verify_gate", "verify_panel_state", "verify_gate_match", "reobserve_region", "reobserve_remote_change"}
                or bool(dict(row.get("mechanic_graph_evidence", {}) or {}).get("remote_region_changed"))
            )
            for row in recent_outcomes
        ) or any(
            max([int(value) for value in list(edge.get("source_round_ids", []) or [])] or [-1]) > last_failed_round
            and (
                int(edge.get("post_visit_remote_change_count", 0) or 0) > 0
                or int(edge.get("post_visit_panel_match_count", 0) or 0) > 0
                or int(edge.get("directed_outcome_support_count", 0) or 0) > 0
            )
            for edge in relevant_edges
        )
    last_failed_without_new_support = bool(last_failed_exit) and not new_support_since_last_failure
    hypothesis_only_chain = bool(prerequisite_paths) and all(bool(path.hypothesis_only) for path in prerequisite_paths)

    score = 0.0
    score += 0.32 if verified_trigger_contact else 0.0
    score += 0.24 if remote_change_support else 0.0
    score += 0.22 if panel_or_gate_confirmation else 0.0
    score += 0.08 if planner_usable_hypothesis_count > 0 else 0.0
    score += 0.08 if new_support_since_last_failure else 0.0
    score -= min(0.22, 0.06 * contradiction_burden)
    if hypothesis_only_chain:
        score -= 0.14
    if last_failed_without_new_support:
        score -= 0.34
    missing_prerequisites = []
    if not verified_trigger_contact:
        missing_prerequisites.append("verify_trigger_contact")
    if not remote_change_support:
        missing_prerequisites.append("reobserve_remote_change")
    if not panel_or_gate_confirmation:
        missing_prerequisites.append("verify_panel_or_gate")
    return {
        "exit_node_id": str(exit_node_id),
        "readiness_score": max(0.0, min(1.0, score)),
        "has_verified_trigger_contact": bool(verified_trigger_contact),
        "has_remote_change_support": bool(remote_change_support),
        "has_panel_or_gate_confirmation": bool(panel_or_gate_confirmation),
        "has_new_support_since_last_exit_attempt": bool(new_support_since_last_failure),
        "last_exit_attempt_failed_without_new_support": bool(last_failed_without_new_support),
        "last_failed_exit_round_id": last_failed_round if last_failed_round >= 0 else None,
        "contradiction_burden": int(contradiction_burden),
        "hypothesis_only_chain": bool(hypothesis_only_chain),
        "missing_prerequisite_types": missing_prerequisites,
        "verified_support_count": int(verified_trigger_contact) + int(remote_change_support) + int(panel_or_gate_confirmation),
    }


def _stable_edge_id_for_readiness(row: dict) -> str:
    return f"readiness:{row.get('src_node_id')}:{row.get('edge_kind')}:{row.get('dst_node_id')}:{row.get('condition_key')}"
