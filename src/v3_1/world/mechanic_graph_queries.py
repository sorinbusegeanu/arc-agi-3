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
