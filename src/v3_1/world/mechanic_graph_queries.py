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
    ranked = sorted(paths, key=lambda row: (row.hypothesis_only, row.contradiction_count, -row.support_strength))
    return GraphQueryResult(paths=tuple(ranked[:10]))
