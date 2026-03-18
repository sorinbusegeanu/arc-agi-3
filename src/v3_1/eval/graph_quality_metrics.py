from __future__ import annotations

from collections import Counter


def compute_graph_quality_metrics(round_graph_snapshots: list[dict]) -> dict:
    snapshots = [dict(row or {}) for row in list(round_graph_snapshots or [])]
    node_count_by_round = {}
    edge_count_by_family = Counter()
    repeated_support_edge_count = 0
    contradiction_count_by_family = Counter()
    counterfactual_supported_edge_count = 0
    exit_linked_path_count = 0
    usable_path_to_exit_count = 0
    total_edges = 0
    for snapshot in snapshots:
        round_id = int(snapshot.get("round_id", len(node_count_by_round) + 1) or 0)
        nodes = dict(snapshot.get("nodes_by_id", {}) or {})
        edges = dict(snapshot.get("edges_by_id", {}) or {})
        node_count_by_round[str(round_id)] = len(nodes)
        for edge in edges.values():
            edge_kind = str(edge.get("edge_kind") or "unknown")
            edge_count_by_family[edge_kind] += 1
            total_edges += 1
            if int(edge.get("support_count", 0) or 0) >= 2:
                repeated_support_edge_count += 1
            contradiction_count_by_family[edge_kind] += int(edge.get("contradiction_count", 0) or 0)
            if int(edge.get("counterfactual_support_count", 0) or 0) > 0:
                counterfactual_supported_edge_count += 1
        for path in list(snapshot.get("paths_to_exit", []) or []):
            if list(path.get("node_ids", []) or []):
                exit_linked_path_count += 1
            if float(path.get("execution_feasibility_score", 0.0) or 0.0) >= 0.5 and int(path.get("contradiction_count", 0) or 0) <= 0:
                usable_path_to_exit_count += 1
    return {
        "graph_node_count_by_round": node_count_by_round,
        "graph_edge_count_by_family": dict(edge_count_by_family),
        "repeated_support_edge_rate": float(repeated_support_edge_count) / float(max(1, total_edges)),
        "contradiction_rate_by_edge_family": {
            family: float(count) / float(max(1, edge_count_by_family.get(family, 0)))
            for family, count in contradiction_count_by_family.items()
        },
        "counterfactual_supported_edge_rate": float(counterfactual_supported_edge_count) / float(max(1, total_edges)),
        "exit_linked_path_count": exit_linked_path_count,
        "usable_path_to_exit_rate": float(usable_path_to_exit_count) / float(max(1, exit_linked_path_count)),
    }
