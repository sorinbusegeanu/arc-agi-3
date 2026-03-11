from __future__ import annotations

from typing import Dict, List

from codex_baseline_v2.shared.schemas import MechanicEdgeV1, MechanicGraphStateV1, MechanicNodeV1


def node_map(graph: MechanicGraphStateV1 | None) -> Dict[str, MechanicNodeV1]:
    if graph is None:
        return {}
    return {node.node_id: node for node in graph.nodes}


def outgoing_edges(graph: MechanicGraphStateV1 | None, node_id: str) -> List[MechanicEdgeV1]:
    if graph is None:
        return []
    return [edge for edge in graph.edges if edge.src_node_id == node_id]


def incoming_edges(graph: MechanicGraphStateV1 | None, node_id: str) -> List[MechanicEdgeV1]:
    if graph is None:
        return []
    return [edge for edge in graph.edges if edge.dst_node_id == node_id]
