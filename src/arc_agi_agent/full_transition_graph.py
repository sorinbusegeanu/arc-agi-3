from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .full_explorer_types import TransitionEdge, TransitionGraph, TransitionNode


class FullTransitionGraphStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, TransitionNode] = {}
        self.edges: Dict[Tuple[str, str, int, int, str], TransitionEdge] = {}
        self._adj: Dict[str, List[Tuple[str, int, int, str]]] = {}

    def add_node(self, node: TransitionNode) -> None:
        if node.state not in self.nodes:
            self.nodes[node.state] = node
            self._adj.setdefault(node.state, [])

    def add_edge(
        self,
        from_state: str,
        action_id: str,
        x: int,
        y: int,
        to_state: str,
        changed_cells: int,
        changed_bbox_area: int,
        event_signatures: List[str],
        step_idx: int,
    ) -> None:
        key = (from_state, action_id, x, y, to_state)
        edge = self.edges.get(key)
        if edge is None:
            edge = TransitionEdge(
                from_state=from_state,
                action_id=action_id,
                x=x,
                y=y,
                to_state=to_state,
                count=0,
                avg_changed_cells=0.0,
                avg_changed_bbox_area=0.0,
                event_signature_histogram={},
                example_steps=[],
            )
            self.edges[key] = edge
            self._adj.setdefault(from_state, []).append((action_id, x, y, to_state))

        edge.count += 1
        edge.avg_changed_cells = _running_avg(edge.avg_changed_cells, changed_cells, edge.count)
        edge.avg_changed_bbox_area = _running_avg(edge.avg_changed_bbox_area, changed_bbox_area, edge.count)
        for sig in event_signatures:
            edge.event_signature_histogram[sig] = edge.event_signature_histogram.get(sig, 0) + 1
        if len(edge.example_steps) < 3:
            edge.example_steps.append(step_idx)

    def bfs_path(
        self,
        start: str,
        target: str,
        max_depth: int,
    ) -> Optional[List[Tuple[str, int, int, str]]]:
        if start == target:
            return []
        if start not in self._adj:
            return None
        visited = {start}
        queue: deque[Tuple[str, List[Tuple[str, int, int, str]]]] = deque()
        queue.append((start, []))
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action_id, x, y, nxt in self._adj.get(node, []):
                if nxt in visited:
                    continue
                new_path = path + [(action_id, x, y, nxt)]
                if nxt == target:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return None

    def to_graph(self) -> TransitionGraph:
        return TransitionGraph(nodes=dict(self.nodes), edges=dict(self.edges))


def _running_avg(prev: float, value: int, count: int) -> float:
    if count <= 1:
        return float(value)
    return prev + (value - prev) / count
