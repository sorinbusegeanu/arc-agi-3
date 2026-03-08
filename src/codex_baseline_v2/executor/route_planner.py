from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import BlackboardStateV2, CandidatePOIV2
from codex_baseline_v2.shared.utils import BBox


@dataclass(frozen=True)
class RoutePlanV2:
    next_subgoal: Optional[Tuple[int, int]]
    distance_estimate: Optional[float]
    distance_prev: Optional[float]
    distance_delta: Optional[float]
    progress_valid: bool
    progress_reason: str
    fallback_mode: Optional[str]
    confidence: float
    stalled: bool
    blocked: bool


def _build_graph(traversable_map: Optional[Dict[str, any]]) -> Dict[Tuple[int, int], int]:
    if not traversable_map:
        return {}
    points = traversable_map.get("points", [])
    return {(int(p["x"]), int(p["y"])): int(p.get("visits", 1)) for p in points if "x" in p and "y" in p}


def _bfs_distance(start: Tuple[int, int], graph: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    dist: Dict[Tuple[int, int], int] = {start: 0}
    q: deque = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (nx, ny) not in graph:
                continue
            if (nx, ny) in dist:
                continue
            dist[(nx, ny)] = dist[(x, y)] + 1
            q.append((nx, ny))
    return dist


def _target_cells(target: BBox) -> List[Tuple[int, int]]:
    cells = []
    for x in range(target.x1, target.x2 + 1):
        for y in range(target.y1, target.y2 + 1):
            cells.append((x, y))
    return cells


def _distance_to_bbox(point: Tuple[int, int], bbox: BBox) -> float:
    x, y = point
    if bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
        return 0.0
    dx = max(bbox.x1 - x, 0, x - bbox.x2)
    dy = max(bbox.y1 - y, 0, y - bbox.y2)
    return float(dx + dy)


def plan_route(
    blackboard: BlackboardStateV2,
    target: CandidatePOIV2,
    current: Tuple[int, int],
    prev_distance: Optional[float],
) -> RoutePlanV2:
    graph = _build_graph(blackboard.traversable_map)
    fallback_mode = None
    if not graph or current not in graph:
        fallback_mode = "euclidean"
        distance = _distance_to_bbox(current, target.bbox)
        delta = None if prev_distance is None else prev_distance - distance
        return RoutePlanV2(
            next_subgoal=None,
            distance_estimate=distance,
            distance_prev=prev_distance,
            distance_delta=delta,
            progress_valid=True,
            progress_reason="fallback_euclidean",
            fallback_mode=fallback_mode,
            confidence=0.2,
            stalled=False,
            blocked=False,
        )
    dist = _bfs_distance(current, graph)
    target_cells = _target_cells(target.bbox)
    best = None
    for cell in target_cells:
        if cell in dist:
            d = dist[cell]
            best = d if best is None or d < best else best
    if best is None:
        return RoutePlanV2(
            next_subgoal=None,
            distance_estimate=None,
            distance_prev=prev_distance,
            distance_delta=None,
            progress_valid=False,
            progress_reason="no_graph_path",
            fallback_mode="graph",
            confidence=0.3,
            stalled=False,
            blocked=True,
        )
    # pick a neighbor closer to target as next subgoal
    next_cell = current
    for nx, ny in ((current[0] - 1, current[1]), (current[0] + 1, current[1]), (current[0], current[1] - 1), (current[0], current[1] + 1)):
        if (nx, ny) in dist and dist[(nx, ny)] < dist.get(next_cell, dist[(nx, ny)] + 1):
            next_cell = (nx, ny)
    delta = None if prev_distance is None else prev_distance - float(best)
    return RoutePlanV2(
        next_subgoal=next_cell,
        distance_estimate=float(best),
        distance_prev=prev_distance,
        distance_delta=delta,
        progress_valid=True,
        progress_reason="graph_distance",
        fallback_mode=None,
        confidence=0.7,
        stalled=False,
        blocked=False,
    )
