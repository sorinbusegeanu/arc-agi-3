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


def _build_graph(traversable_map: Optional[Dict[str, object]]) -> Dict[Tuple[int, int], int]:
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
            if (nx, ny) not in graph or (nx, ny) in dist:
                continue
            dist[(nx, ny)] = dist[(x, y)] + 1
            q.append((nx, ny))
    return dist


def _target_cells(target: BBox) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(target.x1, target.x2 + 1) for y in range(target.y1, target.y2 + 1)]


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
    if not graph or current not in graph:
        distance = _distance_to_bbox(current, target.bbox)
        delta = None if prev_distance is None else prev_distance - distance
        return RoutePlanV2(
            next_subgoal=None,
            distance_estimate=distance,
            distance_prev=distance,
            distance_delta=delta,
            progress_valid=False,
            progress_reason="fallback_euclidean",
            fallback_mode="euclidean",
            confidence=0.2,
            stalled=False,
            blocked=True,
        )

    target_cells = _target_cells(target.bbox)
    traversable_targets = [cell for cell in target_cells if cell in graph]
    if traversable_targets:
        anchor = min(traversable_targets, key=lambda cell: abs(cell[0] - current[0]) + abs(cell[1] - current[1]))
        distance_map = _bfs_distance(anchor, graph)
        progress_reason = "graph_to_target"
        fallback_mode = None
    else:
        anchor = min(graph.keys(), key=lambda cell: _distance_to_bbox(cell, target.bbox))
        distance_map = _bfs_distance(anchor, graph)
        progress_reason = "graph_to_anchor"
        fallback_mode = "anchor"

    if current not in distance_map:
        return RoutePlanV2(
            next_subgoal=None,
            distance_estimate=None,
            distance_prev=None,
            distance_delta=None,
            progress_valid=False,
            progress_reason="current_off_path",
            fallback_mode=fallback_mode,
            confidence=0.3,
            stalled=False,
            blocked=True,
        )

    current_distance = float(distance_map[current])
    best_neighbor: Optional[Tuple[int, int]] = None
    best_distance: Optional[float] = None
    for neighbor in ((current[0] - 1, current[1]), (current[0] + 1, current[1]), (current[0], current[1] - 1), (current[0], current[1] + 1)):
        if neighbor not in distance_map:
            continue
        neighbor_distance = float(distance_map[neighbor])
        if best_distance is None or neighbor_distance < best_distance:
            best_neighbor = neighbor
            best_distance = neighbor_distance

    if best_neighbor is None:
        return RoutePlanV2(
            next_subgoal=None,
            distance_estimate=None,
            distance_prev=current_distance,
            distance_delta=None,
            progress_valid=False,
            progress_reason="no_routed_neighbor",
            fallback_mode=fallback_mode,
            confidence=0.3,
            stalled=False,
            blocked=True,
        )

    distance_delta = current_distance - float(best_distance)
    stalled = bool(distance_delta <= 0.0)
    return RoutePlanV2(
        next_subgoal=best_neighbor,
        distance_estimate=float(best_distance),
        distance_prev=current_distance,
        distance_delta=distance_delta,
        progress_valid=True,
        progress_reason=progress_reason,
        fallback_mode=fallback_mode,
        confidence=0.8 if not stalled else 0.4,
        stalled=stalled,
        blocked=False,
    )
