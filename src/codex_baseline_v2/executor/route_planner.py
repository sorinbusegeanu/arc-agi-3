from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import BlackboardStateV2, CandidatePOIV2
from codex_baseline_v2.shared.utils import point_neighbors4, point_manhattan


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
    route_edge_ids: List[str] = field(default_factory=list)


def _edge_cost(edge, blackboard: BlackboardStateV2) -> float:
    cfg = blackboard.metadata.get("routing_config", {}) if isinstance(blackboard.metadata, dict) else {}
    blocked_penalty = float(cfg.get("blocked_edge_penalty", 2.0))
    unknown_penalty = float(cfg.get("unknown_edge_penalty", 0.5))
    transition_penalty = float(cfg.get("transition_edge_penalty", 1.0))
    cost = 1.0
    cost += blocked_penalty * float(edge.blocked_count > edge.success_count)
    cost += unknown_penalty * float(edge.confidence < 0.4)
    cost += transition_penalty * float(edge.transition_type == "transition")
    return cost


def _access_cells(blackboard: BlackboardStateV2, target: CandidatePOIV2) -> List[Tuple[int, int]]:
    for profile in blackboard.target_access_table:
        if profile.poi_id == target.poi_id and profile.access_cells:
            return list(profile.access_cells)
    centroid = (int(round(target.centroid[0])), int(round(target.centroid[1])))
    return point_neighbors4(centroid)


def build_route_context(blackboard: BlackboardStateV2, target: CandidatePOIV2) -> Tuple[Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, str]]], List[Tuple[int, int]]]:
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, str]]] = {}
    for edge in blackboard.navigation_edges:
        adjacency.setdefault(edge.src_cell, []).append((edge.dst_cell, _edge_cost(edge, blackboard), edge.edge_id))
    return adjacency, _access_cells(blackboard, target)


def _shortest_path(
    blackboard: BlackboardStateV2,
    start: Tuple[int, int],
    goals: List[Tuple[int, int]],
    adjacency: Optional[Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, str]]]] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], List[str], Optional[float]]:
    if adjacency is None:
        adjacency = {}
        for edge in blackboard.navigation_edges:
            adjacency.setdefault(edge.src_cell, []).append((edge.dst_cell, _edge_cost(edge, blackboard), edge.edge_id))
    goal_set = set(goals)
    heap: List[Tuple[float, Tuple[int, int], List[Tuple[int, int]], List[str]]] = [(0.0, start, [start], [])]
    best: Dict[Tuple[int, int], float] = {start: 0.0}
    while heap:
        cost, node, path, ids = heapq.heappop(heap)
        if node in goal_set:
            return path, ids, cost
        if cost > best.get(node, float("inf")):
            continue
        for nxt, edge_cost_value, edge_id in adjacency.get(node, []):
            new_cost = cost + edge_cost_value
            if new_cost >= best.get(nxt, float("inf")):
                continue
            best[nxt] = new_cost
            heapq.heappush(heap, (new_cost, nxt, path + [nxt], ids + [edge_id]))
    return None, [], None


def plan_route(
    blackboard: BlackboardStateV2,
    target: CandidatePOIV2,
    current_cell: Tuple[int, int],
    prev_distance: Optional[float] = None,
    route_context: Optional[Tuple[Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, str]]], List[Tuple[int, int]]]] = None,
) -> RoutePlanV2:
    if route_context is None:
        adjacency, goals = build_route_context(blackboard, target)
    else:
        adjacency, goals = route_context
    path, edge_ids, distance = _shortest_path(blackboard, current_cell, goals, adjacency=adjacency)
    if path and len(path) >= 2:
        next_subgoal = path[1]
        distance_prev = float(len(path) - 1)
        distance_estimate = float(len(path) - 2)
        delta = distance_prev - distance_estimate
        return RoutePlanV2(
            next_subgoal=next_subgoal,
            distance_estimate=distance_estimate,
            distance_prev=distance_prev,
            distance_delta=delta,
            progress_valid=True,
            progress_reason="navigation_path",
            fallback_mode=None,
            confidence=0.8,
            stalled=delta <= 0.0,
            blocked=False,
            route_edge_ids=edge_ids,
        )
    if path and len(path) == 1:
        return RoutePlanV2(
            next_subgoal=current_cell,
            distance_estimate=0.0,
            distance_prev=0.0,
            distance_delta=0.0 if prev_distance is None else prev_distance,
            progress_valid=True,
            progress_reason="already_at_access_cell",
            fallback_mode=None,
            confidence=0.9,
            stalled=False,
            blocked=False,
            route_edge_ids=edge_ids,
        )
    fallback_candidates = sorted(goals, key=lambda cell: point_manhattan(current_cell, cell))
    if fallback_candidates:
        next_subgoal = fallback_candidates[0]
        dist_prev = float(point_manhattan(current_cell, next_subgoal))
        return RoutePlanV2(
            next_subgoal=next_subgoal,
            distance_estimate=max(0.0, dist_prev - 1.0),
            distance_prev=dist_prev,
            distance_delta=(prev_distance - max(0.0, dist_prev - 1.0)) if prev_distance is not None else None,
            progress_valid=False,
            progress_reason="local_probe_subgoal",
            fallback_mode="local_probe",
            confidence=0.3,
            stalled=False,
            blocked=False,
            route_edge_ids=[],
        )
    return RoutePlanV2(
        next_subgoal=None,
        distance_estimate=None,
        distance_prev=prev_distance,
        distance_delta=None,
        progress_valid=False,
        progress_reason="no_route",
        fallback_mode="none",
        confidence=0.0,
        stalled=True,
        blocked=True,
        route_edge_ids=[],
    )
