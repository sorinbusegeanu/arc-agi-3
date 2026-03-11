from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import RoutingConfigV2
from codex_baseline_v2.shared.schemas import (
    AvatarTrackHypothesisV2,
    CandidatePOIV2,
    NavigationEdgeV2,
    NavigationStateCellV2,
    ReachabilityRecordV2,
    TargetAccessProfileV2,
    SCHEMA_VERSION,
)
from codex_baseline_v2.shared.utils import point_manhattan


def _edge_cost(edge: NavigationEdgeV2, cfg: RoutingConfigV2) -> float:
    cost = 1.0
    cost += cfg.blocked_edge_penalty * float(edge.blocked_count > edge.success_count)
    cost += cfg.unknown_edge_penalty * float(edge.confidence < 0.4)
    cost += cfg.transition_edge_penalty * float(edge.transition_type == "transition")
    return cost


def _best_avatar_anchor(avatar_tracks: List[AvatarTrackHypothesisV2]) -> Optional[Tuple[int, int]]:
    if not avatar_tracks:
        return None
    best = max(avatar_tracks, key=lambda track: track.posterior)
    return (int(round(best.centroid[0])), int(round(best.centroid[1])))


def _dijkstra(
    start: Tuple[int, int],
    goal_cells: List[Tuple[int, int]],
    navigation_edges: List[NavigationEdgeV2],
    cfg: RoutingConfigV2,
) -> Tuple[Optional[float], List[str]]:
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, str]]] = {}
    for edge in navigation_edges:
        adjacency.setdefault(edge.src_cell, []).append((edge.dst_cell, _edge_cost(edge, cfg), edge.edge_id))
    goal_set = set(goal_cells)
    heap: List[Tuple[float, Tuple[int, int], List[str]]] = [(0.0, start, [])]
    best_cost: Dict[Tuple[int, int], float] = {start: 0.0}
    while heap:
        cost, cell, route_ids = heapq.heappop(heap)
        if cell in goal_set:
            return cost, route_ids
        if cost > best_cost.get(cell, float("inf")):
            continue
        for nxt, edge_cost_value, edge_id in adjacency.get(cell, []):
            new_cost = cost + edge_cost_value
            if new_cost >= best_cost.get(nxt, float("inf")):
                continue
            best_cost[nxt] = new_cost
            heapq.heappush(heap, (new_cost, nxt, route_ids + [edge_id]))
    return None, []


def classify_reachability(
    pois: List[CandidatePOIV2],
    avatar_tracks: List[AvatarTrackHypothesisV2],
    navigation_cells: List[NavigationStateCellV2],
    navigation_edges: List[NavigationEdgeV2],
    access_profiles: List[TargetAccessProfileV2],
    cfg: RoutingConfigV2,
) -> List[ReachabilityRecordV2]:
    anchor = _best_avatar_anchor(avatar_tracks)
    access_by_poi = {profile.poi_id: profile for profile in access_profiles}
    known_cells = {cell.cell for cell in navigation_cells}
    results: List[ReachabilityRecordV2] = []
    for poi in pois:
        profile = access_by_poi.get(poi.poi_id)
        access_cells = [cell for cell in (profile.access_cells if profile else []) if cell in known_cells]
        if anchor is None:
            status = "uncertain"
            reason = "avatar_missing"
            distance = None
            route_edge_ids: List[str] = []
            confidence = 0.2
        elif not access_cells:
            status = "cross_area_only" if poi.area_id else "uncertain"
            reason = "no_access_cells"
            distance = None
            route_edge_ids = []
            confidence = 0.25
        else:
            distance, route_edge_ids = _dijkstra(anchor, access_cells, navigation_edges, cfg)
            if distance is None:
                status = "blocked"
                reason = "no_graph_path"
                confidence = 0.3
            elif distance <= 2.0:
                status = "reachable"
                reason = "graph_path"
                confidence = 0.85
            else:
                status = "unreachable"
                reason = "graph_path_far"
                confidence = 0.55
        results.append(
            ReachabilityRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=poi.game_id,
                poi_id=poi.poi_id,
                status=status,
                confidence=confidence,
                distance_estimate=distance,
                evidence_refs=[],
                reason_code=reason,
                area_id=poi.area_id,
                route_edge_ids=route_edge_ids,
                access_profile_id=profile.poi_id if profile else None,
                progress_confidence=(profile.confidence if profile else 0.0),
            )
        )
    return results
