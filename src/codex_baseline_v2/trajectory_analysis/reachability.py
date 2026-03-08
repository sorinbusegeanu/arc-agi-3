from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import CandidatePOIV2, ReachabilityRecordV2, SCHEMA_VERSION


def _build_graph(traversable_map: Optional[Dict[str, object]]) -> Dict[Tuple[int, int], int]:
    if not traversable_map:
        return {}
    points = traversable_map.get("points", []) if isinstance(traversable_map, dict) else []
    graph: Dict[Tuple[int, int], int] = {}
    for p in points:
        if not isinstance(p, dict):
            continue
        if "x" in p and "y" in p:
            graph[(int(p["x"]), int(p["y"]))] = int(p.get("visits", 1))
    return graph


def _bfs_distance_map(sources: List[Tuple[int, int]], graph: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    if not sources or not graph:
        return {}
    q: deque = deque()
    dist: Dict[Tuple[int, int], int] = {}
    for s in sources:
        if s in graph:
            dist[s] = 0
            q.append(s)
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


def _poi_distance(dist_map: Dict[Tuple[int, int], int], poi: CandidatePOIV2) -> Optional[int]:
    if not dist_map:
        return None
    best = None
    for x in range(poi.bbox.x1, poi.bbox.x2 + 1):
        for y in range(poi.bbox.y1, poi.bbox.y2 + 1):
            if (x, y) in dist_map:
                d = dist_map[(x, y)]
                best = d if best is None or d < best else best
    return best


def classify_reachability(
    pois: List[CandidatePOIV2],
    avatar_centroids: List[Tuple[float, float]],
    traversable_map: Optional[Dict[str, object]],
) -> List[ReachabilityRecordV2]:
    graph = _build_graph(traversable_map)
    has_avatar = bool(avatar_centroids)
    reachability: List[ReachabilityRecordV2] = []
    if not has_avatar:
        for poi in pois:
            status = "likely_hud" if poi.object_class == "hud_like" else "unknown_avatar"
            reason = "hud_like_region" if status == "likely_hud" else "avatar_missing"
            reachability.append(
                ReachabilityRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=poi.game_id,
                    poi_id=poi.poi_id,
                    status=status,
                    confidence=0.2 if status == "likely_hud" else 0.3,
                    distance_estimate=None,
                    evidence_refs=[],
                    reason_code=reason,
                )
            )
        return reachability
    if not graph:
        for poi in pois:
            status = "likely_hud" if poi.object_class == "hud_like" else "unknown_traversable"
            reason = "hud_like_region" if status == "likely_hud" else "traversable_missing"
            reachability.append(
                ReachabilityRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=poi.game_id,
                    poi_id=poi.poi_id,
                    status=status,
                    confidence=0.2 if status == "likely_hud" else 0.3,
                    distance_estimate=None,
                    evidence_refs=[],
                    reason_code=reason,
                )
            )
        return reachability
    sources = [(int(round(x)), int(round(y))) for x, y in avatar_centroids]
    dist_map = _bfs_distance_map(sources, graph)
    for poi in pois:
        if poi.object_class == "hud_like":
            reachability.append(
                ReachabilityRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=poi.game_id,
                    poi_id=poi.poi_id,
                    status="likely_hud",
                    confidence=0.2,
                    distance_estimate=None,
                    evidence_refs=[],
                    reason_code="hud_like_region",
                )
            )
            continue
        distance = _poi_distance(dist_map, poi)
        if distance is None:
            reachability.append(
                ReachabilityRecordV2(
                    schema_version=SCHEMA_VERSION,
                    game_id=poi.game_id,
                    poi_id=poi.poi_id,
                    status="insufficient_evidence",
                    confidence=0.35,
                    distance_estimate=None,
                    evidence_refs=[],
                    reason_code="no_graph_path",
                )
            )
            continue
        status = "reachable_now" if distance <= 2 else "unreachable_now"
        reachability.append(
            ReachabilityRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=poi.game_id,
                poi_id=poi.poi_id,
                status=status,
                confidence=0.8 if status == "reachable_now" else 0.6,
                distance_estimate=float(distance),
                evidence_refs=[],
                reason_code="graph_distance",
            )
        )
    return reachability
