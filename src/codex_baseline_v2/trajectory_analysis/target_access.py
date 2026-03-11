from __future__ import annotations

from typing import Dict, List, Tuple

from codex_baseline_v2.shared.config import RoutingConfigV2
from codex_baseline_v2.shared.schemas import (
    AvatarTrackHypothesisV2,
    CandidatePOIV2,
    NavigationEdgeV2,
    NavigationStateCellV2,
    TargetAccessProfileV2,
    SCHEMA_VERSION,
)
from codex_baseline_v2.shared.utils import point_manhattan, point_neighbors4


def _bbox_cells(poi: CandidatePOIV2) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for y in range(poi.bbox.y1, poi.bbox.y2 + 1):
        for x in range(poi.bbox.x1, poi.bbox.x2 + 1):
            out.append((x, y))
    return out


def _edge_lookup(navigation_edges: List[NavigationEdgeV2]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for edge in navigation_edges:
        counts[edge.dst_cell] = counts.get(edge.dst_cell, 0) + edge.success_count
    return counts


def infer_target_access_profiles(
    pois: List[CandidatePOIV2],
    navigation_cells: List[NavigationStateCellV2],
    navigation_edges: List[NavigationEdgeV2],
    avatar_tracks: List[AvatarTrackHypothesisV2],
    cfg: RoutingConfigV2,
) -> list[TargetAccessProfileV2]:
    known_cells = {row.cell: row for row in navigation_cells}
    inbound_counts = _edge_lookup(navigation_edges)
    avatar_anchor = None
    if avatar_tracks:
        best = max(avatar_tracks, key=lambda row: (row.posterior, row.support_count))
        avatar_anchor = (int(round(best.centroid[0])), int(round(best.centroid[1])))
    profiles: List[TargetAccessProfileV2] = []

    for poi in pois:
        poi_cells = set(_bbox_cells(poi))
        centroid = (int(round(poi.centroid[0])), int(round(poi.centroid[1])))
        overlap_cells = [cell for cell in poi_cells if cell in known_cells]
        adjacent_cells = [cell for cell in {n for cell in poi_cells for n in point_neighbors4(cell)} if cell in known_cells and cell not in poi_cells]
        access_cells = sorted(set(overlap_cells + adjacent_cells))
        if overlap_cells:
            contact_mode = "stand_on" if len(poi_cells) == 1 else "overlap_contact"
        elif adjacent_cells:
            contact_mode = "adjacent_contact"
        elif avatar_anchor is not None and point_manhattan(avatar_anchor, centroid) <= cfg.local_probe_radius + 1:
            contact_mode = "proximity_only"
        else:
            contact_mode = "unknown"
        preferred_sides: List[str] = []
        blocked_sides: List[str] = []
        scored_access = sorted(
            access_cells,
            key=lambda cell: (
                -known_cells[cell].visit_count,
                -inbound_counts.get(cell, 0),
                point_manhattan(cell, centroid),
            ),
        )
        if scored_access:
            best_cell = scored_access[0]
            if best_cell[0] < poi.bbox.x1:
                preferred_sides.append("left")
            if best_cell[0] > poi.bbox.x2:
                preferred_sides.append("right")
            if best_cell[1] < poi.bbox.y1:
                preferred_sides.append("top")
            if best_cell[1] > poi.bbox.y2:
                preferred_sides.append("bottom")
        for side, probe in (
            ("left", (poi.bbox.x1 - 1, int(round(poi.centroid[1])))),
            ("right", (poi.bbox.x2 + 1, int(round(poi.centroid[1])))),
            ("top", (int(round(poi.centroid[0])), poi.bbox.y1 - 1)),
            ("bottom", (int(round(poi.centroid[0])), poi.bbox.y2 + 1)),
        ):
            if probe not in known_cells:
                blocked_sides.append(side)
        confidence = 0.2
        if access_cells:
            confidence = 0.55 + min(0.35, len(access_cells) * 0.05)
        if preferred_sides:
            confidence += 0.05
        if contact_mode == "unknown":
            confidence = min(confidence, 0.35)
        profiles.append(
            TargetAccessProfileV2(
                schema_version=SCHEMA_VERSION,
                game_id=poi.game_id,
                poi_id=poi.poi_id,
                contact_mode=contact_mode,
                access_cells=scored_access[:8],
                blocked_sides=blocked_sides,
                preferred_sides=preferred_sides,
                stand_off_distance=max(0, cfg.local_probe_radius - 1 if contact_mode == "proximity_only" else 0),
                confidence=min(1.0, confidence),
            )
        )
    return profiles
