from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from codex_baseline_v2.shared.config import AreaModelConfigV2
from codex_baseline_v2.shared.schemas import AreaStateV2, ChangeEventV2, NavigationEdgeV2, TopologyDeltaV2, SCHEMA_VERSION


def _edge_set(edges: List[NavigationEdgeV2]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return {(row.src_cell, row.dst_cell) for row in edges if row.success_count > 0}


def _changed_cells(event: ChangeEventV2) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for region in event.region_deltas:
        for y in range(region.bbox.y1, region.bbox.y2 + 1):
            for x in range(region.bbox.x1, region.bbox.x2 + 1):
                out.add((x, y))
    return out


def _path_delta(new_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]], removed_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Optional[float]:
    if not new_edges and not removed_edges:
        return None
    return float(len(new_edges) - len(removed_edges))


def infer_topology_deltas(
    events: List[ChangeEventV2],
    navigation_before: List[NavigationEdgeV2],
    navigation_after: List[NavigationEdgeV2],
    areas: List[AreaStateV2],
    cfg: AreaModelConfigV2,
) -> list[TopologyDeltaV2]:
    del areas, cfg
    before = _edge_set(navigation_before)
    after = _edge_set(navigation_after)
    opened = after - before
    closed = before - after
    deltas: List[TopologyDeltaV2] = []
    for event in events:
        if event.event_type not in {"transition", "mixed", "object_state_change"}:
            continue
        affected = _changed_cells(event)
        local_opened = sorted(edge for edge in opened if edge[0] in affected or edge[1] in affected)
        local_closed = sorted(edge for edge in closed if edge[0] in affected or edge[1] in affected)
        if not local_opened and not local_closed and event.locality != "cross_area_transition":
            continue
        deltas.append(
            TopologyDeltaV2(
                schema_version=SCHEMA_VERSION,
                game_id=event.game_id,
                delta_id="topology:%s" % event.event_id,
                event_id=event.event_id,
                pre_area_id=event.pre_area_id,
                post_area_id=event.post_area_id,
                new_edges=local_opened,
                removed_edges=local_closed,
                opened_chokepoints=sorted({edge[1] for edge in local_opened}),
                closed_chokepoints=sorted({edge[1] for edge in local_closed}),
                connectivity_changed=bool(local_opened or local_closed or event.locality == "cross_area_transition"),
                path_length_delta=_path_delta(set(local_opened), set(local_closed)),
                confidence=0.7 if (local_opened or local_closed) else 0.5,
            )
        )
    return deltas
