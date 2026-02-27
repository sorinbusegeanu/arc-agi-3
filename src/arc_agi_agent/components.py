from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .types import BBox, Component, ObjectDelta


def extract_components(
    grid: np.ndarray,
    colors: Iterable[int],
    connectivity: int,
    min_area: int,
    max_objects: int,
    grid_name: Optional[str] = None,
) -> List[Component]:
    comps: List[Component] = []
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    neighbors = _neighbors_8 if connectivity == 8 else _neighbors_4
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            color = int(grid[y, x])
            if color not in colors:
                visited[y, x] = True
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in neighbors(cy, cx):
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and int(grid[ny, nx]) == color:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(coords)
            if area < min_area:
                continue
            ys = [c[0] for c in coords]
            xs = [c[1] for c in coords]
            bbox: BBox = (min(ys), min(xs), max(ys), max(xs))
            centroid = (float(np.mean(ys)), float(np.mean(xs)))
            comp_id = f"{color}:{len(comps)}"
            comps.append(
                Component(
                    id=comp_id,
                    color=color,
                    area=area,
                    bbox=bbox,
                    centroid=centroid,
                    grid_name=grid_name,
                )
            )
            if len(comps) >= max_objects:
                return comps
    return comps


def track_components(
    prev: List[Component],
    curr: List[Component],
    iou_threshold: float,
    iou_soft_threshold: float,
    centroid_distance_threshold: float,
) -> List[ObjectDelta]:
    deltas: List[ObjectDelta] = []
    unmatched_prev = set(range(len(prev)))
    for curr_idx, curr_comp in enumerate(curr):
        best_match = None
        best_iou = -1.0
        best_dist = float("inf")
        for prev_idx in list(unmatched_prev):
            prev_comp = prev[prev_idx]
            if prev_comp.color != curr_comp.color:
                continue
            iou = bbox_iou(prev_comp.bbox, curr_comp.bbox)
            dist = centroid_distance(prev_comp.centroid, curr_comp.centroid)
            if _is_match(iou, dist, iou_threshold, iou_soft_threshold, centroid_distance_threshold):
                if iou > best_iou or (iou == best_iou and dist < best_dist):
                    best_match = prev_idx
                    best_iou = iou
                    best_dist = dist
        if best_match is not None:
            prev_comp = prev[best_match]
            unmatched_prev.discard(best_match)
            dy = curr_comp.centroid[0] - prev_comp.centroid[0]
            dx = curr_comp.centroid[1] - prev_comp.centroid[1]
            event = "moved" if abs(dy) > 0 or abs(dx) > 0 else "static"
            deltas.append(
                ObjectDelta(
                    object_id=curr_comp.id,
                    color=curr_comp.color,
                    prev_bbox=prev_comp.bbox,
                    curr_bbox=curr_comp.bbox,
                    dy=dy,
                    dx=dx,
                    event=event,
                )
            )
        else:
            deltas.append(
                ObjectDelta(
                    object_id=curr_comp.id,
                    color=curr_comp.color,
                    prev_bbox=None,
                    curr_bbox=curr_comp.bbox,
                    dy=0.0,
                    dx=0.0,
                    event="appeared",
                )
            )
    for prev_idx in unmatched_prev:
        prev_comp = prev[prev_idx]
        deltas.append(
            ObjectDelta(
                object_id=prev_comp.id,
                color=prev_comp.color,
                prev_bbox=prev_comp.bbox,
                curr_bbox=None,
                dy=0.0,
                dx=0.0,
                event="disappeared",
            )
        )
    return deltas


def _neighbors_4(y: int, x: int) -> List[Tuple[int, int]]:
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def _neighbors_8(y: int, x: int) -> List[Tuple[int, int]]:
    return [
        (y - 1, x - 1),
        (y - 1, x),
        (y - 1, x + 1),
        (y, x - 1),
        (y, x + 1),
        (y + 1, x - 1),
        (y + 1, x),
        (y + 1, x + 1),
    ]


def bbox_iou(a: BBox, b: BBox) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    inter_y0 = max(ay0, by0)
    inter_x0 = max(ax0, bx0)
    inter_y1 = min(ay1, by1)
    inter_x1 = min(ax1, bx1)
    if inter_y1 < inter_y0 or inter_x1 < inter_x0:
        return 0.0
    inter_area = (inter_y1 - inter_y0 + 1) * (inter_x1 - inter_x0 + 1)
    area_a = (ay1 - ay0 + 1) * (ax1 - ax0 + 1)
    area_b = (by1 - by0 + 1) * (bx1 - bx0 + 1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _is_match(
    iou: float,
    dist: float,
    iou_threshold: float,
    iou_soft_threshold: float,
    centroid_distance_threshold: float,
) -> bool:
    if iou >= iou_threshold:
        return True
    if iou >= iou_soft_threshold and dist <= centroid_distance_threshold:
        return True
    return False
