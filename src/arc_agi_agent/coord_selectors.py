from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .full_explorer_config import FullExplorerConfig
from .types import BBox


@dataclass(frozen=True)
class CoordCandidate:
    x: int
    y: int
    selector: str


@dataclass
class ComponentWithCells:
    color: int
    cells: List[Tuple[int, int]]
    bbox: BBox
    centroid: Tuple[float, float]


def build_coords(
    grid: np.ndarray,
    bg_color: int,
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[CoordCandidate]:
    components = _extract_components_with_cells(grid, bg_color)
    selectors = {
        "object_centroids": _object_centroids,
        "object_bbox_corners": _object_bbox_corners,
        "adjacent_boundary_cells": _adjacent_boundary_cells,
        "region_frontier_cells": _region_frontier_cells,
        "grid_corners": _grid_corners,
        "grid_edges_midpoints": _grid_edges_midpoints,
        "changed_bbox_focus": _changed_bbox_focus,
        "color_hotspots": _color_hotspots,
    }

    candidates: List[CoordCandidate] = []
    for selector_name in cfg.selector_priority_order:
        selector_fn = selectors.get(selector_name)
        if selector_fn is None:
            continue
        coords = selector_fn(grid, bg_color, components, diff_bbox, cfg)
        coords_sorted = sorted(coords, key=lambda c: (c[1], c[0]))
        for x, y in coords_sorted:
            candidates.append(CoordCandidate(x=x, y=y, selector=selector_name))

    unique: Dict[Tuple[int, int], CoordCandidate] = {}
    for cand in candidates:
        key = (cand.x, cand.y)
        if key not in unique:
            unique[key] = cand

    ordered = list(unique.values())
    if len(ordered) > cfg.max_coords_per_state:
        ordered = ordered[: cfg.max_coords_per_state]
    return ordered


def _object_centroids(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    coords = []
    for comp in components:
        cx = int(round(comp.centroid[1]))
        cy = int(round(comp.centroid[0]))
        if _in_bounds(grid, cx, cy):
            coords.append((cx, cy))
    return coords


def _object_bbox_corners(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    coords = []
    for comp in components:
        y0, x0, y1, x1 = comp.bbox
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for x, y in corners:
            if _in_bounds(grid, x, y):
                coords.append((x, y))
    return coords


def _adjacent_boundary_cells(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    coords = []
    h, w = grid.shape
    for comp in components:
        cell_set = set(comp.cells)
        for y, x in comp.cells:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in cell_set:
                    coords.append((nx, ny))
    return coords


def _region_frontier_cells(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    coords = []
    for comp in components:
        frontier = _frontier_cells(grid, comp)
        if not frontier:
            continue
        stride = max(1, len(frontier) // cfg.max_perimeter_points_per_region)
        selected = frontier[::stride][: cfg.max_perimeter_points_per_region]
        coords.extend((x, y) for y, x in selected)
    return coords


def _grid_corners(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    h, w = grid.shape
    return [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]


def _grid_edges_midpoints(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    h, w = grid.shape
    return [(w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]


def _changed_bbox_focus(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    if diff_bbox is None:
        return []
    y0, x0, y1, x1 = diff_bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (cx, cy)]
    coords = [(x, y) for x, y in corners if _in_bounds(grid, x, y)]
    return coords


def _color_hotspots(
    grid: np.ndarray,
    bg_color: int,
    components: List[ComponentWithCells],
    diff_bbox: Optional[BBox],
    cfg: FullExplorerConfig,
) -> List[Tuple[int, int]]:
    colors = _top_colors(grid, bg_color, cfg.topK_colors)
    coords: List[Tuple[int, int]] = []
    comps_by_color: Dict[int, List[ComponentWithCells]] = {}
    for comp in components:
        comps_by_color.setdefault(comp.color, []).append(comp)
    for color in colors:
        comps = comps_by_color.get(color, [])
        comps_sorted = sorted(comps, key=lambda c: (len(c.cells), c.bbox))
        for comp in comps_sorted[: cfg.samples_per_color]:
            cx = int(round(comp.centroid[1]))
            cy = int(round(comp.centroid[0]))
            if (cx, cy) in coords:
                y, x = min(comp.cells)
                coords.append((x, y))
            else:
                coords.append((cx, cy))
    return coords


def _top_colors(grid: np.ndarray, bg_color: int, k: int) -> List[int]:
    values, counts = np.unique(grid, return_counts=True)
    pairs = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != bg_color]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return [p[0] for p in pairs[:k]]


def _frontier_cells(grid: np.ndarray, comp: ComponentWithCells) -> List[Tuple[int, int]]:
    h, w = grid.shape
    cells = set(comp.cells)
    frontier = []
    for y, x in comp.cells:
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in cells:
                frontier.append((y, x))
                break
    frontier.sort()
    return frontier


def _extract_components_with_cells(grid: np.ndarray, bg_color: int) -> List[ComponentWithCells]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[ComponentWithCells] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            color = int(grid[y, x])
            visited[y, x] = True
            if color == bg_color:
                continue
            stack = [(y, x)]
            cells = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        if int(grid[ny, nx]) == color:
                            stack.append((ny, nx))
                            cells.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            bbox: BBox = (min(ys), min(xs), max(ys), max(xs))
            centroid = (float(np.mean(ys)), float(np.mean(xs)))
            components.append(
                ComponentWithCells(color=color, cells=sorted(cells), bbox=bbox, centroid=centroid)
            )
    components.sort(key=lambda c: (c.color, c.bbox, len(c.cells)))
    return components


def _in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]
