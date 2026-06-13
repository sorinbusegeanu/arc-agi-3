from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Delta:
    id: int
    changed_cells: int
    changed_positions: list[tuple[int, int]]
    colors_added: list[int]
    colors_removed: list[int]
    centroid_before_x: float
    centroid_before_y: float
    centroid_after_x: float
    centroid_after_y: float
    dx: float
    dy: float


class DeltaExtractor:
    def extract(self, before_grid: np.ndarray, after_grid: np.ndarray, delta_id: int) -> Delta:
        return extract_delta(before_grid, after_grid, delta_id=delta_id)


def extract_delta(before_grid: np.ndarray, after_grid: np.ndarray, delta_id: int = 0) -> Delta:
    before = np.asarray(before_grid, dtype=int)
    after = np.asarray(after_grid, dtype=int)
    if before.shape != after.shape:
        raise ValueError(f"before/after shape mismatch: {before.shape} != {after.shape}")
    if before.ndim != 2:
        raise ValueError(f"expected 2D ARC grids, got shape {before.shape}")

    changed_mask = before != after
    positions_array = np.argwhere(changed_mask)
    changed_positions = [(int(y), int(x)) for y, x in positions_array]
    changed_cells = len(changed_positions)

    before_colors = set(int(value) for value in np.unique(before))
    after_colors = set(int(value) for value in np.unique(after))
    colors_added = sorted(after_colors - before_colors)
    colors_removed = sorted(before_colors - after_colors)

    before_centroid, after_centroid = _signed_change_centroids(before, after, changed_mask)
    dx = after_centroid[0] - before_centroid[0]
    dy = after_centroid[1] - before_centroid[1]

    return Delta(
        id=int(delta_id),
        changed_cells=int(changed_cells),
        changed_positions=changed_positions,
        colors_added=colors_added,
        colors_removed=colors_removed,
        centroid_before_x=float(before_centroid[0]),
        centroid_before_y=float(before_centroid[1]),
        centroid_after_x=float(after_centroid[0]),
        centroid_after_y=float(after_centroid[1]),
        dx=float(dx),
        dy=float(dy),
    )


def _signed_change_centroids(
    before: np.ndarray,
    after: np.ndarray,
    changed_mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    signed_change = before.astype(float) - after.astype(float)
    before_positions = np.argwhere(changed_mask & (signed_change > 0))
    after_positions = np.argwhere(changed_mask & (signed_change < 0))
    if before_positions.size > 0 and after_positions.size > 0:
        return _centroid(before_positions), _centroid(after_positions)
    fallback = np.argwhere(changed_mask)
    centroid = _centroid(fallback)
    return centroid, centroid


def _centroid(positions: np.ndarray) -> tuple[float, float]:
    if positions.size == 0:
        return (0.0, 0.0)
    y_mean = float(np.mean(positions[:, 0]))
    x_mean = float(np.mean(positions[:, 1]))
    return (x_mean, y_mean)
