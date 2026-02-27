from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .types import BBox


def grid_hash(grids: List[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for grid in grids:
        hasher.update(grid.tobytes())
        hasher.update(str(grid.shape).encode("utf-8"))
    return hasher.hexdigest()


def palette(grid: np.ndarray) -> List[int]:
    return sorted(int(v) for v in np.unique(grid))


def color_histogram(grid: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def bg_candidates(
    grid: np.ndarray, weights: Dict[str, float]
) -> List[Tuple[int, float]]:
    h, w = grid.shape
    total = h * w
    hist = color_histogram(grid)
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    border_hist = {int(v): int(c) for v, c in zip(*np.unique(border, return_counts=True))}
    candidates: List[Tuple[int, float]] = []
    for color, count in hist.items():
        freq_score = count / total
        border_score = border_hist.get(color, 0) / max(1, border.size)
        connected_score = _largest_component_fraction(grid, color)
        score = (
            weights.get("frequency", 0.0) * freq_score
            + weights.get("border", 0.0) * border_score
            + weights.get("connectedness", 0.0) * connected_score
        )
        candidates.append((color, float(score)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _largest_component_fraction(grid: np.ndarray, color: int) -> float:
    mask = grid == color
    if not mask.any():
        return 0.0
    visited = np.zeros(mask.shape, dtype=bool)
    max_area = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and not visited[y, x]:
                area = 0
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                max_area = max(max_area, area)
    return max_area / mask.size


def diff_mask(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return curr != prev


def changed_bbox(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def changed_colors(curr: np.ndarray, prev: np.ndarray) -> Dict[str, int]:
    diff = curr != prev
    ys, xs = np.where(diff)
    counts: Dict[str, int] = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        key = f"{int(prev[y, x])}->{int(curr[y, x])}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def symmetry_scores(grid: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    scores["horizontal"] = _sym_score(grid, np.flipud(grid))
    scores["vertical"] = _sym_score(grid, np.fliplr(grid))
    scores["diag"] = _sym_score(grid, grid.T) if grid.shape[0] == grid.shape[1] else 0.0
    scores["anti_diag"] = (
        _sym_score(grid, np.fliplr(np.flipud(grid)).T)
        if grid.shape[0] == grid.shape[1]
        else 0.0
    )
    scores["rot_180"] = _sym_score(grid, np.rot90(grid, 2))
    scores["rot_90"] = _sym_score(grid, np.rot90(grid, 1)) if grid.shape[0] == grid.shape[1] else 0.0
    return scores


def _sym_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float((a == b).sum() / a.size)


def periodicity(grid: np.ndarray, max_period: int) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    results.extend(_axis_periodicity(grid, axis=0, max_period=max_period))
    results.extend(_axis_periodicity(grid, axis=1, max_period=max_period))
    return results


def _axis_periodicity(
    grid: np.ndarray, axis: int, max_period: int
) -> List[Dict[str, float]]:
    size = grid.shape[axis]
    max_p = min(max_period, size)
    results: List[Dict[str, float]] = []
    for p in range(1, max_p + 1):
        if size % p != 0:
            continue
        ok = True
        if axis == 0:
            for i in range(p, size):
                if not np.array_equal(grid[i], grid[i % p]):
                    ok = False
                    break
        else:
            for i in range(p, size):
                if not np.array_equal(grid[:, i], grid[:, i % p]):
                    ok = False
                    break
        if ok:
            results.append({"axis": axis, "period": p, "score": 1.0})
    return results


def border_color_invariant(grid: np.ndarray) -> Optional[int]:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[:, 0]
    right = grid[:, -1]
    border = np.concatenate([top, bottom, left, right])
    values = np.unique(border)
    if values.size == 1:
        return int(values[0])
    return None


def count_active_regions(mask: np.ndarray) -> Dict[str, int]:
    ys, xs = np.where(mask)
    return {"count": int(ys.size)}


def bbox_area(bbox: Optional[BBox]) -> int:
    if bbox is None:
        return 0
    y0, x0, y1, x1 = bbox
    return max(0, int(y1) - int(y0) + 1) * max(0, int(x1) - int(x0) + 1)


def grid_from_ascii(ascii_grid: str) -> np.ndarray:
    lines = ascii_grid.splitlines()
    if not lines:
        return np.zeros((0, 0), dtype=int)
    content = lines[1:]
    glyphs = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mapping = {ch: idx for idx, ch in enumerate(glyphs)}
    rows = []
    for line in content:
        if not line.strip():
            continue
        row_str = line[3:]
        row = [mapping.get(ch, 0) for ch in row_str]
        rows.append(row)
    return np.array(rows, dtype=int)
