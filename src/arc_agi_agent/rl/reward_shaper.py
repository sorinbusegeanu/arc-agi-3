from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np

from .canonical_grid import stable_hash_grid
from ..grid_utils import changed_bbox as changed_bbox_inclusive, diff_mask
from ..viz import save_grid_image

logger = logging.getLogger(__name__)

_MATCH_POI_SCORE_THRESHOLD = 0.3
_MATCH_POI_MIN_BBOX_SIZE = 3
_MATCH_POI_DEBUG_CANDIDATE_LOG_LIMIT = 4


def _debug_match_poi_enabled() -> bool:
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def _write_match_poi_debug_pngs(
    game_id: str,
    grid_prev: np.ndarray,
    grid_curr: np.ndarray,
) -> Optional[Dict[str, str]]:
    if grid_prev.ndim != 2 or grid_curr.ndim != 2:
        return None
    out_dir = os.path.join(os.getcwd(), "runs", "match_poi_debug")
    os.makedirs(out_dir, exist_ok=True)
    prev_hash = stable_hash_grid(grid_prev)
    curr_hash = stable_hash_grid(grid_curr)
    prefix = f"{str(game_id or 'unknown')}_{prev_hash[:12]}_{curr_hash[:12]}"
    prev_path = os.path.join(out_dir, f"{prefix}_prev.png")
    curr_path = os.path.join(out_dir, f"{prefix}_new.png")
    saved_prev = save_grid_image(prev_path, np.asarray(grid_prev, dtype=np.int64))
    saved_curr = save_grid_image(curr_path, np.asarray(grid_curr, dtype=np.int64))
    if not saved_prev or not saved_curr:
        return None
    return {
        "prev": prev_path,
        "new": curr_path,
    }


def _default_cfg() -> Dict[str, Any]:
    return {
        "movement_cell_thresh": 55,
        "noop_cell_thresh": 10,
        "r_effect_movement": 0.01,
        "r_effect_screen": 0.1,
        "match_poi": 0.5,
        "revert_penalty": 0.2,
        "beta_potential": 0.05,
        "gamma": 0.995,
        "step_penalty_rate": 0.0005,
        "step_penalty_cap": 0.1,
        "flash_changed_total_thresh": 0.5,
        "flash_changed_masked_thresh": 0.5,
        "flash_hist_l1_thresh": 0.15,
    }


def _match_poi_blocks(
    grid_prev: np.ndarray,
    grid_curr: np.ndarray,
    grid_prev_prev: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    debug_enabled = _debug_match_poi_enabled()

    def _debug_log(message: str, *args: Any) -> None:
        if debug_enabled:
            logger.debug(message, *args)

    if grid_prev.ndim != 2 or grid_curr.ndim != 2:
        _debug_log(
            "match_poi_skip reason=invalid_ndim prev_ndim=%s curr_ndim=%s",
            getattr(grid_prev, "ndim", None),
            getattr(grid_curr, "ndim", None),
        )
        return None
    if grid_prev.shape != grid_curr.shape:
        _debug_log(
            "match_poi_skip reason=shape_mismatch prev_shape=%s curr_shape=%s",
            tuple(grid_prev.shape),
            tuple(grid_curr.shape),
        )
        return None

    diff = np.asarray(diff_mask(grid_curr, grid_prev), dtype=bool)
    if not bool(np.any(diff)):
        _debug_log("match_poi_skip reason=no_changed_bbox")
        return None

    changed_cells = int(diff.sum())
    height, width = int(grid_curr.shape[0]), int(grid_curr.shape[1])
    blue_colors = {9, 10}
    dark_colors = {0, 4, 5}
    proposal_top_k = 12
    perimeter_dominance_threshold = 0.72

    def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        return int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)

    def _bbox_is_large_enough(bbox: tuple[int, int, int, int]) -> bool:
        h, w = _bbox_hw(bbox)
        return h >= _MATCH_POI_MIN_BBOX_SIZE and w >= _MATCH_POI_MIN_BBOX_SIZE

    def _bbox_label(bbox: tuple[int, int, int, int]) -> str:
        h, w = _bbox_hw(bbox)
        return f"[y={bbox[0]},x={bbox[1]},h={h},w={w}]"

    def _clip_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        y0, x0, y1, x1 = bbox
        y0 = max(0, min(height - 1, int(y0)))
        x0 = max(0, min(width - 1, int(x0)))
        y1 = max(y0, min(height - 1, int(y1)))
        x1 = max(x0, min(width - 1, int(x1)))
        return y0, x0, y1, x1

    def _expand_bbox(bbox: tuple[int, int, int, int], pad_y: int, pad_x: Optional[int] = None) -> tuple[int, int, int, int]:
        pad_x = pad_y if pad_x is None else pad_x
        return _clip_bbox((bbox[0] - pad_y, bbox[1] - pad_x, bbox[2] + pad_y, bbox[3] + pad_x))

    def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        y0, x0, y1, x1 = bbox
        return np.ascontiguousarray(grid[y0 : y1 + 1, x0 : x1 + 1], dtype=np.int64)

    def _dominant_color(arr: np.ndarray, fallback: int = 0) -> int:
        flat = np.asarray(arr, dtype=np.int64).reshape(-1)
        if flat.size == 0:
            return int(fallback)
        vals, counts = np.unique(flat, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def _changed_component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        seen = np.zeros(mask.shape, dtype=bool)
        boxes: list[tuple[int, int, int, int]] = []
        for sy in range(mask.shape[0]):
            for sx in range(mask.shape[1]):
                if not mask[sy, sx] or seen[sy, sx]:
                    continue
                stack = [(sy, sx)]
                seen[sy, sx] = True
                min_y = max_y = sy
                min_x = max_x = sx
                while stack:
                    cy, cx = stack.pop()
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                boxes.append((int(min_y), int(min_x), int(max_y), int(max_x)))
        return boxes

    def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        gap_y = max(0, max(a[0] - b[2] - 1, b[0] - a[2] - 1))
        gap_x = max(0, max(a[1] - b[3] - 1, b[1] - a[3] - 1))
        return max(gap_y, gap_x)

    def _merge_close_bboxes(boxes: list[tuple[int, int, int, int]], gap_thresh: int = 4) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for idx, box in enumerate(boxes):
            if used[idx]:
                continue
            group = [box]
            used[idx] = True
            changed = True
            while changed:
                changed = False
                current = (
                    min(b[0] for b in group),
                    min(b[1] for b in group),
                    max(b[2] for b in group),
                    max(b[3] for b in group),
                )
                for jdx, other in enumerate(boxes):
                    if used[jdx]:
                        continue
                    if _bbox_gap(current, other) <= gap_thresh:
                        group.append(other)
                        used[jdx] = True
                        changed = True
            merged.append(
                (
                    min(b[0] for b in group),
                    min(b[1] for b in group),
                    max(b[2] for b in group),
                    max(b[3] for b in group),
                )
            )
        return merged

    def _frame_objectness(crop: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(crop, dtype=np.int64)
        h, w = arr.shape
        if h == 0 or w == 0:
            return {
                "border_ratio": 0.0,
                "dark_ratio": 0.0,
                "blue_ratio": 0.0,
                "rect_fill": 0.0,
                "frame_indicator": 0.0,
            }
        if h == 1 or w == 1:
            dom = _dominant_color(arr)
            blue_ratio = float(np.mean(np.isin(arr, list(blue_colors))))
            dark_ratio = float(np.mean(np.isin(arr, list(dark_colors))))
            return {
                "border_ratio": 1.0,
                "dark_ratio": dark_ratio,
                "blue_ratio": blue_ratio,
                "rect_fill": 1.0 if dom != 0 else 0.0,
                "frame_indicator": 0.5,
            }
        border = np.concatenate([arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]])
        inner = arr[1:-1, 1:-1] if h > 2 and w > 2 else arr
        border_color = _dominant_color(border)
        inner_color = _dominant_color(inner, fallback=border_color)
        border_ratio = float(np.mean(border == border_color)) if border.size else 0.0
        dark_ratio = float(np.mean(np.isin(inner, list(dark_colors)))) if inner.size else 0.0
        blue_ratio = float(np.mean(np.isin(inner, list(blue_colors)))) if inner.size else 0.0
        rect_fill = float(np.mean(arr != border_color))
        frame_indicator = 0.5 * border_ratio + 0.3 * float(border_color != inner_color) + 0.2 * max(0.0, 1.0 - rect_fill)
        return {
            "border_ratio": border_ratio,
            "dark_ratio": dark_ratio,
            "blue_ratio": blue_ratio,
            "rect_fill": rect_fill,
            "frame_indicator": float(max(0.0, min(1.0, frame_indicator))),
        }

    def _perimeter_stats(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> Dict[str, float]:
        crop = _crop(grid, bbox)
        h, w = crop.shape
        if h < 2 or w < 2:
            return {
                "dominant_fraction": 0.0,
                "coverage": 0.0,
                "rectangularity": 0.0,
                "interior_occupancy": 0.0,
                "interior_diversity": 0.0,
                "frame_quality": 0.0,
            }
        perimeter_mask = np.zeros((h, w), dtype=bool)
        perimeter_mask[0, :] = True
        perimeter_mask[-1, :] = True
        perimeter_mask[:, 0] = True
        perimeter_mask[:, -1] = True
        perimeter = crop[perimeter_mask]
        vals, counts = np.unique(perimeter, return_counts=True)
        dom_idx = int(np.argmax(counts)) if counts.size else 0
        dom_color = int(vals[dom_idx]) if vals.size else 0
        dominant_fraction = float(counts[dom_idx] / max(1, perimeter.size)) if counts.size else 0.0
        coverage = float(np.mean(perimeter == dom_color)) if perimeter.size else 0.0
        rectangularity = float(perimeter_mask.sum()) / float(max(1, 2 * h + 2 * w - 4))
        inner = crop[1:-1, 1:-1] if h > 2 and w > 2 else crop
        interior_occupancy = float(np.mean(inner != dom_color)) if inner.size else 0.0
        interior_diversity = float(len(np.unique(inner)) / max(1, min(8, inner.size))) if inner.size else 0.0
        frame_quality = (
            0.55 * dominant_fraction
            + 0.15 * coverage
            + 0.15 * rectangularity
            + 0.10 * min(1.0, interior_occupancy * 2.0)
            + 0.05 * min(1.0, interior_diversity)
        )
        return {
            "dominant_fraction": dominant_fraction,
            "coverage": coverage,
            "rectangularity": rectangularity,
            "interior_occupancy": interior_occupancy,
            "interior_diversity": interior_diversity,
            "frame_quality": float(max(0.0, min(1.0, frame_quality))),
        }

    def complete_bbox_by_perimeter_dominance(grid: np.ndarray, seed_bbox: tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        best_bbox = None
        best_stats = None
        best_score = -1.0
        for pad_top in range(0, 5):
            for pad_bottom in range(0, 5):
                for pad_left in range(0, 5):
                    for pad_right in range(0, 5):
                        cand = _clip_bbox((
                            seed_bbox[0] - pad_top,
                            seed_bbox[1] - pad_left,
                            seed_bbox[2] + pad_bottom,
                            seed_bbox[3] + pad_right,
                        ))
                        if not _bbox_is_large_enough(cand):
                            continue
                        stats = _perimeter_stats(grid, cand)
                        area_bonus = min(
                            1.0,
                            float(_bbox_hw(seed_bbox)[0] * _bbox_hw(seed_bbox)[1]) / float(max(1, _bbox_hw(cand)[0] * _bbox_hw(cand)[1])),
                        )
                        score = (
                            0.55 * stats["dominant_fraction"]
                            + 0.15 * stats["coverage"]
                            + 0.10 * stats["rectangularity"]
                            + 0.15 * min(1.0, stats["interior_occupancy"] * 2.0)
                            + 0.05 * area_bonus
                        )
                        if score > best_score:
                            best_score = score
                            best_bbox = cand
                            best_stats = stats
        if best_bbox is None or best_stats is None:
            return None
        if float(best_stats["dominant_fraction"]) < perimeter_dominance_threshold:
            return None
        if float(best_stats["interior_occupancy"]) <= 0.05:
            return None
        return {
            "bbox": best_bbox,
            "perimeter_stats": best_stats,
            "completion_score": float(best_score),
        }

    def _build_sources() -> list[Dict[str, Any]]:
        raw_boxes = _changed_component_bboxes(diff)
        if not raw_boxes:
            return []
        merged_boxes = _merge_close_bboxes(raw_boxes, gap_thresh=4)
        sources: list[Dict[str, Any]] = []
        seen_bbox_keys: set[tuple[tuple[int, int, int, int], str]] = set()

        def _add_source(
            mode: str,
            raw_bbox: tuple[int, int, int, int],
            source_bbox: tuple[int, int, int, int],
            template_grid: np.ndarray,
            source_grid_name: str,
        ) -> None:
            clipped = _clip_bbox(source_bbox)
            if not _bbox_is_large_enough(clipped):
                return
            key = (clipped, source_grid_name)
            if key in seen_bbox_keys:
                return
            seen_bbox_keys.add(key)
            sources.append(
                {
                    "source_mode": mode,
                    "raw_bbox": raw_bbox,
                    "source_bbox": clipped,
                    "template_grid": template_grid,
                    "source_grid_name": source_grid_name,
                }
            )

        for raw_bbox in raw_boxes:
            if not _bbox_is_large_enough(raw_bbox):
                continue
        for merged_bbox in merged_boxes:
            raw_ref = merged_bbox
            if not _bbox_is_large_enough(merged_bbox):
                continue
            curr_completed = complete_bbox_by_perimeter_dominance(grid_curr, merged_bbox)
            if curr_completed is not None:
                _add_source("completed_perimeter_curr", raw_ref, tuple(curr_completed["bbox"]), grid_curr, "grid_curr")
            if curr_completed is None:
                _add_source("merged_seed_fallback", raw_ref, merged_bbox, grid_curr, "grid_curr")
        return sources

    def _extract_object_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
        border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        bg_color = _dominant_color(border)
        fg_mask = np.asarray(grid != bg_color, dtype=bool)
        raw_boxes = _changed_component_bboxes(fg_mask)
        if not raw_boxes:
            return []
        merged_boxes = _merge_close_bboxes(raw_boxes, gap_thresh=2)
        candidates: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for bbox in merged_boxes:
            if not _bbox_is_large_enough(bbox):
                continue
            completed = complete_bbox_by_perimeter_dominance(grid, bbox)
            final_bbox = tuple(completed["bbox"]) if completed is not None else bbox
            if not _bbox_is_large_enough(final_bbox):
                continue
            if final_bbox in seen:
                continue
            seen.add(final_bbox)
            candidates.append(final_bbox)
        return candidates

    def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        src = np.asarray(arr, dtype=np.int64)
        if src.ndim != 2 or src.size == 0 or out_h <= 0 or out_w <= 0:
            return np.zeros((max(0, out_h), max(0, out_w)), dtype=np.int64)
        y_idx = np.floor(np.arange(out_h) * (src.shape[0] / float(out_h))).astype(int)
        x_idx = np.floor(np.arange(out_w) * (src.shape[1] / float(out_w))).astype(int)
        y_idx = np.clip(y_idx, 0, src.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, src.shape[1] - 1)
        return src[y_idx][:, x_idx]

    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return (0.5 * float(mask.shape[0] - 1), 0.5 * float(mask.shape[1] - 1))
        return float(np.mean(ys)), float(np.mean(xs))

    def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(mask, dtype=bool)
        src_y0 = max(0, -dy)
        src_y1 = min(mask.shape[0], mask.shape[0] - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(mask.shape[1], mask.shape[1] - dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
        return out

    def _best_shift_iou(a_mask: np.ndarray, b_mask: np.ndarray, max_shift: int = 2) -> float:
        if a_mask.size == 0 or b_mask.size == 0:
            return 0.0
        best = 0.0
        cy_a, cx_a = _centroid(a_mask)
        cy_b, cx_b = _centroid(b_mask)
        base_dy = int(round(cy_a - cy_b))
        base_dx = int(round(cx_a - cx_b))
        for dy in range(base_dy - max_shift, base_dy + max_shift + 1):
            for dx in range(base_dx - max_shift, base_dx + max_shift + 1):
                shifted = _shift_mask(b_mask, dy, dx)
                inter = float(np.logical_and(a_mask, shifted).sum())
                union = float(np.logical_or(a_mask, shifted).sum())
                score = inter / union if union > 0 else 0.0
                if score > best:
                    best = score
        return best

    def _best_shift_layout_similarity(a_grid: np.ndarray, b_grid: np.ndarray, mask: np.ndarray, max_shift: int = 2) -> float:
        if a_grid.size == 0 or b_grid.size == 0 or mask.size == 0 or not bool(mask.any()):
            return 0.0
        best = 0.0
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                shifted = _shift_mask(mask, dy, dx)
                overlap = np.logical_and(mask, shifted)
                denom = float(overlap.sum())
                if denom <= 0:
                    continue
                score = float(np.sum((a_grid == _shift_int_grid(b_grid, dy, dx)) & overlap)) / denom
                if score > best:
                    best = score
        return best

    def _shift_int_grid(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.int64)
        src_y0 = max(0, -dy)
        src_y1 = min(arr.shape[0], arr.shape[0] - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(arr.shape[1], arr.shape[1] - dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
        return out

    def _normalize_layers(crop: np.ndarray, out_h: int = 18, out_w: int = 18) -> Dict[str, Any]:
        arr = np.asarray(crop, dtype=np.int64)
        h, w = arr.shape
        border_mask = np.zeros((h, w), dtype=bool)
        if h > 0 and w > 0:
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
        border_vals = arr[border_mask] if border_mask.any() else arr.reshape(-1)
        border_color = _dominant_color(border_vals)
        inner = arr[1:-1, 1:-1] if h > 2 and w > 2 else arr
        inner_color = _dominant_color(inner, fallback=border_color)
        border_layer = _resize_nearest(np.where(border_mask, arr == border_color, 0).astype(np.int64), out_h, out_w).astype(bool)
        inner_dark_mask = np.isin(arr, list(dark_colors))
        fg_mask = arr != inner_color
        symbol_mask = np.logical_and(fg_mask, np.logical_not(arr == border_color))
        blue_mask = np.isin(arr, list(blue_colors))
        color_layout = _resize_nearest(arr, out_h, out_w)
        return {
            "color_layout": color_layout,
            "fg_mask": _resize_nearest(fg_mask.astype(np.int64), out_h, out_w).astype(bool),
            "symbol_mask": _resize_nearest(symbol_mask.astype(np.int64), out_h, out_w).astype(bool),
            "blue_mask": _resize_nearest(blue_mask.astype(np.int64), out_h, out_w).astype(bool),
            "border_frame": border_layer,
            "inner_dark_ratio": float(np.mean(inner_dark_mask)) if inner_dark_mask.size else 0.0,
            "frame_indicator": _frame_objectness(arr)["frame_indicator"],
            "blue_ratio": float(np.mean(blue_mask)) if blue_mask.size else 0.0,
        }

    def _overlap_fraction(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> float:
        iy0 = max(a_bbox[0], b_bbox[0])
        ix0 = max(a_bbox[1], b_bbox[1])
        iy1 = min(a_bbox[2], b_bbox[2])
        ix1 = min(a_bbox[3], b_bbox[3])
        if iy1 < iy0 or ix1 < ix0:
            return 0.0
        inter = float((iy1 - iy0 + 1) * (ix1 - ix0 + 1))
        a_area = float(max(1, (_bbox_hw(a_bbox)[0] * _bbox_hw(a_bbox)[1])))
        b_area = float(max(1, (_bbox_hw(b_bbox)[0] * _bbox_hw(b_bbox)[1])))
        return inter / max(1.0, min(a_area, b_area))

    def _center_similarity(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> float:
        ay = 0.5 * float(a_bbox[0] + a_bbox[2])
        ax = 0.5 * float(a_bbox[1] + a_bbox[3])
        by = 0.5 * float(b_bbox[0] + b_bbox[2])
        bx = 0.5 * float(b_bbox[1] + b_bbox[3])
        dist = math.hypot(ay - by, ax - bx)
        diag = math.hypot(max(1, height - 1), max(1, width - 1))
        return max(0.0, 1.0 - (dist / max(1.0, diag)))

    def _size_similarity(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> float:
        ah, aw = _bbox_hw(a_bbox)
        bh, bw = _bbox_hw(b_bbox)
        return 0.5 * (min(ah, bh) / float(max(ah, bh)) + min(aw, bw) / float(max(aw, bw)))

    def _aspect_similarity(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> float:
        ah, aw = _bbox_hw(a_bbox)
        bh, bw = _bbox_hw(b_bbox)
        a_aspect = float(aw) / float(max(1, ah))
        b_aspect = float(bw) / float(max(1, bh))
        return min(a_aspect, b_aspect) / float(max(a_aspect, b_aspect, 1e-6))

    def _proposal_shapes(source_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
        sh, sw = _bbox_hw(source_bbox)
        shapes: set[tuple[int, int]] = set()
        for pad in (0, 1, 3):
            for dh in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    h = sh + 2 * pad + dh
                    w = sw + 2 * pad + dw
                    if h >= _MATCH_POI_MIN_BBOX_SIZE and w >= _MATCH_POI_MIN_BBOX_SIZE and h <= height and w <= width:
                        shapes.add((h, w))
        for ah in (0.9, 1.0, 1.1):
            for aw in (0.9, 1.0, 1.1):
                h = int(round(sh * ah))
                w = int(round(sw * aw))
                if h >= _MATCH_POI_MIN_BBOX_SIZE and w >= _MATCH_POI_MIN_BBOX_SIZE and h <= height and w <= width:
                    shapes.add((h, w))
        for pad in (2, 4):
            h = sh + 2 * pad
            w = sw + 2 * pad
            if h <= height and w <= width:
                shapes.add((h, w))
        return sorted(shapes)[:18]

    def _quick_similarity(source_layers: Dict[str, Any], candidate_layers: Dict[str, Any]) -> Dict[str, float]:
        fg = _best_shift_iou(source_layers["fg_mask"], candidate_layers["fg_mask"], max_shift=1)
        symbol = _best_shift_iou(source_layers["symbol_mask"], candidate_layers["symbol_mask"], max_shift=1)
        border = 1.0 - abs(source_layers["frame_indicator"] - candidate_layers["frame_indicator"])
        blue = 1.0 - abs(source_layers["blue_ratio"] - candidate_layers["blue_ratio"])
        dark = 1.0 - abs(source_layers["inner_dark_ratio"] - candidate_layers["inner_dark_ratio"])
        return {
            "fg": fg,
            "symbol": symbol,
            "border": max(0.0, border),
            "blue": max(0.0, blue),
            "dark": max(0.0, dark),
        }

    def _proposal_candidates(source: Dict[str, Any], object_candidates: list[tuple[int, int, int, int]]) -> list[Dict[str, Any]]:
        source_bbox = source["source_bbox"]
        source_layers = source["source_layers"]
        proposals: list[Dict[str, Any]] = []
        for cand_bbox in object_candidates:
            if not _bbox_is_large_enough(cand_bbox):
                continue
            candidate_layers = _normalize_layers(_crop(grid_curr, cand_bbox))
            quick = _quick_similarity(source_layers, candidate_layers)
            frame_feat = _frame_objectness(_crop(grid_curr, cand_bbox))
            overlap_penalty = _overlap_fraction(source_bbox, cand_bbox)
            proposal_score = (
                0.30 * quick["fg"]
                + 0.20 * quick["symbol"]
                + 0.10 * frame_feat["frame_indicator"]
                + 0.10 * frame_feat["dark_ratio"]
                + 0.10 * frame_feat["blue_ratio"]
                + 0.10 * quick["border"]
                + 0.10 * (1.0 - overlap_penalty)
            )
            proposals.append(
                {
                    "bbox": cand_bbox,
                    "proposal_score": float(proposal_score),
                    "proposal_frame_indicator": float(frame_feat["frame_indicator"]),
                    "proposal_dark_ratio": float(frame_feat["dark_ratio"]),
                    "proposal_blue_ratio": float(frame_feat["blue_ratio"]),
                    "proposal_overlap_penalty": float(overlap_penalty),
                }
            )
        proposals.sort(key=lambda item: float(item["proposal_score"]), reverse=True)
        trimmed: list[Dict[str, Any]] = []
        seen_bbox: set[tuple[int, int, int, int]] = set()
        for item in proposals:
            bbox = tuple(int(v) for v in item["bbox"])
            if bbox in seen_bbox:
                continue
            seen_bbox.add(bbox)
            trimmed.append(item)
            if len(trimmed) >= proposal_top_k:
                break
        return trimmed

    def _detail_score(source: Dict[str, Any], cand_bbox: tuple[int, int, int, int]) -> Dict[str, float]:
        source_bbox = source["source_bbox"]
        source_layers = source["source_layers"]
        cand_layers = _normalize_layers(_crop(grid_curr, cand_bbox))
        fg_similarity = _best_shift_iou(source_layers["fg_mask"], cand_layers["fg_mask"], max_shift=2)
        symbol_mask_iou = _best_shift_iou(source_layers["symbol_mask"], cand_layers["symbol_mask"], max_shift=2)
        color_layout_similarity = _best_shift_layout_similarity(
            source_layers["color_layout"],
            cand_layers["color_layout"],
            np.logical_or(source_layers["fg_mask"], cand_layers["fg_mask"]),
            max_shift=2,
        )
        border_frame_similarity = max(0.0, 1.0 - abs(source_layers["frame_indicator"] - cand_layers["frame_indicator"]))
        dark_ratio_similarity = max(0.0, 1.0 - abs(source_layers["inner_dark_ratio"] - cand_layers["inner_dark_ratio"]))
        center_similarity = _center_similarity(source_bbox, cand_bbox)
        size_similarity = _size_similarity(source_bbox, cand_bbox)
        aspect_similarity = _aspect_similarity(source_bbox, cand_bbox)
        overlap_penalty = _overlap_fraction(source_bbox, cand_bbox)
        total = (
            0.31 * fg_similarity
            + 0.28 * color_layout_similarity
            + 0.22 * symbol_mask_iou
            + 0.04 * border_frame_similarity
            + 0.05 * dark_ratio_similarity
            + 0.04 * center_similarity
            + 0.04 * size_similarity
            + 0.02 * aspect_similarity
            - 0.12 * overlap_penalty
        )
        return {
            "foreground_mask_similarity": float(fg_similarity),
            "color_layout_similarity": float(color_layout_similarity),
            "symbol_mask_iou": float(symbol_mask_iou),
            "border_frame_indicator_similarity": float(border_frame_similarity),
            "inner_dark_area_ratio_similarity": float(dark_ratio_similarity),
            "center_similarity": float(center_similarity),
            "size_similarity": float(size_similarity),
            "aspect_similarity": float(aspect_similarity),
            "overlap_penalty": float(overlap_penalty),
            "score": float(total),
        }

    sources = _build_sources()
    if not sources:
        _debug_log("match_poi_skip reason=no_changed_components")
        return None
    object_candidates = _extract_object_candidates(grid_curr)
    for source in sources:
        source["source_layers"] = _normalize_layers(_crop(source["template_grid"], source["source_bbox"]))
    _debug_log(
        "match_poi_sources count=%s changed_cells=%s candidate_objects=%s source_modes=%s",
        len(sources),
        changed_cells,
        len(object_candidates),
        ",".join(f"{src['source_mode']}:{_bbox_label(src['source_bbox'])}" for src in sources),
    )

    best_match: Optional[Dict[str, Any]] = None
    for source_idx, source in enumerate(sources):
        source_bbox = tuple(int(v) for v in source["source_bbox"])
        raw_bbox = tuple(int(v) for v in source["raw_bbox"])
        _debug_log(
            "match_poi_source source_idx=%s source_mode=%s raw_bbox=%s expanded_bbox=%s template_grid=%s",
            source_idx,
            str(source["source_mode"]),
            _bbox_label(raw_bbox),
            _bbox_label(source_bbox),
            str(source["source_grid_name"]),
        )
        proposals = _proposal_candidates(source, object_candidates)
        _debug_log(
            "match_poi_proposals source_idx=%s source_mode=%s proposal_count=%s top_proposals=%s",
            source_idx,
            str(source["source_mode"]),
            len(proposals),
            ";".join(
                f"{_bbox_label(tuple(int(v) for v in p['bbox']))}:p={float(p['proposal_score']):.4f}"
                for p in proposals[:5]
            ),
        )
        source_best: Optional[Dict[str, Any]] = None
        detailed_eval_count = 0
        prior_rejects = 0
        candidate_logs_emitted = 0
        for proposal in proposals:
            cand_bbox = tuple(int(v) for v in proposal["bbox"])
            terms = _detail_score(source, cand_bbox)
            detailed_eval_count += 1
            payload = {
                "source_idx": int(source_idx),
                "source_mode": str(source["source_mode"]),
                "raw_bbox": [int(v) for v in raw_bbox],
                "changed_bbox": [int(v) for v in source_bbox],
                "shape": [int(v) for v in _bbox_hw(source_bbox)],
                "a": {"y": int(source_bbox[0]), "x": int(source_bbox[1])},
                "b": {"y": int(cand_bbox[0]), "x": int(cand_bbox[1])},
                "candidate_shape": [int(v) for v in _bbox_hw(cand_bbox)],
                "source_grid_name": str(source["source_grid_name"]),
                "scale": float(max(_bbox_hw(cand_bbox)[0] / float(max(1, _bbox_hw(source_bbox)[0])), _bbox_hw(cand_bbox)[1] / float(max(1, _bbox_hw(source_bbox)[1])))),
                "proposal_score": float(proposal["proposal_score"]),
                "proposal_frame_indicator": float(proposal["proposal_frame_indicator"]),
                "proposal_dark_ratio": float(proposal["proposal_dark_ratio"]),
                "proposal_blue_ratio": float(proposal["proposal_blue_ratio"]),
                **terms,
            }
            should_log_candidate = candidate_logs_emitted < _MATCH_POI_DEBUG_CANDIDATE_LOG_LIMIT
            if source_best is None or float(payload["score"]) > float(source_best["score"]):
                source_best = payload
                should_log_candidate = True
            if should_log_candidate:
                candidate_logs_emitted += 1
                _debug_log(
                    "match_poi_candidate source_idx=%s source_mode=%s raw_bbox=%s expanded_bbox=%s candidate_bbox=%s proposal_score=%.4f score=%.4f fg=%.4f color=%.4f symbol=%.4f border=%.4f dark=%.4f center=%.4f size=%.4f aspect=%.4f overlap=%.4f",
                    source_idx,
                    str(source["source_mode"]),
                    _bbox_label(raw_bbox),
                    _bbox_label(source_bbox),
                    _bbox_label(cand_bbox),
                    float(proposal["proposal_score"]),
                    float(terms["score"]),
                    float(terms["foreground_mask_similarity"]),
                    float(terms["color_layout_similarity"]),
                    float(terms["symbol_mask_iou"]),
                    float(terms["border_frame_indicator_similarity"]),
                    float(terms["inner_dark_area_ratio_similarity"]),
                    float(terms["center_similarity"]),
                    float(terms["size_similarity"]),
                    float(terms["aspect_similarity"]),
                    float(terms["overlap_penalty"]),
                )
        _debug_log(
            "match_poi_summary source_idx=%s source_mode=%s raw_bbox=%s expanded_bbox=%s proposal_count=%s detailed_eval_count=%s prior_rejects=%s best_score=%.4f threshold=%.4f",
            source_idx,
            str(source["source_mode"]),
            _bbox_label(raw_bbox),
            _bbox_label(source_bbox),
            len(proposals),
            detailed_eval_count,
            prior_rejects,
            float((source_best or {}).get("score", 0.0)),
            float(_MATCH_POI_SCORE_THRESHOLD),
        )
        if source_best is None:
            _debug_log(
                "match_poi_reject source_idx=%s source_mode=%s reason=no_valid_candidate raw_bbox=%s expanded_bbox=%s",
                source_idx,
                str(source["source_mode"]),
                _bbox_label(raw_bbox),
                _bbox_label(source_bbox),
            )
            continue
        if float(source_best["score"]) < _MATCH_POI_SCORE_THRESHOLD:
            _debug_log(
                "match_poi_reject source_idx=%s source_mode=%s reason=below_threshold raw_bbox=%s expanded_bbox=%s best_candidate=%s best_score=%.4f",
                source_idx,
                str(source["source_mode"]),
                _bbox_label(raw_bbox),
                _bbox_label(source_bbox),
                _bbox_label((
                    int(source_best["b"]["y"]),
                    int(source_best["b"]["x"]),
                    int(source_best["b"]["y"]) + int(source_best["candidate_shape"][0]) - 1,
                    int(source_best["b"]["x"]) + int(source_best["candidate_shape"][1]) - 1,
                )),
                float(source_best["score"]),
            )
        if best_match is None or float(source_best["score"]) > float(best_match["score"]):
            best_match = source_best
            _debug_log(
                "match_poi_best_update source_idx=%s source_mode=%s reason=highest_score raw_bbox=%s expanded_bbox=%s candidate_bbox=%s score=%.4f",
                source_idx,
                str(source["source_mode"]),
                _bbox_label(raw_bbox),
                _bbox_label(source_bbox),
                _bbox_label((
                    int(source_best["b"]["y"]),
                    int(source_best["b"]["x"]),
                    int(source_best["b"]["y"]) + int(source_best["candidate_shape"][0]) - 1,
                    int(source_best["b"]["x"]) + int(source_best["candidate_shape"][1]) - 1,
                )),
                float(source_best["score"]),
            )

    if best_match is not None and float(best_match.get("score", 0.0)) >= _MATCH_POI_SCORE_THRESHOLD:
        return best_match
    return None


class RewardShaper:
    def reset_hud_cache(self) -> None:
        pass  # HUD masking removed; kept for call-site compatibility.

    def effect_from_transition(
        self,
        game_id: str,
        grid_prev: np.ndarray,
        grid_curr: np.ndarray,
        cfg_ctx: Dict[str, Any],
    ) -> Dict[str, float]:
        if grid_prev.shape != grid_curr.shape:
            grid_prev = grid_curr.copy()
        delta = np.asarray(grid_curr != grid_prev, dtype=bool)
        h, w = int(grid_curr.shape[0]), int(grid_curr.shape[1])
        area = max(1, h * w)
        raw_count = int(delta.sum())
        if raw_count > area // 2:
            raw_count = 0

        changed_total_frac = float(delta.sum()) / float(area)
        flash_changed_total_thresh = float(cfg_ctx.get("flash_changed_total_thresh", 0.5))
        flash_changed_masked_thresh = float(cfg_ctx.get("flash_changed_masked_thresh", 0.5))
        flash_hist_l1_thresh = float(cfg_ctx.get("flash_hist_l1_thresh", 0.15))

        hist_l1 = 1.0
        flash_event = False
        if changed_total_frac >= flash_changed_total_thresh:
            try:
                h_prev = np.bincount(grid_prev.reshape(-1).clip(0, 10), minlength=11).astype(np.float64)
                h_curr = np.bincount(grid_curr.reshape(-1).clip(0, 10), minlength=11).astype(np.float64)
                hist_l1 = float(np.abs(h_prev - h_curr).sum() / float(area))
                flash_event = (changed_total_frac >= flash_changed_masked_thresh) and (hist_l1 <= flash_hist_l1_thresh)
            except Exception:
                flash_event = False

        return {
            "cells_changed": float(raw_count),
            "changed_total_frac": changed_total_frac,
            "flash_event": float(1.0 if flash_event else 0.0),
            "hist_l1": float(hist_l1),
            "H": float(h),
            "W": float(w),
        }

    def compute(
        self,
        event: Dict[str, Any],
        done: bool,
        win: bool,
        t: int = 0,
        visit_counts: Optional[Dict[str, int]] = None,
        state_hash_t_minus_2: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**_default_cfg(), **(cfg or {})}
        ctx = ctx or {}

        grid_prev = np.asarray(ctx.get("grid_prev"))
        grid_curr = np.asarray(ctx.get("grid_curr"))
        grid_prev_prev_raw = ctx.get("grid_prev_prev")
        grid_prev_prev = np.asarray(grid_prev_prev_raw) if grid_prev_prev_raw is not None else None

        # --- Effect transition (cell diff + flash detection) ---
        pre = ctx.get("effect_transition")
        if isinstance(pre, dict) and "cells_changed" in pre:
            cells_changed = int(pre.get("cells_changed", 0))
            flash_event = bool(float(pre.get("flash_event", 0.0)))
        else:
            eff = self.effect_from_transition(
                str(ctx.get("game_id", "")), grid_prev, grid_curr,
                {
                    "flash_changed_total_thresh": float(cfg_eff["flash_changed_total_thresh"]),
                    "flash_changed_masked_thresh": float(cfg_eff["flash_changed_masked_thresh"]),
                    "flash_hist_l1_thresh": float(cfg_eff["flash_hist_l1_thresh"]),
                },
            )
            cells_changed = int(eff["cells_changed"])
            flash_event = bool(float(eff["flash_event"]))

        # --- State hash ---
        if grid_curr.ndim == 2:
            state_hash = stable_hash_grid(grid_curr)
        else:
            state_hash = str(
                (event.get("state_hash_after_filtered") or event.get("state_hash_after") or "")
                if isinstance(event, dict) else ""
            )

        # --- m_noop: treat as no-op if fewer than noop_cell_thresh cells changed ---
        noop_cell_thresh = int(cfg_eff.get("noop_cell_thresh", 10))
        m_noop = 0 if cells_changed < noop_cell_thresh else 1

        # --- r_win ---
        r_win = 1.0 if bool(win) else 0.0

        # --- r_effect ---
        movement_thresh = int(cfg_eff.get("movement_cell_thresh", 55))
        r_effect_movement = float(cfg_eff.get("r_effect_movement", 0.01))
        r_effect_screen = float(cfg_eff.get("r_effect_screen", 0.1))
        if flash_event or cells_changed == 0:
            r_effect = 0.0
        elif cells_changed <= movement_thresh:
            r_effect = r_effect_movement
        else:
            r_effect = r_effect_screen

        # --- r_match_poi (new duplicated non-overlapping >=3x3 pattern appears) ---
        match_poi_info: Optional[Dict[str, Any]] = None
        if not flash_event and cells_changed >= max(noop_cell_thresh, _MATCH_POI_MIN_BBOX_SIZE * _MATCH_POI_MIN_BBOX_SIZE):
            match_poi_info = _match_poi_blocks(grid_prev, grid_curr, grid_prev_prev=grid_prev_prev)
        r_match_poi = (
            float(cfg_eff.get("match_poi", 0.5)) * float(match_poi_info.get("score", 0.0))
            if match_poi_info is not None
            else 0.0
        )
        if match_poi_info is not None:
            shape = match_poi_info.get("shape") or [0, 0]
            candidate_shape = match_poi_info.get("candidate_shape") or shape
            a = match_poi_info.get("a") or {}
            b = match_poi_info.get("b") or {}
            debug_pngs = None
            if _debug_match_poi_enabled():
                try:
                    debug_pngs = _write_match_poi_debug_pngs(
                        str(ctx.get("game_id", "")),
                        grid_prev,
                        grid_curr,
                    )
                except Exception:
                    logger.exception("match_poi_debug_png_write_failed game_id=%s", str(ctx.get("game_id", "")))
            logger.info(
                "match_poi_hit game_id=%s bbox_a=[y=%s,x=%s,h=%s,w=%s] bbox_b=[y=%s,x=%s,h=%s,w=%s] scale=%s score=%.4f embed=%.4f center=%.4f size=%.4f aspect=%.4f prev_png=%s new_png=%s",
                str(ctx.get("game_id", "")),
                int(a.get("y", 0)),
                int(a.get("x", 0)),
                int(shape[0]) if len(shape) > 0 else 0,
                int(shape[1]) if len(shape) > 1 else 0,
                int(b.get("y", 0)),
                int(b.get("x", 0)),
                int(candidate_shape[0]) if len(candidate_shape) > 0 else 0,
                int(candidate_shape[1]) if len(candidate_shape) > 1 else 0,
                int(match_poi_info.get("scale", 1)),
                float(match_poi_info.get("score", 0.0)),
                float(match_poi_info.get("embedding_cosine", 0.0)),
                float(match_poi_info.get("center_similarity", 0.0)),
                float(match_poi_info.get("size_similarity", 0.0)),
                float(match_poi_info.get("aspect_similarity", 0.0)),
                (debug_pngs or {}).get("prev", ""),
                (debug_pngs or {}).get("new", ""),
            )

        # --- r_revert (A→B→A penalty) ---
        revert_flag = bool(
            state_hash
            and state_hash_t_minus_2
            and state_hash == state_hash_t_minus_2
        )
        r_revert = -float(cfg_eff.get("revert_penalty", 0.2)) if revert_flag else 0.0

        # --- r_potential ---
        vc = visit_counts if visit_counts is not None else {}
        n_curr = int(vc.get(state_hash, 0)) if state_hash else 0
        if grid_prev.ndim == 2:
            prev_hash = stable_hash_grid(grid_prev)
        else:
            prev_hash = str(
                (event.get("state_hash_before_filtered") or event.get("state_hash_before") or "")
                if isinstance(event, dict) else ""
            )
        n_prev = int(vc.get(prev_hash, 0)) if prev_hash else 0

        r_potential = 0.0

        if state_hash and visit_counts is not None and not flash_event:
            visit_counts[state_hash] = n_curr + 1

        # --- r_step (unconditional growing time penalty) ---
        rate = float(cfg_eff.get("step_penalty_rate", 0.0005))
        cap = float(cfg_eff.get("step_penalty_cap", 0.1))
        r_step = -min(rate * float(t), cap)

        # --- r_noop penalty ---
        r_noop = -float(cfg_eff.get("noop_penalty", 0.1)) if m_noop == 0 else 0.0

        # --- Total ---
        r_total = r_win + float(m_noop) * (r_effect + r_match_poi + r_revert + r_potential) + r_step + r_noop

        return {
            "schema_version": "REWARD_V1",
            "r_total": float(r_total),
            "terms": {
                "r_win": float(r_win),
                "r_effect": float(r_effect),
                "r_match_poi": float(r_match_poi),
                "r_revert": float(r_revert),
                "r_potential": float(r_potential),
                "r_step": float(r_step),
                "r_noop": float(r_noop),
                "m_noop": int(m_noop),
                "flash_event": bool(flash_event),
                "effect_flag": bool(cells_changed > 0 and not flash_event),
                "revert_flag": bool(revert_flag),
                "match_poi": bool(match_poi_info is not None),
                "match_poi_info": match_poi_info,
                "delta_c": int(cells_changed),
                "cells_changed": int(cells_changed),
                "state_hash": str(state_hash),
            },
            "aux": {
                "mode_target": 1 if (cells_changed > 0 and not flash_event) or win else 0,
                "mode_weight": float(cfg_eff.get("controller_aux_weight", 0.2)),
            },
        }
