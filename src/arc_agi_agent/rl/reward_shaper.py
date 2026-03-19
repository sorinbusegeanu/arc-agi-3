from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from .canonical_grid import stable_hash_grid
from ..grid_utils import changed_bbox as changed_bbox_inclusive, diff_mask

logger = logging.getLogger(__name__)

_MATCH_POI_SCORE_THRESHOLD = 0.85


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
    if grid_prev.ndim != 2 or grid_curr.ndim != 2:
        return None
    if grid_prev.shape != grid_curr.shape:
        return None
    diff = diff_mask(grid_curr, grid_prev)
    bbox = changed_bbox_inclusive(diff)
    if bbox is None:
        return None
    y0, x0, y1, x1 = [int(v) for v in bbox]
    patch_h = y1 - y0 + 1
    patch_w = x1 - x0 + 1
    if patch_h < 3 or patch_w < 3:
        return None

    patch = np.ascontiguousarray(grid_curr[y0 : y1 + 1, x0 : x1 + 1])
    height, width = int(grid_curr.shape[0]), int(grid_curr.shape[1])

    def _inner_normalize_crop(arr: np.ndarray) -> np.ndarray:
        crop = np.asarray(arr, dtype=np.int64)
        if crop.ndim != 2 or crop.size == 0:
            return np.zeros((0, 0), dtype=np.int64)
        if crop.shape[0] > 2 and crop.shape[1] > 2:
            crop = crop[1:-1, 1:-1]
        if crop.size == 0:
            return np.zeros((0, 0), dtype=np.int64)
        border_vals = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
        if border_vals.size > 0:
            vals, counts = np.unique(border_vals, return_counts=True)
            border_color = int(vals[int(np.argmax(counts))])
            mask = crop != border_color
            ys, xs = np.where(mask)
            if ys.size > 0 and xs.size > 0:
                crop = crop[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        return np.ascontiguousarray(crop, dtype=np.int64)

    def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        src = np.asarray(arr, dtype=np.int64)
        if src.ndim != 2 or src.size == 0 or out_h <= 0 or out_w <= 0:
            return np.zeros((max(0, out_h), max(0, out_w)), dtype=np.int64)
        y_idx = np.floor(np.arange(out_h) * (src.shape[0] / float(out_h))).astype(int)
        x_idx = np.floor(np.arange(out_w) * (src.shape[1] / float(out_w))).astype(int)
        y_idx = np.clip(y_idx, 0, src.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, src.shape[1] - 1)
        return src[y_idx][:, x_idx]

    def _pattern_similarity_resized(left: np.ndarray, right: np.ndarray) -> float:
        left_n = _inner_normalize_crop(left)
        right_n = _inner_normalize_crop(right)
        if left_n.size == 0 or right_n.size == 0:
            return 0.0
        out_h = max(int(left_n.shape[0]), int(right_n.shape[0]))
        out_w = max(int(left_n.shape[1]), int(right_n.shape[1]))
        if out_h <= 0 or out_w <= 0:
            return 0.0
        left_r = _resize_nearest(left_n, out_h, out_w)
        right_r = _resize_nearest(right_n, out_h, out_w)
        return float(np.mean(left_r == right_r))

    def _center_similarity(a_y0: int, a_x0: int, a_y1: int, a_x1: int, b_y0: int, b_x0: int, b_y1: int, b_x1: int) -> float:
        ay = 0.5 * float(a_y0 + a_y1)
        ax = 0.5 * float(a_x0 + a_x1)
        by = 0.5 * float(b_y0 + b_y1)
        bx = 0.5 * float(b_x0 + b_x1)
        dist = math.hypot(ay - by, ax - bx)
        diag = math.hypot(max(1, height - 1), max(1, width - 1))
        return max(0.0, 1.0 - (dist / max(1.0, diag)))

    def _size_similarity(a_h: int, a_w: int, b_h: int, b_w: int) -> float:
        h_ratio = min(a_h, b_h) / float(max(a_h, b_h))
        w_ratio = min(a_w, b_w) / float(max(a_w, b_w))
        return max(0.0, min(1.0, 0.5 * (h_ratio + w_ratio)))

    def _aspect_similarity(a_h: int, a_w: int, b_h: int, b_w: int) -> float:
        a_aspect = float(a_w) / float(max(1, a_h))
        b_aspect = float(b_w) / float(max(1, b_h))
        ratio = min(a_aspect, b_aspect) / float(max(a_aspect, b_aspect, 1e-6))
        return max(0.0, min(1.0, ratio))

    def _match_score(a_crop: np.ndarray, b_crop: np.ndarray, a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> Dict[str, float]:
        ay0, ax0, ay1, ax1 = a_bbox
        by0, bx0, by1, bx1 = b_bbox
        embedding_cosine = _pattern_similarity_resized(a_crop, b_crop)
        center_similarity = _center_similarity(ay0, ax0, ay1, ax1, by0, bx0, by1, bx1)
        size_similarity = _size_similarity(ay1 - ay0 + 1, ax1 - ax0 + 1, by1 - by0 + 1, bx1 - bx0 + 1)
        aspect_similarity = _aspect_similarity(ay1 - ay0 + 1, ax1 - ax0 + 1, by1 - by0 + 1, bx1 - bx0 + 1)
        total = (
            0.60 * embedding_cosine
            + 0.20 * center_similarity
            + 0.15 * size_similarity
            + 0.05 * aspect_similarity
        )
        return {
            "embedding_cosine": float(embedding_cosine),
            "center_similarity": float(center_similarity),
            "size_similarity": float(size_similarity),
            "aspect_similarity": float(aspect_similarity),
            "score": float(total),
        }

    def _rectangles_overlap(a_y0: int, a_x0: int, a_y1: int, a_x1: int, b_y0: int, b_x0: int, b_y1: int, b_x1: int) -> bool:
        return not (a_y1 < b_y0 or b_y1 < a_y0 or a_x1 < b_x0 or b_x1 < a_x0)

    def _scaled_match(base: np.ndarray, candidate: np.ndarray) -> Optional[int]:
        base_h, base_w = int(base.shape[0]), int(base.shape[1])
        cand_h, cand_w = int(candidate.shape[0]), int(candidate.shape[1])
        if cand_h == base_h and cand_w == base_w:
            return 1 if np.array_equal(candidate, base) else None
        if cand_h % base_h != 0 or cand_w % base_w != 0:
            return None
        scale_y = cand_h // base_h
        scale_x = cand_w // base_w
        if scale_y != scale_x or scale_y <= 0:
            return None
        scale = int(scale_y)
        for py in range(base_h):
            for px in range(base_w):
                block = candidate[py * scale : (py + 1) * scale, px * scale : (px + 1) * scale]
                if not np.all(block == base[py, px]):
                    return None
        return scale

    seen_shapes: set[tuple[int, int]] = set()
    candidate_shapes: list[tuple[int, int, int]] = [(patch_h, patch_w, 1)]
    for scale in (2, 3):
        cand_h = patch_h * scale
        cand_w = patch_w * scale
        if cand_h <= height and cand_w <= width:
            candidate_shapes.append((cand_h, cand_w, scale))
    for scale in (2, 3):
        if patch_h % scale != 0 or patch_w % scale != 0:
            continue
        cand_h = patch_h // scale
        cand_w = patch_w // scale
        if cand_h < 3 or cand_w < 3:
            continue
        candidate_shapes.append((cand_h, cand_w, -scale))

    best_match: Optional[Dict[str, Any]] = None
    for cand_h, cand_w, declared_scale in candidate_shapes:
        if (cand_h, cand_w) in seen_shapes:
            continue
        seen_shapes.add((cand_h, cand_w))
        if cand_h < 3 or cand_w < 3 or cand_h > height or cand_w > width:
            continue
        max_y = height - cand_h + 1
        max_x = width - cand_w + 1
        for top in range(max_y):
            for left in range(max_x):
                cand_y1 = top + cand_h - 1
                cand_x1 = left + cand_w - 1
                if _rectangles_overlap(y0, x0, y1, x1, top, left, cand_y1, cand_x1):
                    continue
                candidate = grid_curr[top : top + cand_h, left : left + cand_w]
                if declared_scale > 0:
                    scale = _scaled_match(patch, candidate)
                    if scale is None or int(scale) != int(declared_scale):
                        continue
                else:
                    inverse_scale = _scaled_match(candidate, patch)
                    if inverse_scale is None or int(inverse_scale) != int(-declared_scale):
                        continue
                    scale = int(declared_scale)
                score_terms = _match_score(
                    patch,
                    candidate,
                    (y0, x0, y1, x1),
                    (top, left, cand_y1, cand_x1),
                )
                if (
                    grid_prev_prev is not None
                    and isinstance(grid_prev_prev, np.ndarray)
                    and grid_prev_prev.shape == grid_curr.shape
                ):
                    prev_prev_a = grid_prev_prev[y0 : y1 + 1, x0 : x1 + 1]
                    prev_prev_b = grid_prev_prev[top : top + cand_h, left : left + cand_w]
                    prior_terms = _match_score(
                        prev_prev_a,
                        prev_prev_b,
                        (y0, x0, y1, x1),
                        (top, left, cand_y1, cand_x1),
                    )
                    if float(prior_terms.get("score", 0.0)) >= _MATCH_POI_SCORE_THRESHOLD:
                        continue
                payload = {
                    "shape": [int(patch_h), int(patch_w)],
                    "a": {"y": int(y0), "x": int(x0)},
                    "b": {"y": int(top), "x": int(left)},
                    "changed_bbox": [int(y0), int(x0), int(y1), int(x1)],
                    "candidate_shape": [int(cand_h), int(cand_w)],
                    "scale": int(scale),
                    **score_terms,
                }
                if best_match is None or float(payload.get("score", 0.0)) > float(best_match.get("score", 0.0)):
                    best_match = payload
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
        if not flash_event and cells_changed > 0:
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
            logger.info(
                "match_poi_hit game_id=%s bbox_a=[y=%s,x=%s,h=%s,w=%s] bbox_b=[y=%s,x=%s,h=%s,w=%s] scale=%s score=%.4f embed=%.4f center=%.4f size=%.4f aspect=%.4f",
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
