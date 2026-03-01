from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

import numpy as np

from .canonical_grid import stable_hash_grid

logger = logging.getLogger(__name__)


def _default_cfg() -> Dict[str, Any]:
    return {
        "alpha_novel": 0.05,
        "beta_effect": 0.02,
        "delta_loop": 0.05,
        "loop_window_N": 25,
        "controller_aux_weight": 0.2,
    }


class RewardShaper:
    def __init__(self) -> None:
        self._hud_masks: Dict[str, np.ndarray] = {}
        self._hud_mask_load_logged: Set[str] = set()

    def reset_hud_cache(self) -> None:
        self._hud_masks.clear()
        self._hud_mask_load_logged.clear()

    @staticmethod
    def _rle_to_mask(payload: Dict[str, Any], expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
        shape = payload.get("shape")
        runs = payload.get("runs")
        if not (isinstance(shape, list) and len(shape) == 2 and isinstance(runs, list)):
            return None
        h = int(shape[0])
        w = int(shape[1])
        if (h, w) != expected_shape:
            return None
        flat = np.zeros((h * w,), dtype=bool)
        for run in runs:
            if not (isinstance(run, list) and len(run) == 2):
                continue
            start = int(run[0])
            ln = int(run[1])
            if ln <= 0:
                continue
            end = min(h * w, start + ln)
            if start < 0 or start >= h * w:
                continue
            flat[start:end] = True
        return flat.reshape((h, w))

    def _load_hud_mask(self, game_id: str, grid_shape: tuple[int, int], cfg_ctx: Dict[str, Any]) -> np.ndarray:
        gid = str(game_id)
        if gid in self._hud_masks:
            m = self._hud_masks[gid]
            if m.shape == grid_shape:
                return m
            # Grid shape changed; drop cached mask and rebuild from spec.
            self._hud_masks.pop(gid, None)
        mask = np.zeros(grid_shape, dtype=bool)
        specs = cfg_ctx.get("hud_specs", {})
        spec = specs.get(gid) if isinstance(specs, dict) else None
        if isinstance(spec, dict):
            if isinstance(spec.get("bboxes"), list):
                for bb in spec.get("bboxes", []):
                    if not (isinstance(bb, list) and len(bb) == 4):
                        continue
                    y0, x0, y1, x1 = [int(v) for v in bb]
                    y0 = max(0, min(grid_shape[0], y0))
                    y1 = max(0, min(grid_shape[0], y1))
                    x0 = max(0, min(grid_shape[1], x0))
                    x1 = max(0, min(grid_shape[1], x1))
                    if y1 > y0 and x1 > x0:
                        mask[y0:y1, x0:x1] = True
            elif isinstance(spec.get("hud_bbox"), list) and len(spec["hud_bbox"]) == 4:
                y0, x0, y1, x1 = [int(v) for v in spec["hud_bbox"]]
                y0 = max(0, min(grid_shape[0], y0))
                y1 = max(0, min(grid_shape[0], y1))
                x0 = max(0, min(grid_shape[1], x0))
                x1 = max(0, min(grid_shape[1], x1))
                if y1 > y0 and x1 > x0:
                    mask[y0:y1, x0:x1] = True
            if not np.any(mask) and isinstance(spec.get("hud_mask_rle"), dict):
                dec = self._rle_to_mask(spec["hud_mask_rle"], expected_shape=grid_shape)
                if dec is not None:
                    mask = dec
        self._hud_masks[gid] = mask
        if gid not in self._hud_mask_load_logged:
            mask_bbox = None
            if np.any(mask):
                ys, xs = np.where(mask)
                mask_bbox = [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]
            logger.info(
                "hud_mask_loaded game_id=%s shape=%s area=%s bbox=%s source=in_memory_spec",
                gid,
                list(grid_shape),
                int(mask.sum()),
                mask_bbox,
            )
            self._hud_mask_load_logged.add(gid)
        return mask

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
        h = int(grid_curr.shape[0])
        w = int(grid_curr.shape[1])
        den = int(max(1, h))
        area = int(max(1, h * w))
        raw_count = int(delta.sum())
        if raw_count > (area // 2):
            raw_count = 0
        use_hud_mask = bool(cfg_ctx.get("use_hud_mask", True))
        if use_hud_mask:
            hud_mask = self._load_hud_mask(str(game_id), (h, w), cfg_ctx)
        else:
            hud_mask = np.zeros((h, w), dtype=bool)
        if hud_mask.shape != delta.shape:
            hud_mask = np.zeros_like(delta, dtype=bool)
        delta_masked = np.asarray(delta & ~hud_mask, dtype=bool)
        changed_cells_masked_count = int(delta_masked.sum())
        if changed_cells_masked_count > (area // 2):
            changed_cells_masked_count = 0
        hud_overlap_count = int(np.logical_and(delta, hud_mask).sum())
        changed_cells_outside_count = int(np.logical_and(delta, ~hud_mask).sum())
        changed_cells_bbox_count = 0
        if use_hud_mask:
            specs = cfg_ctx.get("hud_specs", {})
            spec = specs.get(str(game_id)) if isinstance(specs, dict) else None
            if isinstance(spec, dict) and isinstance(spec.get("hud_bbox"), list) and len(spec["hud_bbox"]) == 4:
                y0, x0, y1, x1 = [int(v) for v in spec["hud_bbox"]]
                y0 = max(0, min(h, y0))
                y1 = max(0, min(h, y1))
                x0 = max(0, min(w, x0))
                x1 = max(0, min(w, x1))
                if y1 > y0 and x1 > x0:
                    changed_cells_bbox_count = int(delta[y0:y1, x0:x1].sum())
        return {
            "effect_changed_cells_masked": float(changed_cells_masked_count) / float(den),
            "changed_cells_masked_count": float(changed_cells_masked_count),
            "changed_cells_raw": float(raw_count),
            "hud_mask_area": float(hud_mask.sum()),
            "hud_overlap_count": float(hud_overlap_count),
            "changed_cells_outside_count": float(changed_cells_outside_count),
            "changed_cells_bbox_count": float(changed_cells_bbox_count),
            "H": float(h),
            "W": float(w),
            "den": float(den),
        }

    def compute(
        self,
        event: Dict[str, Any],
        done: bool,
        win: bool,
        seen_hashes: Optional[Set[str]],
        recent_hashes: Optional[list[str]],
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**_default_cfg(), **(cfg or {})}
        cfg_ctx = dict(ctx or {})
        seen = seen_hashes or set()
        recent = recent_hashes or []

        grid_prev = np.asarray(cfg_ctx.get("grid_prev"), dtype=np.int64)
        grid_curr = np.asarray(cfg_ctx.get("grid_curr"), dtype=np.int64)
        has_grids = grid_prev.ndim == 2 and grid_curr.ndim == 2

        if has_grids:
            game_id = str(cfg_ctx.get("game_id") or event.get("game_id") or "")
            pre = cfg_ctx.get("effect_transition")
            if isinstance(pre, dict) and "effect_changed_cells_masked" in pre:
                effect_changed_cells = float(pre.get("effect_changed_cells_masked", 0.0))
                h = int(pre.get("H", int(grid_curr.shape[0])))
                w = int(pre.get("W", int(grid_curr.shape[1])))
                den = int(pre.get("den", max(1, h * w)))
                changed_cells_raw = int(pre.get("changed_cells_raw", 0))
            else:
                eff = self.effect_from_transition(game_id, grid_prev, grid_curr, cfg_ctx)
                effect_changed_cells = float(eff["effect_changed_cells_masked"])
                h = int(eff["H"])
                w = int(eff["W"])
                den = int(eff["den"])
                changed_cells_raw = int(eff["changed_cells_raw"])
            state_hash = stable_hash_grid(grid_curr)
        else:
            grid_delta = event.get("grid_delta", {}) if isinstance(event, dict) else {}
            effect_changed_cells = float(grid_delta.get("changed_cells_count_filtered", grid_delta.get("changed_cells_count", 0)) or 0)
            state_hash = str(
                (event.get("state_hash_after_filtered") if isinstance(event, dict) else None)
                or (event.get("state_hash_after") if isinstance(event, dict) else None)
                or ""
            )

        action_key = event.get("action_key", {}) if isinstance(event, dict) else {}
        action_id = str(action_key.get("id") or action_key.get("action_id") or "")

        r_win = 1.0 if bool(win) else 0.0
        r_env = 0.0
        effect_flag = float(effect_changed_cells) > 0.0
        novel_flag = bool(state_hash and state_hash not in seen)
        repeat_flag = bool(state_hash and state_hash in recent)

        negative_step = float(cfg_eff.get("negative_step", 0.5))
        r_effect = float(cfg_eff["beta_effect"]) * float(effect_changed_cells - negative_step)
        r_novel = 0.0
        r_loop = 0.0

        r_total = float(r_win + r_env + r_effect + r_novel + r_loop)

        if action_id.upper() == "ACTION6":
            mode_target = 2
        elif float(effect_changed_cells) > 0.0 or win:
            mode_target = 1
        else:
            mode_target = 0

        return {
            "schema_version": "REWARD_V1",
            "r_total": r_total,
            "terms": {
                "r_win": float(r_win),
                "r_env": float(r_env),
                "r_effect": float(r_effect),
                "r_novel": float(r_novel),
                "r_loop": float(r_loop),
                "effect_flag_raw": bool(effect_flag),
                "effect_flag_filtered": bool(effect_flag),
                "novel_flag_raw": bool(novel_flag),
                "novel_flag_filtered": bool(novel_flag),
                "repeat_flag_raw": bool(repeat_flag),
                "repeat_flag_filtered": bool(repeat_flag),
                "effect_changed_cells": float(effect_changed_cells),
                "state_hash": str(state_hash),
            },
            "aux": {
                "mode_target": int(mode_target),
                "mode_weight": float(cfg_eff.get("controller_aux_weight", 0.2)),
            },
        }
