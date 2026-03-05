from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from .canonical_grid import stable_hash_grid

logger = logging.getLogger(__name__)


def _default_cfg() -> Dict[str, Any]:
    return {
        "movement_cell_thresh": 55,
        "noop_cell_thresh": 10,
        "r_effect_movement": 0.01,
        "r_effect_screen": 0.1,
        "revert_penalty": 0.2,
        "beta_potential": 0.05,
        "gamma": 0.995,
        "step_penalty_rate": 0.0005,
        "step_penalty_cap": 0.1,
        "flash_changed_total_thresh": 0.5,
        "flash_changed_masked_thresh": 0.5,
        "flash_hist_l1_thresh": 0.15,
    }


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

        # --- Total ---
        r_total = r_win + float(m_noop) * (r_effect + r_revert + r_potential) + r_step

        return {
            "schema_version": "REWARD_V1",
            "r_total": float(r_total),
            "terms": {
                "r_win": float(r_win),
                "r_effect": float(r_effect),
                "r_revert": float(r_revert),
                "r_potential": float(r_potential),
                "r_step": float(r_step),
                "m_noop": int(m_noop),
                "flash_event": bool(flash_event),
                "effect_flag": bool(cells_changed > 0 and not flash_event),
                "revert_flag": bool(revert_flag),
                "delta_c": int(cells_changed),
                "cells_changed": int(cells_changed),
                "state_hash": str(state_hash),
            },
            "aux": {
                "mode_target": 1 if (cells_changed > 0 and not flash_event) or win else 0,
                "mode_weight": float(cfg_eff.get("controller_aux_weight", 0.2)),
            },
        }
