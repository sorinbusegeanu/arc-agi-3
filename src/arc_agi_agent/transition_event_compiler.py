from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid_utils import changed_bbox as bbox_inclusive, changed_colors, diff_mask, palette
from .logger import get_logger
from .normalize import normalize_observation
from .state_hash import hash_state, hash_state_filtered
from .transition_event_compiler_config import TransitionEventCompilerConfig
from .transition_event_compiler_types import TransitionEventV1

logger = get_logger(__name__)
_RL_UI_BAR_CFG_CACHE: Optional[Dict[str, Any]] = None


def compile_transition_event(
    prev_observation: Any,
    observation: Any,
    action_taken: Dict[str, Any],
    fp_prev_report: Dict[str, Any],
    fp_curr_report: Dict[str, Any],
    ctx: Dict[str, Any],
    cfg: Optional[TransitionEventCompilerConfig] = None,
    env_frame_names: Optional[List[str]] = None,
    prev_grid_norm: Optional[Any] = None,
    next_grid_norm: Optional[Any] = None,
    prev_meta_norm: Optional[Dict[str, Any]] = None,
    next_meta_norm: Optional[Dict[str, Any]] = None,
    ui_exclusion_mask: Optional[Any] = None,
) -> TransitionEventV1:
    cfg = cfg or TransitionEventCompilerConfig()
    lite_mode = _resolve_lite_mode(ctx)

    pre_norm_fast_path = prev_grid_norm is not None and next_grid_norm is not None
    if pre_norm_fast_path:
        prev_grids: List[Any] = [np.asarray(prev_grid_norm)]
        curr_grids: List[Any] = [np.asarray(next_grid_norm)]
        prev_meta = dict(prev_meta_norm) if isinstance(prev_meta_norm, dict) else {}
        curr_meta = dict(next_meta_norm) if isinstance(next_meta_norm, dict) else {}
        grid_names = env_frame_names or ["grid_0"]
    else:
        prev_norm = normalize_observation(prev_observation, schema_warnings=[])
        curr_norm = normalize_observation(observation, schema_warnings=[])
        prev_grids = list(prev_norm.grids)
        curr_grids = list(curr_norm.grids)
        prev_meta = dict(prev_norm.meta or {})
        curr_meta = dict(curr_norm.meta or {})
        grid_names = env_frame_names or curr_norm.grid_names or [f"grid_{i}" for i in range(len(curr_grids))]
        if len(grid_names) < len(curr_grids):
            grid_names = grid_names + [f"grid_{i}" for i in range(len(grid_names), len(curr_grids))]

    primary_name = grid_names[0] if grid_names else "grid_0"
    if cfg.enable_multigrid:
        grid_names_used = list(grid_names)
        aggregation = "UNION_DIFFS"
    else:
        grid_names_used = [primary_name]
        aggregation = "PRIMARY_ONLY"

    state_hash_before = _state_hash(prev_grids, grid_names_used, prev_meta, cfg)
    state_hash_after = _state_hash(curr_grids, grid_names_used, curr_meta, cfg)

    primary_idx = grid_names.index(primary_name) if primary_name in grid_names else 0
    ui_cfg = _resolve_ui_bar_cfg(cfg, ctx)
    if curr_grids:
        ui_mask = np.zeros_like(np.asarray(curr_grids[0]), dtype=bool)
    elif prev_grids:
        ui_mask = np.zeros_like(np.asarray(prev_grids[0]), dtype=bool)
    else:
        ui_mask = np.zeros((0, 0), dtype=bool)
    ui_bboxes: List[Tuple[int, int, int, int]] = []
    if ui_exclusion_mask is not None:
        mask = np.asarray(ui_exclusion_mask, dtype=bool)
        if mask.shape == ui_mask.shape:
            ui_mask = mask
            ys, xs = np.where(ui_mask)
            if ys.size > 0 and xs.size > 0:
                ui_bboxes = [(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)]
    primary_idx_valid = (
        primary_idx >= 0
        and primary_idx < len(prev_grids)
        and primary_idx < len(curr_grids)
    )
    if not ui_bboxes and bool(ui_cfg.get("enabled", False)) and primary_idx_valid:
        ui_mask, ui_bboxes = detect_ui_progress_bar_mask(
            np.asarray(prev_grids[primary_idx]),
            np.asarray(curr_grids[primary_idx]),
            ui_cfg,
        )

    state_hash_before_filtered = _state_hash_filtered(
        prev_grids,
        grid_names_used,
        prev_meta,
        cfg,
        primary_name=primary_name,
        ui_mask=ui_mask,
    )
    state_hash_after_filtered = _state_hash_filtered(
        curr_grids,
        grid_names_used,
        curr_meta,
        cfg,
        primary_name=primary_name,
        ui_mask=ui_mask,
    )

    grid_delta = _grid_delta(
        prev_grids,
        curr_grids,
        grid_names_used,
        grid_names,
        cfg,
        primary_name=primary_name,
        ui_mask=ui_mask,
        lite_mode=lite_mode,
    )

    if lite_mode:
        event_signatures = [{"sig_id": "lite", "confidence": 1.0}]
        object_deltas = {"moved": [], "appeared_count": 0, "disappeared_count": 0, "split_count": 0, "merge_count": 0}
    else:
        event_signatures = _event_signatures(fp_curr_report)
        object_deltas = _object_deltas(fp_curr_report)
    meta_delta = _meta_delta(prev_meta, curr_meta, cfg)

    action_key = _action_key(action_taken)

    event = TransitionEventV1(
        schema_version="TRANSITION_EVENT_V1",
        game_id=ctx.get("game_id"),
        seed=ctx.get("seed"),
        step_idx=ctx.get("step_idx"),
        action_key=action_key,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        state_hash_before_filtered=state_hash_before_filtered,
        state_hash_after_filtered=state_hash_after_filtered,
        ui_exclusion_mask_rle=None if lite_mode else (_mask_to_rle(ui_mask) if bool(ui_mask.any()) else None),
        ui_exclusion_bboxes=[(int(y0), int(x0), int(y1), int(x1)) for (y0, x0, y1, x1) in ui_bboxes],
        frame_policy={
            "primary_grid_name": primary_name,
            "grid_names_used": grid_names_used,
            "aggregation": aggregation,
        },
        grid_delta=grid_delta,
        event_signatures=event_signatures,
        object_deltas=object_deltas,
        meta_delta={**meta_delta, "done": ctx.get("done"), "win": ctx.get("win")},
    )

    if cfg.log_transition_events:
        logger.info(
            "TransitionEvent %s action=%s changed=%s sig=%s policy=%s",
            ctx.get("step_idx"),
            action_key.get("id"),
            grid_delta.get("changed_cells_count"),
            event_signatures[0]["sig_id"] if event_signatures else "unknown",
            aggregation,
        )
    return event


def to_json(event: TransitionEventV1) -> Dict[str, Any]:
    payload = asdict(event)
    return payload


def _action_key(action_taken: Dict[str, Any]) -> Dict[str, Any]:
    kind = "SIMPLE"
    action_id = None
    x = None
    y = None
    if isinstance(action_taken, dict):
        action_id = action_taken.get("action_id") or action_taken.get("id") or action_taken.get("name")
        if action_taken.get("type") == "coord" or action_taken.get("kind") == "coord":
            kind = "COORD"
            x = action_taken.get("x")
            y = action_taken.get("y")
        elif action_taken.get("type") == "reset":
            kind = "RESET"
    return {
        "kind": kind,
        "id": action_id,
        "x": x,
        "y": y,
    }


def _state_hash(
    grids: List[Any],
    grid_names_used: List[str],
    meta: Dict[str, Any],
    cfg: TransitionEventCompilerConfig,
) -> str:
    hasher = hashlib.sha256()
    for name, grid in zip(grid_names_used, grids[: len(grid_names_used)]):
        arr = np.asarray(grid)
        hasher.update(str(name).encode("utf-8"))
        hasher.update(str(getattr(arr, "shape", None)).encode("utf-8"))
        hasher.update(arr.tobytes())
    for key in sorted(cfg.hash_meta_whitelist):
        value = meta.get(key, None) if isinstance(meta, dict) else None
        hasher.update(str(key).encode("utf-8"))
        hasher.update(str(value).encode("utf-8"))
    return hasher.hexdigest()


def _state_hash_filtered(
    grids: List[Any],
    grid_names_used: List[str],
    meta: Dict[str, Any],
    cfg: TransitionEventCompilerConfig,
    primary_name: str,
    ui_mask: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    for name, grid in zip(grid_names_used, grids[: len(grid_names_used)]):
        hasher.update(str(name).encode("utf-8"))
        arr = np.asarray(grid)
        hasher.update(str(getattr(arr, "shape", None)).encode("utf-8"))
        if name == primary_name and arr.shape == ui_mask.shape:
            digest = hash_state_filtered(arr, ui_mask)
        else:
            digest = hash_state(arr)
        hasher.update(digest.encode("utf-8"))
    for key in sorted(cfg.hash_meta_whitelist):
        value = meta.get(key, None) if isinstance(meta, dict) else None
        hasher.update(str(key).encode("utf-8"))
        hasher.update(str(value).encode("utf-8"))
    return hasher.hexdigest()


def _grid_delta(
    prev_grids: List[Any],
    curr_grids: List[Any],
    grid_names_used: List[str],
    all_names: List[str],
    cfg: TransitionEventCompilerConfig,
    primary_name: str,
    ui_mask: np.ndarray,
    lite_mode: bool = False,
) -> Dict[str, Any]:
    changed_cells_total = 0
    changed_cells_total_filtered = 0
    bbox_union: Optional[Tuple[int, int, int, int]] = None
    color_counts: Dict[str, int] = {}
    palette_before: set[int] = set()
    palette_after: set[int] = set()

    for idx, name in enumerate(all_names):
        if name not in grid_names_used:
            continue
        if idx >= len(prev_grids) or idx >= len(curr_grids):
            continue
        prev_grid = prev_grids[idx]
        curr_grid = curr_grids[idx]
        diff = diff_mask(curr_grid, prev_grid)
        changed_cells = int(diff.sum())
        changed_cells_total += changed_cells
        if name == primary_name and np.asarray(diff).shape == ui_mask.shape:
            changed_cells_total_filtered += int(np.logical_and(np.asarray(diff, dtype=bool), ~ui_mask).sum())
        else:
            changed_cells_total_filtered += changed_cells
        if not lite_mode:
            bbox = bbox_inclusive(diff)
            if bbox is not None:
                bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    y0, x0, y1, x1 = bbox_union
                    by0, bx0, by1, bx1 = bbox
                    bbox_union = (min(y0, by0), min(x0, bx0), max(y1, by1), max(x1, bx1))
            for key, count in changed_colors(curr_grid, prev_grid).items():
                color_counts[key] = color_counts.get(key, 0) + int(count)
            palette_before.update(palette(prev_grid))
            palette_after.update(palette(curr_grid))

    palette_added: List[int] = []
    palette_removed: List[int] = []
    changed_colors_list: List[Dict[str, Any]] = []
    if not lite_mode:
        palette_added = sorted(palette_after - palette_before)
        palette_removed = sorted(palette_before - palette_after)
        changed_colors_list = _top_changed_colors(color_counts, cfg.changed_colors_topM)

    return {
        "changed_cells_count": changed_cells_total,
        "changed_cells_count_raw": changed_cells_total,
        "changed_cells_count_filtered": changed_cells_total_filtered,
        "changed_bbox": bbox_union,
        "changed_colors": changed_colors_list,
        "palette_added": palette_added,
        "palette_removed": palette_removed,
    }


def _default_ui_bar_cfg() -> Dict[str, Any]:
    return {
        "enabled": False,
        "min_row_diff_frac": 0.6,
        "min_row_diff_abs": 8,
        "max_band_height": 3,
        "min_total_diff_frac": 0.6,
        "min_run_frac": 0.5,
        "min_run_abs": 8,
        "pad_cols": 1,
    }


def _load_ui_bar_from_rl_config_file() -> Dict[str, Any]:
    global _RL_UI_BAR_CFG_CACHE
    if _RL_UI_BAR_CFG_CACHE is not None:
        return dict(_RL_UI_BAR_CFG_CACHE)
    defaults = _default_ui_bar_cfg()
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "v4_5", "config", "agents_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reward = data.get("reward", {}) if isinstance(data, dict) else {}
        ui_bar = reward.get("ui_bar", {}) if isinstance(reward, dict) else {}
        out = {**defaults, **(ui_bar if isinstance(ui_bar, dict) else {})}
    except Exception:
        out = defaults
    _RL_UI_BAR_CFG_CACHE = dict(out)
    return out


def _resolve_ui_bar_cfg(cfg: TransitionEventCompilerConfig, ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = _default_ui_bar_cfg()
    out.update(_load_ui_bar_from_rl_config_file())
    cfg_ui = getattr(cfg, "ui_bar", None)
    if isinstance(cfg_ui, dict):
        out.update(cfg_ui)
    if isinstance(ctx.get("reward"), dict) and isinstance(ctx["reward"].get("ui_bar"), dict):
        out.update(ctx["reward"]["ui_bar"])
    return out


def _resolve_lite_mode(ctx: Dict[str, Any]) -> bool:
    pipeline = ctx.get("pipeline", {}) if isinstance(ctx, dict) else {}
    pipeline_mode = str(pipeline.get("mode", "")) if isinstance(pipeline, dict) else ""
    default = str(pipeline_mode).lower() == "rl_only"
    te_cfg = ctx.get("transition_event", {}) if isinstance(ctx, dict) else {}
    if isinstance(te_cfg, dict) and "lite_mode" in te_cfg:
        return bool(te_cfg.get("lite_mode"))
    return default


def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(mask, dtype=bool)
    flat = arr.reshape(-1)
    runs: List[List[int]] = []
    i = 0
    n = int(flat.shape[0])
    while i < n:
        if not flat[i]:
            i += 1
            continue
        start = i
        j = i
        while j < n and bool(flat[j]):
            j += 1
        runs.append([int(start), int(j - start)])
        i = j
    return {"shape": [int(arr.shape[0]), int(arr.shape[1])], "runs": runs}


def _largest_contiguous_run(xs: np.ndarray) -> int:
    if xs.size == 0:
        return 0
    s = np.sort(xs)
    best = 1
    cur = 1
    for i in range(1, s.size):
        if int(s[i]) == int(s[i - 1]) + 1:
            cur += 1
        else:
            if cur > best:
                best = cur
            cur = 1
    if cur > best:
        best = cur
    return int(best)


def detect_ui_progress_bar_mask(prev_grid: Any, next_grid: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    prev_arr = np.asarray(prev_grid)
    next_arr = np.asarray(next_grid)
    if prev_arr.shape != next_arr.shape or prev_arr.ndim != 2:
        return np.zeros_like(next_arr, dtype=bool), []

    h, w = int(next_arr.shape[0]), int(next_arr.shape[1])
    diff = prev_arr != next_arr
    total_diff = int(diff.sum())
    if h == 0 or w == 0:
        return np.zeros_like(next_arr, dtype=bool), []

    row_counts = np.asarray(diff.sum(axis=1), dtype=np.int64)
    row_thr = max(int(np.floor(w * float(cfg.get("min_row_diff_frac", 0.6)))), int(cfg.get("min_row_diff_abs", 8)))
    candidate = row_counts >= row_thr
    if not bool(candidate.any()):
        return np.zeros_like(next_arr, dtype=bool), []

    max_band_height = max(1, int(cfg.get("max_band_height", 3)))
    best_key: Optional[Tuple[int, int, int]] = None
    best_band: Optional[Tuple[int, int]] = None
    for y0 in range(h):
        for bh in range(1, max_band_height + 1):
            y1 = y0 + bh
            if y1 > h:
                break
            if not bool(candidate[y0:y1].all()):
                continue
            score = int(row_counts[y0:y1].sum())
            key = (score, -y0, -bh)
            if best_key is None or key > best_key:
                best_key = key
                best_band = (y0, y1)

    if best_band is None:
        return np.zeros_like(next_arr, dtype=bool), []

    y0, y1 = best_band
    band_diff = diff[y0:y1, :]
    total_band_diff = int(band_diff.sum())

    min_total_diff_frac = float(cfg.get("min_total_diff_frac", 0.6))
    if total_diff > 0:
        if (float(total_band_diff) / float(total_diff)) < min_total_diff_frac:
            return np.zeros_like(next_arr, dtype=bool), []

    min_run = max(int(np.floor(w * float(cfg.get("min_run_frac", 0.5)))), int(cfg.get("min_run_abs", 8)))
    for y in range(y0, y1):
        xs = np.where(diff[y])[0]
        if _largest_contiguous_run(xs) < min_run:
            return np.zeros_like(next_arr, dtype=bool), []

    ui_mask = np.zeros_like(diff, dtype=bool)
    pad_cols = max(0, int(cfg.get("pad_cols", 1)))
    xs_band = np.where(band_diff)[1]
    if xs_band.size > 0:
        x0 = max(0, int(xs_band.min()) - pad_cols)
        x1 = min(w, int(xs_band.max()) + 1 + pad_cols)
    else:
        x0, x1 = 0, w
    ui_mask[y0:y1, x0:x1] = True
    return ui_mask, [(int(y0), int(x0), int(y1), int(x1))]


def _top_changed_colors(counts: Dict[str, int], top_m: int) -> List[Dict[str, Any]]:
    entries = []
    for key, count in counts.items():
        if "->" not in key:
            continue
        left, right = key.split("->", 1)
        try:
            entries.append({"from": int(left), "to": int(right), "count": int(count)})
        except Exception:
            continue
    entries.sort(key=lambda e: (-e["count"], e["from"], e["to"]))
    return entries[:top_m]


def _event_signatures(fp_curr_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    diff = fp_curr_report.get("diff_summary") or {}
    sigs = diff.get("event_signatures") or []
    if not sigs:
        return [{"sig_id": "unknown", "confidence": 1.0}]
    normalized = []
    for sig in sigs:
        if isinstance(sig, dict):
            sig_id = sig.get("kind") or sig.get("sig_id")
            confidence = sig.get("confidence", 0.0)
        else:
            sig_id = str(sig)
            confidence = 0.0
        if not sig_id:
            continue
        normalized.append({"sig_id": str(sig_id), "confidence": float(confidence)})
    normalized.sort(key=lambda e: (-e["confidence"], e["sig_id"]))
    return normalized


def _object_deltas(fp_curr_report: Dict[str, Any]) -> Dict[str, Any]:
    diff = fp_curr_report.get("diff_summary") or {}
    deltas = diff.get("per_object_deltas") or []
    moved = []
    appeared = 0
    disappeared = 0
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        event = delta.get("event")
        if event == "moved":
            moved.append({"object_id": delta.get("object_id"), "dy": delta.get("dy"), "dx": delta.get("dx")})
        elif event == "appeared":
            appeared += 1
        elif event == "disappeared":
            disappeared += 1
    return {
        "moved": moved,
        "appeared_count": appeared,
        "disappeared_count": disappeared,
        "split_count": 0,
        "merge_count": 0,
    }


def _meta_delta(prev_meta: Optional[Dict[str, Any]], curr_meta: Optional[Dict[str, Any]], cfg: TransitionEventCompilerConfig) -> Dict[str, Any]:
    prev_meta = prev_meta or {}
    curr_meta = curr_meta or {}
    def _sorted_actions(value: Any) -> List[str]:
        if isinstance(value, list):
            return sorted(str(v) for v in value)
        return []

    meta_keys_used = sorted(cfg.hash_meta_whitelist)
    return {
        "available_actions_before": _sorted_actions(prev_meta.get("available_actions")),
        "available_actions_after": _sorted_actions(curr_meta.get("available_actions")),
        "terminal_before": prev_meta.get("terminal"),
        "terminal_after": curr_meta.get("terminal"),
        "reward_before": prev_meta.get("reward"),
        "reward_after": curr_meta.get("reward"),
        "meta_keys_used": meta_keys_used,
    }
