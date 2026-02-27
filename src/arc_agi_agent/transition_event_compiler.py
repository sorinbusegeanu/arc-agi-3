from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .grid_utils import changed_bbox as bbox_inclusive, changed_colors, diff_mask, palette
from .logger import get_logger
from .normalize import normalize_observation
from .transition_event_compiler_config import TransitionEventCompilerConfig
from .transition_event_compiler_types import TransitionEventV1

logger = get_logger(__name__)


def compile_transition_event(
    prev_observation: Any,
    observation: Any,
    action_taken: Dict[str, Any],
    fp_prev_report: Dict[str, Any],
    fp_curr_report: Dict[str, Any],
    ctx: Dict[str, Any],
    cfg: Optional[TransitionEventCompilerConfig] = None,
    env_frame_names: Optional[List[str]] = None,
) -> TransitionEventV1:
    cfg = cfg or TransitionEventCompilerConfig()
    prev_norm = normalize_observation(prev_observation, schema_warnings=[])
    curr_norm = normalize_observation(observation, schema_warnings=[])

    grid_names = env_frame_names or curr_norm.grid_names or [f"grid_{i}" for i in range(len(curr_norm.grids))]
    if len(grid_names) < len(curr_norm.grids):
        grid_names = grid_names + [f"grid_{i}" for i in range(len(grid_names), len(curr_norm.grids))]

    primary_name = grid_names[0] if grid_names else "grid_0"
    if cfg.enable_multigrid:
        grid_names_used = list(grid_names)
        aggregation = "UNION_DIFFS"
    else:
        grid_names_used = [primary_name]
        aggregation = "PRIMARY_ONLY"

    state_hash_before = _state_hash(prev_norm.grids, grid_names_used, prev_norm.meta or {}, cfg)
    state_hash_after = _state_hash(curr_norm.grids, grid_names_used, curr_norm.meta or {}, cfg)

    grid_delta = _grid_delta(prev_norm.grids, curr_norm.grids, grid_names_used, grid_names, cfg)

    event_signatures = _event_signatures(fp_curr_report)
    object_deltas = _object_deltas(fp_curr_report)
    meta_delta = _meta_delta(prev_norm.meta, curr_norm.meta, cfg)

    action_key = _action_key(action_taken)

    event = TransitionEventV1(
        schema_version="TRANSITION_EVENT_V1",
        game_id=ctx.get("game_id"),
        seed=ctx.get("seed"),
        step_idx=ctx.get("step_idx"),
        action_key=action_key,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        frame_policy={
            "primary_grid_name": primary_name,
            "grid_names_used": grid_names_used,
            "aggregation": aggregation,
        },
        grid_delta=grid_delta,
        event_signatures=event_signatures,
        object_deltas=object_deltas,
        meta_delta=meta_delta,
    )

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
        hasher.update(str(name).encode("utf-8"))
        hasher.update(str(getattr(grid, "shape", None)).encode("utf-8"))
        hasher.update(grid.tobytes())
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
) -> Dict[str, Any]:
    changed_cells_total = 0
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

    palette_added = sorted(palette_after - palette_before)
    palette_removed = sorted(palette_before - palette_after)

    changed_colors_list = _top_changed_colors(color_counts, cfg.changed_colors_topM)

    return {
        "changed_cells_count": changed_cells_total,
        "changed_bbox": bbox_union,
        "changed_colors": changed_colors_list,
        "palette_added": palette_added,
        "palette_removed": palette_removed,
    }


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
