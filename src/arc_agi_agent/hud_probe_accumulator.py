from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _GameProbeState:
    steps_seen: int
    deltas: Deque[np.ndarray] = field(default_factory=deque)
    change_counts: Optional[np.ndarray] = None
    finalized: bool = False
    final_spec: Optional[Dict[str, Any]] = None


def _edge_band_mask(shape: Tuple[int, int], margin: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    m = max(0, int(margin))
    z = np.zeros((h, w), dtype=bool)
    if h <= 0 or w <= 0:
        return z
    z[: min(h, m + 1), :] = True
    z[max(0, h - m - 1) :, :] = True
    z[:, : min(w, m + 1)] = True
    z[:, max(0, w - m - 1) :] = True
    return z


def _components_4n(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            comp: List[Tuple[int, int]] = []
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in dirs:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            comps.append(comp)
    return comps


def _bbox_from_comp(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    ys = [p[0] for p in comp]
    xs = [p[1] for p in comp]
    y0, y1 = min(ys), max(ys) + 1
    x0, x1 = min(xs), max(xs) + 1
    return int(y0), int(x0), int(y1), int(x1)


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


class HudProbeAccumulator:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self._states: Dict[str, _GameProbeState] = {}

    def observe(self, game_id: str, grid_prev: Any, grid_curr: Any) -> None:
        gid = str(game_id)
        prev = np.asarray(grid_prev, dtype=np.int64)
        curr = np.asarray(grid_curr, dtype=np.int64)
        if prev.ndim != 2 or curr.ndim != 2:
            return
        if prev.shape != curr.shape:
            prev = curr.copy()
        delta = np.asarray(curr != prev, dtype=bool)

        st = self._states.get(gid)
        window = max(2, int(self.cfg.get("hud_detect_window", 30)))
        if st is None or st.change_counts is None or st.change_counts.shape != delta.shape:
            st = _GameProbeState(steps_seen=0, deltas=deque(maxlen=window), change_counts=np.zeros(delta.shape, dtype=np.int32))
            self._states[gid] = st
        if st.finalized:
            return

        if len(st.deltas) >= window:
            oldest = st.deltas.popleft()
            st.change_counts -= oldest.astype(np.int32)
        st.deltas.append(delta)
        st.change_counts += delta.astype(np.int32)
        st.steps_seen += 1

    def finalize(self, game_id: str) -> Optional[Dict[str, Any]]:
        gid = str(game_id)
        st = self._states.get(gid)
        if st is None:
            return None
        if st.finalized:
            return st.final_spec
        if st.change_counts is None:
            st.finalized = True
            st.final_spec = None
            return None

        min_steps = int(self.cfg.get("hud_probe_min_steps", self.cfg.get("hud_detect_window", 30)))
        window_len = len(st.deltas)
        if window_len < max(2, min_steps):
            st.finalized = True
            st.final_spec = None
            return None

        h, w = int(st.change_counts.shape[0]), int(st.change_counts.shape[1])
        edge = _edge_band_mask((h, w), int(self.cfg.get("hud_edge_margin", 4)))
        change_counts_edge = np.where(edge, st.change_counts, 0)

        cell_rate = change_counts_edge.astype(np.float32) / float(max(1, window_len))
        cell_support_threshold = 1.0 / float(max(1, window_len))
        support = np.asarray(cell_rate >= cell_support_threshold, dtype=bool)

        min_area = max(1, int(self.cfg.get("hud_min_component_area", 20)))
        candidates: List[Tuple[int, int, int, int]] = []
        for comp in _components_4n(support):
            area = len(comp)
            if area < min_area:
                continue
            candidates.append(_bbox_from_comp(comp))

        min_changed_per_step = max(1, int(self.cfg.get("hud_min_changed_cells_per_step", 1)))
        activity_thr = float(self.cfg.get("hud_change_rate_threshold", 0.8))
        kept: List[Tuple[int, int, int, int]] = []
        bbox_scores: List[Dict[str, Any]] = []
        for (y0, x0, y1, x1) in candidates:
            active_steps = 0
            edge_roi = edge[y0:y1, x0:x1]
            for d in st.deltas:
                active_cells = int(np.logical_and(d[y0:y1, x0:x1], edge_roi).sum())
                if active_cells >= min_changed_per_step:
                    active_steps += 1
            region_activity = float(active_steps) / float(max(1, window_len))
            bbox_scores.append({"bbox": [int(y0), int(x0), int(y1), int(x1)], "region_activity": region_activity})
            if region_activity >= activity_thr:
                kept.append((y0, x0, y1, x1))

        if not kept:
            st.finalized = True
            st.final_spec = None
            return None

        hud_mask = np.zeros((h, w), dtype=bool)
        for (y0, x0, y1, x1) in kept:
            hud_mask[y0:y1, x0:x1] = True

        ys, xs = np.where(hud_mask)
        union_bbox = [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1] if ys.size > 0 else None
        hud_area = int(hud_mask.sum())

        spec = {
            "game_id": gid,
            "shape": [h, w],
            "window_len": int(window_len),
            "bboxes": [[int(y0), int(x0), int(y1), int(x1)] for (y0, x0, y1, x1) in kept],
            "bbox_scores": bbox_scores,
            "hud_bbox": union_bbox,
            "hud_mask_rle": _mask_to_rle(hud_mask),
            "hud_area": hud_area,
            "hud_area_frac": float(hud_area) / float(max(1, h * w)),
            "steps_seen": int(st.steps_seen),
        }
        st.finalized = True
        st.final_spec = spec
        return spec

    def finalize_all(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for gid in sorted(self._states.keys()):
            spec = self.finalize(gid)
            if spec is not None:
                out[gid] = spec
        return out

    def save_all(self, cache_dir: str) -> Dict[str, str]:
        os.makedirs(cache_dir, exist_ok=True)
        out: Dict[str, str] = {}
        for gid, spec in self.finalize_all().items():
            path = os.path.join(cache_dir, f"{gid}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spec, f, separators=(",", ":"))
            out[gid] = path
        return out
