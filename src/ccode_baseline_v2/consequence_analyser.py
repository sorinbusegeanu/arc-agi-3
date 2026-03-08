"""consequence_analyser.py — Module 3.

Fires when agent reaches a POI. Measures pixel diff + histogram shift,
classifies consequence, detects GAME_WON / LEVEL_CHANGE.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .structs import ConsequenceResult, POIRecord
from .config import PIXEL_DIFF_THRESHOLD, HISTOGRAM_SHIFT_THR, K_PROXIMITY_PX


# ── Object state delta ───────────────────────────────────────────────────────

@dataclass
class ObjectStateDelta:
    poi_id: str
    bbox: Tuple[int, int, int, int]
    pixel_hash_before: str
    pixel_hash_after: str
    changed: bool            # pixel_hash_before != pixel_hash_after
    changed_ratio: float     # fraction of bbox pixels that changed


def _pixel_hash(crop: np.ndarray) -> str:
    """MD5 of raw pixel bytes — stable visual fingerprint of a bbox region."""
    return hashlib.md5(np.asarray(crop).tobytes()).hexdigest()[:12]


def extract_object_deltas(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    pois: List[POIRecord],
) -> List[ObjectStateDelta]:
    """For each POI bbox, crop both frames and compute pixel hash + diff ratio.

    Called after a BIG_CHANGE/SMALL_CHANGE consequence to identify which specific
    objects changed and record their new visual state (pixel_hash).
    """
    fb = np.asarray(frame_before)
    fa = np.asarray(frame_after)
    h, w = fb.shape[:2]
    deltas: List[ObjectStateDelta] = []
    for poi in pois:
        y0, x0, y1, x1 = poi.bbox
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        if y1 <= y0 or x1 <= x0:
            continue
        crop_before = fb[y0:y1, x0:x1]
        crop_after  = fa[y0:y1, x0:x1]
        h_before = _pixel_hash(crop_before)
        h_after  = _pixel_hash(crop_after)
        ratio = float(np.sum(crop_before != crop_after)) / max(crop_before.size, 1)
        deltas.append(ObjectStateDelta(
            poi_id=poi.poi_id,
            bbox=poi.bbox,
            pixel_hash_before=h_before,
            pixel_hash_after=h_after,
            changed=(h_before != h_after),
            changed_ratio=ratio,
        ))
    return deltas


# ── Signal 1 — Pixel Diff ────────────────────────────────────────────────────

def _pixel_diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that differ between frames a and b."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return 1.0
    total = max(1, a.size)
    return float((a != b).sum()) / total


# ── Signal 2 — Histogram Shift (primary) ────────────────────────────────────

def _histogram_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between 16-bin color histograms of frames a and b.

    0.0 = identical histograms. 1.0 = orthogonal (completely different room).
    """
    a_flat = np.asarray(a).reshape(-1).clip(0, 15).astype(np.int64)
    b_flat = np.asarray(b).reshape(-1).clip(0, 15).astype(np.int64)
    h_a = np.bincount(a_flat, minlength=16).astype(np.float64)
    h_b = np.bincount(b_flat, minlength=16).astype(np.float64)
    norm_a = np.linalg.norm(h_a)
    norm_b = np.linalg.norm(h_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    cosine_sim = float(np.dot(h_a, h_b) / (norm_a * norm_b))
    return max(0.0, 1.0 - cosine_sim)


# ── Classification ───────────────────────────────────────────────────────────

def _classify(pixel_ratio: float, hist_shift: float,
              pixel_thr: float = PIXEL_DIFF_THRESHOLD,
              hist_thr: float = HISTOGRAM_SHIFT_THR) -> str:
    # GAME_WON: very large histogram shift + significant pixel change → non-gameplay screen
    if pixel_ratio > pixel_thr and hist_shift > 0.5:
        return "GAME_WON"
    # LEVEL_CHANGE: large histogram shift (new room), but game continues
    if hist_shift > hist_thr:
        return "LEVEL_CHANGE"
    # BIG_CHANGE: significant pixel diff in same room
    if pixel_ratio > pixel_thr:
        return "BIG_CHANGE"
    # SMALL_CHANGE: localised diff (item pickup, toggle)
    if pixel_ratio > 0.01:
        return "SMALL_CHANGE"
    return "NO_CHANGE"


# ── ConsequenceAnalyser ──────────────────────────────────────────────────────

class ConsequenceAnalyser:
    """Compares frame_before vs frame_after and returns a ConsequenceResult."""

    def __init__(self, cfg: dict):
        self._pixel_thr  = float(cfg.get("pixel_diff_threshold", PIXEL_DIFF_THRESHOLD))
        self._hist_thr   = float(cfg.get("histogram_shift_thr",  HISTOGRAM_SHIFT_THR))
        self._k_prox     = int(cfg.get("k_proximity_px",         K_PROXIMITY_PX))

    def analyse(self, frame_before: np.ndarray, frame_after: np.ndarray) -> ConsequenceResult:
        pixel_ratio = _pixel_diff_ratio(frame_before, frame_after)
        hist_shift  = _histogram_shift(frame_before, frame_after)
        label = _classify(pixel_ratio, hist_shift, self._pixel_thr, self._hist_thr)
        return ConsequenceResult(
            label=label,
            pixel_diff_ratio=pixel_ratio,
            histogram_shift=hist_shift,
        )

    def is_near_poi(self, position: Tuple[int, int], poi: POIRecord) -> bool:
        """True if position (x, y) is within K_PROXIMITY_PX of poi bbox centroid."""
        y0, x0, y1, x1 = poi.bbox
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        px, py = position
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= self._k_prox
