"""poi_detector.py — Module 2.

Identifies all visually distinct objects (POIs) from a batch of episodes.
Runs offline; no env interaction.

Pipeline:
  Step 1 — Background/Foreground separation
  Step 2 — Connected-component BBox clustering per frame
  Step 3 — Aggregate bboxes across frames → canonical POIs
  Step 4 — Sprite Detection (action-correlation → tag SELF)
  Step 5 — Reachability filter
"""
from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import logging

from .structs import EpisodeRecord, POIRecord
from .config import (
    MIN_BBOX_AREA, K_PROXIMITY_PX, K_PROXIMITY_REACHABLE, MAX_SPRITE_AREA, CONFIDENCE_NEW,
    CLUSTER_CENTROID_DIST, HUD_FREQ_THRESH, STATIC_VAR_THRESH,
    SECONDARY_BG_THRESH, SPRITE_CORR_THRESH, MIN_MOTION_FRAMES,
    SELF_CORRELATION_THRESHOLD, MIN_MOVEMENT_STEPS,
)

logger = logging.getLogger(__name__)


# ── Connected components ──────────────────────────────────────────────────────

def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (labeled, n_labels) on a 2D bool mask.
    Uses scipy if available, else falls back to BFS flood fill.
    """
    try:
        from scipy.ndimage import label as _scipy_label
        labeled, n = _scipy_label(mask)
        return labeled.astype(np.int32), int(n)
    except ImportError:
        pass

    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    n = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c] and labeled[r, c] == 0:
                n += 1
                queue = [(r, c)]
                labeled[r, c] = n
                while queue:
                    y, x = queue.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = n
                            queue.append((ny, nx))
    return labeled, n


# ── Background detection ──────────────────────────────────────────────────────

def _find_bg_colors(frames: List[np.ndarray]) -> List[int]:
    """Return the 1–2 most frequent pixel values across all frames (background colors)."""
    all_vals = np.concatenate([np.asarray(f).reshape(-1) for f in frames])
    counts = np.bincount(all_vals.clip(0, 15).astype(np.int64), minlength=16)
    total = max(1, all_vals.size)

    primary = int(counts.argmax())
    bg = [primary]

    counts2 = counts.copy()
    counts2[primary] = 0
    second = int(counts2.argmax())
    if counts2[second] / total > SECONDARY_BG_THRESH:
        bg.append(second)

    return bg


# ── Per-frame component extraction ───────────────────────────────────────────

def _bbox_from_mask(comp_mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.where(comp_mask.any(axis=1))[0]
    cols = np.where(comp_mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def _dominant_colors(
    frame: np.ndarray,
    comp_mask: np.ndarray,
    top_k: int = 3,
    min_freq: float = 0.20,   # color must be >= 20% of bbox pixels to be included
) -> List[int]:
    """Return up to top_k dominant colors that each cover >= min_freq of the component.

    The min_freq threshold prevents border/background bleed (e.g. a single row of
    color 11 at the edge of the bbox) from contaminating the color signature.
    """
    pixels = frame[comp_mask]
    if pixels.size == 0:
        return [0]
    counts = np.bincount(pixels.clip(0, 15).astype(np.int64), minlength=16)
    min_count = max(1, math.ceil(pixels.size * min_freq))
    order = counts.argsort()[::-1]
    result = [int(c) for c in order[:top_k] if counts[c] >= min_count]
    # Always return at least the single most frequent color
    if not result:
        result = [int(order[0])]
    return result


def _cluster_fg_components(frame: np.ndarray, bg_colors: List[int]) -> List[dict]:
    """Return list of {bbox, color_sig, area, cx, cy} for each FG component >= MIN_BBOX_AREA."""
    frame = np.asarray(frame)
    mask = np.ones(frame.shape[:2], dtype=bool)
    for c in bg_colors:
        mask &= (frame != c)

    labeled, n = _connected_components(mask)
    result = []
    for lbl in range(1, n + 1):
        comp_mask = labeled == lbl
        area = int(comp_mask.sum())
        if area < MIN_BBOX_AREA:
            continue
        bbox = _bbox_from_mask(comp_mask)
        y0, x0, y1, x1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        color_sig = _dominant_colors(frame, comp_mask)
        result.append({"bbox": bbox, "color_sig": color_sig, "area": area, "cx": cx, "cy": cy})
    return result


# ── Centroid utility ──────────────────────────────────────────────────────────

def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


# ── Motion detection ──────────────────────────────────────────────────────────

def _has_motion(centroid_series: List[Tuple[float, float]]) -> bool:
    """True if centroid moves more than 0.5 cells between any two consecutive appearances."""
    for i in range(1, len(centroid_series)):
        da = centroid_series[i][0] - centroid_series[i - 1][0]
        db = centroid_series[i][1] - centroid_series[i - 1][1]
        if (da ** 2 + db ** 2) ** 0.5 > 0.5:
            return True
    return False


# ── Cross-frame POI aggregation ───────────────────────────────────────────────

def _aggregate_pois(
    frame_components: List[List[dict]],   # per-frame list of component dicts
    n_total_frames: int,
    version: int = 0,
    cluster_dist: float = CLUSTER_CENTROID_DIST,
    hud_freq: float = HUD_FREQ_THRESH,
    static_var: float = STATIC_VAR_THRESH,
) -> Tuple[List[POIRecord], Dict[str, dict]]:
    """
    Greedy centroid-based clustering of bboxes across all frames.

    Returns:
        pois          — List[POIRecord] with preliminary tags
        centroid_data — Dict[poi_id → {centroids, frame_indices, var, freq}]
    """
    # Each cluster tracks a running mean centroid and per-appearance data
    clusters: List[dict] = []

    for frame_idx, comps in enumerate(frame_components):
        for comp in comps:
            cx, cy = comp["cx"], comp["cy"]

            # Find nearest existing cluster within cluster_dist
            best_cl = None
            best_d = cluster_dist
            for cl in clusters:
                d = math.sqrt((cx - cl["mcx"]) ** 2 + (cy - cl["mcy"]) ** 2)
                if d < best_d:
                    best_d = d
                    best_cl = cl

            if best_cl is not None:
                best_cl["centroids"].append((cx, cy))
                best_cl["bboxes"].append(comp["bbox"])
                best_cl["color_sigs"].extend(comp["color_sig"])
                best_cl["frame_indices"].add(frame_idx)
                # Update running mean
                n = len(best_cl["centroids"])
                best_cl["mcx"] = sum(c[0] for c in best_cl["centroids"]) / n
                best_cl["mcy"] = sum(c[1] for c in best_cl["centroids"]) / n
            else:
                clusters.append({
                    "centroids":    [(cx, cy)],
                    "bboxes":       [comp["bbox"]],
                    "color_sigs":   list(comp["color_sig"]),
                    "frame_indices": {frame_idx},
                    "mcx": cx,
                    "mcy": cy,
                })

    pois: List[POIRecord] = []
    centroid_data: Dict[str, dict] = {}

    for cl in clusters:
        freq = len(cl["frame_indices"]) / max(1, n_total_frames)
        cxs = [c[0] for c in cl["centroids"]]
        cys = [c[1] for c in cl["centroids"]]
        var = math.sqrt(float(np.var(cxs)) + float(np.var(cys)))

        # Representative bbox: last seen
        rep_bbox = cl["bboxes"][-1]

        # Color signature: most common colors across appearances
        color_counts = Counter(cl["color_sigs"])
        color_sig = [c for c, _ in color_counts.most_common(3)]

        # Preliminary tag
        if freq >= hud_freq and var < 2.0:
            tag = "HUD"
        else:
            tag = "UNKNOWN"   # static or dynamic; SpriteDetector refines

        poi_id = str(uuid.uuid4())
        ikey = _make_identity_key(rep_bbox, color_sig, tag=tag)
        motion = _has_motion(cl["centroids"])
        poi = POIRecord(
            poi_id=poi_id,
            bbox=rep_bbox,
            color_signature=color_sig,
            tag=tag,
            reachable=False,
            visited=False,
            consequence=None,
            confidence=CONFIDENCE_NEW,
            version=version,
            identity_key=ikey,
            motion_detected=motion,
        )
        pois.append(poi)
        centroid_data[poi_id] = {
            "centroids":     cl["centroids"],
            "frame_indices": sorted(cl["frame_indices"]),
            "var":           var,
            "freq":          freq,
        }

    return pois, centroid_data


# ── Reachability filter ───────────────────────────────────────────────────────

def _filter_reachable(
    pois: List[POIRecord],
    episodes: List[EpisodeRecord],
    cfg: dict,
) -> List[POIRecord]:
    """Mark POIs reachable if any trajectory passes within K_PROXIMITY_REACHABLE of centroid.

    If no valid sprite positions are available, all non-SELF/HUD POIs are assumed reachable.
    """
    k = int(cfg.get("k_proximity_reachable", K_PROXIMITY_REACHABLE))

    has_positions = any(
        pos is not None
        for ep in episodes
        for pos in ep.positions
    )

    for poi in pois:
        if poi.tag in ("SELF", "HUD"):
            continue
        if not has_positions:
            poi.reachable = True
            continue
        poi_cy = (poi.bbox[0] + poi.bbox[2]) / 2.0
        poi_cx = (poi.bbox[1] + poi.bbox[3]) / 2.0
        reachable = False
        for ep in episodes:
            for pos in ep.positions:
                if pos is None:
                    continue
                px, py = pos   # (col, row)
                if math.sqrt((px - poi_cx) ** 2 + (py - poi_cy) ** 2) <= k:
                    reachable = True
                    break
            if reachable:
                break
        poi.reachable = reachable
    return pois


# ── Stable identity key ───────────────────────────────────────────────────────

def _make_identity_key(bbox: Tuple, color_signature: List[int], tag: str = "") -> str:
    """Stable hash across analysis cycles.

    SELF: keyed by color + area_bin only — position changes every episode.
    All others: keyed by quantised position + color.
    """
    if tag == "SELF":
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        area_bin = area // 4
        key_str = f"SELF:{sorted(color_signature)}:area{area_bin}"
    else:
        quantised = tuple(v // 4 for v in bbox)
        key_str = f"{quantised}:{sorted(color_signature)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


# ── Action-displacement correlation (Bug 2) ───────────────────────────────────

_MOVEMENT_ACTIONS: Dict[str, Tuple[int, int]] = {
    "ACTION1": (0, -1),   # up:    dy < 0
    "ACTION2": (0, +1),   # down:  dy > 0
    "ACTION3": (-1, 0),   # left:  dx < 0
    "ACTION4": (+1, 0),   # right: dx > 0
}


def _action_to_expected_delta(action_str: str) -> Optional[Tuple[int, int]]:
    """Returns expected (dx, dy) for movement actions only. None for non-movement."""
    s = action_str.upper()
    for key, direction in _MOVEMENT_ACTIONS.items():
        if key in s:
            return direction
    return None


def _action_correlation(
    displacements: List[Tuple[float, float]],
    actions: List[str],
    min_steps: int = MIN_MOVEMENT_STEPS,
) -> float:
    """Correlation over movement steps only (filters out ACTION5/ACTION6/RESET).
    Returns Pearson r between (expected_dx+expected_dy) and (actual_dx+actual_dy).
    """
    expected: List[float] = []
    actual: List[float] = []
    for (dx, dy), action in zip(displacements, actions):
        delta = _action_to_expected_delta(action)
        if delta is None:
            continue
        ex, ey = delta
        expected.append(float(ex + ey))
        actual.append(float(dx + dy))

    if len(expected) < min_steps:
        return 0.0

    e_arr = np.array(expected, dtype=float)
    a_arr = np.array(actual, dtype=float)
    if e_arr.std() < 1e-6 or a_arr.std() < 1e-6:
        logger.debug(
            "correlation e_std=%.4f a_std=%.4f n_movement_steps=%d r=0.000 (std too low)",
            float(e_arr.std()), float(a_arr.std()), len(expected),
        )
        return 0.0
    result = float(np.corrcoef(e_arr, a_arr)[0, 1])
    logger.debug(
        "correlation e_std=%.4f a_std=%.4f n_movement_steps=%d r=%.3f",
        float(e_arr.std()), float(a_arr.std()), len(expected), result,
    )
    return result


# ── Bbox area + IoU helpers ───────────────────────────────────────────────────

def _bbox_area(bbox: Tuple) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_iou(a: Tuple, b: Tuple) -> float:
    """Intersection-over-union of two (y0,x0,y1,x1) bboxes."""
    iy0 = max(a[0], b[0]); ix0 = max(a[1], b[1])
    iy1 = min(a[2], b[2]); ix1 = min(a[3], b[3])
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter == 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter) / max(union, 1)


# ── SpriteDetector ────────────────────────────────────────────────────────────

class SpriteDetector:
    """Determines which POI is the player sprite (SELF) via action-correlation."""

    def __init__(self, bg_colors: List[int], cfg: dict):
        self._bg_colors = bg_colors
        self._k_px = int(cfg.get("k_proximity_px", K_PROXIMITY_PX))
        self._corr_thresh = float(cfg.get("self_correlation_threshold", SELF_CORRELATION_THRESHOLD))
        self._min_steps = int(cfg.get("min_movement_steps", MIN_MOVEMENT_STEPS))
        self._max_sprite_area = int(cfg.get("max_sprite_area", MAX_SPRITE_AREA))
        self._self_color_sig: Optional[List[int]] = None
        # diagnostic
        self.last_corr_scores: List[float] = []
        self.self_tagged_by: str = ""        # "store_reuse" | "correlation" | "size_fallback" | ""
        self.self_bbox_updated: bool = False  # True when store_reuse updated SELF bbox this cycle

    def detect_self(
        self,
        episodes: List[EpisodeRecord],
        pois: List[POIRecord],
        centroid_data: Dict[str, dict],
        frame_episode_map: List[Tuple[int, int]],
        store=None,   # Optional[HypothesisStore] — checked for existing SELF
    ) -> Optional[str]:
        """Return poi_id of the SELF sprite, or None if undetermined.

        Priority:
          1. Reuse existing SELF from store (prevents accumulation of wrong SELF tags)
          2. Action-displacement correlation
          3. Size fallback — only when store has no SELF at all
        """
        # 1. Check store for existing SELF
        self.self_bbox_updated = False
        existing_self_in_store: List = []
        if store is not None:
            existing_self_in_store = [p for p in store.get_all() if p.tag == "SELF"]

        # Evict ghost SELFs: if multiple SELF records exist, keep only the most recently
        # updated one. A ghost SELF has a stale version that stopped incrementing.
        if store is not None and len(existing_self_in_store) > 1:
            existing_self_in_store.sort(key=lambda s: s.version, reverse=True)
            for ghost in existing_self_in_store[1:]:
                logger.warning(
                    "SELF_evict ghost poi=%s version=%d bbox=%s",
                    ghost.poi_id[:8], ghost.version, ghost.bbox,
                )
                store.remove(ghost.poi_id)
            existing_self_in_store = existing_self_in_store[:1]

        if existing_self_in_store:
            # Bug F fix: stored SELF uses color+area key; candidates have position-based keys.
            # Build lookup using SELF-scheme key for each candidate so they can be matched.
            current_by_self_key = {
                _make_identity_key(c.bbox, c.color_signature, tag="SELF"): c
                for c in pois
            }
            for s in existing_self_in_store:
                if s.identity_key in current_by_self_key:
                    current = current_by_self_key[s.identity_key]
                    old_key = s.identity_key
                    # Bug G fix: ALWAYS update geometry BEFORE returning — sprite moves
                    s.bbox = current.bbox
                    s.color_signature = current.color_signature
                    s.version = (store.version + 1) if store is not None else s.version
                    s.identity_key = _make_identity_key(s.bbox, s.color_signature, tag="SELF")
                    # Keep _by_identity index consistent if key shifted (area_bin change)
                    if store is not None and s.identity_key != old_key:
                        store._by_identity.pop(old_key, None)
                        store._by_identity[s.identity_key] = s
                    self.self_tagged_by = "store_reuse"
                    self._self_color_sig = current.color_signature
                    self.last_corr_scores = []
                    self.self_bbox_updated = True
                    logger.debug(
                        "SELF_reused poi_id=%s updated_bbox=%s version=%d",
                        s.poi_id[:8], s.bbox, s.version,
                    )
                    return s.poi_id
            # Existing SELF not matched by color+area key this cycle — re-detect
            logger.debug("SELF store SELF not in current candidates — re-detecting")

        # 2. Correlation detection
        self_id = self._correlation_detect(episodes, pois, centroid_data, frame_episode_map)
        if self_id:
            return self_id

        # 3. Size fallback — only when no SELF exists anywhere in store
        if not existing_self_in_store:
            return self._size_fallback(pois)

        # Existing SELF is stale/missing — do not create a new wrong one
        self.self_tagged_by = ""
        return None

    def _correlation_detect(
        self,
        episodes: List[EpisodeRecord],
        pois: List[POIRecord],
        centroid_data: Dict[str, dict],
        frame_episode_map: List[Tuple[int, int]],
    ) -> Optional[str]:
        fmap: Dict[int, Tuple[int, int]] = {i: v for i, v in enumerate(frame_episode_map)}

        best_poi_id: Optional[str] = None
        best_corr = self._corr_thresh
        corr_scores: List[float] = []

        for poi in pois:
            if poi.tag == "HUD":
                continue
            cd = centroid_data.get(poi.poi_id)
            if cd is None or cd["var"] < 0.5:
                continue  # static — unlikely SELF

            displacements: List[Tuple[float, float]] = []
            actions_mv: List[str] = []

            frame_idxs = cd["frame_indices"]
            for i in range(len(frame_idxs) - 1):
                fi = frame_idxs[i]
                fj = frame_idxs[i + 1]
                if fj != fi + 1:
                    continue
                if fi not in fmap:
                    continue
                ep_idx, step_idx = fmap[fi]
                if ep_idx >= len(episodes):
                    continue
                ep = episodes[ep_idx]
                if step_idx >= len(ep.actions):
                    continue
                if i >= len(cd["centroids"]) or i + 1 >= len(cd["centroids"]):
                    continue

                cx0, cy0 = cd["centroids"][i]
                cx1, cy1 = cd["centroids"][i + 1]
                displacements.append((cx1 - cx0, cy1 - cy0))
                actions_mv.append(ep.actions[step_idx])

            corr = _action_correlation(displacements, actions_mv, self._min_steps)
            corr_scores.append(corr)

            if corr > best_corr:
                best_corr = corr
                best_poi_id = poi.poi_id

        self.last_corr_scores = sorted(corr_scores, reverse=True)[:3]

        if best_poi_id is not None:
            self.self_tagged_by = "correlation"
            matched = next(p for p in pois if p.poi_id == best_poi_id)
            self._self_color_sig = matched.color_signature
            return best_poi_id
        return None

    def _size_fallback(self, pois: List[POIRecord]) -> Optional[str]:
        """Tag largest small moving non-HUD component as SELF. Area-capped to exclude walls."""
        moving = [
            p for p in pois
            if p.tag not in ("SELF", "HUD")
            and _bbox_area(p.bbox) <= self._max_sprite_area
            and p.motion_detected
        ]
        if not moving:
            return None
        moving.sort(key=lambda p: _bbox_area(p.bbox), reverse=True)
        self.self_tagged_by = "size_fallback"
        self._self_color_sig = moving[0].color_signature
        logger.warning(
            "SELF size_fallback: poi_id=%s color=%s area=%d bbox=%s",
            moving[0].poi_id[:8], moving[0].color_signature,
            _bbox_area(moving[0].bbox), moving[0].bbox,
        )
        return moving[0].poi_id

    def extract_centroid(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return (x, y) centroid of SELF bbox in this frame via color matching."""
        if self._self_color_sig is None:
            return None
        comps = _cluster_fg_components(frame, self._bg_colors)
        if not comps:
            return None
        target_set = set(self._self_color_sig[:1])
        best_comp = max(comps, key=lambda c: len(set(c["color_sig"]) & target_set))
        return int(best_comp["cx"]), int(best_comp["cy"])


# ── POIDetector ───────────────────────────────────────────────────────────────

class POIDetector:
    """Identifies all visually distinct POIs from a batch of episodes."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sprite_detector: Optional[SpriteDetector] = None  # set after detect()

    def detect(self, episodes: List[EpisodeRecord], store=None) -> List[POIRecord]:
        """Run full pipeline. Returns list of POIRecord.

        store: Optional HypothesisStore — used by SpriteDetector to reuse existing SELF.
        """
        if not episodes:
            return []

        # Collect all frames with episode/step tracking
        all_frames: List[np.ndarray] = []
        frame_episode_map: List[Tuple[int, int]] = []   # frame_idx → (ep_idx, step_idx)
        for ep_idx, ep in enumerate(episodes):
            for step_idx, frame in enumerate(ep.frames):
                all_frames.append(np.asarray(frame))
                frame_episode_map.append((ep_idx, step_idx))

        if not all_frames:
            return []

        # Step 1 — Background
        bg_colors = _find_bg_colors(all_frames)

        # Step 2 — Per-frame component extraction
        frame_components = [_cluster_fg_components(f, bg_colors) for f in all_frames]

        # Step 3 — Aggregate into canonical POIs
        pois, centroid_data = _aggregate_pois(
            frame_components, len(all_frames),
            version=0,
            cluster_dist=float(self.cfg.get("cluster_centroid_dist", CLUSTER_CENTROID_DIST)),
            hud_freq=float(self.cfg.get("hud_freq_thresh", HUD_FREQ_THRESH)),
            static_var=float(self.cfg.get("static_var_thresh", STATIC_VAR_THRESH)),
        )

        # Step 4 — Sprite detection
        sprite_det = SpriteDetector(bg_colors, self.cfg)
        self.sprite_detector = sprite_det
        self_poi_id = sprite_det.detect_self(
            episodes, pois, centroid_data, frame_episode_map, store=store
        )

        # Build list of SELF bboxes (from store + newly detected) to guard against collisions
        self_bboxes: List[Tuple] = []
        if store is not None:
            self_bboxes = [p.bbox for p in store.get_all() if p.tag == "SELF"]

        for poi in pois:
            if poi.poi_id == self_poi_id:
                poi.tag = "SELF"
                # Bug F fix: recompute identity key using SELF-scheme (color+area, not position)
                poi.identity_key = _make_identity_key(poi.bbox, poi.color_signature, tag="SELF")
                self_bboxes.append(poi.bbox)
            elif centroid_data.get(poi.poi_id, {}).get("var", 0.0) > float(
                self.cfg.get("static_var_thresh", STATIC_VAR_THRESH)
            ) and poi.tag != "HUD":
                # Fix 2: do not tag as ENEMY if bbox substantially overlaps an existing SELF
                overlaps_self = any(_bbox_iou(poi.bbox, sb) > 0.5 for sb in self_bboxes)
                if overlaps_self:
                    logger.debug(
                        "enemy_tag_skipped: poi=%s bbox=%s overlaps SELF",
                        poi.poi_id[:8], poi.bbox,
                    )
                else:
                    poi.tag = "ENEMY"

        # Fix I: remove ANY non-SELF POI whose bbox substantially overlaps a SELF bbox
        if self_bboxes:
            removed_ids: set = set()
            for poi in pois:
                if poi.tag == "SELF":
                    continue
                if any(_bbox_iou(poi.bbox, sb) > 0.5 for sb in self_bboxes):
                    removed_ids.add(poi.poi_id)
                    logger.info(
                        "fix_I_poi_removed: poi=%s tag=%s bbox=%s overlaps SELF",
                        poi.poi_id[:8], poi.tag, poi.bbox,
                    )
            if removed_ids:
                pois = [p for p in pois if p.poi_id not in removed_ids]

        # Step 5 — Reachability filter
        pois = _filter_reachable(pois, episodes, cfg=self.cfg)

        return pois
