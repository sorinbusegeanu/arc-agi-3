"""match_detector.py — Module 7.

After any BIG_CHANGE: scan all POI pairs and check if any two share the same
pixel_hash (or are pixel-similar within APPROX_THRESHOLD).

If a match is found, the pattern-match win-precondition may be satisfied.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from .structs import POIRecord

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    matched: bool
    poi_id_a: Optional[str]     # the object that changed
    poi_id_b: Optional[str]     # the reference object it now matches
    match_score: float          # 0.0–1.0; 1.0 = exact pixel hash match
    confidence: str             # "exact" | "approximate" | "none"


class MatchDetector:
    """Scan all POI pairs in the store for pixel-level visual matches.

    Compares every pair of non-SELF non-HUD POIs that have a recorded pixel_hash.
    A match at >= APPROX_THRESHOLD indicates the pattern-match condition may be met.
    """

    APPROX_THRESHOLD = 0.85

    def check(
        self,
        store,   # HypothesisStore — duck-typed to avoid circular import
        frame_curr: np.ndarray,
    ) -> MatchResult:
        """Return the best-scoring POI pair match found in the current frame."""
        pois: List[POIRecord] = [
            p for p in store.get_all()
            if p.tag not in ("SELF", "HUD") and p.pixel_hash is not None
        ]

        no_match = MatchResult(
            matched=False, poi_id_a=None, poi_id_b=None,
            match_score=0.0, confidence="none",
        )

        if len(pois) < 2:
            return no_match

        frame = np.asarray(frame_curr)
        best = no_match

        for i, a in enumerate(pois):
            for b in pois[i + 1:]:
                if a.poi_id == b.poi_id:
                    continue
                score = self._compare(a, b, frame)
                if score > best.match_score:
                    if score == 1.0:
                        confidence = "exact"
                    elif score >= self.APPROX_THRESHOLD:
                        confidence = "approximate"
                    else:
                        confidence = "none"
                    best = MatchResult(
                        matched=(score >= self.APPROX_THRESHOLD),
                        poi_id_a=a.poi_id,
                        poi_id_b=b.poi_id,
                        match_score=score,
                        confidence=confidence,
                    )

        if best.matched:
            logger.info(
                "match_detector MATCH poi_a=%s poi_b=%s score=%.3f confidence=%s",
                best.poi_id_a[:8] if best.poi_id_a else None,
                best.poi_id_b[:8] if best.poi_id_b else None,
                best.match_score, best.confidence,
            )

        return best

    def _compare(self, a: POIRecord, b: POIRecord, frame: np.ndarray) -> float:
        """Pixel agreement score between bbox regions of a and b in the current frame."""
        # Fast path: exact hash match
        if a.pixel_hash == b.pixel_hash:
            return 1.0

        h, w = frame.shape[:2]
        y0a, x0a = max(0, a.bbox[0]), max(0, a.bbox[1])
        y1a, x1a = min(h, a.bbox[2]), min(w, a.bbox[3])
        y0b, x0b = max(0, b.bbox[0]), max(0, b.bbox[1])
        y1b, x1b = min(h, b.bbox[2]), min(w, b.bbox[3])

        crop_a = frame[y0a:y1a, x0a:x1a]
        crop_b = frame[y0b:y1b, x0b:x1b]

        if crop_a.size == 0 or crop_b.size == 0:
            return 0.0

        ha, wa = crop_a.shape[:2]
        hb, wb = crop_b.shape[:2]

        if (ha, wa) != (hb, wb):
            # Only compare if sizes match within 1 cell
            if abs(ha - hb) > 1 or abs(wa - wb) > 1:
                return 0.0
            min_h, min_w = min(ha, hb), min(wa, wb)
            crop_a = crop_a[:min_h, :min_w]
            crop_b = crop_b[:min_h, :min_w]

        agree = int(np.sum(crop_a == crop_b))
        total = crop_a.size
        return float(agree) / max(total, 1)
