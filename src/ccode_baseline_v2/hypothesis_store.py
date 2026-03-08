"""hypothesis_store.py — Module 4.

Versioned, persistent POI map. Updated after each analysis cycle.
Bug 3 fix: identity_key-based stable merge prevents POI inflation across versions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from .structs import POIRecord, ConsequenceResult
from .config import (
    CONFIDENCE_NEW, CONFIDENCE_BIG, CONFIDENCE_NONE_DELTA,
    STALE_VERSIONS, STALE_VERSIONS_UNVISITED,
)


# ── Confidence update rules ──────────────────────────────────────────────────

def _update_confidence(poi: POIRecord, result: ConsequenceResult) -> float:
    if result.label == "BIG_CHANGE":
        return CONFIDENCE_BIG
    if result.label == "SMALL_CHANGE":
        return min(poi.confidence + 0.2, 1.0)
    if result.label == "NO_CHANGE":
        return max(poi.confidence + CONFIDENCE_NONE_DELTA, 0.0)
    # GAME_WON, LEVEL_CHANGE, UNREACHABLE → no change
    return poi.confidence


# ── HypothesisStore ──────────────────────────────────────────────────────────

class HypothesisStore:
    """Versioned map of all known POIs. Updated after each analysis cycle."""

    def __init__(self):
        self._pois: Dict[str, POIRecord] = {}
        self._by_identity: Dict[str, POIRecord] = {}   # identity_key → POIRecord
        self.version: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, new_pois: List[POIRecord], version: int) -> Dict[str, int]:
        """Merge new_pois into store using stable identity_key.

        Returns {"merged": n_updated, "new": n_created}.
        Preserves confidence + visited on existing POIs.
        """
        self.version = version
        seen_keys: set = set()
        n_merged = 0
        n_new = 0

        for new_poi in new_pois:
            key = new_poi.identity_key
            seen_keys.add(key)

            existing = self._by_identity.get(key)
            if existing is not None:
                # Update geometry + tag; preserve confidence + visited
                existing.bbox = new_poi.bbox
                existing.color_signature = new_poi.color_signature
                if new_poi.tag != "UNKNOWN":
                    if existing.tag != "SELF" and new_poi.tag == "SELF":
                        # Tag changed to SELF — recompute identity key with SELF-scheme
                        old_key = existing.identity_key
                        area = (existing.bbox[2] - existing.bbox[0] + 1) * (existing.bbox[3] - existing.bbox[1] + 1)
                        area_bin = area // 4
                        new_key = hashlib.md5(
                            f"SELF:{sorted(existing.color_signature)}:area{area_bin}".encode()
                        ).hexdigest()[:12]
                        if new_key != old_key:
                            self._by_identity.pop(old_key, None)
                            self._by_identity[new_key] = existing
                            seen_keys.discard(old_key)
                            seen_keys.add(new_key)
                        existing.identity_key = new_key
                    existing.tag = new_poi.tag
                existing.reachable = new_poi.reachable or existing.reachable
                existing.version = version
                existing.depriority = False
                n_merged += 1
            else:
                new_poi.confidence = float(new_poi.confidence or CONFIDENCE_NEW)
                new_poi.version = version
                new_poi.depriority = False
                self._pois[new_poi.poi_id] = new_poi
                self._by_identity[key] = new_poi
                n_new += 1

        # Deprioritise absent POIs — unvisited get more time before deprioritisation
        for poi in self._pois.values():
            if poi.identity_key in seen_keys:
                continue
            versions_absent = version - poi.version
            if poi.visited:
                if versions_absent >= STALE_VERSIONS:
                    poi.depriority = True
            else:
                if versions_absent >= STALE_VERSIONS_UNVISITED:
                    poi.depriority = True

        return {"merged": n_merged, "new": n_new}

    def get_all(self) -> List[POIRecord]:
        """Return all POIs in the store."""
        return list(self._pois.values())

    def get_targets(self) -> List[POIRecord]:
        """Unvisited, reachable, non-SELF, non-HUD POIs, sorted by confidence DESC."""
        result = [
            poi for poi in self._pois.values()
            if not poi.visited
            and poi.tag not in ("SELF", "HUD")
            and poi.reachable
            and not poi.depriority
        ]
        result.sort(key=lambda p: p.confidence, reverse=True)
        return result

    def record_consequence(self, poi_id: str, result: ConsequenceResult) -> None:
        """Update confidence + consequence label after visiting a POI."""
        if poi_id not in self._pois:
            return
        poi = self._pois[poi_id]
        poi.confidence = _update_confidence(poi, result)
        poi.consequence = result.label
        poi.visited = True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "version": self.version,
            "pois": [
                {
                    "poi_id":          poi.poi_id,
                    "bbox":            list(poi.bbox),
                    "color_signature": poi.color_signature,
                    "tag":             poi.tag,
                    "reachable":       poi.reachable,
                    "visited":         poi.visited,
                    "consequence":     poi.consequence,
                    "confidence":      poi.confidence,
                    "version":         poi.version,
                    "identity_key":    poi.identity_key,
                    "depriority":      poi.depriority,
                }
                for poi in self._pois.values()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = int(data.get("version", 0))
        self._pois = {}
        self._by_identity = {}
        for rec in data.get("pois", []):
            poi = POIRecord(
                poi_id=rec["poi_id"],
                bbox=tuple(rec["bbox"]),
                color_signature=rec["color_signature"],
                tag=rec["tag"],
                reachable=rec["reachable"],
                visited=rec["visited"],
                consequence=rec.get("consequence"),
                confidence=float(rec["confidence"]),
                version=int(rec["version"]),
                identity_key=rec.get("identity_key", ""),
                depriority=bool(rec.get("depriority", False)),
            )
            self._pois[poi.poi_id] = poi
            if poi.identity_key:
                self._by_identity[poi.identity_key] = poi
