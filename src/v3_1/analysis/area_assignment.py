from __future__ import annotations

from v3_1.utils.ids import stable_digest


def _area_features(summary: dict) -> dict:
    stable_signatures = tuple(sorted(obj["signature"] for obj in summary.get("objects", []) if obj["kind"] != "hud_like"))
    return {
        "width": int(summary.get("width", 0)),
        "height": int(summary.get("height", 0)),
        "background_color": int(summary.get("background", {}).get("color", 0)),
        "palette": tuple(summary.get("palette", [])),
        "stable_signatures": stable_signatures,
        "state_hash": str(summary.get("state_identity", {}).get("state_hash", "")),
    }


def canonical_area_signature(summary: dict) -> str:
    return stable_digest(_area_features(summary))


def _area_match_score(lhs: dict, rhs: dict) -> float:
    lhs_palette = set(lhs["palette"])
    rhs_palette = set(rhs["palette"])
    palette_overlap = len(lhs_palette & rhs_palette) / float(max(1, len(lhs_palette | rhs_palette)))
    object_overlap = len(set(lhs["stable_signatures"]) & set(rhs["stable_signatures"])) / float(max(1, len(set(lhs["stable_signatures"]) | set(rhs["stable_signatures"]))))
    score = 0.0
    if lhs["state_hash"] and lhs["state_hash"] == rhs["state_hash"]:
        score += 0.55
    if lhs["background_color"] == rhs["background_color"]:
        score += 0.15
    score += 0.15 * palette_overlap
    score += 0.15 * object_overlap
    return score


def assign_area(summary: dict, known_areas: list[dict] | None = None) -> dict:
    features = _area_features(summary)
    area_signature = canonical_area_signature(summary)
    best = None
    best_score = -1.0
    for area in known_areas or []:
        score = _area_match_score(features, area["features"])
        if score > best_score:
            best = area
            best_score = score
    if best is not None and best_score >= 0.65:
        return {
            "area_id": best["area_id"],
            "area_signature": best["area_signature"],
            "width": features["width"],
            "height": features["height"],
            "palette": list(features["palette"]),
            "background_color": features["background_color"],
            "state_hash": features["state_hash"],
            "visit_count": 1,
            "features": features,
            "match_score": best_score,
        }
    return {
        "area_id": f"area:{area_signature}",
        "area_signature": area_signature,
        "width": features["width"],
        "height": features["height"],
        "palette": list(features["palette"]),
        "background_color": features["background_color"],
        "state_hash": features["state_hash"],
        "visit_count": 1,
        "features": features,
        "match_score": 1.0,
    }
