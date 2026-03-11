from __future__ import annotations

from v3_1.analysis.object_extraction import summarize_object_persistence
from v3_1.utils.ids import stable_digest


def detect_pois(step_summaries: list[dict], avatar_tracking: dict) -> list[dict]:
    step_objects = [list(summary.get("objects", [])) for summary in step_summaries]
    persistence = summarize_object_persistence(step_objects)
    avatar_signatures = {track["signature"] for track in avatar_tracking.get("tracks", [])}
    pois: list[dict] = []

    for signature, stats in persistence.items():
        exemplar = None
        for objects in step_objects:
            exemplar = next((row for row in objects if row["signature"] == signature), None)
            if exemplar is not None:
                break
        if exemplar is None:
            continue
        rejection_reasons: list[str] = []
        demotion_reasons: list[str] = []
        utility = 0.0
        novelty = 0.0

        if exemplar["kind"] == "hud_like":
            rejection_reasons.append("hud_like")
        if signature in avatar_signatures:
            demotion_reasons.append("avatar_like")
        if stats["persistence"] < 0.2:
            rejection_reasons.append("low_persistence")
        if exemplar["touches_border"] and exemplar["kind"] != "mobile_candidate":
            demotion_reasons.append("border_touching")
        if exemplar["area"] <= 2:
            demotion_reasons.append("tiny")

        utility += min(0.45, stats["persistence"])
        utility += min(0.25, exemplar["confidence"])
        utility += 0.15 if "candidate_avatar" not in exemplar["type_hints"] else 0.0
        novelty += 0.2 if stats["count"] == 1 else 0.0
        novelty += 0.15 if exemplar["primary_color"] not in (summary.get("background", {}).get("color") for summary in step_summaries) else 0.0
        confidence = max(0.0, utility + novelty - (0.2 * len(demotion_reasons)) - (0.4 * len(rejection_reasons)))

        if rejection_reasons:
            continue

        poi_id = f"poi:{stable_digest({'signature': signature, 'centroid': exemplar['centroid']})}"
        pois.append(
            {
                "entity_id": poi_id,
                "poi_id": poi_id,
                "kind": "poi",
                "signature": signature,
                "centroid": list(exemplar["centroid"]),
                "bbox": dict(exemplar["bbox"]),
                "area": int(exemplar["area"]),
                "primary_color": int(exemplar["primary_color"]),
                "type_hints": list(exemplar["type_hints"]),
                "persistence": float(stats["persistence"]),
                "utility": float(utility),
                "novelty": float(novelty),
                "confidence": min(1.0, confidence),
                "observations": int(stats["count"]),
                "demotion_reasons": list(demotion_reasons),
                "canonical_descriptor": {
                    "signature": signature,
                    "kind": exemplar["kind"],
                    "primary_color": exemplar["primary_color"],
                    "bbox_size": [exemplar["width"], exemplar["height"]],
                },
            }
        )
    pois.sort(key=lambda row: (-float(row["confidence"]), -float(row["utility"]), row["entity_id"]))
    deduped: list[dict] = []
    seen_positions: set[tuple[int, int, int]] = set()
    for poi in pois:
        centroid = poi["centroid"]
        position_key = (poi["primary_color"], int(round(centroid[0])), int(round(centroid[1])))
        if position_key in seen_positions:
            continue
        seen_positions.add(position_key)
        deduped.append(poi)
    return deduped
