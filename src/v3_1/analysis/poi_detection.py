from __future__ import annotations

from math import sqrt

from v3_1.analysis.object_extraction import summarize_object_persistence
from v3_1.utils.ids import stable_digest

EFFECT_NORMALIZER = 50.0
MAP_DIAGONAL = 90.0
MOTION_NORMALIZER = 15.0


def _signature_objects(step_objects: list[list[dict]], signature: str) -> list[dict]:
    rows = []
    for objects in step_objects:
        for obj in objects:
            if obj.get("signature") == signature:
                rows.append(obj)
    return rows


def _motion_stats(signature_rows: list[dict]) -> tuple[float, float]:
    centroids = [row.get("centroid") for row in signature_rows if isinstance(row.get("centroid"), (list, tuple)) and len(row.get("centroid")) == 2]
    if not centroids:
        return 0.0, 0.0
    xs = [float(row[0]) for row in centroids]
    ys = [float(row[1]) for row in centroids]
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    variance_x = sum((value - mean_x) ** 2 for value in xs) / float(len(xs))
    variance_y = sum((value - mean_y) ** 2 for value in ys) / float(len(ys))
    motion_variance = variance_x + variance_y
    motion_score = min(1.0, motion_variance / MOTION_NORMALIZER)
    return motion_variance, motion_score


def _avatar_like_candidate(*, signature: str, exemplar: dict, avatar_signatures: set[str]) -> bool:
    type_hints = {str(value or "") for value in list(exemplar.get("type_hints", []) or [])}
    kind = str(exemplar.get("kind") or "")
    return (
        signature in avatar_signatures
        or "candidate_avatar" in type_hints
        or kind == "mobile_candidate"
    )


def detect_pois(step_summaries: list[dict], avatar_tracking: dict, step_rows: list[dict] | None = None) -> list[dict]:
    step_objects = [list(summary.get("objects", [])) for summary in step_summaries]
    persistence = summarize_object_persistence(step_objects)
    avatar_signatures = {track["signature"] for track in avatar_tracking.get("tracks", [])}
    pois: list[dict] = []
    step_rows = list(step_rows or [])
    first_summary = step_summaries[0] if step_summaries else {}
    grid_width = int(first_summary.get("width", 0) or 0)
    grid_height = int(first_summary.get("height", 0) or 0)
    grid_area = max(1, grid_width * grid_height)
    background_colors = {summary.get("background", {}).get("color") for summary in step_summaries}
    avatar_cell = None
    if step_rows:
        first_avatar_cell = step_rows[0].get("avatar_cell")
        if isinstance(first_avatar_cell, (list, tuple)) and len(first_avatar_cell) == 2:
            avatar_cell = [float(first_avatar_cell[0]), float(first_avatar_cell[1])]

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
        bbox = dict(exemplar.get("bbox", {}) or {})
        bbox_width = int(bbox.get("x2", 0)) - int(bbox.get("x1", 0)) + 1 if bbox else 0
        bbox_height = int(bbox.get("y2", 0)) - int(bbox.get("y1", 0)) + 1 if bbox else 0
        bbox_area = max(0, bbox_width * bbox_height)
        bbox_area_ratio = bbox_area / float(grid_area)
        area_ratio = int(exemplar.get("area", 0)) / float(grid_area)
        poi_class = "interactive"
        signature_rows = _signature_objects(step_objects, signature)
        motion_variance, motion_score = _motion_stats(signature_rows)
        poi_id = f"poi:{stable_digest({'signature': signature, 'centroid': exemplar['centroid']})}"
        movement_attempts = 0
        interact_attempts = 0
        click_attempts = 0
        movement_effect_sum = 0
        interact_effect_sum = 0
        click_effect_sum = 0
        movement_effect_score = 0.0
        interact_effect_score = 0.0
        click_effect_score = 0.0
        candidate_effect_score = 0.0
        candidate_effect_mode = "move"
        centroid = list(exemplar.get("centroid", [0.0, 0.0]))
        distance_from_avatar = 0.0
        distance_score = 0.0
        remote_effect_support = 0

        for row in step_rows:
            if row.get("target_entity_id") != poi_id:
                continue
            action_family = str(row.get("action_family") or "unknown")
            changed_cells = int(row.get("changed_cells", 0) or 0)
            telemetry = dict(row.get("telemetry", {}) or {})
            if action_family == "move":
                movement_attempts += 1
                movement_effect_sum += changed_cells
            elif action_family == "interact":
                interact_attempts += 1
                interact_effect_sum += changed_cells
            elif action_family == "click_at":
                click_attempts += 1
                click_effect_sum += changed_cells
            if isinstance(telemetry.get("effect_region"), dict) and telemetry.get("effect_region"):
                remote_effect_support += 1

        if exemplar["kind"] == "hud_like":
            rejection_reasons.append("hud_like")
        avatar_like = _avatar_like_candidate(signature=signature, exemplar=exemplar, avatar_signatures=avatar_signatures)
        if avatar_like:
            demotion_reasons.append("avatar_like")
        if stats["persistence"] < 0.2:
            rejection_reasons.append("low_persistence")
        if bbox_area_ratio > 0.12:
            rejection_reasons.append("bbox_too_large")
        if grid_width > 0 and bbox_width > 0.45 * grid_width:
            rejection_reasons.append("bbox_too_wide")
        if grid_height > 0 and bbox_height > 0.45 * grid_height:
            rejection_reasons.append("bbox_too_tall")
        if exemplar["primary_color"] in background_colors and bbox_area_ratio > 0.03:
            rejection_reasons.append("background_dominant")
        if area_ratio > 0.08 and stats["persistence"] > 0.6:
            rejection_reasons.append("persistent_floor_structure")
        if exemplar["area"] < 1 or exemplar["area"] > 25:
            rejection_reasons.append("invalid_poi_size")
        if exemplar["touches_border"] and exemplar["kind"] != "mobile_candidate":
            demotion_reasons.append("border_touching")
        if exemplar["area"] <= 2:
            demotion_reasons.append("tiny")
        if avatar_like and interact_attempts <= 0 and click_attempts <= 0 and remote_effect_support <= 0:
            rejection_reasons.append("avatar_like_without_interaction_support")
        if exemplar["kind"] == "structure":
            poi_class = "structure"
        elif exemplar["kind"] == "mobile_candidate":
            poi_class = "mobile"
        pattern_id = exemplar.get("pattern_id")
        pattern_descriptor = dict(exemplar.get("pattern_descriptor", {}) or {})
        symbolic_structure_score = float(exemplar.get("symbolic_structure_score", 0.0) or 0.0)
        identity_confidence = float(exemplar.get("identity_confidence", 0.0) or 0.0)
        identity_status = str(exemplar.get("identity_status") or "new_entity")

        if pattern_id:
            utility += 0.12
        utility += 0.25 * symbolic_structure_score
        if identity_status == "match_existing":
            utility += 0.12
        if identity_status == "ambiguous_match":
            novelty += 0.04
        utility += 0.18 * identity_confidence
        if float(pattern_descriptor.get("symmetry_score", 0.0) or 0.0) >= 0.5:
            utility += 0.04
        if "compact_structure_candidate" in exemplar["type_hints"]:
            utility += 0.05

        if poi_class != "mobile" and pattern_id and identity_confidence >= 0.55:
            poi_class = "structure"

        if avatar_cell is not None and len(centroid) == 2:
            dx = float(centroid[0]) - float(avatar_cell[0])
            dy = float(centroid[1]) - float(avatar_cell[1])
            distance_from_avatar = sqrt((dx * dx) + (dy * dy))
            distance_norm = distance_from_avatar / MAP_DIAGONAL
            distance_score = 1.0 - min(1.0, distance_norm)

        utility += min(0.45, stats["persistence"])
        utility += min(0.25, exemplar["confidence"])
        utility += 0.15 if "candidate_avatar" not in exemplar["type_hints"] else 0.0
        utility += 0.2 * distance_score
        utility += 0.25 * motion_score
        novelty += 0.2 if stats["count"] == 1 else 0.0
        novelty += 0.15 if exemplar["primary_color"] not in background_colors else 0.0

        if rejection_reasons:
            continue

        if movement_attempts > 0:
            movement_effect_score = movement_effect_sum / float(movement_attempts)
        if interact_attempts > 0:
            interact_effect_score = interact_effect_sum / float(interact_attempts)
        if click_attempts > 0:
            click_effect_score = click_effect_sum / float(click_attempts)
        movement_effect_score = min(1.0, movement_effect_score / EFFECT_NORMALIZER)
        interact_effect_score = min(1.0, interact_effect_score / EFFECT_NORMALIZER)
        click_effect_score = min(1.0, click_effect_score / EFFECT_NORMALIZER)
        if interact_attempts > 0:
            candidate_effect_mode = "interact"
            candidate_effect_score = interact_effect_score
        elif click_attempts > 0:
            candidate_effect_mode = "click_at"
            candidate_effect_score = click_effect_score
        else:
            candidate_effect_mode = "move"
            candidate_effect_score = movement_effect_score
        utility += 0.35 * candidate_effect_score
        confidence = max(0.0, utility + novelty - (0.2 * len(demotion_reasons)) - (0.4 * len(rejection_reasons)))
        pois.append(
            {
                "entity_id": poi_id,
                "poi_id": poi_id,
                "kind": "poi",
                "signature": signature,
                "centroid": list(exemplar["centroid"]),
                "bbox": dict(exemplar["bbox"]),
                "area": int(exemplar["area"]),
                "bbox_area": bbox_area,
                "bbox_area_ratio": bbox_area_ratio,
                "primary_color": int(exemplar["primary_color"]),
                "type_hints": list(exemplar["type_hints"]),
                "poi_class": poi_class,
                "persistence": float(stats["persistence"]),
                "utility": float(utility),
                "novelty": float(novelty),
                "movement_attempts": int(movement_attempts),
                "interact_attempts": int(interact_attempts),
                "click_attempts": int(click_attempts),
                "movement_effect_sum": int(movement_effect_sum),
                "interact_effect_sum": int(interact_effect_sum),
                "click_effect_sum": int(click_effect_sum),
                "movement_effect_score": float(movement_effect_score),
                "interact_effect_score": float(interact_effect_score),
                "click_effect_score": float(click_effect_score),
                "candidate_effect_score": float(candidate_effect_score),
                "candidate_effect_mode": candidate_effect_mode,
                "distance_from_avatar": float(distance_from_avatar),
                "distance_score": float(distance_score),
                "motion_variance": float(motion_variance),
                "motion_score": float(motion_score),
                "confidence": min(1.0, confidence),
                "observations": int(stats["count"]),
                "demotion_reasons": list(demotion_reasons),
                "canonical_descriptor": {
                    "signature": signature,
                    "kind": exemplar["kind"],
                    "primary_color": exemplar["primary_color"],
                    "bbox_size": [exemplar["width"], exemplar["height"]],
                },
                "pattern_id": pattern_id,
                "pattern_descriptor": pattern_descriptor,
                "identity_confidence": identity_confidence,
                "identity_status": identity_status,
                "symbolic_structure_score": symbolic_structure_score,
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
