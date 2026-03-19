from __future__ import annotations

from math import sqrt

from v3_1.analysis.object_extraction import summarize_object_persistence
from v3_1.utils.ids import stable_digest

EFFECT_NORMALIZER = 50.0
MAP_DIAGONAL = 90.0
MOTION_NORMALIZER = 15.0
NORMAL_MIN_AREA = 3
NORMAL_MIN_EXTENT = 2
NORMAL_MIN_PERSISTENCE = 0.3
MICRO_PROMOTION_MIN_PERSISTENCE = 0.65
MICRO_PROMOTION_MIN_EFFECT = 0.35
CLUSTER_RADIUS = 3.25
LOW_CONFIDENCE_PER_AREA_CAP = 2


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


def _avatar_like_candidate(*, signature: str, exemplar: dict, avatar_signatures: set[str]) -> tuple[bool, dict]:
    type_hints = {str(value or "") for value in list(exemplar.get("type_hints", []) or [])}
    kind = str(exemplar.get("kind") or "")
    area = int(exemplar.get("area", 0) or 0)
    bbox = dict(exemplar.get("bbox", {}) or {})
    bbox_width, bbox_height, _ = _bbox_metrics(bbox)
    direct_avatar_track = signature in avatar_signatures
    compact_avatar_shape = "candidate_avatar" in type_hints and kind == "mobile_candidate" and area <= 12 and bbox_width <= 4 and bbox_height <= 4
    broad_mobile_hint = kind == "mobile_candidate"
    avatar_like = bool(direct_avatar_track or compact_avatar_shape)
    return avatar_like, {
        "direct_avatar_track": direct_avatar_track,
        "compact_avatar_shape": compact_avatar_shape,
        "broad_mobile_hint": broad_mobile_hint,
        "type_hints": sorted(type_hints),
        "kind": kind,
    }


def _bbox_metrics(bbox: dict) -> tuple[int, int, int]:
    bbox_width = int(bbox.get("x2", 0)) - int(bbox.get("x1", 0)) + 1 if bbox else 0
    bbox_height = int(bbox.get("y2", 0)) - int(bbox.get("y1", 0)) + 1 if bbox else 0
    bbox_area = max(0, bbox_width * bbox_height)
    return bbox_width, bbox_height, bbox_area


def _bbox_contains(outer: dict, inner: dict, *, margin: int = 0) -> bool:
    return (
        int(inner.get("x1", 0)) >= int(outer.get("x1", 0)) - margin
        and int(inner.get("y1", 0)) >= int(outer.get("y1", 0)) - margin
        and int(inner.get("x2", 0)) <= int(outer.get("x2", 0)) + margin
        and int(inner.get("y2", 0)) <= int(outer.get("y2", 0)) + margin
    )


def _bbox_overlap_or_adjacent(left: dict, right: dict, *, margin: int = 1) -> bool:
    return not (
        int(left.get("x2", 0)) < int(right.get("x1", 0)) - margin
        or int(right.get("x2", 0)) < int(left.get("x1", 0)) - margin
        or int(left.get("y2", 0)) < int(right.get("y1", 0)) - margin
        or int(right.get("y2", 0)) < int(left.get("y1", 0)) - margin
    )


def _centroid_distance(left: list[float], right: list[float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return sqrt((dx * dx) + (dy * dy))


def _descriptor_similarity(left: dict, right: dict) -> float:
    if not left and not right:
        return 1.0
    palette_left = tuple(left.get("palette", ()) or ())
    palette_right = tuple(right.get("palette", ()) or ())
    size_left = tuple(left.get("bbox_size", ()) or ())
    size_right = tuple(right.get("bbox_size", ()) or ())
    pattern_match = 1.0 if left.get("pattern_id") and left.get("pattern_id") == right.get("pattern_id") else 0.0
    palette_score = 1.0 if palette_left and palette_left == palette_right else 0.0
    size_score = 1.0 if size_left and size_left == size_right else 0.0
    return max(pattern_match, (0.55 * pattern_match) + (0.3 * palette_score) + (0.15 * size_score))


def _poi_bucket(exemplar: dict, *, candidate_effect_score: float, remote_effect_support: int) -> tuple[str, bool]:
    type_hints = {str(value or "") for value in list(exemplar.get("type_hints", []) or [])}
    area = int(exemplar.get("area", 0) or 0)
    bbox_width, bbox_height, _ = _bbox_metrics(dict(exemplar.get("bbox", {}) or {}))
    if area < NORMAL_MIN_AREA or bbox_width < NORMAL_MIN_EXTENT or bbox_height < NORMAL_MIN_EXTENT:
        return "micro_point", False
    if "compact_structure_candidate" in type_hints or "structural_candidate" in type_hints or exemplar.get("kind") == "structure":
        return "structural", True
    if remote_effect_support > 0 or candidate_effect_score >= 0.2:
        return "interactable_object", True
    if "symbol_candidate" in type_hints:
        return "region_trigger", False
    return "debug_only", False


def _micro_promotable(*, stats: dict, candidate_effect_score: float, remote_effect_support: int, interact_attempts: int, click_attempts: int) -> bool:
    return bool(
        float(stats.get("persistence", 0.0) or 0.0) >= MICRO_PROMOTION_MIN_PERSISTENCE
        and (
            candidate_effect_score >= MICRO_PROMOTION_MIN_EFFECT
            or remote_effect_support > 0
            or interact_attempts > 0
            or click_attempts > 0
        )
    )


def _find_parent_region(candidate: dict, raw_candidates: list[dict]) -> dict | None:
    bbox = dict(candidate.get("bbox", {}) or {})
    for parent in sorted(raw_candidates, key=lambda row: (-int(row.get("area", 0) or 0), str(row.get("raw_candidate_id") or ""))):
        if parent is candidate:
            continue
        if int(parent.get("area", 0) or 0) < max(8, int(candidate.get("area", 0) or 0) * 4):
            continue
        parent_bbox = dict(parent.get("bbox", {}) or {})
        if not _bbox_contains(parent_bbox, bbox, margin=1):
            continue
        if str(parent.get("poi_bucket") or "") in {"micro_point", "debug_only"}:
            continue
        if str(parent.get("primary_color")) != str(candidate.get("primary_color")) and not parent.get("pattern_id"):
            continue
        return parent
    return None


def _cluster_candidates(raw_candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    clusters: list[list[dict]] = []
    merge_debug: list[dict] = []
    for candidate in sorted(raw_candidates, key=lambda row: (-float(row.get("confidence", 0.0)), -float(row.get("utility", 0.0)), str(row.get("raw_candidate_id") or ""))):
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            if str(candidate.get("poi_bucket")) != str(anchor.get("poi_bucket")):
                continue
            if str(candidate.get("poi_class")) != str(anchor.get("poi_class")):
                continue
            if _descriptor_similarity(dict(candidate.get("canonical_descriptor", {}) or {}), dict(anchor.get("canonical_descriptor", {}) or {})) < 0.5:
                continue
            if not (
                _bbox_overlap_or_adjacent(dict(candidate.get("bbox", {}) or {}), dict(anchor.get("bbox", {}) or {}), margin=1)
                or _centroid_distance(list(candidate.get("centroid", [0.0, 0.0])), list(anchor.get("centroid", [0.0, 0.0]))) <= CLUSTER_RADIUS
            ):
                continue
            cluster.append(candidate)
            candidate["merge_target_id"] = anchor.get("raw_candidate_id")
            merge_debug.append(
                {
                    "raw_candidate_id": candidate.get("raw_candidate_id"),
                    "merge_target_id": anchor.get("raw_candidate_id"),
                    "reason": "spatial_canonicalization",
                }
            )
            placed = True
            break
        if not placed:
            clusters.append([candidate])
    canonical: list[dict] = []
    for cluster in clusters:
        anchor = dict(cluster[0])
        merged_bbox = {
            "x1": min(int(dict(row.get("bbox", {}) or {}).get("x1", 0)) for row in cluster),
            "y1": min(int(dict(row.get("bbox", {}) or {}).get("y1", 0)) for row in cluster),
            "x2": max(int(dict(row.get("bbox", {}) or {}).get("x2", 0)) for row in cluster),
            "y2": max(int(dict(row.get("bbox", {}) or {}).get("y2", 0)) for row in cluster),
        }
        merged_centroid = [
            sum(float(list(row.get("centroid", [0.0, 0.0]))[0]) for row in cluster) / float(len(cluster)),
            sum(float(list(row.get("centroid", [0.0, 0.0]))[1]) for row in cluster) / float(len(cluster)),
        ]
        merged_observations = sum(int(row.get("observations", 0) or 0) for row in cluster)
        merged_visits = sum(int(row.get("visit_count", row.get("observations", 0)) or 0) for row in cluster)
        anchor["bbox"] = merged_bbox
        anchor["centroid"] = merged_centroid
        anchor["observations"] = merged_observations
        anchor["visit_count"] = merged_visits
        anchor["raw_candidate_ids"] = [str(row.get("raw_candidate_id") or "") for row in cluster]
        anchor["merged_cluster_size"] = len(cluster)
        anchor["exported_canonical_poi_id"] = anchor.get("poi_id")
        canonical.append(anchor)
    return canonical, merge_debug


def detect_pois(step_summaries: list[dict], avatar_tracking: dict, step_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    step_objects = [list(summary.get("objects", [])) for summary in step_summaries]
    persistence = summarize_object_persistence(step_objects)
    avatar_signatures = {track["signature"] for track in avatar_tracking.get("tracks", [])}
    raw_candidates: list[dict] = []
    rejected_candidates: list[dict] = []
    rejection_reason_counts: dict[str, int] = {}
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

    for canonical_track_id, stats in persistence.items():
        exemplar = dict(stats.get("exemplar", {}) or {})
        signature = str(exemplar.get("signature") or stats.get("signature") or "")
        if exemplar is None or not signature:
            continue
        rejection_reasons: list[str] = []
        demotion_reasons: list[str] = []
        utility = 0.0
        novelty = 0.0
        bbox = dict(exemplar.get("bbox", {}) or {})
        bbox_width, bbox_height, bbox_area = _bbox_metrics(bbox)
        bbox_area_ratio = bbox_area / float(grid_area)
        area_ratio = int(exemplar.get("area", 0)) / float(grid_area)
        poi_class = "interactable"
        signature_rows = [dict(row) for row in list(stats.get("object_rows", []) or [])]
        motion_variance, motion_score = _motion_stats(signature_rows)
        raw_candidate_id = f"poi_raw:{stable_digest({'track': canonical_track_id, 'centroid': exemplar['centroid']})}"
        poi_id = f"poi:{stable_digest({'track': canonical_track_id})}"
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

        if movement_attempts > 0:
            movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts)) / EFFECT_NORMALIZER)
        if interact_attempts > 0:
            interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts)) / EFFECT_NORMALIZER)
        if click_attempts > 0:
            click_effect_score = min(1.0, (click_effect_sum / float(click_attempts)) / EFFECT_NORMALIZER)
        if interact_attempts > 0:
            candidate_effect_mode = "interact"
            candidate_effect_score = interact_effect_score
        elif click_attempts > 0:
            candidate_effect_mode = "click_at"
            candidate_effect_score = click_effect_score
        else:
            candidate_effect_score = movement_effect_score

        if exemplar["kind"] == "hud_like":
            rejection_reasons.append("hud_like")
        avatar_like, avatar_like_inputs = _avatar_like_candidate(signature=signature, exemplar=exemplar, avatar_signatures=avatar_signatures)
        if avatar_like:
            demotion_reasons.append("avatar_like")
        raw_signature_persistence = float(stats.get("raw_signature_persistence", 0.0) or 0.0)
        canonical_track_persistence = float(stats.get("canonical_track_persistence", stats.get("persistence", 0.0)) or 0.0)
        canonical_track_count = int(stats.get("canonical_track_count", stats.get("count", 0)) or 0)
        if bbox_area_ratio > 0.12:
            rejection_reasons.append("bbox_too_large")
        if grid_width > 0 and bbox_width > 0.45 * grid_width:
            rejection_reasons.append("bbox_too_wide")
        if grid_height > 0 and bbox_height > 0.45 * grid_height:
            rejection_reasons.append("bbox_too_tall")
        if exemplar["primary_color"] in background_colors and bbox_area_ratio > 0.03:
            rejection_reasons.append("background_dominant")
        if area_ratio > 0.08 and float(stats.get("persistence", 0.0) or 0.0) > 0.6:
            rejection_reasons.append("persistent_floor_structure")
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
            distance_score = 1.0 - min(1.0, distance_from_avatar / MAP_DIAGONAL)
        utility += min(0.45, canonical_track_persistence)
        utility += min(0.25, float(exemplar.get("confidence", 0.0) or 0.0))
        utility += 0.15 if "candidate_avatar" not in exemplar["type_hints"] else 0.0
        utility += 0.2 * distance_score
        utility += 0.25 * motion_score
        novelty += 0.2 if int(stats.get("count", 0) or 0) == 1 else 0.0
        novelty += 0.15 if exemplar["primary_color"] not in background_colors else 0.0
        utility += 0.35 * candidate_effect_score
        confidence = max(0.0, utility + novelty - (0.2 * len(demotion_reasons)))

        poi_bucket, planner_visible = _poi_bucket(exemplar, candidate_effect_score=candidate_effect_score, remote_effect_support=remote_effect_support)
        structural_track = poi_bucket == "structural"
        persistence_threshold = NORMAL_MIN_PERSISTENCE
        if structural_track:
            if canonical_track_count >= 10 or len(list(stats.get("raw_signatures", []) or [])) >= 4:
                persistence_threshold = 0.12
            else:
                persistence_threshold = 0.18
        if canonical_track_persistence < persistence_threshold:
            rejection_reasons.append("low_persistence")
        if avatar_like and poi_bucket != "structural" and poi_class == "mobile" and interact_attempts <= 0 and click_attempts <= 0 and remote_effect_support <= 0:
            rejection_reasons.append("avatar_like_without_interaction_support")
        is_micro = poi_bucket == "micro_point"
        if is_micro and not _micro_promotable(
            stats=stats,
            candidate_effect_score=candidate_effect_score,
            remote_effect_support=remote_effect_support,
            interact_attempts=interact_attempts,
            click_attempts=click_attempts,
        ):
            rejection_reasons.append("tiny_rejected_by_default")
        if not is_micro and (int(exemplar["area"]) < NORMAL_MIN_AREA or bbox_width < NORMAL_MIN_EXTENT or bbox_height < NORMAL_MIN_EXTENT):
            rejection_reasons.append("below_normal_poi_minimums")

        candidate = {
            "raw_candidate_id": raw_candidate_id,
            "source_object_id": str(exemplar.get("object_id") or ""),
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
            "poi_bucket": poi_bucket,
            "planner_visible": bool(planner_visible and not rejection_reasons),
            "persistence": canonical_track_persistence,
            "persistence_count": canonical_track_count,
            "raw_signature_persistence": raw_signature_persistence,
            "canonical_track_persistence": canonical_track_persistence,
            "raw_signature_count": int(stats.get("raw_signature_count", 0) or 0),
            "canonical_track_count": canonical_track_count,
            "canonical_track_id": canonical_track_id,
            "raw_source_ids": list(stats.get("raw_source_ids", []) or []),
            "raw_signatures": list(stats.get("raw_signatures", []) or []),
            "utility": float(utility),
            "novelty": float(novelty),
            "movement_attempts": int(movement_attempts),
            "interact_attempts": int(interact_attempts),
            "click_attempts": int(click_attempts),
            "interaction_support": int(interact_attempts + click_attempts),
            "movement_effect_sum": int(movement_effect_sum),
            "interact_effect_sum": int(interact_effect_sum),
            "click_effect_sum": int(click_effect_sum),
            "movement_effect_score": float(movement_effect_score),
            "interact_effect_score": float(interact_effect_score),
            "click_effect_score": float(click_effect_score),
            "candidate_effect_score": float(candidate_effect_score),
            "candidate_effect_mode": candidate_effect_mode,
            "remote_effect_support": int(remote_effect_support),
            "distance_from_avatar": float(distance_from_avatar),
            "distance_score": float(distance_score),
            "motion_variance": float(motion_variance),
            "motion_score": float(motion_score),
            "confidence": min(1.0, confidence),
            "observations": canonical_track_count,
            "visit_count": canonical_track_count,
            "demotion_reasons": list(demotion_reasons),
            "candidate_avatar": bool("candidate_avatar" in set(list(exemplar.get("type_hints", []) or []))),
            "mobile_candidate": bool(exemplar.get("kind") == "mobile_candidate"),
            "avatar_like_inputs": avatar_like_inputs,
            "avatar_like": bool(avatar_like),
            "canonical_descriptor": {
                "signature": signature,
                "kind": exemplar["kind"],
                "primary_color": exemplar["primary_color"],
                "bbox_size": [exemplar["width"], exemplar["height"]],
                "palette": tuple(exemplar.get("palette", ()) or ()),
                "pattern_id": pattern_id,
            },
            "pattern_id": pattern_id,
            "pattern_descriptor": pattern_descriptor,
            "identity_confidence": identity_confidence,
            "identity_status": identity_status,
            "symbolic_structure_score": symbolic_structure_score,
            "rejection_reason": "",
            "decisive_rejection_reason": "",
            "parent_region_id": None,
            "merge_target_id": None,
            "exported_canonical_poi_id": None,
        }
        raw_candidates.append(candidate)
        if rejection_reasons:
            candidate["rejection_reason"] = ",".join(rejection_reasons)
            candidate["decisive_rejection_reason"] = str(rejection_reasons[0] or "")
            rejected_candidates.append(dict(candidate))
            for reason in rejection_reasons:
                rejection_reason_counts[str(reason)] = rejection_reason_counts.get(str(reason), 0) + 1

    viable_candidates = [row for row in raw_candidates if not row.get("rejection_reason")]
    parent_attached_count = 0
    for candidate in viable_candidates:
        parent = _find_parent_region(candidate, viable_candidates)
        if parent is not None and int(candidate.get("area", 0) or 0) <= 3:
            candidate["parent_region_id"] = str(parent.get("raw_candidate_id") or "")
            candidate["rejection_reason"] = "attached_to_parent_region"
            rejected_candidates.append(dict(candidate))
            rejection_reason_counts["attached_to_parent_region"] = rejection_reason_counts.get("attached_to_parent_region", 0) + 1
            parent_attached_count += 1
    viable_candidates = [row for row in viable_candidates if not row.get("rejection_reason")]

    canonical_candidates, merge_debug = _cluster_candidates(viable_candidates)
    by_area: dict[str, list[dict]] = {}
    for poi in canonical_candidates:
        area_key = str(poi.get("bbox", {}).get("y1", 0) // 8) + ":" + str(poi.get("bbox", {}).get("x1", 0) // 8)
        by_area.setdefault(area_key, []).append(poi)
    final_pois: list[dict] = []
    area_capped_count = 0
    for area_key, rows in by_area.items():
        strong = sorted(rows, key=lambda row: (-float(row.get("planner_visible", False)), -float(row.get("confidence", 0.0)), -float(row.get("utility", 0.0)), str(row.get("poi_id") or "")))
        weak_kept = 0
        for row in strong:
            if row.get("planner_visible"):
                final_pois.append(row)
                continue
            if weak_kept < LOW_CONFIDENCE_PER_AREA_CAP:
                final_pois.append(row)
                weak_kept += 1
                continue
            row["rejection_reason"] = "low_confidence_area_cap"
            rejected_candidates.append(dict(row))
            rejection_reason_counts["low_confidence_area_cap"] = rejection_reason_counts.get("low_confidence_area_cap", 0) + 1
            area_capped_count += 1

    raw_by_id = {str(row.get("raw_candidate_id") or ""): dict(row) for row in raw_candidates}
    for poi in final_pois:
        poi["planner_visible"] = bool(poi.get("planner_visible")) and str(poi.get("poi_bucket") or "") in {"structural", "interactable_object"}
        poi["noise_rejected"] = False
        poi["poi_debug"] = {
            "raw_candidates": [raw_by_id[candidate_id] for candidate_id in list(poi.get("raw_candidate_ids", [])) if candidate_id in raw_by_id],
            "rejected_tiny_candidates": [row for row in rejected_candidates if str(row.get("rejection_reason") or "").startswith("tiny_")],
            "merged_clusters": [row for row in merge_debug if str(row.get("merge_target_id") or "") == str(poi.get("raw_candidate_id") or "")],
            "final_exported_canonical_poi_id": poi.get("poi_id"),
        }
    for row in rejected_candidates:
        row["noise_rejected"] = True

    pois = sorted(
        final_pois,
        key=lambda row: (-float(row.get("planner_visible", False)), -float(row.get("confidence", 0.0)), -float(row.get("utility", 0.0)), str(row.get("entity_id") or "")),
    )
    debug_artifact = {
        "candidate_source_count": len(persistence),
        "raw_candidate_count": len(raw_candidates),
        "viable_candidate_count": len(viable_candidates),
        "canonical_cluster_count": len(canonical_candidates),
        "final_exported_count": len(pois),
        "parent_attached_count": parent_attached_count,
        "area_cap_drop_count": area_capped_count,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "raw_poi_candidates": [dict(row) for row in raw_candidates],
        "rejected_tiny_candidates": [dict(row) for row in rejected_candidates if "tiny" in str(row.get("rejection_reason") or "")],
        "merged_clusters": list(merge_debug),
        "final_exported_canonical_pois": [
            {
                "poi_id": row.get("poi_id"),
                "raw_candidate_ids": list(row.get("raw_candidate_ids", [])),
                "planner_visible": bool(row.get("planner_visible")),
                "poi_bucket": str(row.get("poi_bucket") or ""),
                "confidence": float(row.get("confidence", 0.0)),
                "utility": float(row.get("utility", 0.0)),
                "bbox": dict(row.get("bbox", {}) or {}),
            }
            for row in pois
        ],
        "rejected_candidates": [dict(row) for row in rejected_candidates],
    }
    for poi in pois:
        poi["poi_detection_debug"] = debug_artifact
    return pois, debug_artifact
