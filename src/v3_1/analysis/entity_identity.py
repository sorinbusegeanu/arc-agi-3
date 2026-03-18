from __future__ import annotations

from typing import Any


IDENTITY_OUTCOMES = {
    "match_existing",
    "new_entity",
    "ambiguous_match",
    "split_candidate",
    "merge_candidate",
}


def _bbox_center(bbox: dict[str, Any] | None) -> tuple[float, float]:
    row = dict(bbox or {})
    return (
        (float(row.get("x1", 0) or 0) + float(row.get("x2", 0) or 0)) / 2.0,
        (float(row.get("y1", 0) or 0) + float(row.get("y2", 0) or 0)) / 2.0,
    )


def _size_similarity(current: dict[str, Any], prior: dict[str, Any]) -> float:
    current_area = max(1.0, float(current.get("area", 0) or 0))
    prior_area = max(1.0, float(prior.get("area", 0) or 0))
    return max(0.0, 1.0 - (abs(current_area - prior_area) / max(current_area, prior_area)))


def _structure_like(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or "")
    hints = {str(value) for value in list(row.get("type_hints", []) or []) if value}
    return bool(
        kind in {"world_object", "structure"}
        and "candidate_avatar" not in hints
        and "mobile_candidate" not in hints
    )


def _location_similarity(current: dict[str, Any], prior: dict[str, Any]) -> float:
    cx, cy = _bbox_center(current.get("bbox"))
    px, py = _bbox_center(prior.get("bbox"))
    distance = abs(cx - px) + abs(cy - py)
    distance_budget = 24.0 if _structure_like(current) and _structure_like(prior) else 16.0
    return max(0.0, 1.0 - min(1.0, distance / distance_budget))


def _appearance_similarity(current: dict[str, Any], prior: dict[str, Any]) -> float:
    score = 0.0
    if str(current.get("signature") or "") and str(current.get("signature") or "") == str(prior.get("signature") or ""):
        score += 0.45
    if int(current.get("primary_color", -1)) == int(prior.get("primary_color", -2)):
        score += 0.2
    if str(current.get("kind") or "") == str(prior.get("kind") or ""):
        score += 0.15
    current_descriptor = dict(current.get("stable_descriptor", {}) or {})
    prior_descriptor = dict(prior.get("stable_descriptor", {}) or {})
    if current_descriptor and prior_descriptor and current_descriptor == prior_descriptor:
        score += 0.15
    if str(current.get("pattern_id") or "") and str(current.get("pattern_id") or "") == str(prior.get("pattern_id") or ""):
        score += 0.2
    if (
        current_descriptor
        and prior_descriptor
        and current_descriptor.get("kind") == prior_descriptor.get("kind")
        and current_descriptor.get("bbox_size") == prior_descriptor.get("bbox_size")
    ):
        score += 0.1
    return min(1.0, score)


def identity_score(current: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    temporal_continuity = 1.0 if prior else 0.0
    motion_consistency = _location_similarity(current, prior)
    appearance_features = _appearance_similarity(current, prior)
    relative_location = motion_consistency
    behavior_continuity = 1.0 if str(current.get("kind") or "") == str(prior.get("kind") or "") else 0.0
    size_shape_continuity = _size_similarity(current, prior)
    type_hint_overlap = 0.0
    current_hints = set(str(value) for value in list(current.get("type_hints", []) or []) if value)
    prior_hints = set(str(value) for value in list(prior.get("type_hints", []) or []) if value)
    if current_hints or prior_hints:
        type_hint_overlap = float(len(current_hints & prior_hints)) / float(max(1, len(current_hints | prior_hints)))
    if _structure_like(current) and _structure_like(prior):
        score = (
            (0.18 * temporal_continuity)
            + (0.14 * motion_consistency)
            + (0.32 * appearance_features)
            + (0.10 * relative_location)
            + (0.08 * behavior_continuity)
            + (0.14 * size_shape_continuity)
            + (0.04 * type_hint_overlap)
        )
    else:
        score = (
            (0.22 * temporal_continuity)
            + (0.18 * motion_consistency)
            + (0.24 * appearance_features)
            + (0.14 * relative_location)
            + (0.08 * behavior_continuity)
            + (0.14 * size_shape_continuity)
            + (0.06 * type_hint_overlap)
        )
    return {
        "score": min(1.0, max(0.0, score)),
        "temporal_continuity": temporal_continuity,
        "motion_consistency": motion_consistency,
        "appearance_features": appearance_features,
        "relative_location": relative_location,
        "behavior_continuity": behavior_continuity,
        "size_shape_continuity": size_shape_continuity,
        "type_hint_overlap": type_hint_overlap,
    }


def assign_identity(current: dict[str, Any], priors: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for prior in list(priors or []):
        metrics = identity_score(current, dict(prior or {}))
        scored.append(
            {
                "prior_id": str(prior.get("object_id") or prior.get("entity_id") or prior.get("signature") or ""),
                **metrics,
            }
        )
    scored.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("prior_id") or "")))
    top = scored[0] if scored else None
    second = scored[1] if len(scored) > 1 else None
    min_match_threshold = 0.34 if _structure_like(current) else 0.4
    if top is None or float(top.get("score", 0.0) or 0.0) < min_match_threshold:
        return {
            "identity_status": "new_entity",
            "identity_confidence": float(top.get("score", 0.0) or 0.0) if top else 0.0,
            "identity_candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "matched_prior_id": None,
            "identity_metrics": top or {},
        }
    ambiguity_gap = 0.06 if _structure_like(current) and float(top.get("appearance_features", 0.0) or 0.0) >= 0.55 else 0.1
    if second is not None and abs(float(top.get("score", 0.0) or 0.0) - float(second.get("score", 0.0) or 0.0)) < ambiguity_gap:
        return {
            "identity_status": "ambiguous_match",
            "identity_confidence": float(top.get("score", 0.0) or 0.0),
            "identity_candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "matched_prior_id": None,
            "identity_metrics": top,
        }
    if float(top.get("score", 0.0) or 0.0) < (0.5 if _structure_like(current) else 0.58) and second is not None:
        return {
            "identity_status": "ambiguous_match",
            "identity_confidence": float(top.get("score", 0.0) or 0.0),
            "identity_candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "candidate_prior_ids": [row["prior_id"] for row in scored[:3] if row.get("prior_id")],
            "matched_prior_id": None,
            "identity_metrics": top,
        }
    return {
        "identity_status": "match_existing",
        "identity_confidence": float(top.get("score", 0.0) or 0.0),
        "identity_candidate_prior_ids": [top["prior_id"]] + [row["prior_id"] for row in scored[1:3] if row.get("prior_id")],
        "candidate_prior_ids": [top["prior_id"]] + [row["prior_id"] for row in scored[1:3] if row.get("prior_id")],
        "matched_prior_id": top["prior_id"],
        "identity_metrics": top,
    }
