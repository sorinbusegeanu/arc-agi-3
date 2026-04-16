from __future__ import annotations

from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import POICandidate


def rank_poi_candidates(
    candidates: tuple[dict[str, Any], ...],
    avatar_bbox: tuple[int, int, int, int] | None,
) -> tuple[tuple[POICandidate, ...], int]:
    max_x = max((int(item["bbox"][2]) for item in candidates), default=0)
    max_y = max((int(item["bbox"][3]) for item in candidates), default=0)
    width = int(max_x + 1)
    height = int(max_y + 1)
    ranked: list[POICandidate] = []
    for index, candidate in enumerate(candidates):
        seen_steps = tuple(candidate.get("seen_step_indices", ()))
        persistence = min(1.0, len(seen_steps) / 3.0)
        source_kind = str(candidate.get("source_kind", ""))
        dual_source_bonus = 1.0 if "+" in source_kind else 0.6
        min_distance = float(candidate.get("min_avatar_distance", 999.0))
        separation = max(0.0, min(1.0, min_distance / 6.0))
        area = int(candidate.get("area", 0))
        area_score = max(0.0, min(1.0, area / 8.0))
        compactness = _compactness(candidate["bbox"], area)
        recurrence = min(1.0, len(candidate.get("seen_step_indices", ())) / 4.0)
        histogram_stability = _histogram_stability(candidate.get("value_histogram", {}))
        overlap_penalty = _bbox_iou(candidate["bbox"], avatar_bbox) if avatar_bbox is not None else 0.0
        border_locked = _is_border_locked_candidate(candidate["bbox"], area, width, height)
        interior_support = _has_interior_support(candidate["bbox"], width, height)
        multi_step_bonus = min(1.0, len(seen_steps) / 4.0)
        border_penalty = 0.85 if border_locked else (0.35 if not interior_support else 0.0)

        score = (
            0.18 * persistence
            + 0.14 * dual_source_bonus
            + 0.13 * separation
            + 0.10 * compactness
            + 0.10 * recurrence
            + 0.08 * area_score
            + 0.10 * histogram_stability
            + 0.20 * (1.0 if interior_support else 0.0)
            + 0.12 * multi_step_bonus
            - 0.20 * overlap_penalty
            - border_penalty
        )
        score = max(0.0, min(1.0, score))
        ambiguity_flags: list[str] = []
        if area <= 1 or len(candidate.get("seen_step_indices", ())) <= 1:
            ambiguity_flags.append("tiny_or_one_frame")
        if overlap_penalty > 0.0:
            ambiguity_flags.append("avatar_overlap")
        if border_locked:
            ambiguity_flags.append("border_locked")
        elif not interior_support:
            ambiguity_flags.append("weak_interior_support")

        ranked.append(
            POICandidate(
                poi_id=str(candidate.get("poi_id", f"poi_{index:03d}")),
                bbox=tuple(candidate["bbox"]),
                center=tuple(candidate["center"]),
                area=area,
                value_histogram=dict(sorted(candidate.get("value_histogram", {}).items())),
                seen_step_indices=tuple(sorted(int(v) for v in candidate.get("seen_step_indices", ()))),
                support_episode_indices=tuple(sorted(int(v) for v in candidate.get("support_episode_indices", ()) or ())),
                source_kind=source_kind,
                near_avatar_steps=tuple(sorted(int(v) for v in candidate.get("near_avatar_steps", ()))),
                min_avatar_distance=min_distance,
                confidence=float(score),
                ambiguity_flags=tuple(ambiguity_flags),
            )
        )

    ranked.sort(key=lambda item: (-item.confidence, -len(item.seen_step_indices), item.bbox, item.poi_id))
    ambiguous = 0
    if len(ranked) > 1 and abs(ranked[0].confidence - ranked[1].confidence) <= 0.06:
        ambiguous = 2
        ranked[0] = _with_flag(ranked[0], "close_score")
        ranked[1] = _with_flag(ranked[1], "close_score")
    return tuple(ranked), ambiguous


def _with_flag(candidate: POICandidate, flag: str) -> POICandidate:
    if flag in set(candidate.ambiguity_flags):
        return candidate
    return POICandidate(
        poi_id=candidate.poi_id,
        bbox=candidate.bbox,
        center=candidate.center,
        area=candidate.area,
        value_histogram=candidate.value_histogram,
        seen_step_indices=candidate.seen_step_indices,
        support_episode_indices=candidate.support_episode_indices,
        source_kind=candidate.source_kind,
        near_avatar_steps=candidate.near_avatar_steps,
        min_avatar_distance=candidate.min_avatar_distance,
        confidence=candidate.confidence,
        ambiguity_flags=tuple(candidate.ambiguity_flags) + (flag,),
    )


def _compactness(bbox: tuple[int, int, int, int], area: int) -> float:
    width = max(1, bbox[2] - bbox[0] + 1)
    height = max(1, bbox[3] - bbox[1] + 1)
    box_area = width * height
    return max(0.0, min(1.0, area / max(box_area, 1)))


def _histogram_stability(histogram: dict[int, int]) -> float:
    if not histogram:
        return 0.0
    total = sum(int(v) for v in histogram.values())
    if total <= 0:
        return 0.0
    dominant = max(int(v) for v in histogram.values())
    return max(0.0, min(1.0, dominant / total))


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int] | None) -> float:
    if right is None:
        return 0.0
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    left_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    right_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    union = max(left_area + right_area - inter, 1)
    return inter / union


def _is_border_locked_candidate(
    bbox: tuple[int, int, int, int],
    area: int,
    width: int,
    height: int,
) -> bool:
    if width <= 0 or height <= 0:
        return False
    thickness = _border_band_thickness(width, height)
    x0, y0, x1, y1 = bbox
    fully_in_band = (
        y1 < thickness
        or x1 < thickness
    )
    if not fully_in_band:
        return False
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    strip_like = bw <= 2 or bh <= 2 or bw >= 3 * bh or bh >= 3 * bw
    tiny = int(area) <= 8
    return bool(tiny or strip_like)


def _has_interior_support(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    if width <= 0 or height <= 0:
        return False
    thickness = _border_band_thickness(width, height)
    x0, y0, _, _ = bbox
    return bool(x0 >= thickness and y0 >= thickness)


def _border_band_thickness(width: int, height: int) -> int:
    return max(1, min(3, min(width, height) // 20 + 1))
