from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any


def merge_poi_candidates(
    changed_candidates: tuple[dict[str, Any], ...],
    static_candidates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    for candidate in tuple(changed_candidates) + tuple(static_candidates):
        matched = _find_match(merged, candidate)
        if matched is None:
            merged.append(
                {
                    "bbox": candidate["bbox"],
                    "center": candidate["center"],
                    "area": int(candidate.get("area", 0)),
                    "value_histogram": dict(candidate.get("value_histogram", {})),
                    "seen_step_indices": set(candidate.get("seen_step_indices", (candidate.get("step_index"),))),
                    "source_kinds": {str(candidate.get("source_kind", "unknown"))},
                    "near_avatar_steps": set((candidate.get("step_index"),) if candidate.get("near_avatar", False) else ()),
                    "min_avatar_distance": float(candidate.get("min_avatar_distance", 999.0)),
                }
            )
            continue

        matched["bbox"] = _merge_bbox(matched["bbox"], candidate["bbox"])
        matched["center"] = _center_from_bbox(matched["bbox"])
        matched["area"] = max(int(matched["area"]), int(candidate.get("area", 0)))
        matched["seen_step_indices"].update(candidate.get("seen_step_indices", (candidate.get("step_index"),)))
        matched["source_kinds"].add(str(candidate.get("source_kind", "unknown")))
        if candidate.get("near_avatar", False):
            step_index = candidate.get("step_index")
            if step_index is not None:
                matched["near_avatar_steps"].add(int(step_index))
        matched["min_avatar_distance"] = min(float(matched["min_avatar_distance"]), float(candidate.get("min_avatar_distance", 999.0)))

        merged_hist = Counter(matched["value_histogram"])
        merged_hist.update(candidate.get("value_histogram", {}))
        matched["value_histogram"] = dict(sorted(merged_hist.items()))

    out: list[dict[str, Any]] = []
    max_x = max((item["bbox"][2] for item in merged), default=0)
    max_y = max((item["bbox"][3] for item in merged), default=0)
    width = int(max_x + 1)
    height = int(max_y + 1)
    for index, item in enumerate(merged):
        if _is_border_locked_bbox(item["bbox"], width, height, int(item.get("area", 0))):
            continue
        out.append(
            {
                "poi_id": f"poi_{index:03d}",
                "bbox": item["bbox"],
                "center": item["center"],
                "area": int(item["area"]),
                "value_histogram": dict(item["value_histogram"]),
                "seen_step_indices": tuple(sorted(int(v) for v in item["seen_step_indices"] if v is not None)),
                "source_kind": "+".join(sorted(item["source_kinds"])),
                "near_avatar_steps": tuple(sorted(item["near_avatar_steps"])),
                "min_avatar_distance": float(item["min_avatar_distance"]),
            }
        )
    out.sort(key=lambda item: (item["bbox"], -item["area"], item["poi_id"]))
    return tuple(out)


def _find_match(existing: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    best = None
    best_score = -1.0
    for item in existing:
        iou = _bbox_iou(item["bbox"], candidate["bbox"])
        dist = _distance(item["center"], candidate["center"])
        hist_sim = _hist_sim(item["value_histogram"], dict(candidate.get("value_histogram", {})))
        score = 0.45 * iou + 0.25 * max(0.0, 1.0 - dist / 5.0) + 0.30 * hist_sim
        if score > best_score:
            best = item
            best_score = score
    if best is not None and best_score >= 0.55:
        return best
    return None


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
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


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _hist_sim(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return float(overlap / max(total, 1))


def _merge_bbox(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _is_border_locked_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    area: int,
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


def _border_band_thickness(width: int, height: int) -> int:
    return max(1, min(3, min(width, height) // 20 + 1))
