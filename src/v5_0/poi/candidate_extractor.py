from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import AvatarCandidate, AvatarIdentificationReport, AvatarSelectedResult, ProbeTransitionRecord


def extract_poi_components(
    transitions: tuple[ProbeTransitionRecord, ...],
    selected_avatar: AvatarSelectedResult,
    avatar_candidates: tuple[AvatarCandidate, ...] | None = None,
    avatar_report: AvatarIdentificationReport | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    avatar_candidates = avatar_candidates if avatar_candidates is not None else tuple(avatar_report.candidates) if avatar_report is not None else ()
    avatar_bbox = selected_avatar.selected_bbox
    avatar_hist = _avatar_histogram(avatar_candidates, selected_avatar.selected_candidate_id)
    avatar_support_steps = _avatar_support_steps(avatar_candidates, selected_avatar.selected_candidate_id)

    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for record in transitions:
        if record.pre_frame is None or record.post_frame is None:
            dropped["missing_frame"] += 1
            continue
        if record.blocked_action:
            dropped["blocked_action"] += 1
            continue

        changed = _changed_cells(record.pre_frame, record.post_frame)
        for cells in _connected_components(changed):
            bbox = _bbox(cells)
            center = _center(cells)
            histogram = Counter(int(record.post_frame[y][x]) for x, y in cells)
            area = len(cells)
            frame_height = min(len(record.pre_frame), len(record.post_frame))
            frame_width = min(len(record.pre_frame[0]), len(record.post_frame[0])) if frame_height else 0
            if _is_border_locked_component(bbox, area, frame_width, frame_height):
                dropped["border_locked_nonworld_candidate"] += 1
                continue

            overlap = _bbox_iou(bbox, avatar_bbox) if avatar_bbox is not None else 0.0
            distance = _center_distance(center, _center_from_bbox(avatar_bbox)) if avatar_bbox is not None else 999.0
            hist_similarity = _histogram_similarity(dict(histogram), avatar_hist)
            on_avatar_step = int(record.step_index) in avatar_support_steps

            reject = False
            # First, reject if this component exactly matches the avatar bbox
            if avatar_bbox is not None and bbox == avatar_bbox:
                dropped["avatar_exact_match"] += 1
                reject = True
            # Strengthen avatar filtering: high histogram similarity OR significant overlap
            elif hist_similarity >= 0.8:  # Very similar histogram → likely avatar
                dropped["avatar_high_histogram_similarity"] += 1
                reject = True
            elif hist_similarity >= 0.5 and distance <= 3.5:
                dropped["avatar_histogram_similarity"] += 1
                reject = True
            elif overlap >= 0.2:
                dropped["avatar_overlap"] += 1
                reject = True
            elif distance <= 1.5 and on_avatar_step:
                dropped["avatar_adjacent_on_avatar_step"] += 1
                reject = True

            if reject:
                continue

            kept.append(
                {
                    "bbox": bbox,
                    "center": center,
                    "area": area,
                    "step_index": int(record.step_index),
                    "value_histogram": dict(sorted(histogram.items())),
                    "source_kind": "changed_component",
                    "near_avatar": distance <= 2.5,
                    "min_avatar_distance": float(distance),
                }
            )
    kept.sort(key=lambda item: (item["step_index"], item["bbox"], item["area"]))
    return tuple(kept), dict(sorted(dropped.items()))


def _avatar_histogram(candidates: tuple[AvatarCandidate, ...], selected_id: str | None) -> dict[int, int]:
    if selected_id is None:
        return {}
    for candidate in candidates:
        if candidate.candidate_id == selected_id:
            return dict(candidate.value_histogram_post)
    return {}


def _avatar_support_steps(candidates: tuple[AvatarCandidate, ...], selected_id: str | None) -> set[int]:
    if selected_id is None:
        return set()
    for candidate in candidates:
        if candidate.candidate_id == selected_id:
            return set(int(v) for v in candidate.support_step_indices)
    return set()


def _changed_cells(pre: tuple[tuple[int, ...], ...], post: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    changed: set[tuple[int, int]] = set()
    h = min(len(pre), len(post))
    for y in range(h):
        w = min(len(pre[y]), len(post[y]))
        for x in range(w):
            if int(pre[y][x]) != int(post[y][x]):
                changed.add((x, y))
    return changed


def _connected_components(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    remaining = set(cells)
    out: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in remaining:
                    remaining.remove(n)
                    comp.add(n)
                    stack.append(n)
        out.append(tuple(sorted(comp)))
    out.sort(key=lambda comp: (_bbox(comp), len(comp)))
    return tuple(out)


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _center_from_bbox(bbox: tuple[int, int, int, int] | None) -> tuple[float, float]:
    if bbox is None:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


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


def _histogram_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    total = sum(max(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    if total <= 0:
        return 0.0
    return float(overlap / total)


def _is_border_locked_component(
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
        or y0 >= max(0, height - thickness)
        or x1 < thickness
        or x0 >= max(0, width - thickness)
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
