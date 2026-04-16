from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import AvatarSelectedResult, ProbeTransitionRecord


def build_static_object_inventory(
    transitions: tuple[ProbeTransitionRecord, ...],
    selected_avatar: AvatarSelectedResult,
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    frames: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    for frame_index, record in enumerate(transitions):
        if record.pre_frame is not None:
            frames.append((int(frame_index * 2), record.pre_frame))
        if record.post_frame is not None:
            frames.append((int(frame_index * 2 + 1), record.post_frame))

    avatar_bbox = selected_avatar.selected_bbox
    clusters: list[dict[str, Any]] = []
    dropped = Counter()

    for step_index, frame in frames:
        if not frame:
            continue
        background = _dominant_value(frame)
        foreground = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
        for component in _connected_components(foreground):
            bbox = _bbox(component)
            area = len(component)
            if _bbox_iou(bbox, avatar_bbox) >= 0.2:
                dropped["avatar_overlap"] += 1
                continue
            if _is_border_locked_component(bbox, area, len(frame[0]), len(frame)):
                dropped["border_locked_static_candidate"] += 1
                continue
            hist = Counter(int(frame[y][x]) for x, y in component)
            center = _center(component)
            matched = _find_matching_cluster(clusters, bbox, center, hist)
            if matched is None:
                clusters.append(
                    {
                        "bbox": bbox,
                        "center": center,
                        "area": area,
                        "value_histogram": dict(sorted(hist.items())),
                        "seen_step_indices": {int(step_index)},
                        "source_kind": "static_inventory",
                    }
                )
            else:
                matched["seen_step_indices"].add(int(step_index))
                matched["bbox"] = _merge_bbox(matched["bbox"], bbox)
                matched["center"] = _center_from_bbox(matched["bbox"])
                matched["area"] = max(int(matched["area"]), area)
                merged_hist = Counter(matched["value_histogram"])
                merged_hist.update(hist)
                matched["value_histogram"] = dict(sorted(merged_hist.items()))

    items: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster["seen_step_indices"]) < 2:
            dropped["unstable_single_frame"] += 1
            continue
        items.append(
            {
                "bbox": cluster["bbox"],
                "center": cluster["center"],
                "area": cluster["area"],
                "value_histogram": dict(cluster["value_histogram"]),
                "seen_step_indices": tuple(sorted(cluster["seen_step_indices"])),
                "source_kind": "static_inventory",
            }
        )
    items.sort(key=lambda item: (item["bbox"], -len(item["seen_step_indices"]), -item["area"]))
    return tuple(items), dict(sorted(dropped.items()))


def _find_matching_cluster(clusters, bbox, center, hist):
    best = None
    best_score = -1.0
    for cluster in clusters:
        iou = _bbox_iou(bbox, cluster["bbox"])
        dist = _distance(center, cluster["center"])
        sim = _hist_sim(dict(hist), cluster["value_histogram"])
        score = 0.45 * iou + 0.30 * max(0.0, 1.0 - dist / 6.0) + 0.25 * sim
        if score > best_score:
            best_score = score
            best = cluster
    if best is not None and best_score >= 0.5:
        return best
    return None


def _dominant_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counts = Counter(int(value) for row in frame for value in row)
    if not counts:
        return 0
    return int(counts.most_common(1)[0][0])


def _connected_components(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    remaining = set(cells)
    components: list[tuple[tuple[int, int], ...]] = []
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
        components.append(tuple(sorted(comp)))
    components.sort(key=lambda comp: (_bbox(comp), len(comp)))
    return tuple(components)


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _merge_bbox(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


def _center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
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


def _hist_sim(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return float(overlap / max(total, 1))


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
