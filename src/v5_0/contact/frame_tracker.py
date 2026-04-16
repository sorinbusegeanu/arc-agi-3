from __future__ import annotations

from collections import Counter
from math import sqrt

_DOMINANT_VALUE_CACHE: dict[int, tuple[object, int]] = {}


def track_avatar_bbox_in_frame(
    frame: tuple[tuple[int, ...], ...] | None,
    reference_bbox: tuple[int, int, int, int] | None,
    reference_histogram: dict[int, int] | None = None,
    frontier_reanchor: bool = False,
) -> tuple[int, int, int, int] | None:
    return _track_bbox(frame, reference_bbox, reference_histogram, for_poi=False, frontier_reanchor=frontier_reanchor)


def track_poi_bbox_in_frame(
    frame: tuple[tuple[int, ...], ...] | None,
    reference_bbox: tuple[int, int, int, int] | None,
    reference_histogram: dict[int, int] | None = None,
    frontier_reanchor: bool = False,
) -> tuple[int, int, int, int] | None:
    return _track_bbox(frame, reference_bbox, reference_histogram, for_poi=True, frontier_reanchor=frontier_reanchor)


def detect_screen_change(pre_frame: tuple[tuple[int, ...], ...] | None, post_frame: tuple[tuple[int, ...], ...] | None) -> bool:
    if pre_frame is None or post_frame is None:
        return False
    return any(
        int(pre_frame[y][x]) != int(post_frame[y][x])
        for y in range(min(len(pre_frame), len(post_frame)))
        for x in range(min(len(pre_frame[y]), len(post_frame[y])))
    )


def detect_screen_change_outside_hud_mask(
    pre_frame: tuple[tuple[int, ...], ...] | None,
    post_frame: tuple[tuple[int, ...], ...] | None,
    hud_mask=None,
) -> bool:
    if hud_mask is None:
        return detect_screen_change(pre_frame, post_frame)
    if pre_frame is None or post_frame is None:
        return False
    h = min(len(pre_frame), len(post_frame))
    w = min(len(pre_frame[0]), len(post_frame[0])) if h else 0
    if h == 0 or w == 0:
        return False

    if isinstance(hud_mask, dict):
        rows_active = set(int(v) for v in hud_mask.get("rows_active", ()))
        cols_active = set(int(v) for v in hud_mask.get("cols_active", ()))
    else:
        rows_active = set(int(v) for v in getattr(hud_mask, "rows_active", ()))
        cols_active = set(int(v) for v in getattr(hud_mask, "cols_active", ()))

    for y in range(h):
        for x in range(w):
            if int(pre_frame[y][x]) == int(post_frame[y][x]):
                continue
            if y in rows_active and x in cols_active:
                continue
            return True
    return False


def detect_hud_only_change(pre_frame: tuple[tuple[int, ...], ...] | None, post_frame: tuple[tuple[int, ...], ...] | None, border_width: int = 2) -> bool:
    if pre_frame is None or post_frame is None:
        return False
    if not detect_screen_change(pre_frame, post_frame):
        return False
    h = min(len(pre_frame), len(post_frame))
    w = min(len(pre_frame[0]), len(post_frame[0])) if h else 0
    # Conservative rule: when there is no interior region, do not classify as HUD-only.
    if h <= (2 * border_width) or w <= (2 * border_width):
        return False
    for y in range(h):
        for x in range(w):
            if int(pre_frame[y][x]) == int(post_frame[y][x]):
                continue
            if border_width <= x < (w - border_width) and border_width <= y < (h - border_width):
                return False
    return True


def detect_contact(
    avatar_bbox: tuple[int, int, int, int] | None,
    poi_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if avatar_bbox is None or poi_bbox is None:
        return False
    if _bbox_iou(avatar_bbox, poi_bbox) > 0.0:
        return True
    return _bbox_gap(avatar_bbox, poi_bbox) <= 1.0


def detect_object_removed(
    previous_poi_bbox: tuple[int, int, int, int] | None,
    current_poi_bbox: tuple[int, int, int, int] | None,
    contact_or_approach: bool,
) -> bool:
    return bool(previous_poi_bbox is not None and current_poi_bbox is None and contact_or_approach)


def detect_new_object_appeared(
    pre_frame: tuple[tuple[int, ...], ...] | None,
    post_frame: tuple[tuple[int, ...], ...] | None,
    avatar_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    if pre_frame is None or post_frame is None:
        return False
    h = min(len(pre_frame), len(post_frame))
    w = min(len(pre_frame[0]), len(post_frame[0])) if h else 0
    pre_background = _dominant_value(pre_frame)
    appeared_cells: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            pre_v = int(pre_frame[y][x])
            post_v = int(post_frame[y][x])
            if pre_v == post_v:
                continue
            if post_v == pre_background:
                continue
            appeared_cells.add((x, y))
    for component in _connected_components(appeared_cells):
        box = _bbox(component)
        if avatar_bbox is not None and _bbox_iou(box, avatar_bbox) > 0.0:
            continue
        if len(component) >= 2:
            return True
    return False


def _track_bbox(frame, reference_bbox, reference_histogram, for_poi: bool, frontier_reanchor: bool = False):
    if frame is None:
        return None
    components = tuple()
    if reference_bbox is not None:
        components = _components_non_background_near(frame, _center_from_bbox(reference_bbox), radius=18.0)
    if not components:
        components = _components_non_background(frame)
    if not components:
        return None
    if reference_bbox is None:
        return _bbox(max(components, key=len))

    best = None
    best_score = -1.0
    best_iou = 0.0
    best_dist = float("inf")
    ref_center = _center_from_bbox(reference_bbox)
    ref_area = _bbox_area(reference_bbox)
    best_area_ratio = 1.0
    for component in components:
        box = _bbox(component)
        center = _center_from_bbox(box)
        iou = _bbox_iou(box, reference_bbox)
        dist = sqrt((center[0] - ref_center[0]) ** 2 + (center[1] - ref_center[1]) ** 2)
        proximity = max(0.0, 1.0 - dist / 8.0)
        hist = _histogram_for_component(frame, component)
        hist_sim = _hist_similarity(hist, reference_histogram or {}) if reference_histogram else 0.5
        area = _bbox_area(box)
        area_ratio = max(area, ref_area) / max(1.0, min(area, ref_area))
        area_similarity = max(0.0, min(1.0, 1.0 - (area_ratio - 1.0) / 3.0))
        expansion_penalty = max(0.0, (area / max(1.0, ref_area)) - 2.5)
        score = 0.34 * iou + 0.24 * proximity + 0.22 * hist_sim + 0.20 * area_similarity - 0.18 * expansion_penalty
        if score > best_score:
            best = box
            best_score = score
            best_iou = iou
            best_dist = dist
            best_area_ratio = area / max(1.0, ref_area)

    if best is None:
        return None
    if _is_large_geometry_mismatch(
        for_poi=for_poi,
        score=best_score,
        iou=best_iou,
        center_distance=best_dist,
        area_ratio=best_area_ratio,
    ):
        if frontier_reanchor:
            return find_best_component_match_in_frame(
                frame=frame,
                reference_bbox=reference_bbox,
                reference_histogram=reference_histogram,
                for_poi=for_poi,
            )
        return None
    if for_poi and best_area_ratio > 3.0:
        if _is_bbox_visible(frame, reference_bbox):
            return reference_bbox
        return None
    if for_poi and best_score < 0.25:
        if _is_bbox_visible(frame, reference_bbox):
            return reference_bbox
        return None
    return best


def find_plausible_component_match_in_frame(
    *,
    frame: tuple[tuple[int, ...], ...] | None,
    reference_bbox: tuple[int, int, int, int] | None,
    reference_histogram: dict[int, int] | None = None,
    for_poi: bool,
) -> tuple[int, int, int, int] | None:
    return find_best_component_match_in_frame(
        frame=frame,
        reference_bbox=reference_bbox,
        reference_histogram=reference_histogram,
        for_poi=for_poi,
        max_distance=(8.5 if for_poi else 7.5),
    )


def find_best_component_match_in_frame(
    frame,
    reference_bbox,
    reference_histogram=None,
    for_poi: bool = False,
    max_distance=None,
    preferred_center=None,
    recent_bbox=None,
    previous_center=None,
    recent_motion=None,
    local_window_radius=None,
) -> tuple[int, int, int, int] | None:
    if frame is None:
        return None
    components = tuple()
    if reference_bbox is not None:
        local_center = preferred_center or previous_center or _center_from_bbox(reference_bbox)
        components = _candidate_components_for_reference_near(
            frame,
            reference_histogram,
            center=local_center,
            radius=float(local_window_radius) if local_window_radius is not None else (10.0 if for_poi else 8.0),
        )
    if not components:
        components = _candidate_components_for_reference(frame, reference_histogram)
    if not components:
        return None
    if reference_bbox is None:
        candidate = _bbox(max(components, key=len))
        return candidate if _bbox_area(candidate) >= 1 else None

    ref_center = _center_from_bbox(reference_bbox)
    pref_center = tuple(preferred_center) if preferred_center is not None else None
    prev_center = pref_center if pref_center is not None else (tuple(previous_center) if previous_center is not None else ref_center)
    ref_area = _bbox_area(reference_bbox)
    recent_center = _center_from_bbox(recent_bbox) if recent_bbox is not None else None
    best_box = None
    best_score = -1.0
    max_dist = float(max_distance) if max_distance is not None else (14.0 if for_poi else 9.0)
    local_radius = float(local_window_radius) if local_window_radius is not None else (6.5 if for_poi else 4.5)
    predicted_center = None
    if recent_motion is not None:
        try:
            predicted_center = (float(prev_center[0]) + float(recent_motion[0]), float(prev_center[1]) + float(recent_motion[1]))
        except Exception:
            predicted_center = None
    for component in components:
        box = _bbox(component)
        center = _center_from_bbox(box)
        dist = sqrt((center[0] - ref_center[0]) ** 2 + (center[1] - ref_center[1]) ** 2)
        if dist > max_dist:
            continue
        proximity = max(0.0, 1.0 - dist / max(1.0, max_dist))
        local_dist = sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
        local_bonus = max(0.0, 1.0 - local_dist / max(1.0, local_radius))
        if local_dist > (2.5 * local_radius):
            local_bonus = 0.0
        motion_bonus = 0.0
        if predicted_center is not None:
            pred_dist = sqrt((center[0] - predicted_center[0]) ** 2 + (center[1] - predicted_center[1]) ** 2)
            motion_bonus = max(0.0, 1.0 - pred_dist / max(1.0, local_radius + 2.0))
        recent_bbox_bonus = 0.0
        if recent_center is not None:
            recent_dist = sqrt((center[0] - recent_center[0]) ** 2 + (center[1] - recent_center[1]) ** 2)
            recent_bbox_bonus = max(0.0, 1.0 - recent_dist / max(1.0, local_radius + 1.0))
        hist = _histogram_for_component(frame, component)
        hist_sim = _hist_similarity(hist, reference_histogram or {}) if reference_histogram else 0.5
        area = _bbox_area(box)
        ratio = max(area, ref_area) / max(1.0, min(area, ref_area))
        area_similarity = max(0.0, 1.0 - (ratio - 1.0) / 3.5)
        width = max(1, box[2] - box[0] + 1)
        height = max(1, box[3] - box[1] + 1)
        ref_w = max(1, reference_bbox[2] - reference_bbox[0] + 1)
        ref_h = max(1, reference_bbox[3] - reference_bbox[1] + 1)
        wh_similarity = 1.0 - (abs(width - ref_w) + abs(height - ref_h)) / float(max(ref_w + ref_h, 1))
        wh_similarity = max(0.0, min(1.0, wh_similarity))
        compactness = min(width, height) / float(max(width, height))
        distance_weight = 0.30 if not for_poi else 0.20
        score = (
            (0.30 * hist_sim)
            + (distance_weight * proximity)
            + (0.18 * area_similarity)
            + (0.10 * wh_similarity)
            + (0.04 * compactness)
            + (0.10 * local_bonus)
            + (0.08 * motion_bonus)
            + (0.08 * recent_bbox_bonus)
        )
        if score > best_score:
            best_score = score
            best_box = box
    if best_box is None:
        return None
    threshold = 0.35 if for_poi else 0.38
    if best_score < threshold:
        # conservative plausibility fallback for world-change cases:
        # if a nearby component remains reasonably similar, prefer it over None.
        ref_center = _center_from_bbox(reference_bbox)
        c = _center_from_bbox(best_box)
        dist = sqrt((c[0] - ref_center[0]) ** 2 + (c[1] - ref_center[1]) ** 2)
        area_ratio = max(_bbox_area(best_box), _bbox_area(reference_bbox)) / max(1.0, min(_bbox_area(best_box), _bbox_area(reference_bbox)))
        plausible = dist <= (9.5 if for_poi else 7.0) and area_ratio <= (4.5 if for_poi else 3.5)
        if not plausible:
            return None
    return best_box


def reacquire_avatar_bbox_in_frame(
    frame,
    reference_bbox,
    reference_histogram=None,
    preferred_center=None,
    recent_bbox=None,
    previous_center=None,
    recent_motion=None,
) -> tuple[int, int, int, int] | None:
    local = find_best_component_match_in_frame(
        frame=frame,
        reference_bbox=reference_bbox,
        reference_histogram=reference_histogram,
        for_poi=False,
        max_distance=6.5,
        preferred_center=preferred_center,
        recent_bbox=recent_bbox,
        previous_center=previous_center,
        recent_motion=recent_motion,
        local_window_radius=4.5,
    )
    if local is not None:
        return local
    return find_best_component_match_in_frame(
        frame=frame,
        reference_bbox=reference_bbox,
        reference_histogram=reference_histogram,
        for_poi=False,
        max_distance=10.0,
        preferred_center=preferred_center,
        recent_bbox=recent_bbox,
        previous_center=previous_center,
        recent_motion=recent_motion,
        local_window_radius=6.0,
    )


def reacquire_poi_bbox_in_frame(
    frame,
    reference_bbox,
    reference_histogram=None,
    preferred_center=None,
    recent_bbox=None,
    previous_center=None,
    recent_motion=None,
) -> tuple[int, int, int, int] | None:
    local = find_best_component_match_in_frame(
        frame=frame,
        reference_bbox=reference_bbox,
        reference_histogram=reference_histogram,
        for_poi=True,
        max_distance=8.0,
        preferred_center=preferred_center,
        recent_bbox=recent_bbox,
        previous_center=previous_center,
        recent_motion=recent_motion,
        local_window_radius=6.0,
    )
    if local is not None:
        return local
    return find_best_component_match_in_frame(
        frame=frame,
        reference_bbox=reference_bbox,
        reference_histogram=reference_histogram,
        for_poi=True,
        max_distance=14.0,
        preferred_center=preferred_center,
        recent_bbox=recent_bbox,
        previous_center=previous_center,
        recent_motion=recent_motion,
        local_window_radius=8.0,
    )


def _is_large_geometry_mismatch(*, for_poi: bool, score: float, iou: float, center_distance: float, area_ratio: float) -> bool:
    if for_poi:
        if iou < 0.05 and center_distance > 10.0:
            return True
        if iou < 0.05 and area_ratio > 5.0:
            return True
        return score < 0.15 and center_distance > 8.0
    if iou < 0.03 and center_distance > 9.0:
        return True
    return score < 0.10 and area_ratio > 4.0


def _components_non_background(frame):
    background = _dominant_value(frame)
    cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
    return _connected_components(cells)


def _candidate_components_for_reference(frame, reference_histogram):
    import time, sys
    t0 = time.time()
    prioritized = _components_for_reference_values(frame, reference_histogram)
    if not prioritized:
        result = _components_non_background(frame)
        return result
    seen = set()
    ordered = []
    all_bg = _components_non_background(frame)
    for component in tuple(prioritized) + tuple(all_bg):
        key = tuple(component)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(component)
    return tuple(ordered)


def _candidate_components_for_reference_near(frame, reference_histogram, *, center, radius: float):
    prioritized = _components_for_reference_values_near(frame, reference_histogram, center=center, radius=radius)
    if prioritized:
        return prioritized
    return _components_non_background_near(frame, center, radius=radius)


def _components_for_reference_values(frame, reference_histogram):
    if frame is None:
        return tuple()
    background = _dominant_value(frame)
    if reference_histogram:
        ranked_values = [
            int(value)
            for value, count in sorted(
                ((int(v), int(c)) for v, c in dict(reference_histogram).items() if int(c) > 0),
                key=lambda item: (-item[1], item[0]),
            )
            if int(value) != background
        ]
    else:
        counts = Counter(
            int(cell_value)
            for row in frame
            for cell_value in row
            if int(cell_value) != background
        )
        ranked_values = [int(value) for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    components = []
    for value in ranked_values[:3]:
        cells = {
            (x, y)
            for y, row in enumerate(frame)
            for x, cell_value in enumerate(row)
            if int(cell_value) == int(value)
        }
        components.extend(_connected_components(cells))
    return tuple(components)


def _components_for_reference_values_near(frame, reference_histogram, *, center, radius: float):
    if frame is None or center is None:
        return tuple()
    background = _dominant_value(frame)
    if reference_histogram:
        ranked_values = [
            int(value)
            for value, count in sorted(
                ((int(v), int(c)) for v, c in dict(reference_histogram).items() if int(c) > 0),
                key=lambda item: (-item[1], item[0]),
            )
            if int(value) != background
        ]
    else:
        ranked_values = tuple()
    if not ranked_values:
        return tuple()
    x0, y0, x1, y1 = _window_bounds(frame, center, radius)
    components = []
    value_set = set(ranked_values[:3])
    cells_by_value: dict[int, set[tuple[int, int]]] = {value: set() for value in value_set}
    for y in range(y0, y1 + 1):
        row = frame[y]
        for x in range(x0, x1 + 1):
            value = int(row[x])
            if value in value_set:
                cells_by_value[value].add((x, y))
    for value in ranked_values[:3]:
        components.extend(_connected_components(cells_by_value.get(int(value), set())))
    return tuple(components)


def _components_non_background_near(frame, center, *, radius: float):
    if frame is None or center is None:
        return tuple()
    background = _dominant_value(frame)
    x0, y0, x1, y1 = _window_bounds(frame, center, radius)
    cells = {
        (x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
        if int(frame[y][x]) != background
    }
    return _connected_components(cells)


def _window_bounds(frame, center, radius: float):
    cx, cy = float(center[0]), float(center[1])
    r = max(1.0, float(radius))
    height = len(frame)
    width = len(frame[0]) if height else 0
    x0 = max(0, int(cx - r))
    y0 = max(0, int(cy - r))
    x1 = min(max(0, width - 1), int(cx + r))
    y1 = min(max(0, height - 1), int(cy + r))
    return x0, y0, x1, y1


def _connected_components(cells, max_components=50):
    """
    Find connected components with a limit to prevent expensive computation on large frames.

    Args:
        cells: Set of (x, y) cells to analyze
        max_components: Maximum number of components to find before early exit

    Returns:
        Tuple of components (each component is a tuple of (x, y) cells)
    """
    remaining = set(cells)
    components = []
    while remaining and len(components) < max_components:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in remaining:
                    remaining.remove(n)
                    component.add(n)
                    stack.append(n)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _dominant_value(frame):
    cache_key = id(frame)
    cached = _DOMINANT_VALUE_CACHE.get(cache_key)
    if cached is not None and cached[0] is frame:
        return int(cached[1])
    counts = Counter(int(v) for row in frame for v in row)
    value = int(counts.most_common(1)[0][0]) if counts else 0
    if len(_DOMINANT_VALUE_CACHE) > 128:
        _DOMINANT_VALUE_CACHE.clear()
    _DOMINANT_VALUE_CACHE[cache_key] = (frame, value)
    return value


def _bbox(component):
    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    return (min(xs), min(ys), max(xs), max(ys))


def _center_from_bbox(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_area(bbox):
    return max(1, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))


def _bbox_iou(left, right):
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    l_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    r_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    return inter / max(l_area + r_area - inter, 1)


def _bbox_gap(left, right):
    dx = max(0, max(left[0] - right[2] - 1, right[0] - left[2] - 1))
    dy = max(0, max(left[1] - right[3] - 1, right[1] - left[3] - 1))
    return sqrt(dx * dx + dy * dy)


def _histogram_for_component(frame, component):
    c = Counter(int(frame[y][x]) for x, y in component)
    return dict(c)


def _hist_similarity(left, right):
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return overlap / max(total, 1)


def _is_bbox_visible(frame, bbox):
    if frame is None or bbox is None:
        return False
    background = _dominant_value(frame)
    x0, y0, x1, y1 = bbox
    hits = 0
    total = 0
    for y in range(max(0, y0), min(len(frame), y1 + 1)):
        for x in range(max(0, x0), min(len(frame[0]), x1 + 1)):
            total += 1
            if int(frame[y][x]) != background:
                hits += 1
    if total <= 0:
        return False
    return (hits / total) >= 0.2
