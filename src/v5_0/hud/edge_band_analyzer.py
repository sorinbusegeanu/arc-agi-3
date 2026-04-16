from __future__ import annotations

from collections import Counter
from math import sqrt

from v5_0.contracts.avatar_types import ProbeTransitionRecord

DEFAULT_EDGE_BAND_THICKNESS = 4


def extract_edge_band_components(
    transitions: tuple[ProbeTransitionRecord, ...],
    selected_avatar=None,
    poi_report=None,
    edge_band_thickness: int = DEFAULT_EDGE_BAND_THICKNESS,
) -> tuple[dict[str, object], ...]:
    if not transitions:
        return ()

    clusters: list[dict[str, object]] = []
    for record in transitions:
        if record.pre_frame is not None:
            for component in _components_in_edge_bands(record.pre_frame, edge_band_thickness):
                _merge_component(clusters, component, int(record.step_index), changed=False)
        if record.post_frame is not None:
            for component in _components_in_edge_bands(record.post_frame, edge_band_thickness):
                _merge_component(clusters, component, int(record.step_index), changed=False)
        if record.pre_frame is not None and record.post_frame is not None:
            for component in _changed_components_in_edge_bands(record.pre_frame, record.post_frame, edge_band_thickness):
                _merge_component(clusters, component, int(record.step_index), changed=True)

    out: list[dict[str, object]] = []
    for index, component in enumerate(clusters):
        bbox = component["bbox"]
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        out.append(
            {
                "hud_region_id": f"edge_{index:03d}",
                "bbox": bbox,
                "center": center,
                "area": int(area),
                "edge_side": component["edge_side"],
                "value_histogram": dict(sorted(component["value_histogram"].items())),
                "seen_step_indices": tuple(sorted(component["seen_step_indices"])),
                "change_step_indices": tuple(sorted(component["change_step_indices"])),
                "edge_locked_fraction": float(component.get("edge_locked_fraction", 0.0)),
            }
        )
    out.sort(key=lambda item: (item["edge_side"], item["bbox"], item["hud_region_id"]))
    return tuple(out)


def _merge_component(
    clusters: list[dict[str, object]],
    component: dict[str, object],
    step_index: int,
    changed: bool,
) -> None:
    best_index: int | None = None
    best_score = -1.0
    for index, existing in enumerate(clusters):
        if existing["edge_side"] != component["edge_side"]:
            continue
        iou = _bbox_iou(existing["bbox"], component["bbox"])
        center_dist = _bbox_center_distance(existing["bbox"], component["bbox"])
        center_sim = max(0.0, 1.0 - center_dist / 6.0)
        hist_sim = _hist_similarity(existing["value_histogram"], component["value_histogram"])
        score = 0.35 * iou + 0.35 * center_sim + 0.30 * hist_sim
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None or best_score < 0.35:
        clusters.append(
            {
                "bbox": component["bbox"],
                "edge_side": component["edge_side"],
                "value_histogram": Counter(component["value_histogram"]),
                "seen_step_indices": {step_index},
                "change_step_indices": {step_index} if changed else set(),
                "edge_locked_fraction": float(component.get("edge_locked_fraction", 0.0)),
                "merge_count": 1,
            }
        )
        return

    current = clusters[best_index]
    current["bbox"] = _union_bbox(current["bbox"], component["bbox"])
    current["value_histogram"].update(component["value_histogram"])
    current["seen_step_indices"].add(step_index)
    if changed:
        current["change_step_indices"].add(step_index)
    prev_count = int(current.get("merge_count", 1))
    new_count = prev_count + 1
    prev_frac = float(current.get("edge_locked_fraction", 0.0))
    new_frac = float(component.get("edge_locked_fraction", 0.0))
    current["edge_locked_fraction"] = ((prev_frac * prev_count) + new_frac) / max(new_count, 1)
    current["merge_count"] = new_count


def _components_in_edge_bands(
    frame: tuple[tuple[int, ...], ...],
    thickness: int,
) -> tuple[dict[str, object], ...]:
    height = len(frame)
    width = len(frame[0]) if height else 0
    if height == 0 or width == 0:
        return ()

    background = _dominant_value(frame)
    side_cells: dict[str, set[tuple[int, int]]] = {
        "top": set(),
        "bottom": set(),
        "left": set(),
        "right": set(),
    }

    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            if int(value) == background:
                continue
            side = _edge_side_for_cell(x, y, width, height, thickness)
            if side is None:
                continue
            side_cells[side].add((x, y))

    out: list[dict[str, object]] = []
    for side in ("top", "bottom", "left", "right"):
        for component in _connected_components(side_cells[side]):
            out.append(
                {
                    "bbox": _bbox(component),
                    "edge_side": side,
                    "value_histogram": _histogram(frame, component),
                    "edge_locked_fraction": _edge_locked_fraction(component, side, width, height, thickness),
                }
            )
    return tuple(out)


def _changed_components_in_edge_bands(
    pre_frame: tuple[tuple[int, ...], ...],
    post_frame: tuple[tuple[int, ...], ...],
    thickness: int,
) -> tuple[dict[str, object], ...]:
    height = min(len(pre_frame), len(post_frame))
    width = min(len(pre_frame[0]), len(post_frame[0])) if height else 0
    if height == 0 or width == 0:
        return ()

    side_cells: dict[str, set[tuple[int, int]]] = {
        "top": set(),
        "bottom": set(),
        "left": set(),
        "right": set(),
    }
    for y in range(height):
        for x in range(width):
            if int(pre_frame[y][x]) == int(post_frame[y][x]):
                continue
            side = _edge_side_for_cell(x, y, width, height, thickness)
            if side is None:
                continue
            side_cells[side].add((x, y))

    out: list[dict[str, object]] = []
    for side in ("top", "bottom", "left", "right"):
        for component in _connected_components(side_cells[side]):
            out.append(
                {
                    "bbox": _bbox(component),
                    "edge_side": side,
                    "value_histogram": _histogram(post_frame, component),
                    "edge_locked_fraction": _edge_locked_fraction(component, side, width, height, thickness),
                }
            )
    return tuple(out)


def _edge_side_for_cell(x: int, y: int, width: int, height: int, thickness: int) -> str | None:
    candidates: list[tuple[int, str]] = []
    if y < thickness:
        candidates.append((y, "top"))
    if y >= max(0, height - thickness):
        candidates.append((height - y - 1, "bottom"))
    if x < thickness:
        candidates.append((x, "left"))
    if x >= max(0, width - thickness):
        candidates.append((width - x - 1, "right"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _connected_components(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    remaining = set(cells)
    components: list[tuple[tuple[int, int], ...]] = []
    while remaining:
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


def _dominant_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counts = Counter(int(v) for row in frame for v in row)
    return int(counts.most_common(1)[0][0]) if counts else 0


def _histogram(frame: tuple[tuple[int, ...], ...], component: tuple[tuple[int, int], ...]) -> dict[int, int]:
    hist = Counter(int(frame[y][x]) for x, y in component)
    return dict(hist)


def _bbox(component: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    return (min(xs), min(ys), max(xs), max(ys))


def _union_bbox(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
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


def _bbox_center_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lc = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    rc = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    return sqrt((lc[0] - rc[0]) ** 2 + (lc[1] - rc[1]) ** 2)


def _hist_similarity(left: dict[int, int] | Counter[int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return float(overlap / max(total, 1))


def _edge_locked_fraction(
    component: tuple[tuple[int, int], ...],
    side: str,
    width: int,
    height: int,
    thickness: int,
) -> float:
    if not component:
        return 0.0
    locked = 0
    for x, y in component:
        if side == "top" and y < thickness:
            locked += 1
        elif side == "bottom" and y >= max(0, height - thickness):
            locked += 1
        elif side == "left" and x < thickness:
            locked += 1
        elif side == "right" and x >= max(0, width - thickness):
            locked += 1
    return float(locked / max(1, len(component)))
