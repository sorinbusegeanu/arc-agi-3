from __future__ import annotations

from math import sqrt

from v5_0.contracts.avatar_types import HUDMask


def reject_avatar_like_edge_components(
    edge_band_components: tuple[dict[str, object], ...],
    selected_avatar,
    avatar_report=None,
    episode_transitions=None,
) -> tuple[tuple[dict[str, object], ...], int]:
    avatar_bbox = getattr(selected_avatar, "selected_bbox", None)
    avatar_hist = _selected_avatar_histogram(avatar_report, selected_avatar)
    support_steps = set(_selected_avatar_support_steps(avatar_report, selected_avatar))

    kept: list[dict[str, object]] = []
    rejected = 0
    for component in edge_band_components:
        bbox = component["bbox"]
        overlap = _bbox_iou(bbox, avatar_bbox) if avatar_bbox is not None else 0.0
        center_dist = _bbox_center_distance(bbox, avatar_bbox) if avatar_bbox is not None else 999.0
        hist_sim = _hist_similarity(component.get("value_histogram", {}), avatar_hist)
        seen_steps = set(int(v) for v in component.get("seen_step_indices", ()))
        support_overlap = bool(support_steps & seen_steps)

        if overlap > 0.0:
            rejected += 1
            continue
        if avatar_bbox is not None and center_dist <= 1.5 and support_overlap:
            rejected += 1
            continue
        if hist_sim >= 0.9 and support_overlap:
            rejected += 1
            continue
        kept.append(component)
    return tuple(kept), rejected


def reject_world_like_edge_components(
    edge_band_components: tuple[dict[str, object], ...],
    selected_avatar,
    avatar_reports=None,
    poi_report=None,
    episode_transitions=None,
    edge_band_thickness: int = 2,
) -> tuple[tuple[dict[str, object], ...], int]:
    world_change_steps = _world_change_steps(episode_transitions or (), edge_band_thickness)
    poi_boxes = tuple(getattr(item, "bbox", None) for item in getattr(poi_report, "candidates", ()))
    frame = None
    if episode_transitions:
        frame = next((r.pre_frame for r in episode_transitions if r.pre_frame is not None), None)
        if frame is None:
            frame = next((r.post_frame for r in episode_transitions if r.post_frame is not None), None)
    height = len(frame) if frame is not None else 0
    width = len(frame[0]) if frame is not None and frame else 0

    kept: list[dict[str, object]] = []
    rejected = 0
    for component in edge_band_components:
        out_component = dict(component)
        bbox = component["bbox"]
        if frame is not None:
            if _extends_into_world(bbox, frame, edge_band_thickness):
                rejected += 1
                continue

        changed_steps = set(int(v) for v in component.get("change_step_indices", ()))
        coupled_ratio = 0.0
        if changed_steps:
            coupled_ratio = len(changed_steps & world_change_steps) / max(1, len(changed_steps))
        world_motion_penalty = max(0.0, min(1.0, 0.55 * coupled_ratio))
        out_component["world_motion_penalty"] = float(world_motion_penalty)

        if any(box is not None and _bbox_iou(bbox, box) > 0.5 for box in poi_boxes):
            rejected += 1
            continue

        edge_locked = float(component.get("edge_locked_fraction", 0.0)) >= 0.75
        if not edge_locked:
            rejected += 1
            continue

        if _is_static_edge_locked_candidate(out_component, width, height, edge_band_thickness):
            out_component["static_edge_hud"] = True
            kept.append(out_component)
            continue

        out_component["static_edge_hud"] = False
        kept.append(out_component)
    return tuple(kept), rejected


def build_persistent_hud_mask(
    surviving_components: tuple[dict[str, object], ...],
    episode_transitions,
    min_stability_score: float = 0.35,
    min_persistence_steps: int = 2,
) -> HUDMask:
    frame = None
    for record in episode_transitions:
        if record.pre_frame is not None:
            frame = record.pre_frame
            break
        if record.post_frame is not None:
            frame = record.post_frame
            break

    height = len(frame) if frame is not None else 0
    width = len(frame[0]) if frame is not None and frame else 0
    active_cells: set[tuple[int, int]] = set()
    regions: list[str] = []

    for component in surviving_components:
        stability = float(component.get("stability_score", 0.0))
        persistence_steps = len(set(int(v) for v in component.get("seen_step_indices", ())))
        edge_locked = float(component.get("edge_locked_fraction", 0.0)) >= 0.75
        world_penalty = float(component.get("world_motion_penalty", 0.0))
        static_accept = bool(edge_locked and persistence_steps >= int(min_persistence_steps) and world_penalty <= 0.8)
        if not static_accept and stability < min_stability_score:
            continue
        x0, y0, x1, y1 = component["bbox"]
        for y in range(max(0, y0), min(height, y1 + 1)):
            for x in range(max(0, x0), min(width, x1 + 1)):
                active_cells.add((x, y))
        regions.append(str(component["hud_region_id"]))

    rows_active = tuple(sorted({y for _, y in active_cells}))
    cols_active = tuple(sorted({x for x, _ in active_cells}))
    return HUDMask(
        height=int(height),
        width=int(width),
        true_cell_count=len(active_cells),
        rows_active=rows_active,
        cols_active=cols_active,
        regions=tuple(sorted(regions)),
    )


def _world_change_steps(transitions, edge_band_thickness: int) -> set[int]:
    out: set[int] = set()
    for record in transitions:
        if record.pre_frame is None or record.post_frame is None:
            continue
        if _has_non_edge_change(record.pre_frame, record.post_frame, edge_band_thickness):
            out.add(int(record.step_index))
    return out


def _has_non_edge_change(pre, post, edge_band_thickness: int) -> bool:
    h = min(len(pre), len(post))
    w = min(len(pre[0]), len(post[0])) if h else 0
    for y in range(h):
        for x in range(w):
            if int(pre[y][x]) == int(post[y][x]):
                continue
            if edge_band_thickness <= x < (w - edge_band_thickness) and edge_band_thickness <= y < (h - edge_band_thickness):
                return True
    return False


def _extends_into_world(bbox: tuple[int, int, int, int], frame, edge_band_thickness: int) -> bool:
    h = len(frame)
    w = len(frame[0]) if h else 0
    x0, y0, x1, y1 = bbox
    return (
        x0 >= edge_band_thickness
        and x1 < (w - edge_band_thickness)
        and y0 >= edge_band_thickness
        and y1 < (h - edge_band_thickness)
    )


def _is_static_edge_locked_candidate(
    component: dict[str, object],
    width: int,
    height: int,
    edge_band_thickness: int,
) -> bool:
    bbox = component.get("bbox")
    if bbox is None:
        return False
    edge_locked_fraction = float(component.get("edge_locked_fraction", 0.0))
    if edge_locked_fraction < 0.75:
        return False
    if width > 0 and height > 0:
        x0, y0, x1, y1 = bbox
        if x0 >= edge_band_thickness and x1 < (width - edge_band_thickness) and y0 >= edge_band_thickness and y1 < (height - edge_band_thickness):
            return False
    persistence_steps = len(set(int(v) for v in component.get("seen_step_indices", ())))
    if persistence_steps < 2:
        return False
    positional_stability = float(component.get("positional_stability", 0.0))
    same_side_stability = float(component.get("same_side_stability", 0.0))
    world_penalty = float(component.get("world_motion_penalty", 0.0))
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    strip_like = bw <= 2 or bh <= 2 or bw >= 3 * bh or bh >= 3 * bw
    small = bw * bh <= 10
    return bool((strip_like or small) and (positional_stability >= 0.45 or same_side_stability >= 0.45) and world_penalty <= 0.9)


def _selected_avatar_histogram(avatar_report, selected_avatar) -> dict[int, int]:
    if avatar_report is None:
        return {}
    selected_id = getattr(selected_avatar, "selected_candidate_id", None)
    for candidate in getattr(avatar_report, "candidates", ()):
        if getattr(candidate, "candidate_id", None) == selected_id:
            return dict(getattr(candidate, "value_histogram_post", {}))
    return {}


def _selected_avatar_support_steps(avatar_report, selected_avatar) -> tuple[int, ...]:
    if avatar_report is None:
        return ()
    selected_id = getattr(selected_avatar, "selected_candidate_id", None)
    for candidate in getattr(avatar_report, "candidates", ()):
        if getattr(candidate, "candidate_id", None) == selected_id:
            return tuple(int(v) for v in getattr(candidate, "support_step_indices", ()))
    return ()


def _bbox_iou(left, right) -> float:
    if left is None or right is None:
        return 0.0
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


def _bbox_center_distance(left, right) -> float:
    if left is None or right is None:
        return 999.0
    lc = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    rc = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    return sqrt((lc[0] - rc[0]) ** 2 + (lc[1] - rc[1]) ** 2)


def _hist_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return overlap / max(total, 1)
