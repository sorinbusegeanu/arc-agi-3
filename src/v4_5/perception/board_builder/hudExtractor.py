from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot
from typing import Any

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.boardObject import BoardObject
from v4_5.contracts.poiTypes import PoiRecord


@dataclass(frozen=True)
class HudExtractionResult:
    hud_regions: tuple[tuple[int, int, int, int], ...]
    hud_tracks: tuple[dict[str, Any], ...]
    hud_confidence: float
    failure_reason: str | None
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ProgressExtractionResult:
    progress_candidates: tuple[dict[str, Any], ...]
    selected_candidates: tuple[dict[str, Any], ...]
    diagnostics: dict[str, object]
    failure_reason: str | None = None


def extract_hud_regions(frame: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    if not frame:
        return (None, None, None)
    background = _dominant_frame_value(frame)
    components = [
        component
        for component in _connected_components(
            {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
        )
        if _touches_border(component, frame)
    ]
    if not components:
        return (None, None, None)
    scored = sorted(components, key=lambda component: (_border_support(component, frame), len(component)), reverse=True)
    hud_bbox = _bbox(scored[0])
    progress_bbox = None
    life_bbox = None
    for component in scored:
        bbox = _bbox(component)
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
        aspect = max(width / max(1.0, height), height / max(1.0, width))
        if progress_bbox is None and aspect >= 3.0:
            progress_bbox = bbox
        if life_bbox is None and width <= 3 and height <= 3:
            life_bbox = bbox
    return (hud_bbox, progress_bbox, life_bbox)


def extract_hud_candidates(
    *,
    bootstrap_frames: tuple[Any, ...] = (),
    bootstrap_transition_records: tuple[Any, ...] = (),
    avatar_detection_result: AvatarDetectionResult | None,
    poi_extraction_result: Any = None,
    hud_exclusion_regions: tuple[str, ...] = (),
    avatar_bbox_margin: int = 0,
) -> HudExtractionResult:
    records = tuple(_normalize_record(record) for record in bootstrap_transition_records)
    frames = tuple(_normalize_frame(frame) for frame in bootstrap_frames if _normalize_frame(frame) is not None)
    if not frames:
        frames = _ordered_frames(records)
    avatar_track = _avatar_track_for_frames(frames, avatar_detection_result, margin=avatar_bbox_margin)
    avatar_masks = {index: set(item["cells"]) for index, item in avatar_track.items()}
    poi_masks = _poi_masks_for_frames(frames, poi_extraction_result)
    extra_exclusions = _parse_region_cells(hud_exclusion_regions)
    per_frame_components: list[dict[str, Any]] = []
    rejection_counts = Counter()
    for frame_index, frame in enumerate(frames):
        if not frame:
            per_frame_components.append({"frame_index": frame_index, "components": ()})
            continue
        background = _dominant_frame_value(frame)
        avatar_mask = set(avatar_masks.get(frame_index, set()))
        poi_mask = set(poi_masks.get(frame_index, set()))
        all_cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background and (x, y) not in extra_exclusions}
        for component in _connected_components(all_cells):
            avatar_overlap = _mask_overlap_ratio(component, avatar_mask)
            poi_overlap = _mask_overlap_ratio(component, poi_mask)
            if avatar_overlap >= 0.5:
                rejection_counts["avatar_overlap"] += 1
            if poi_overlap >= 0.2:
                rejection_counts["poi_overlap"] += 1
        cells = {cell for cell in all_cells if cell not in avatar_mask and cell not in poi_mask}
        components = []
        for component in _connected_components(cells):
            touches = _touches(frame, component)
            border_distance = _distance_to_border(component, frame)
            value_histogram = dict(sorted(Counter(int(frame[y][x]) for x, y in component).items()))
            avatar_overlap = _mask_overlap_ratio(component, avatar_mask)
            poi_overlap = _mask_overlap_ratio(component, poi_mask)
            avatar_center = avatar_track.get(frame_index, {}).get("center")
            component_center = _center(component)
            record = {
                "bbox": _bbox(component),
                "area": len(component),
                "center": component_center,
                "value_histogram": value_histogram,
                "shape_signature": _shape_signature(component),
                "touches_top": touches["top"],
                "touches_bottom": touches["bottom"],
                "touches_left": touches["left"],
                "touches_right": touches["right"],
                "distance_to_nearest_border": border_distance,
                "frame_index": frame_index,
                "cells": component,
                "avatar_overlap_ratio": avatar_overlap,
                "poi_overlap_ratio": poi_overlap,
                "avatar_center_distance": None if avatar_center is None else _distance(component_center, avatar_center),
                "screen_anchor_score": _screen_anchor_score(touches, border_distance, frame),
            }
            components.append(record)
        per_frame_components.append({"frame_index": frame_index, "components": tuple(components)})
    tracks = _build_hud_tracks(per_frame_components)
    scored_tracks = sorted((_score_hud_track(track, records, avatar_track) for track in tracks), key=lambda item: item["hud_score"], reverse=True)
    selected_tracks = []
    for track in scored_tracks:
        if track["rejection_reason"] is not None:
            rejection_counts[track["rejection_reason"]] += 1
            continue
        if track["hud_score"] < 0.45:
            rejection_counts["low_score"] += 1
            continue
        selected_tracks.append(track)
    selected_tracks = _merge_adjacent_hud_tracks(selected_tracks)
    best_scores = [round(float(track["hud_score"]), 4) for track in scored_tracks[:5]]
    diagnostics = {
        "per_frame_non_background_component_count": {
            int(item["frame_index"]): len(item["components"]) for item in per_frame_components
        },
        "hud_track_count": len(scored_tracks),
        "selected_hud_region_count": len(selected_tracks),
        "rejection_counts": dict(rejection_counts),
        "best_hud_scores": tuple(best_scores),
        "failure_reason": None if selected_tracks else ("no_hud_candidates" if scored_tracks else "no_border_anchored_components"),
    }
    return HudExtractionResult(
        hud_regions=tuple(track["hud_bbox"] for track in selected_tracks),
        hud_tracks=tuple(selected_tracks),
        hud_confidence=0.0 if not selected_tracks else float(selected_tracks[0]["hud_score"]),
        failure_reason=diagnostics["failure_reason"],
        diagnostics=diagnostics,
    )


def extract_progress_candidates(
    *,
    frames: tuple[Any, ...],
    hud_tracks: tuple[dict[str, Any], ...],
    avatar_detection_result: AvatarDetectionResult | None = None,
    poi_extraction_result: Any = None,
) -> ProgressExtractionResult:
    normalized_frames = tuple(_normalize_frame(frame) for frame in frames if _normalize_frame(frame) is not None)
    if not normalized_frames or not hud_tracks:
        return ProgressExtractionResult(
            progress_candidates=(),
            selected_candidates=(),
            diagnostics={"progress_candidate_count": 0, "selected_progress_region_count": 0, "best_progress_scores": (), "failure_reason": "no_hud_tracks"},
            failure_reason="no_hud_tracks",
        )
    avatar_track = _avatar_track_for_frames(normalized_frames, avatar_detection_result, margin=1)
    poi_masks = _poi_masks_for_frames(normalized_frames, poi_extraction_result)
    candidates = []
    for track in hud_tracks:
        candidate = _score_progress_track(track, normalized_frames, avatar_track, poi_masks)
        candidates.append(candidate)
    ranked = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    selected = [candidate for candidate in ranked if candidate["rejection_reason"] is None and candidate["confidence"] >= 0.5]
    diagnostics = {
        "progress_candidate_count": len(ranked),
        "selected_progress_region_count": len(selected),
        "best_progress_scores": tuple(round(float(item["confidence"]), 4) for item in ranked[:5]),
        "failure_reason": None if selected else ("no_progress_candidates" if ranked else "no_hud_tracks"),
    }
    return ProgressExtractionResult(
        progress_candidates=tuple(ranked),
        selected_candidates=tuple(selected),
        diagnostics=diagnostics,
        failure_reason=diagnostics["failure_reason"],
    )


def _normalize_record(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        getter = record.get
    else:
        getter = lambda name, default=None: getattr(record, name, default)
    return {
        "step_index": int(getter("step_index", 0) or 0),
        "action": str(getter("action", "") or ""),
        "pre_frame": _normalize_frame(getter("pre_observation") or getter("pre_observation_ref")),
        "post_frame": _normalize_frame(getter("post_observation") or getter("post_observation_ref") or getter("raw_observation_ref")),
        "invalid_action": bool(getter("invalid_action", False)),
        "blocked_action": bool(getter("blocked_action", False)),
        "terminal": bool(getter("terminal", False)),
    }


def _normalize_frame(frame: Any) -> tuple[tuple[int, ...], ...] | None:
    if frame is None:
        return None
    if isinstance(frame, dict):
        frame = frame.get("frame")
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, (list, tuple)) and frame and isinstance(frame[0], (list, tuple)) and frame[0] and isinstance(frame[0][0], (list, tuple)):
        frame = frame[0]
    if not isinstance(frame, (list, tuple)):
        return None
    rows = []
    for row in frame:
        if not isinstance(row, (list, tuple)):
            return None
        rows.append(tuple(int(value) for value in row))
    return tuple(rows)


def _ordered_frames(records: tuple[dict[str, Any], ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    frames = []
    if records and records[0]["pre_frame"] is not None:
        frames.append(records[0]["pre_frame"])
    for record in records:
        if record["post_frame"] is not None:
            frames.append(record["post_frame"])
    unique_frames = []
    for frame in frames:
        if not unique_frames or unique_frames[-1] != frame:
            unique_frames.append(frame)
    return tuple(unique_frames)


def _avatar_track_for_frames(
    frames: tuple[tuple[tuple[int, ...], ...], ...],
    avatar_result: AvatarDetectionResult | None,
    *,
    margin: int,
) -> dict[int, dict[str, object]]:
    if avatar_result is None or avatar_result.avatar_bbox is None:
        return {}
    track = {}
    for frame_index, frame in enumerate(frames):
        del frame, frame_index
        chosen_cells = set(_cells_for_bbox(avatar_result.avatar_bbox, margin=margin))
        bbox = _bbox(chosen_cells)
        center = _center(chosen_cells)
        track[len(track)] = {"cells": chosen_cells, "bbox": bbox, "center": center}
    return track


def _poi_masks_for_frames(frames: tuple[tuple[tuple[int, ...], ...], ...], poi_result: Any) -> dict[int, set[tuple[int, int]]]:
    pois: list[PoiRecord] = []
    if poi_result is None:
        return {}
    selected = getattr(poi_result, "selected_pois", None)
    if selected is not None:
        pois.extend(
            poi
            for poi in getattr(selected, "pois", ())
            if getattr(poi, "failure_reason", None) is None and not getattr(poi, "rejected_as_avatar_overlap", False)
        )
    elif hasattr(poi_result, "pois"):
        pois.extend(
            poi
            for poi in getattr(poi_result, "pois", ())
            if getattr(poi, "failure_reason", None) is None and not getattr(poi, "rejected_as_avatar_overlap", False)
        )
    elif hasattr(poi_result, "selected_pois"):
        pois.extend(tuple(getattr(poi_result, "selected_pois", ())))
    masks = {}
    for frame_index in range(len(frames)):
        masks[frame_index] = set()
        for poi in pois:
            masks[frame_index].update(_cells_for_bbox(poi.bbox))
    return masks


def _build_hud_tracks(per_frame_components: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    tracks: list[dict[str, Any]] = []
    for frame_group in per_frame_components:
        for component in frame_group["components"]:
            best_track = None
            best_link_score = 0.0
            for track in tracks:
                previous = track["components"][-1]
                if component["frame_index"] != previous["frame_index"] + 1:
                    continue
                link_score = _hud_track_link_score(previous, component)
                if link_score > best_link_score:
                    best_link_score = link_score
                    best_track = track
            if best_track is not None and best_link_score >= 0.5:
                best_track["components"].append(component)
                best_track["link_scores"].append(best_link_score)
            else:
                tracks.append({"components": [component], "link_scores": []})
    return tuple({"components": tuple(track["components"]), "link_scores": tuple(track["link_scores"])} for track in tracks)


def _score_hud_track(track: dict[str, Any], records: tuple[dict[str, Any], ...], avatar_track: dict[int, dict[str, object]]) -> dict[str, Any]:
    components = tuple(track["components"])
    support_steps = tuple(int(component["frame_index"]) for component in components)
    centers = [component["center"] for component in components]
    bboxes = [component["bbox"] for component in components]
    histograms = [component["value_histogram"] for component in components]
    persistence = min(1.0, len(components) / max(1.0, len(avatar_track) or len(components)))
    position_stability = _spatial_stability(centers, bboxes)
    size_stability = _size_stability(components)
    shape_stability = _shape_stability(components)
    value_stability = _value_histogram_stability(histograms)
    border_attachment = sum(component["screen_anchor_score"] for component in components) / len(components)
    motion_penalty = _world_motion_penalty(components, records)
    avatar_penalty = _avatar_correlation_penalty(components, records, avatar_track)
    overlap_penalty = max(max(component["avatar_overlap_ratio"], component["poi_overlap_ratio"]) for component in components)
    compact_objectness = _compact_objectness(components)
    follows_world_motion = motion_penalty >= 0.65
    middle_without_border = border_attachment < 0.35
    weak_persistence = persistence < 0.5
    rejection_reason = None
    if overlap_penalty >= 0.5:
        rejection_reason = "overlap"
    elif follows_world_motion:
        rejection_reason = "world_motion"
    elif weak_persistence:
        rejection_reason = "low_persistence"
    elif middle_without_border:
        rejection_reason = "middle_without_border"
    hud_score = max(
        0.0,
        0.24 * persistence
        + 0.20 * position_stability
        + 0.12 * size_stability
        + 0.10 * shape_stability
        + 0.10 * value_stability
        + 0.18 * border_attachment
        + 0.08 * compact_objectness
        - 0.12 * motion_penalty
        - 0.16 * avatar_penalty
        - 0.10 * overlap_penalty,
    )
    representative = max(components, key=lambda component: (component["screen_anchor_score"], component["area"]))
    life_like = (
        len(components) >= 2
        and max((component["bbox"][2] - component["bbox"][0] + 1) for component in components) <= 3
        and max((component["bbox"][3] - component["bbox"][1] + 1) for component in components) <= 3
        and border_attachment >= 0.5
    )
    return {
        "hud_bbox": representative["bbox"],
        "support_step_indices": support_steps,
        "value_candidates": tuple(int(value) for value, _ in Counter(
            value
            for histogram in histograms
            for value, count in histogram.items()
            for _ in range(int(count))
        ).most_common(3)),
        "hud_score": hud_score,
        "persistence_score": persistence,
        "border_attachment_score": border_attachment,
        "position_stability_score": position_stability,
        "components": components,
        "cells": tuple(sorted({cell for component in components for cell in component["cells"]}, key=lambda item: (item[1], item[0]))),
        "rejection_reason": rejection_reason,
        "life_like": life_like,
    }


def _score_progress_track(
    track: dict[str, Any],
    frames: tuple[tuple[tuple[int, ...], ...], ...],
    avatar_track: dict[int, dict[str, object]],
    poi_masks: dict[int, set[tuple[int, int]]],
) -> dict[str, Any]:
    components = tuple(track["components"])
    bbox = _stable_container_bbox(components)
    width = bbox[2] - bbox[0] + 1
    height = bbox[3] - bbox[1] + 1
    aspect_ratio = max(width / max(1.0, height), height / max(1.0, width))
    outer_stability = _outer_bbox_stability(components, bbox)
    border_touch = any((
        components[0]["touches_top"],
        components[0]["touches_bottom"],
        components[0]["touches_left"],
        components[0]["touches_right"],
    ))
    orientation = "horizontal" if width >= height else "vertical"
    container_value = Counter(
        value
        for component in components
        for value, count in component["value_histogram"].items()
        for _ in range(int(count))
    ).most_common(1)[0][0]
    fill_bboxes = []
    fill_ratios = []
    interior_change_counts = []
    primary_axis_spans = []
    secondary_axis_spans = []
    for frame_index, frame in enumerate(frames):
        if not frame:
            fill_bboxes.append(None)
            fill_ratios.append(0.0)
            interior_change_counts.append(0)
            primary_axis_spans.append(0)
            secondary_axis_spans.append(0)
            continue
        background = _dominant_frame_value(frame)
        avatar_cells = set(avatar_track.get(frame_index, {}).get("cells", ()))
        poi_cells = set(poi_masks.get(frame_index, set()))
        region_cells = {
            (x, y, int(frame[y][x]))
            for y in range(bbox[1], bbox[3] + 1)
            for x in range(bbox[0], bbox[2] + 1)
            if 0 <= y < len(frame)
            and 0 <= x < len(frame[0])
            and int(frame[y][x]) != background
            and (x, y) not in avatar_cells
            and (x, y) not in poi_cells
        }
        fill_cells = {(x, y) for x, y, value in region_cells if value != container_value}
        interior = {
            (x, y) for x, y, value in region_cells
            if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3] and value != container_value
        }
        region_points = {(x, y) for x, y, _ in region_cells}
        target_cells = interior if interior else (fill_cells if fill_cells else region_points)
        fill_bbox = None if not target_cells else _bbox(target_cells)
        fill_bboxes.append(fill_bbox)
        inner_width = max(1, width if height == 1 else width - 2)
        inner_height = max(1, height if width == 1 else height - 2)
        fill_area = len(target_cells)
        fill_ratios.append(fill_area / max(1, inner_width * inner_height))
        interior_change_counts.append(len(target_cells))
        if target_cells:
            xs = [cell[0] for cell in target_cells]
            ys = [cell[1] for cell in target_cells]
            primary_axis_spans.append((max(xs) - min(xs) + 1) if orientation == "horizontal" else (max(ys) - min(ys) + 1))
            secondary_axis_spans.append((max(ys) - min(ys) + 1) if orientation == "horizontal" else (max(xs) - min(xs) + 1))
        else:
            primary_axis_spans.append(0)
            secondary_axis_spans.append(0)
    fill_range = (max(fill_ratios) - min(fill_ratios)) if fill_ratios else 0.0
    axis_consistency = 0.0
    if primary_axis_spans:
        primary_var = max(primary_axis_spans) - min(primary_axis_spans)
        secondary_var = max(secondary_axis_spans) - min(secondary_axis_spans)
        axis_consistency = primary_var / max(1.0, primary_var + secondary_var)
    interior_changes = sum(1 for left, right in zip(interior_change_counts, interior_change_counts[1:]) if left != right)
    compact_icon = aspect_ratio < 2.0
    scattered = any(ratio < 0.15 for ratio in fill_ratios) and fill_range < 0.1
    lacking_container = outer_stability < 0.55 and aspect_ratio < 3.0
    rejection_reason = None
    if compact_icon:
        rejection_reason = "compact_icon"
    elif scattered:
        rejection_reason = "scattered_fill"
    elif lacking_container:
        rejection_reason = "unstable_container"
    confidence = max(
        0.0,
        0.20 * min(1.0, aspect_ratio / 4.0)
        + 0.26 * outer_stability
        + 0.24 * min(1.0, fill_range * 2.0)
        + 0.18 * axis_consistency
        + 0.12 * min(1.0, interior_changes / max(1, len(fill_ratios) - 1))
    )
    return {
        "container_bbox": bbox,
        "fill_bbox_per_frame": tuple(fill_bboxes),
        "fill_ratio_per_frame": tuple(round(float(ratio), 4) for ratio in fill_ratios),
        "orientation": orientation,
        "confidence": confidence,
        "rejection_reason": rejection_reason,
        "source_hud_bbox": track["hud_bbox"],
    }


def _merge_adjacent_hud_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tracks) <= 1:
        return tracks
    merged = []
    used = [False] * len(tracks)
    for index, track in enumerate(tracks):
        if used[index]:
            continue
        current = track
        for other_index in range(index + 1, len(tracks)):
            if used[other_index]:
                continue
            other = tracks[other_index]
            if not _same_border_side(current, other):
                continue
            if _border_gap(current["hud_bbox"], other["hud_bbox"]) > 2:
                continue
            current = _merge_track_pair(current, other)
            used[other_index] = True
        used[index] = True
        merged.append(current)
    return merged


def _merge_track_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    per_frame = {}
    for component in left["components"] + right["components"]:
        frame_index = component["frame_index"]
        if frame_index not in per_frame:
            per_frame[frame_index] = component
            continue
        existing = per_frame[frame_index]
        cells = set(existing["cells"]) | set(component["cells"])
        value_histogram = Counter(existing["value_histogram"]) + Counter(component["value_histogram"])
        merged_component = {
            "bbox": _bbox(cells),
            "area": len(cells),
            "center": _center(cells),
            "value_histogram": dict(sorted(value_histogram.items())),
            "shape_signature": _shape_signature(cells),
            "touches_top": existing["touches_top"] or component["touches_top"],
            "touches_bottom": existing["touches_bottom"] or component["touches_bottom"],
            "touches_left": existing["touches_left"] or component["touches_left"],
            "touches_right": existing["touches_right"] or component["touches_right"],
            "distance_to_nearest_border": min(existing["distance_to_nearest_border"], component["distance_to_nearest_border"]),
            "frame_index": frame_index,
            "cells": cells,
            "avatar_overlap_ratio": max(existing["avatar_overlap_ratio"], component["avatar_overlap_ratio"]),
            "poi_overlap_ratio": max(existing["poi_overlap_ratio"], component["poi_overlap_ratio"]),
            "avatar_center_distance": min(
                value for value in (existing["avatar_center_distance"], component["avatar_center_distance"]) if value is not None
            ) if existing["avatar_center_distance"] is not None or component["avatar_center_distance"] is not None else None,
            "screen_anchor_score": max(existing["screen_anchor_score"], component["screen_anchor_score"]),
        }
        per_frame[frame_index] = merged_component
    merged_components = tuple(per_frame[key] for key in sorted(per_frame))
    return {
        "hud_bbox": _bbox_union([left["hud_bbox"], right["hud_bbox"]]),
        "support_step_indices": tuple(sorted(set(left["support_step_indices"]) | set(right["support_step_indices"]))),
        "value_candidates": tuple(dict.fromkeys(left["value_candidates"] + right["value_candidates"])),
        "hud_score": max(left["hud_score"], right["hud_score"]),
        "persistence_score": max(left["persistence_score"], right["persistence_score"]),
        "border_attachment_score": max(left["border_attachment_score"], right["border_attachment_score"]),
        "position_stability_score": max(left["position_stability_score"], right["position_stability_score"]),
        "components": merged_components,
        "cells": tuple(sorted({cell for component in merged_components for cell in component["cells"]}, key=lambda item: (item[1], item[0]))),
        "rejection_reason": None,
        "life_like": bool(left["life_like"] and right["life_like"]),
    }


def _same_border_side(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_component = left["components"][0]
    right_component = right["components"][0]
    for side in ("touches_top", "touches_bottom", "touches_left", "touches_right"):
        if left_component[side] and right_component[side]:
            return True
    return False


def _border_gap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    horizontal_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]) - 1)
    vertical_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]) - 1)
    return min(horizontal_gap, vertical_gap)


def _hud_track_link_score(previous: dict[str, Any], current: dict[str, Any]) -> float:
    center_distance = _distance(previous["center"], current["center"])
    spatial = max(0.0, 1.0 - (center_distance / 2.0))
    size_similarity = _ratio_similarity(previous["area"], current["area"])
    shape_similarity = _shape_similarity(previous["shape_signature"], current["shape_signature"])
    value_similarity = _value_histogram_similarity(previous["value_histogram"], current["value_histogram"])
    return 0.35 * spatial + 0.20 * size_similarity + 0.20 * shape_similarity + 0.25 * value_similarity


def _spatial_stability(centers: list[tuple[float, float]], bboxes: list[tuple[int, int, int, int]]) -> float:
    if len(centers) <= 1:
        return 0.6
    max_center_delta = max(_distance(left, right) for left, right in zip(centers, centers[1:]))
    bbox_jitter = max(_bbox_distance(left, right) for left, right in zip(bboxes, bboxes[1:]))
    return max(0.0, 1.0 - (0.6 * max_center_delta + 0.4 * bbox_jitter) / 3.0)


def _size_stability(components: tuple[dict[str, Any], ...]) -> float:
    if len(components) <= 1:
        return 0.6
    areas = [component["area"] for component in components]
    return min(_ratio_similarity(left, right) for left, right in zip(areas, areas[1:]))


def _shape_stability(components: tuple[dict[str, Any], ...]) -> float:
    if len(components) <= 1:
        return 0.6
    return sum(
        _shape_similarity(left["shape_signature"], right["shape_signature"])
        for left, right in zip(components, components[1:])
    ) / (len(components) - 1)


def _value_histogram_stability(histograms: list[dict[int, int]]) -> float:
    if len(histograms) <= 1:
        return 0.6
    scores = [_value_histogram_similarity(left, right) for left, right in zip(histograms, histograms[1:])]
    return sum(scores) / len(scores)


def _world_motion_penalty(components: tuple[dict[str, Any], ...], records: tuple[dict[str, Any], ...]) -> float:
    if len(components) <= 1:
        return 0.0
    penalties = []
    for previous, current in zip(components, components[1:]):
        delta = (current["center"][0] - previous["center"][0], current["center"][1] - previous["center"][1])
        action_delta = _action_delta_for_frame(records, current["frame_index"])
        movement = hypot(delta[0], delta[1])
        alignment = _vector_alignment(delta, (-action_delta[0], -action_delta[1]))
        penalties.append(max(0.0, min(1.0, 0.5 * min(1.0, movement / 1.0) + 0.5 * alignment)))
    return sum(penalties) / len(penalties)


def _avatar_correlation_penalty(
    components: tuple[dict[str, Any], ...],
    records: tuple[dict[str, Any], ...],
    avatar_track: dict[int, dict[str, object]],
) -> float:
    if len(components) <= 1:
        return 0.0
    penalties = []
    for previous, current in zip(components, components[1:]):
        avatar_previous = avatar_track.get(previous["frame_index"])
        avatar_current = avatar_track.get(current["frame_index"])
        if avatar_previous is None or avatar_current is None:
            continue
        avatar_delta = (
            avatar_current["center"][0] - avatar_previous["center"][0],
            avatar_current["center"][1] - avatar_previous["center"][1],
        )
        candidate_delta = (current["center"][0] - previous["center"][0], current["center"][1] - previous["center"][1])
        action_delta = _action_delta_for_frame(records, current["frame_index"])
        penalties.append(
            max(
                _vector_alignment(candidate_delta, avatar_delta),
                0.5 * _vector_alignment(candidate_delta, action_delta),
            )
        )
    return 0.0 if not penalties else sum(penalties) / len(penalties)


def _action_delta_for_frame(records: tuple[dict[str, Any], ...], frame_index: int) -> tuple[float, float]:
    if frame_index <= 0 or frame_index - 1 >= len(records):
        return (0.0, 0.0)
    record = records[frame_index - 1]
    if record["invalid_action"] or record["terminal"] or record["blocked_action"]:
        return (0.0, 0.0)
    return {
        "UP": (0.0, -1.0),
        "DOWN": (0.0, 1.0),
        "LEFT": (-1.0, 0.0),
        "RIGHT": (1.0, 0.0),
    }.get(record["action"], (0.0, 0.0))


def _outer_bbox_stability(components: tuple[dict[str, Any], ...], reference_bbox: tuple[int, int, int, int]) -> float:
    scores = []
    for component in components:
        scores.append(1.0 - min(1.0, _bbox_distance(component["bbox"], reference_bbox) / 2.0))
    return sum(scores) / len(scores)


def _stable_container_bbox(components: tuple[dict[str, Any], ...]) -> tuple[int, int, int, int]:
    return _bbox_union([component["bbox"] for component in components])


def _compact_objectness(components: tuple[dict[str, Any], ...]) -> float:
    scores = []
    for component in components:
        bbox = component["bbox"]
        scores.append(component["area"] / max(1, _bbox_area(bbox)))
    return sum(scores) / len(scores)


def _screen_anchor_score(touches: dict[str, bool], distance_to_border: int, frame: tuple[tuple[int, ...], ...]) -> float:
    anchors = sum(1 for value in touches.values() if value)
    max_distance = max(1.0, max(len(frame), len(frame[0])) / 4.0)
    return min(1.0, 0.6 * min(1.0, anchors / 2.0) + 0.4 * max(0.0, 1.0 - distance_to_border / max_distance))


def _touches(frame: tuple[tuple[int, ...], ...], component: set[tuple[int, int]]) -> dict[str, bool]:
    height = len(frame)
    width = len(frame[0]) if height else 0
    return {
        "top": any(y == 0 for _, y in component),
        "bottom": any(y == height - 1 for _, y in component),
        "left": any(x == 0 for x, _ in component),
        "right": any(x == width - 1 for x, _ in component),
    }


def _touches_border(component: set[tuple[int, int]], frame: tuple[tuple[int, ...], ...]) -> bool:
    touches = _touches(frame, component)
    return any(touches.values())


def _border_support(component: set[tuple[int, int]], frame: tuple[tuple[int, ...], ...]) -> float:
    return _screen_anchor_score(_touches(frame, component), _distance_to_border(component, frame), frame)


def _distance_to_border(component: set[tuple[int, int]], frame: tuple[tuple[int, ...], ...]) -> int:
    height = len(frame)
    width = len(frame[0]) if height else 0
    return min(min(x for x, _ in component), min(y for _, y in component), min(width - 1 - x for x, _ in component), min(height - 1 - y for _, y in component))


def _dominant_frame_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counter = Counter(int(value) for row in frame for value in row)
    return counter.most_common(1)[0][0]


def _connected_components(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], ...]:
    remaining = set(cells)
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            x, y = stack.pop()
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return tuple(components)


def _shape_signature(component: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in component)
    min_y = min(y for _, y in component)
    return tuple(sorted((x - min_x, y - min_y) for x, y in component))


def _shape_similarity(left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _value_histogram_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    overlap = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    total = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return overlap / max(1, total)


def _ratio_similarity(left: int | float, right: int | float) -> float:
    if left == right == 0:
        return 1.0
    return min(float(left), float(right)) / max(1.0, float(max(left, right)))


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _center(cells: set[tuple[int, int]]) -> tuple[float, float]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _distance(left: tuple[float, float], right: tuple[float, float] | None) -> float:
    if right is None:
        return 0.0
    return hypot(left[0] - right[0], left[1] - right[1])


def _bbox_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]), abs(left[2] - right[2]), abs(left[3] - right[3]))


def _cells_for_bbox(bbox: tuple[int, int, int, int], margin: int = 0) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y in range(bbox[1] - margin, bbox[3] + margin + 1)
        for x in range(bbox[0] - margin, bbox[2] + margin + 1)
        if x >= 0 and y >= 0
    }


def _parse_region_cells(regions: tuple[str, ...]) -> set[tuple[int, int]]:
    cells = set()
    for region in regions:
        if not isinstance(region, str):
            continue
        for part in region.split("|"):
            if not part.startswith("cell:") or "," not in part:
                continue
            x_str, y_str = part.removeprefix("cell:").split(",", 1)
            cells.add((int(x_str), int(y_str)))
    return cells


def _mask_overlap_ratio(component: set[tuple[int, int]], mask: set[tuple[int, int]]) -> float:
    if not component:
        return 0.0
    return len(component & mask) / len(component)


def _vector_alignment(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_mag = hypot(left[0], left[1])
    right_mag = hypot(right[0], right[1])
    if left_mag == 0.0 or right_mag == 0.0:
        return 0.0
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_mag * right_mag)
    return max(0.0, cosine)
