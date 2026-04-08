from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot
from typing import Any

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.boardObject import BoardObject
from v4_5.contracts.poiTypes import PoiRecord


@dataclass(frozen=True)
class PoiExtractionResult:
    selected_pois: tuple[PoiRecord, ...]
    ranked_candidates: tuple[PoiRecord, ...]
    diagnostics: dict[str, object]
    failure_reason: str | None = None


def extract_pois(frame: tuple[tuple[int, ...], ...]) -> tuple[BoardObject, ...]:
    if not frame:
        return ()
    background = _dominant_frame_value(frame)
    cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
    components = _connected_components(cells)
    return tuple(
        BoardObject(
            object_id=f"poi:{index}",
            object_type="poi",
            bbox=_bbox(component),
            center=_center_from_bbox(_bbox(component)),
            position_x=_center_from_bbox(_bbox(component))[0],
            position_y=_center_from_bbox(_bbox(component))[1],
            color=_dominant_component_value(frame, component),
        )
        for index, component in enumerate(components)
    )


def extract_poi_candidates(
    *,
    bootstrap_transition_records: tuple[Any, ...],
    avatar_detection_result: AvatarDetectionResult | None,
    hud_exclusion_regions: tuple[str, ...] = (),
    life_exclusion_regions: tuple[str, ...] = (),
    progress_exclusion_regions: tuple[str, ...] = (),
    avatar_bbox_margin: int = 0,
) -> PoiExtractionResult:
    records = tuple(_normalize_record(record) for record in bootstrap_transition_records)
    frames = _ordered_frames(records)
    avatar_track = _avatar_track_for_frames(frames, avatar_detection_result, margin=avatar_bbox_margin)
    avatar_cells_per_step = {index: set(item["cells"]) for index, item in avatar_track.items()}
    hud_excluded = _parse_region_cells(hud_exclusion_regions + life_exclusion_regions + progress_exclusion_regions)
    per_frame_components: list[dict[str, Any]] = []
    rejected_for_overlap = 0
    rejected_for_motion = 0
    for frame_index, frame in enumerate(frames):
        if not frame:
            per_frame_components.append({"frame_index": frame_index, "components": ()})
            continue
        background = _dominant_frame_value(frame)
        non_background = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
        components = []
        avatar_exclusion = set(avatar_cells_per_step.get(frame_index, set())) | set(hud_excluded)
        visible_non_avatar = {cell for cell in non_background if cell not in avatar_exclusion}
        avatar_center = None if frame_index not in avatar_track else avatar_track[frame_index]["center"]
        avatar_move = _avatar_movement_for_frame(records, frame_index)
        avatar_values = set(getattr(avatar_detection_result, "avatar_value_candidates", ()) or ())
        for component in _connected_components(visible_non_avatar):
            overlap_ratio = _mask_overlap_ratio(component, avatar_exclusion)
            record = {
                "bbox": _bbox(component),
                "area": len(component),
                "center": _center(component),
                "value_histogram": dict(sorted(Counter(int(frame[y][x]) for x, y in component).items())),
                "shape_signature": _shape_signature(component),
                "frame_index": frame_index,
                "cells": component,
                "border_contact": _border_contact(component, frame),
                "rejected_as_avatar_overlap": overlap_ratio >= 0.2,
                "avatar_overlap_ratio": overlap_ratio,
                "avatar_center_distance": _distance(_center(component), avatar_center) if avatar_center is not None else None,
                "motion_alignment_with_avatar": _movement_alignment(_component_shift_hint(component, frame, background), avatar_move),
                "avatar_value_match": _histogram_overlap(dict(sorted(Counter(int(frame[y][x]) for x, y in component).items())), avatar_values),
            }
            if record["rejected_as_avatar_overlap"]:
                rejected_for_overlap += 1
                continue
            components.append(record)
        per_frame_components.append({"frame_index": frame_index, "components": tuple(components)})
    tracks = _build_non_avatar_tracks(per_frame_components, records, avatar_detection_result)
    ranked_tracks = sorted((_score_track(track, records, avatar_track) for track in tracks), key=lambda item: item["poi_score"], reverse=True)
    ranked_candidates = []
    selected_pois = []
    best_scores = []
    for track in ranked_tracks:
        failure_reason = None
        if track["rejected_as_avatar_overlap"]:
            failure_reason = "avatar_overlap"
        elif track["motion_correlation_rejected"]:
            failure_reason = "motion_correlation"
            rejected_for_motion += 1
        candidate = PoiRecord(
            bbox=track["poi_bbox"],
            center=track["poi_center"],
            colors=tuple(track["value_candidates"]),
            support_step_indices=tuple(track["support_step_indices"]),
            value_candidates=tuple(track["value_candidates"]),
            stability_score=float(track["stability_score"]),
            reachability_score=float(track["reachability_score"]),
            poi_score=float(track["poi_score"]),
            rejected_as_avatar_overlap=bool(track["rejected_as_avatar_overlap"]),
            failure_reason=failure_reason,
        )
        ranked_candidates.append(candidate)
        best_scores.append(round(float(track["poi_score"]), 4))
        if failure_reason is None and not track["rejected_as_avatar_overlap"]:
            selected_pois.append(candidate)
    diagnostics = {
        "per_frame_component_count_after_avatar_exclusion": {
            int(item["frame_index"]): len(item["components"]) for item in per_frame_components
        },
        "non_avatar_track_count": len(ranked_tracks),
        "rejected_for_avatar_overlap_count": int(rejected_for_overlap),
        "rejected_for_motion_correlation_count": int(rejected_for_motion),
        "best_candidate_scores": tuple(best_scores[:5]),
        "selected_poi_count": len(selected_pois),
        "failure_reason": None if selected_pois else ("no_poi_candidates" if ranked_candidates else "no_non_avatar_components"),
    }
    return PoiExtractionResult(
        selected_pois=tuple(selected_pois),
        ranked_candidates=tuple(ranked_candidates),
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
    avatar_values = set(getattr(avatar_result, "avatar_value_candidates", ()) or ())
    track = {}
    previous_center = None
    for frame_index, frame in enumerate(frames):
        chosen_cells = None
        if frame and avatar_values:
            candidate_cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) in avatar_values}
            components = _connected_components(candidate_cells)
            if components:
                chosen_cells = min(
                    components,
                    key=lambda component: (
                        _distance(_center(component), previous_center) if previous_center is not None else _distance(_center(component), avatar_result.avatar_center),
                        -len(component),
                    ),
                )
        if chosen_cells is None:
            chosen_cells = set(_cells_for_bbox(avatar_result.avatar_bbox, margin=margin))
        bbox = _bbox(chosen_cells)
        center = _center(chosen_cells)
        previous_center = center
        track[frame_index] = {"cells": chosen_cells, "bbox": bbox, "center": center}
    return track


def _avatar_movement_for_frame(records: tuple[dict[str, Any], ...], frame_index: int) -> tuple[float, float]:
    if frame_index <= 0 or frame_index - 1 >= len(records):
        return (0.0, 0.0)
    record = records[frame_index - 1]
    if record["invalid_action"] or record["terminal"]:
        return (0.0, 0.0)
    return {
        "UP": (0.0, -1.0),
        "DOWN": (0.0, 1.0),
        "LEFT": (-1.0, 0.0),
        "RIGHT": (1.0, 0.0),
    }.get(record["action"], (0.0, 0.0))


def _build_non_avatar_tracks(per_frame_components: list[dict[str, Any]], records: tuple[dict[str, Any], ...], avatar_result: AvatarDetectionResult | None) -> tuple[dict[str, Any], ...]:
    tracks: list[dict[str, Any]] = []
    for frame_group in per_frame_components:
        for component in frame_group["components"]:
            best_track = None
            best_link_score = 0.0
            for track in tracks:
                previous = track["components"][-1]
                if component["frame_index"] != previous["frame_index"] + 1:
                    continue
                link_score = _track_link_score(previous, component)
                if link_score > best_link_score:
                    best_link_score = link_score
                    best_track = track
            if best_track is not None and best_link_score >= 0.45:
                best_track["components"].append(component)
                best_track["link_scores"].append(best_link_score)
            else:
                tracks.append({"components": [component], "link_scores": []})
    return tuple({"components": tuple(track["components"]), "link_scores": tuple(track["link_scores"])} for track in tracks)


def _score_track(track: dict[str, Any], records: tuple[dict[str, Any], ...], avatar_track: dict[int, dict[str, object]]) -> dict[str, Any]:
    components = tuple(track["components"])
    centers = [component["center"] for component in components]
    bboxes = [component["bbox"] for component in components]
    values = [component["value_histogram"] for component in components]
    support_steps = tuple(int(component["frame_index"]) for component in components)
    stability_score = _spatial_stability(centers, bboxes)
    value_stability = _value_histogram_stability(values)
    objectness = _compact_objectness(components)
    relative_motion = _relative_motion_score(components, records, avatar_track)
    motion_penalty = _motion_correlation_penalty(components, records)
    border_only = all(component["border_contact"] for component in components)
    reachability_score = max(0.0, min(1.0, 0.55 * relative_motion + 0.45 * stability_score))
    poi_score = max(
        0.0,
        0.30 * min(1.0, len(components) / 3.0)
        + 0.22 * stability_score
        + 0.18 * value_stability
        + 0.12 * objectness
        + 0.18 * relative_motion
        - 0.25 * motion_penalty,
    )
    if border_only and len(components) <= 1:
        poi_score = max(0.0, poi_score - 0.35)
    representative = min(components, key=lambda component: component["avatar_overlap_ratio"])
    value_candidates = tuple(int(value) for value, _ in Counter(
        value
        for histogram in values
        for value, count in histogram.items()
        for _ in range(int(count))
    ).most_common(3))
    rejected_overlap = any(component["avatar_overlap_ratio"] >= 0.2 for component in components)
    near_avatar_count = sum(
        1 for component in components if component["avatar_center_distance"] is not None and component["avatar_center_distance"] <= 1.5
    )
    strong_avatar_value_match = any(component["avatar_value_match"] >= 0.75 for component in components)
    residue_count = sum(1 for component in components if component["area"] <= 2 and component["avatar_center_distance"] is not None and component["avatar_center_distance"] <= 1.0)
    residue_like = residue_count >= 2
    motion_correlation_rejected = (
        motion_penalty >= 0.65
        or (len(components) > 1 and stability_score < 0.8)
        or (border_only and len(components) <= 1)
        or (near_avatar_count >= max(2, len(components)) and strong_avatar_value_match)
        or residue_like
    )
    return {
        "poi_bbox": representative["bbox"],
        "poi_center": representative["center"],
        "support_step_indices": support_steps,
        "value_candidates": value_candidates,
        "stability_score": stability_score,
        "reachability_score": reachability_score,
        "poi_score": poi_score,
        "rejected_as_avatar_overlap": rejected_overlap,
        "motion_correlation_rejected": motion_correlation_rejected,
    }


def _track_link_score(previous: dict[str, Any], current: dict[str, Any]) -> float:
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


def _value_histogram_stability(histograms: list[dict[int, int]]) -> float:
    if len(histograms) <= 1:
        return 0.6
    scores = [_value_histogram_similarity(left, right) for left, right in zip(histograms, histograms[1:])]
    return sum(scores) / len(scores)


def _compact_objectness(components: tuple[dict[str, Any], ...]) -> float:
    scores = []
    for component in components:
        bbox = component["bbox"]
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        scores.append(component["area"] / max(1, bbox_area))
    return sum(scores) / len(scores)


def _relative_motion_score(components: tuple[dict[str, Any], ...], records: tuple[dict[str, Any], ...], avatar_track: dict[int, dict[str, object]]) -> float:
    if len(components) <= 1:
        return 0.5
    scores = []
    for previous, current in zip(components, components[1:]):
        avatar_move = _avatar_movement_for_frame(records, current["frame_index"])
        if avatar_move == (0.0, 0.0):
            continue
        candidate_move = (current["center"][0] - previous["center"][0], current["center"][1] - previous["center"][1])
        static_score = max(0.0, 1.0 - (hypot(candidate_move[0], candidate_move[1]) / 1.5))
        prev_avatar = avatar_track.get(previous["frame_index"], {})
        curr_avatar = avatar_track.get(current["frame_index"], {})
        prev_vector = None if "center" not in prev_avatar else (previous["center"][0] - prev_avatar["center"][0], previous["center"][1] - prev_avatar["center"][1])
        curr_vector = None if "center" not in curr_avatar else (current["center"][0] - curr_avatar["center"][0], current["center"][1] - curr_avatar["center"][1])
        relative_consistency = 0.5
        if prev_vector is not None and curr_vector is not None:
            expected = (prev_vector[0] - avatar_move[0], prev_vector[1] - avatar_move[1])
            relative_consistency = max(0.0, 1.0 - (_distance(curr_vector, expected) / 2.0))
        relative_score = 0.5 * static_score + 0.5 * relative_consistency
        scores.append(max(0.0, min(1.0, 0.7 * static_score + 0.3 * relative_score)))
    return sum(scores) / len(scores) if scores else 0.5


def _motion_correlation_penalty(components: tuple[dict[str, Any], ...], records: tuple[dict[str, Any], ...]) -> float:
    if len(components) <= 1:
        return 0.0
    penalties = []
    for previous, current in zip(components, components[1:]):
        candidate_move = (current["center"][0] - previous["center"][0], current["center"][1] - previous["center"][1])
        avatar_move = _avatar_movement_for_frame(records, current["frame_index"])
        penalties.append(_movement_alignment(candidate_move, avatar_move))
    return sum(penalties) / len(penalties) if penalties else 0.0


def _movement_alignment(candidate_move: tuple[float, float], avatar_move: tuple[float, float]) -> float:
    if avatar_move == (0.0, 0.0):
        return 0.0
    candidate_mag = hypot(candidate_move[0], candidate_move[1])
    avatar_mag = hypot(avatar_move[0], avatar_move[1])
    if candidate_mag == 0.0 or avatar_mag == 0.0:
        return 0.0
    dot = candidate_move[0] * avatar_move[0] + candidate_move[1] * avatar_move[1]
    return max(0.0, min(1.0, dot / (candidate_mag * avatar_mag)))


def _component_shift_hint(component: set[tuple[int, int]], frame: tuple[tuple[int, ...], ...], background: int) -> tuple[float, float]:
    del frame, background
    return (0.0, 0.0)


def _mask_overlap_ratio(component: set[tuple[int, int]], exclusion_mask: set[tuple[int, int]]) -> float:
    if not component:
        return 0.0
    return len(component & exclusion_mask) / len(component)


def _parse_region_cells(regions: tuple[str, ...]) -> set[tuple[int, int]]:
    cells = set()
    for region in regions:
        if not isinstance(region, str):
            continue
        parts = region.split("|")
        for part in parts:
            if not part.startswith("cell:") or "," not in part:
                continue
            x_str, y_str = part.removeprefix("cell:").split(",", 1)
            cells.add((int(x_str), int(y_str)))
    return cells


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


def _dominant_frame_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counts = Counter(int(value) for row in frame for value in row)
    return int(counts.most_common(1)[0][0]) if counts else 0


def _dominant_component_value(frame: tuple[tuple[int, ...], ...], component: set[tuple[int, int]]) -> int:
    return int(Counter(int(frame[y][x]) for x, y in component).most_common(1)[0][0])


def _shape_signature(component: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in component)
    min_y = min(y for _, y in component)
    return tuple(sorted((x - min_x, y - min_y) for x, y in component))


def _shape_similarity(left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _value_histogram_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return 0.0
    shared = sum(min(left.get(value, 0), right.get(value, 0)) for value in set(left) | set(right))
    return shared / max(left_total, right_total)


def _histogram_overlap(histogram: dict[int, int], candidate_values: set[int]) -> float:
    if not histogram or not candidate_values:
        return 0.0
    total = sum(histogram.values())
    return sum(count for value, count in histogram.items() if int(value) in candidate_values) / max(1, total)


def _cells_for_bbox(bbox: tuple[int, int, int, int], *, margin: int) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y in range(bbox[1] - margin, bbox[3] + margin + 1)
        for x in range(bbox[0] - margin, bbox[2] + margin + 1)
    }


def _bbox(component: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    return (min(xs), min(ys), max(xs), max(ys))


def _center(component: set[tuple[int, int]]) -> tuple[float, float]:
    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _border_contact(component: set[tuple[int, int]], frame: tuple[tuple[int, ...], ...]) -> bool:
    height = len(frame)
    width = len(frame[0]) if height else 0
    return all(x in {0, width - 1} or y in {0, height - 1} for x, y in component)


def _distance(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    return hypot(left[0] - right[0], left[1] - right[1])


def _bbox_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]), abs(left[2] - right[2]), abs(left[3] - right[3]))


def _ratio_similarity(left: int, right: int) -> float:
    denom = max(abs(int(left)), abs(int(right)), 1)
    return max(0.0, 1.0 - abs(int(left) - int(right)) / denom)
