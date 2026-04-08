from __future__ import annotations

from collections import Counter
from math import hypot
from typing import Any

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.boardObject import BoardObject


_ACTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
_MAX_STEP_CANDIDATES = 3
_MIN_SUPPORT_STEPS = 2
_TRACK_SEPARATION_MARGIN = 0.08


def extract_avatar(frame: tuple[tuple[int, ...], ...]) -> BoardObject | None:
    if not frame:
        return None
    background = _dominant_frame_value(frame)
    cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != background}
    components = _connected_components(cells)
    if len(components) != 1:
        return None
    bbox = _bbox(components[0])
    center = _center_from_bbox(bbox)
    dominant_value = Counter(int(frame[y][x]) for x, y in components[0]).most_common(1)[0][0]
    return BoardObject(
        object_id="avatar:0",
        object_type="avatar",
        bbox=bbox,
        center=center,
        position_x=center[0],
        position_y=center[1],
        color=int(dominant_value),
    )


def extract_avatar_from_transition_records(
    records: tuple[Any, ...],
    *,
    used_fallback: bool = False,
) -> AvatarDetectionResult:
    normalized_records = tuple(_normalize_record(record) for record in records)
    per_step_candidates: list[dict[str, Any]] = []
    valid_step_count = 0
    for record in normalized_records:
        step_index = int(record["step_index"])
        if record["terminal"] or record["invalid_action"] or record["pre_frame"] is None or record["post_frame"] is None:
            per_step_candidates.append({"step_index": step_index, "candidates": ()})
            continue
        valid_step_count += 1
        components = _extract_changed_components(record)
        scored = sorted(
            (_score_candidate(component, record["action"], blocked_action=record["blocked_action"]) for component in components),
            key=lambda item: item["score"],
            reverse=True,
        )
        kept = tuple(item for item in scored[:_MAX_STEP_CANDIDATES] if item["score"] >= 0.35)
        per_step_candidates.append({"step_index": step_index, "candidates": kept})
    candidate_nodes = [
        {"node_id": node_id, **candidate}
        for node_id, candidate in enumerate(
            candidate
            for step in per_step_candidates
            for candidate in step["candidates"]
        )
    ]
    diagnostics = {
        "per_step_candidate_count": {int(step["step_index"]): len(step["candidates"]) for step in per_step_candidates},
        "top_candidate_scores_per_step": {
            int(step["step_index"]): [round(float(candidate["score"]), 4) for candidate in step["candidates"]]
            for step in per_step_candidates
        },
        "total_track_count": 0,
        "best_track_support_steps": (),
        "best_track_confidence": 0.0,
        "failure_reason": None,
    }
    if not candidate_nodes:
        diagnostics["failure_reason"] = "no_moving_candidate"
        return AvatarDetectionResult(failure_reason="no_moving_candidate", used_fallback=used_fallback, diagnostics=diagnostics)
    track_candidates = _build_tracks(candidate_nodes)
    diagnostics["total_track_count"] = len(track_candidates)
    if not track_candidates:
        diagnostics["failure_reason"] = "no_moving_candidate"
        return AvatarDetectionResult(failure_reason="no_moving_candidate", used_fallback=used_fallback, diagnostics=diagnostics)
    best_track = track_candidates[0]
    second_track = track_candidates[1] if len(track_candidates) > 1 else None
    diagnostics["best_track_support_steps"] = tuple(best_track["support_step_indices"])
    diagnostics["best_track_confidence"] = round(float(best_track["confidence"]), 4)
    failure_reason = _failure_reason_for_tracks(
        best_track=best_track,
        second_track=second_track,
        valid_step_count=valid_step_count,
    )
    if failure_reason is not None:
        diagnostics["failure_reason"] = failure_reason
        return AvatarDetectionResult(
            support_actions=tuple(best_track["support_actions"]),
            support_step_indices=tuple(best_track["support_step_indices"]),
            confidence=float(best_track["confidence"]),
            avatar_value_candidates=tuple(best_track["avatar_value_candidates"]),
            failure_reason=failure_reason,
            used_fallback=used_fallback,
            diagnostics=diagnostics,
        )
    diagnostics["failure_reason"] = None
    return AvatarDetectionResult(
        avatar_bbox=best_track["avatar_bbox"],
        avatar_center=best_track["avatar_center"],
        support_actions=tuple(best_track["support_actions"]),
        support_step_indices=tuple(best_track["support_step_indices"]),
        confidence=float(best_track["confidence"]),
        avatar_value_candidates=tuple(best_track["avatar_value_candidates"]),
        failure_reason=None,
        used_fallback=used_fallback,
        diagnostics=diagnostics,
    )


def _normalize_record(record: Any) -> dict[str, Any]:
    return {
        "pre_frame": _normalize_frame(_record_field(record, "pre_observation") or _record_field(record, "pre_observation_ref")),
        "post_frame": _normalize_frame(_record_field(record, "post_observation") or _record_field(record, "post_observation_ref")),
        "action": str(_record_field(record, "action", "") or ""),
        "invalid_action": bool(_record_field(record, "invalid_action", False)),
        "blocked_action": bool(_record_field(record, "blocked_action", False)),
        "terminal": bool(_record_field(record, "terminal", False)),
        "step_index": int(_record_field(record, "step_index", 0) or 0),
    }


def _record_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


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


def _extract_changed_components(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    pre_frame = record["pre_frame"]
    post_frame = record["post_frame"]
    if pre_frame is None or post_frame is None or not pre_frame or not post_frame:
        return ()
    changed = {
        (x, y)
        for y in range(min(len(pre_frame), len(post_frame)))
        for x in range(min(len(pre_frame[y]), len(post_frame[y])))
        if int(pre_frame[y][x]) != int(post_frame[y][x])
    }
    if not changed:
        return ()
    pre_background = _dominant_frame_value(pre_frame)
    post_background = _dominant_frame_value(post_frame)
    components = []
    for cells in _connected_components(changed):
        pre_histogram = Counter(int(pre_frame[y][x]) for x, y in cells)
        post_histogram = Counter(int(post_frame[y][x]) for x, y in cells)
        pre_non_background_cells = tuple(sorted((x, y) for x, y in cells if int(pre_frame[y][x]) != pre_background))
        post_non_background_cells = tuple(sorted((x, y) for x, y in cells if int(post_frame[y][x]) != post_background))
        center_pre = _center(pre_non_background_cells or tuple(sorted(cells)))
        center_post = _center(post_non_background_cells or tuple(sorted(cells)))
        components.append(
            {
                "step_index": record["step_index"],
                "action": record["action"],
                "blocked_action": record["blocked_action"],
                "bbox": _bbox(cells),
                "area": len(cells),
                "center_pre": center_pre,
                "center_post": center_post,
                "pre_value_histogram": dict(sorted(pre_histogram.items())),
                "post_value_histogram": dict(sorted(post_histogram.items())),
                "pre_non_background_cells": pre_non_background_cells,
                "post_non_background_cells": post_non_background_cells,
                "observed_dx": center_post[0] - center_pre[0],
                "observed_dy": center_post[1] - center_pre[1],
                "pre_bbox": _bbox(pre_non_background_cells) if pre_non_background_cells else None,
                "post_bbox": _bbox(post_non_background_cells) if post_non_background_cells else None,
                "pre_background_value": pre_background,
                "post_background_value": post_background,
            }
        )
    return tuple(components)


def _score_candidate(component: dict[str, Any], action: str, *, blocked_action: bool) -> dict[str, Any]:
    expected_dx, expected_dy = _ACTION_DELTAS.get(action, (0, 0))
    observed_dx = float(component["observed_dx"])
    observed_dy = float(component["observed_dy"])
    movement_magnitude = hypot(observed_dx, observed_dy)
    direction_score = 0.5 if blocked_action else _direction_score(observed_dx, observed_dy, expected_dx, expected_dy)
    magnitude_score = 0.5 if blocked_action else max(0.0, 1.0 - abs(movement_magnitude - 1.0))
    pre_cells = tuple(component["pre_non_background_cells"])
    post_cells = tuple(component["post_non_background_cells"])
    shape_score = _shape_similarity(pre_cells, post_cells)
    area_score = _similarity_ratio(len(pre_cells), len(post_cells))
    dominant_values = _dominant_value_candidates(component)
    value_score = _value_stability_score(component["pre_value_histogram"], component["post_value_histogram"], dominant_values)
    fragmentation_score = 1.0 / max(_fragmentation(pre_cells), _fragmentation(post_cells), 1)
    score = (
        0.32 * direction_score
        + 0.22 * magnitude_score
        + 0.18 * shape_score
        + 0.14 * area_score
        + 0.10 * value_score
        + 0.04 * fragmentation_score
    )
    return {
        **component,
        "score": score,
        "direction_score": direction_score,
        "magnitude_score": magnitude_score,
        "shape_score": shape_score,
        "area_score": area_score,
        "value_score": value_score,
        "fragmentation_score": fragmentation_score,
        "dominant_values": dominant_values,
        "action_agreement": 0.5 * direction_score + 0.5 * magnitude_score,
        "size": max(len(pre_cells), len(post_cells), 1),
    }


def _build_tracks(candidate_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_node: dict[int, dict[str, Any]] = {}
    for candidate in candidate_nodes:
        best_path = {
            "candidates": (candidate,),
            "transition_links": (),
            "score_total": candidate["score"],
        }
        for previous_id, previous_path in tuple(best_by_node.items()):
            previous = previous_path["candidates"][-1]
            link_score = _track_link_score(previous, candidate)
            if link_score < 0.35:
                continue
            proposal_score = previous_path["score_total"] + candidate["score"] + link_score
            if proposal_score > best_path["score_total"]:
                best_path = {
                    "candidates": previous_path["candidates"] + (candidate,),
                    "transition_links": previous_path["transition_links"] + (link_score,),
                    "score_total": proposal_score,
                }
        best_by_node[candidate["node_id"]] = best_path
    unique_paths: dict[tuple[int, ...], dict[str, Any]] = {}
    for path in best_by_node.values():
        key = tuple(int(candidate["node_id"]) for candidate in path["candidates"])
        unique_paths[key] = _score_track(path["candidates"], path["transition_links"], path["score_total"])
    return sorted(unique_paths.values(), key=lambda item: item["confidence"], reverse=True)


def _score_track(candidates: tuple[dict[str, Any], ...], transition_links: tuple[float, ...], score_total: float) -> dict[str, Any]:
    support_steps = tuple(int(candidate["step_index"]) for candidate in candidates)
    support_actions = tuple(str(candidate["action"]) for candidate in candidates)
    action_agreement_rate = sum(float(candidate["action_agreement"]) for candidate in candidates) / max(len(candidates), 1)
    continuity = sum(float(value) for value in transition_links) / max(len(transition_links), 1) if transition_links else 0.75
    sizes = [float(candidate["size"]) for candidate in candidates]
    stable_size = 1.0 - ((max(sizes) - min(sizes)) / max(max(sizes), 1.0))
    dominant_signatures = [tuple(candidate["dominant_values"]) for candidate in candidates if candidate["dominant_values"]]
    signature_mode = Counter(dominant_signatures).most_common(1)[0][0] if dominant_signatures else ()
    signature_stability = (
        Counter(dominant_signatures).most_common(1)[0][1] / len(dominant_signatures)
        if dominant_signatures
        else 0.0
    )
    support_strength = min(1.0, len(candidates) / 3.0)
    mean_candidate_score = sum(float(candidate["score"]) for candidate in candidates) / max(len(candidates), 1)
    base_confidence = (
        0.42 * support_strength
        + 0.20 * action_agreement_rate
        + 0.16 * continuity
        + 0.11 * stable_size
        + 0.11 * signature_stability
    )
    confidence = max(0.0, min(1.0, base_confidence * (0.35 + (0.65 * support_strength))))
    representative = max(candidates, key=lambda candidate: candidate["score"])
    avatar_bbox = representative["post_bbox"] or representative["pre_bbox"] or representative["bbox"]
    avatar_center = _center_from_bbox(avatar_bbox)
    value_candidates = tuple(int(value) for value in signature_mode) if signature_mode else tuple(
        int(value)
        for value, _ in Counter(
            value
            for candidate in candidates
            for value in candidate["dominant_values"]
        ).most_common(3)
    )
    return {
        "candidates": candidates,
        "support_step_indices": support_steps,
        "support_actions": support_actions,
        "action_agreement_rate": action_agreement_rate,
        "continuity": continuity,
        "stable_size": stable_size,
        "signature_stability": signature_stability,
        "confidence": confidence,
        "score_total": score_total + mean_candidate_score,
        "avatar_bbox": avatar_bbox,
        "avatar_center": avatar_center,
        "avatar_value_candidates": value_candidates,
    }


def _failure_reason_for_tracks(*, best_track: dict[str, Any], second_track: dict[str, Any] | None, valid_step_count: int) -> str | None:
    if len(best_track["support_step_indices"]) < min(_MIN_SUPPORT_STEPS, max(valid_step_count, 1)):
        return "insufficient_support"
    if best_track["action_agreement_rate"] < 0.55 or best_track["continuity"] < 0.45:
        return "motion_inconsistency"
    if (
        second_track is not None
        and abs(float(best_track["confidence"]) - float(second_track["confidence"])) <= _TRACK_SEPARATION_MARGIN
        and not (
            set(second_track["support_step_indices"]) < set(best_track["support_step_indices"])
            and _track_center_distance(best_track, second_track) <= 1.0
        )
        and _track_center_distance(best_track, second_track) > 1.0
    ):
        return "multiple_near_equal_tracks"
    return None


def _track_center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return hypot(
        float(left["avatar_center"][0]) - float(right["avatar_center"][0]),
        float(left["avatar_center"][1]) - float(right["avatar_center"][1]),
    )


def _track_link_score(previous: dict[str, Any], current: dict[str, Any]) -> float:
    if int(current["step_index"]) <= int(previous["step_index"]):
        return 0.0
    gap_penalty = max(0.0, 1.0 - 0.2 * (int(current["step_index"]) - int(previous["step_index"]) - 1))
    continuity_distance = hypot(
        float(current["center_pre"][0]) - float(previous["center_post"][0]),
        float(current["center_pre"][1]) - float(previous["center_post"][1]),
    )
    spatial_continuity = max(0.0, 1.0 - (continuity_distance / 2.5))
    size_similarity = _similarity_ratio(int(previous["size"]), int(current["size"]))
    previous_values = set(previous["dominant_values"])
    current_values = set(current["dominant_values"])
    value_similarity = 1.0 if previous_values and previous_values == current_values else (0.6 if previous_values & current_values else 0.0)
    return gap_penalty * (0.45 * spatial_continuity + 0.30 * size_similarity + 0.25 * value_similarity)


def _dominant_frame_value(frame: tuple[tuple[int, ...], ...]) -> int:
    counts = Counter(int(value) for row in frame for value in row)
    return int(counts.most_common(1)[0][0]) if counts else 0


def _dominant_value_candidates(component: dict[str, Any]) -> tuple[int, ...]:
    pre_values = Counter(
        int(value)
        for value, count in component["pre_value_histogram"].items()
        if int(value) != int(component["pre_background_value"])
        for _ in range(int(count))
    )
    post_values = Counter(
        int(value)
        for value, count in component["post_value_histogram"].items()
        if int(value) != int(component["post_background_value"])
        for _ in range(int(count))
    )
    merged = pre_values + post_values
    return tuple(int(value) for value, _ in merged.most_common(3))


def _value_stability_score(pre_histogram: dict[int, int], post_histogram: dict[int, int], dominant_values: tuple[int, ...]) -> float:
    if not dominant_values:
        return 0.0
    pre_values = {int(value) for value, count in pre_histogram.items() if int(count) > 0}
    post_values = {int(value) for value, count in post_histogram.items() if int(count) > 0}
    overlap = set(dominant_values) & pre_values & post_values
    if overlap:
        return 1.0
    if set(dominant_values) & (pre_values | post_values):
        return 0.5
    return 0.0


def _direction_score(observed_dx: float, observed_dy: float, expected_dx: int, expected_dy: int) -> float:
    magnitude = hypot(observed_dx, observed_dy)
    if magnitude <= 0.0:
        return 0.0
    if expected_dx == 0 and expected_dy == 0:
        return 0.0
    dot = (observed_dx * expected_dx) + (observed_dy * expected_dy)
    return max(0.0, min(1.0, dot / magnitude))


def _shape_similarity(left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...]) -> float:
    if not left or not right:
        return 0.0
    normalized_left = _normalize_shape(left)
    normalized_right = _normalize_shape(right)
    union = normalized_left | normalized_right
    if not union:
        return 0.0
    return len(normalized_left & normalized_right) / len(union)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return {(x - min_x, y - min_y) for x, y in cells}


def _fragmentation(cells: tuple[tuple[int, int], ...]) -> int:
    if not cells:
        return 0
    return len(_connected_components(set(cells)))


def _similarity_ratio(left: int, right: int) -> float:
    denom = max(abs(int(left)), abs(int(right)), 1)
    return max(0.0, 1.0 - (abs(int(left) - int(right)) / denom))


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


def _bbox(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    points = tuple(cells)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
