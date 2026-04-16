from __future__ import annotations

from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import POICandidate, ProbeTransitionRecord


def build_contact_logs(
    *,
    episode_index: int,
    transitions: tuple[ProbeTransitionRecord, ...],
    poi_candidates: tuple[POICandidate, ...],
    avatar_bbox: tuple[int, int, int, int] | None,
) -> tuple[dict[str, Any], ...]:
    if avatar_bbox is None:
        return ()
    logs: list[dict[str, Any]] = []
    for record in transitions:
        for candidate in poi_candidates:
            poi_bbox = candidate.bbox
            distance = _bbox_distance(avatar_bbox, poi_bbox)
            event_types: list[str] = []
            if distance <= 1.0:
                event_types.append("adjacent")
            if _bbox_iou(avatar_bbox, poi_bbox) > 0.0:
                event_types.append("touch_or_overlap")
            if record.reward_before is not None and record.reward_after is not None and record.reward_before != record.reward_after and distance <= 2.0:
                event_types.append("reward_change_near_poi")
            if int(record.levels_completed_after) != int(record.levels_completed_before) and distance <= 2.0:
                event_types.append("level_change_near_poi")

            if record.pre_frame is not None and record.post_frame is not None and distance <= 2.0:
                pre_seen = _bbox_non_background(record.pre_frame, poi_bbox)
                post_seen = _bbox_non_background(record.post_frame, poi_bbox)
                if pre_seen and not post_seen:
                    event_types.append("disappeared_after_approach")
                if (not pre_seen) and post_seen:
                    event_types.append("appeared_after_approach")

            for event_type in event_types:
                logs.append(
                    {
                        "episode_index": int(episode_index),
                        "step_index": int(record.step_index),
                        "poi_id": candidate.poi_id,
                        "event_type": event_type,
                        "avatar_bbox": avatar_bbox,
                        "poi_bbox": poi_bbox,
                        "reward_before": record.reward_before,
                        "reward_after": record.reward_after,
                        "levels_completed_before": int(record.levels_completed_before),
                        "levels_completed_after": int(record.levels_completed_after),
                    }
                )
    logs.sort(key=lambda item: (item["episode_index"], item["step_index"], item["poi_id"], item["event_type"]))
    return tuple(logs)


def _bbox_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_c = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    right_c = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    return sqrt((left_c[0] - right_c[0]) ** 2 + (left_c[1] - right_c[1]) ** 2)


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


def _bbox_non_background(frame: tuple[tuple[int, ...], ...], bbox: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bbox
    values = []
    for y in range(max(0, y0), min(len(frame), y1 + 1)):
        row = frame[y]
        for x in range(max(0, x0), min(len(row), x1 + 1)):
            values.append(int(row[x]))
    if not values:
        return False
    return any(int(value) != 0 for value in values)
