from __future__ import annotations

from collections import Counter


def _bbox_contains(bbox: dict[str, int], centroid: list[float]) -> bool:
    return bbox["x1"] <= centroid[0] <= bbox["x2"] and bbox["y1"] <= centroid[1] <= bbox["y2"]


def summarize_motion(raw_steps, step_summaries: list[dict], avatar_tracking: dict) -> dict:
    by_step = {row["step_idx"]: row for row in avatar_tracking.get("per_step", [])}
    avatar_path = []
    movement_rows = []
    region_counter: Counter[tuple[int, int, int, int]] = Counter()
    changed_cell_total = 0

    for step_idx, raw_step in enumerate(raw_steps):
        tracking_row = by_step.get(step_idx, {})
        centroid = tracking_row.get("main_centroid")
        previous_centroid = avatar_path[-1] if avatar_path else None
        if centroid is not None:
            avatar_path.append(list(centroid))
        dx = None
        dy = None
        moved = False
        blocked = False
        noop = False
        if centroid is not None and previous_centroid is not None:
            dx = float(centroid[0]) - float(previous_centroid[0])
            dy = float(centroid[1]) - float(previous_centroid[1])
            moved = abs(dx) + abs(dy) > 0.0
            blocked = not moved and not bool(raw_step.done)
            noop = not moved
        summary = step_summaries[step_idx]
        change_regions = list(summary.get("change_regions", []))
        for region in change_regions:
            bbox = region["bbox"]
            region_counter[(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])] += 1
            changed_cell_total += int(region["area"])
        local_change = max((int(region["area"]) for region in change_regions), default=0)
        action_effect_near_avatar = False
        if centroid is not None:
            action_effect_near_avatar = any(_bbox_contains(region["bbox"], centroid) for region in change_regions)
        movement_rows.append(
            {
                "step_idx": step_idx,
                "action": raw_step.action,
                "avatar_centroid": list(centroid) if centroid is not None else None,
                "dx": dx,
                "dy": dy,
                "moved": moved,
                "blocked": blocked,
                "noop": noop,
                "local_change_area": local_change,
                "change_region_count": len(change_regions),
                "action_effect_near_avatar": action_effect_near_avatar,
                "reward": raw_step.reward,
                "done": raw_step.done,
            }
        )

    motion_regions = [
        {
            "bbox": {"x1": key[0], "y1": key[1], "x2": key[2], "y2": key[3]},
            "activation_count": count,
        }
        for key, count in region_counter.items()
    ]
    motion_regions.sort(key=lambda row: (-int(row["activation_count"]), row["bbox"]["y1"], row["bbox"]["x1"]))
    return {
        "avatar_path": avatar_path,
        "movement_rows": movement_rows,
        "blocked_steps": sum(1 for row in movement_rows if row["blocked"]),
        "noop_steps": sum(1 for row in movement_rows if row["noop"]),
        "total_changed_cells": changed_cell_total,
        "motion_regions": motion_regions,
    }
