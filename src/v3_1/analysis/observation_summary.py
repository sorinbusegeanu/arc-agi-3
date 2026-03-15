from __future__ import annotations

from collections import Counter
from typing import Any

from v3_1.analysis.object_extraction import extract_connected_components, extract_objects
from v3_1.utils.ids import stable_digest


def _is_grid(observation: Any) -> bool:
    return isinstance(observation, list) and bool(observation) and all(isinstance(row, list) for row in observation)


def _background_estimate(observation: list[list[int]], components: list[dict]) -> dict:
    height = len(observation)
    width = len(observation[0]) if height else 0
    counts = Counter()
    border_counts = Counter()
    for y, row in enumerate(observation):
        for x, value in enumerate(row):
            color = int(value)
            counts[color] += 1
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                border_counts[color] += 1
    border_total = max(1, (width * 2) + (height * 2) - 4)
    component_mass: Counter[int] = Counter()
    for component in components:
        component_mass[int(component["dominant_color"])] = max(component_mass[int(component["dominant_color"])], int(component["area"]))
    best_color = None
    best_score = -1.0
    total = max(1, width * height)
    for color, count in counts.items():
        frequency = count / float(total)
        border_frequency = border_counts.get(color, 0) / float(border_total)
        connectedness = component_mass.get(color, 0) / float(max(1, count))
        score = (0.45 * frequency) + (0.35 * border_frequency) + (0.20 * connectedness)
        if score > best_score:
            best_color = color
            best_score = score
    return {
        "color": int(best_color) if best_color is not None else 0,
        "confidence": max(0.0, best_score),
        "counts": dict(counts),
    }


def _change_regions(observation: list[list[int]], previous_observation: list[list[int]] | None) -> list[dict]:
    if previous_observation is None:
        return []
    height = min(len(observation), len(previous_observation))
    width = min(len(observation[0]), len(previous_observation[0])) if height else 0
    changed_points: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if observation[y][x] != previous_observation[y][x]:
                changed_points.append((x, y))
    if not changed_points:
        return []
    regions = []
    visited: set[tuple[int, int]] = set()
    point_set = set(changed_points)
    for point in changed_points:
        if point in visited:
            continue
        stack = [point]
        region_points: list[tuple[int, int]] = []
        visited.add(point)
        while stack:
            cx, cy = stack.pop()
            region_points.append((cx, cy))
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if (nx, ny) in point_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        xs = [value[0] for value in region_points]
        ys = [value[1] for value in region_points]
        regions.append(
            {
                "bbox": {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)},
                "area": len(region_points),
                "centroid": [
                    sum(value[0] for value in region_points) / float(len(region_points)),
                    sum(value[1] for value in region_points) / float(len(region_points)),
                ],
            }
        )
    return sorted(regions, key=lambda row: (-int(row["area"]), row["bbox"]["y1"], row["bbox"]["x1"]))


def summarize_observation(observation: Any, previous_observation: Any | None = None) -> dict:
    if not _is_grid(observation):
        return {
            "width": 0,
            "height": 0,
            "palette": [],
            "background": {"color": 0, "confidence": 0.0, "counts": {}},
            "objects": [],
            "avatar_candidates": [],
            "change_regions": [],
            "active_regions": [],
            "state_identity": {"stable_inputs": [], "state_hash": "empty", "background_color": 0},
        }
    grid = [[int(cell) for cell in row] for row in observation]
    height = len(grid)
    width = len(grid[0]) if height else 0
    components = extract_connected_components(grid)
    objects = extract_objects(grid)
    palette = sorted({int(cell) for row in grid for cell in row})
    background = _background_estimate(grid, components)
    change_regions = _change_regions(grid, previous_observation if _is_grid(previous_observation) else None)
    active_regions = [region for region in change_regions if region["area"] > 0]
    avatar_candidates = []
    for obj in objects:
        score = 0.0
        if "candidate_avatar" in obj["type_hints"]:
            score += 0.45
        if obj["kind"] == "mobile_candidate":
            score += 0.25
        if not obj["touches_border"]:
            score += 0.1
        if obj["primary_color"] != background["color"]:
            score += 0.1
        if 1 <= obj["area"] <= max(12, (width * height) // 8):
            score += 0.1
        avatar_candidates.append(
            {
                "object_id": obj["object_id"],
                "signature": obj["signature"],
                "centroid": list(obj["centroid"]),
                "bbox": dict(obj["bbox"]),
                "score": min(1.0, score),
                "type_hints": list(obj["type_hints"]),
                "primary_color": obj["primary_color"],
                "area": int(obj["area"]),
                "width": int(obj["width"]),
                "height": int(obj["height"]),
                "touches_border": bool(obj["touches_border"]),
            }
        )
    avatar_candidates.sort(key=lambda row: (-float(row["score"]), row["object_id"]))
    stable_inputs = [
        background["color"],
        tuple(palette),
        tuple(
            sorted(
                (
                    obj["signature"],
                    obj["kind"],
                    obj["primary_color"],
                    obj["width"],
                    obj["height"],
                    int(obj["bbox"]["x1"]),
                    int(obj["bbox"]["y1"]),
                    int(obj["bbox"]["x2"]),
                    int(obj["bbox"]["y2"]),
                )
                for obj in objects
                if obj["kind"] != "hud_like"
            )
        ),
    ]
    state_identity = {
        "stable_inputs": stable_inputs,
        "state_hash": stable_digest(stable_inputs),
        "background_color": background["color"],
    }
    return {
        "width": width,
        "height": height,
        "palette": palette,
        "background": background,
        "objects": objects,
        "avatar_candidates": avatar_candidates,
        "change_regions": change_regions,
        "active_regions": active_regions,
        "state_identity": state_identity,
    }
