from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any

from v3_1.utils.ids import stable_digest


def _is_grid(observation: Any) -> bool:
    return isinstance(observation, list) and bool(observation) and all(isinstance(row, list) for row in observation)


def _neighbors(x: int, y: int, width: int, height: int):
    if x > 0:
        yield x - 1, y
    if x + 1 < width:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < height:
        yield x, y + 1


def _component_bbox(points: list[tuple[int, int]]) -> dict[str, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def _bbox_centroid(bbox: dict[str, int]) -> list[float]:
    return [
        (float(bbox["x1"]) + float(bbox["x2"])) / 2.0,
        (float(bbox["y1"]) + float(bbox["y2"])) / 2.0,
    ]


def _touches_border(bbox: dict[str, int], width: int, height: int) -> bool:
    return bbox["x1"] == 0 or bbox["y1"] == 0 or bbox["x2"] == width - 1 or bbox["y2"] == height - 1


def _dominant_color_signature(color_histogram: dict[int, int]) -> str:
    parts = [f"{color}:{count}" for color, count in sorted(color_histogram.items())]
    return ",".join(parts)


def _object_kind(*, bbox: dict[str, int], area: int, width: int, height: int, color_count: int, touches_border: bool) -> tuple[str, list[str]]:
    hints: list[str] = []
    bbox_width = bbox["x2"] - bbox["x1"] + 1
    bbox_height = bbox["y2"] - bbox["y1"] + 1
    width_ratio = bbox_width / float(max(1, width))
    height_ratio = bbox_height / float(max(1, height))
    density = area / float(max(1, bbox_width * bbox_height))

    if touches_border and bbox["y1"] == 0 and height_ratio <= 0.25 and width_ratio >= 0.5:
        hints.extend(["hud_like", "wide_top_strip"])
        return "hud_like", hints
    if area <= 2:
        hints.append("tiny")
    if 1 <= area <= max(9, (width * height) // 16):
        hints.append("candidate_avatar")
    if density >= 0.8:
        hints.append("solid")
    if color_count == 1:
        hints.append("single_color")
    if touches_border:
        hints.append("border_touching")
    if area >= max(12, (width * height) // 8):
        hints.append("large_structure")
        return "structure", hints
    if "candidate_avatar" in hints and not touches_border:
        return "mobile_candidate", hints
    return "world_object", hints


def extract_connected_components(observation: Any) -> list[dict]:
    if not _is_grid(observation):
        return []
    height = len(observation)
    width = len(observation[0]) if height else 0
    visited: set[tuple[int, int]] = set()
    components: list[dict] = []

    for y, row in enumerate(observation):
        for x, raw_value in enumerate(row):
            if (x, y) in visited or not isinstance(raw_value, int):
                continue
            color = int(raw_value)
            stack = [(x, y)]
            points: list[tuple[int, int]] = []
            visited.add((x, y))
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                for nx, ny in _neighbors(cx, cy, width, height):
                    if (nx, ny) in visited or observation[ny][nx] != color:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))
            bbox = _component_bbox(points)
            centroid = _bbox_centroid(bbox)
            area = len(points)
            touches_border = _touches_border(bbox, width, height)
            color_histogram = {color: area}
            bbox_width = bbox["x2"] - bbox["x1"] + 1
            bbox_height = bbox["y2"] - bbox["y1"] + 1
            descriptor = {
                "primary_color": color,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "area": area,
                "density": area / float(max(1, bbox_width * bbox_height)),
                "touches_border": touches_border,
            }
            kind, type_hints = _object_kind(
                bbox=bbox,
                area=area,
                width=width,
                height=height,
                color_count=len(color_histogram),
                touches_border=touches_border,
            )
            signature_payload = {
                "color_signature": _dominant_color_signature(color_histogram),
                "descriptor": descriptor,
                "kind": kind,
            }
            components.append(
                {
                    "component_id": f"component:{stable_digest({'color': color, 'points': points})}",
                    "bbox": bbox,
                    "centroid": centroid,
                    "area": area,
                    "pixels": points,
                    "color_histogram": color_histogram,
                    "dominant_color": color,
                    "touches_border": touches_border,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "descriptor": descriptor,
                    "signature": stable_digest(signature_payload),
                    "kind": kind,
                    "type_hints": type_hints,
                }
            )
    return components


def extract_objects(observation: Any) -> list[dict]:
    if not _is_grid(observation):
        return []
    height = len(observation)
    width = len(observation[0]) if height else 0
    components = extract_connected_components(observation)
    objects: list[dict] = []
    for index, component in enumerate(sorted(components, key=lambda row: (-int(row["area"]), row["component_id"]))):
        histogram = component["color_histogram"]
        color_mass = sum(histogram.values())
        mean_color = sum(color * count for color, count in histogram.items()) / float(max(1, color_mass))
        variance = sum(((float(color) - mean_color) ** 2) * count for color, count in histogram.items()) / float(max(1, color_mass))
        bbox = component["bbox"]
        aspect_ratio = component["bbox_width"] / float(max(1, component["bbox_height"]))
        object_payload = {
            "object_id": f"object:{component['signature']}:{index}",
            "component_id": component["component_id"],
            "kind": component["kind"],
            "bbox": bbox,
            "centroid": component["centroid"],
            "area": component["area"],
            "bbox_area": component["bbox_width"] * component["bbox_height"],
            "width": component["bbox_width"],
            "height": component["bbox_height"],
            "aspect_ratio": aspect_ratio,
            "density": component["descriptor"]["density"],
            "touches_border": component["touches_border"],
            "primary_color": component["dominant_color"],
            "palette": sorted(histogram.keys()),
            "color_histogram": dict(histogram),
            "color_mean": mean_color,
            "color_std": sqrt(max(0.0, variance)),
            "stable_descriptor": {
                "signature": component["signature"],
                "kind": component["kind"],
                "primary_color": component["dominant_color"],
                "bbox_size": [component["bbox_width"], component["bbox_height"]],
                "area": component["area"],
                "density": component["descriptor"]["density"],
            },
            "signature": component["signature"],
            "type_hints": list(component["type_hints"]),
            "confidence": min(1.0, 0.15 + (component["area"] / float(max(1, width * height))) + (0.1 if not component["touches_border"] else 0.0)),
            "observations": 1,
        }
        objects.append(object_payload)
    return objects


def summarize_object_persistence(step_objects: list[list[dict]]) -> dict[str, dict]:
    by_signature: dict[str, dict] = {}
    for step_idx, objects in enumerate(step_objects):
        for obj in objects:
            signature = str(obj["signature"])
            prior = by_signature.get(signature)
            if prior is None:
                by_signature[signature] = {
                    "signature": signature,
                    "kind": obj["kind"],
                    "first_seen_step": step_idx,
                    "last_seen_step": step_idx,
                    "step_indices": [step_idx],
                    "total_area": int(obj["area"]),
                    "count": 1,
                    "primary_color": obj["primary_color"],
                    "type_hints": Counter(obj["type_hints"]),
                }
                continue
            prior["last_seen_step"] = step_idx
            prior["step_indices"].append(step_idx)
            prior["total_area"] += int(obj["area"])
            prior["count"] += 1
            prior["type_hints"].update(obj["type_hints"])
    for row in by_signature.values():
        row["persistence"] = row["count"] / float(max(1, len(step_objects)))
        row["mean_area"] = row["total_area"] / float(max(1, row["count"]))
        row["type_hints"] = sorted(row["type_hints"])
    return by_signature
