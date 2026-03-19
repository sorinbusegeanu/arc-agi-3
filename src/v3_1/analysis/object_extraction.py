from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any

from v3_1.analysis.pattern_identity import patch_crop, repeated_texture_marker, stable_descriptor, stable_pattern_id
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


def _bbox_contains(outer: dict[str, int], inner: dict[str, int], *, margin: int = 0) -> bool:
    return (
        int(inner["x1"]) >= int(outer["x1"]) - margin
        and int(inner["y1"]) >= int(outer["y1"]) - margin
        and int(inner["x2"]) <= int(outer["x2"]) + margin
        and int(inner["y2"]) <= int(outer["y2"]) + margin
    )


def _bbox_overlap_or_adjacent(left: dict[str, int], right: dict[str, int], *, margin: int = 1) -> bool:
    return not (
        int(left["x2"]) < int(right["x1"]) - margin
        or int(right["x2"]) < int(left["x1"]) - margin
        or int(left["y2"]) < int(right["y1"]) - margin
        or int(right["y2"]) < int(left["y1"]) - margin
    )


def _centroid_distance(left: list[float], right: list[float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return sqrt((dx * dx) + (dy * dy))


def _palette_overlap(left: list[int], right: list[int]) -> float:
    left_set = {int(value) for value in list(left or [])}
    right_set = {int(value) for value in list(right or [])}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set)) / float(max(1, len(left_set | right_set)))


def _size_similarity(left: dict, right: dict) -> float:
    left_area = max(1, int(left.get("area", 0) or 0))
    right_area = max(1, int(right.get("area", 0) or 0))
    area_ratio = min(left_area, right_area) / float(max(left_area, right_area))
    left_width = max(1, int(left.get("width", 0) or 0))
    right_width = max(1, int(right.get("width", 0) or 0))
    left_height = max(1, int(left.get("height", 0) or 0))
    right_height = max(1, int(right.get("height", 0) or 0))
    width_ratio = min(left_width, right_width) / float(max(left_width, right_width))
    height_ratio = min(left_height, right_height) / float(max(left_height, right_height))
    return (0.45 * area_ratio) + (0.275 * width_ratio) + (0.275 * height_ratio)


def _strip_like_alignment(left: dict, right: dict) -> bool:
    left_bbox = dict(left.get("bbox", {}) or {})
    right_bbox = dict(right.get("bbox", {}) or {})
    if not left_bbox or not right_bbox:
        return False
    return (
        abs(int(left_bbox.get("y1", 0) or 0) - int(right_bbox.get("y1", 0) or 0)) <= 1
        and abs(int(left_bbox.get("y2", 0) or 0) - int(right_bbox.get("y2", 0) or 0)) <= 1
        and abs(int(left.get("height", 0) or 0) - int(right.get("height", 0) or 0)) <= 1
        and (
            _bbox_overlap_or_adjacent(left_bbox, right_bbox, margin=3)
            or _centroid_distance(list(left.get("centroid", [0.0, 0.0])), list(right.get("centroid", [0.0, 0.0]))) <= 6.0
        )
    )


def _canonical_track_match_score(track: dict, obj: dict, *, step_idx: int) -> float:
    last_obj = dict(track.get("last_object", {}) or {})
    if not last_obj:
        return 0.0
    last_step = int(track.get("last_seen_step", -999) or -999)
    if step_idx - last_step > 2:
        return 0.0
    overlap_score = 1.0 if _bbox_overlap_or_adjacent(dict(last_obj.get("bbox", {}) or {}), dict(obj.get("bbox", {}) or {}), margin=1) else 0.0
    distance_score = max(0.0, 1.0 - (_centroid_distance(list(last_obj.get("centroid", [0.0, 0.0])), list(obj.get("centroid", [0.0, 0.0]))) / 6.5))
    size_score = _size_similarity(last_obj, obj)
    palette_score = _palette_overlap(list(last_obj.get("palette", []) or []), list(obj.get("palette", []) or []))
    pattern_score = 1.0 if last_obj.get("pattern_id") and last_obj.get("pattern_id") == obj.get("pattern_id") else 0.0
    border_strip_score = 1.0 if _strip_like_alignment(last_obj, obj) else 0.0
    color_score = 1.0 if int(last_obj.get("primary_color", -1) or -1) == int(obj.get("primary_color", -2) or -2) else 0.0
    temporal_score = 1.0 if step_idx - last_step <= 1 else 0.75
    return (
        0.18 * overlap_score
        + 0.16 * distance_score
        + 0.18 * size_score
        + 0.12 * palette_score
        + 0.16 * pattern_score
        + 0.12 * border_strip_score
        + 0.04 * color_score
        + 0.04 * temporal_score
    )


def _suppress_decorative_components(components: list[dict], observation: Any) -> tuple[list[dict], dict[str, list[dict]]]:
    suppressed_children: dict[str, list[dict]] = defaultdict(list)
    kept: list[dict] = []
    descriptors: dict[str, dict] = {}
    for component in components:
        bbox = dict(component.get("bbox", {}) or {})
        patch = patch_crop(observation, [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]) if bbox else []
        descriptors[str(component.get("component_id"))] = stable_descriptor(patch) if patch else {}
    sorted_components = sorted(components, key=lambda row: (-int(row.get("area", 0) or 0), str(row.get("component_id") or "")))
    suppressed_ids: set[str] = set()
    for component in reversed(sorted_components):
        component_id = str(component.get("component_id") or "")
        if component_id in suppressed_ids:
            continue
        area = int(component.get("area", 0) or 0)
        bbox = dict(component.get("bbox", {}) or {})
        bbox_width = int(component.get("bbox_width", 0) or 0)
        bbox_height = int(component.get("bbox_height", 0) or 0)
        descriptor = descriptors.get(component_id, {})
        if not (
            area <= 3
            or (bbox_width <= 2 and bbox_height <= 2)
            or repeated_texture_marker(descriptor)
        ):
            continue
        for parent in sorted_components:
            parent_id = str(parent.get("component_id") or "")
            if parent_id == component_id or parent_id in suppressed_ids:
                continue
            parent_area = int(parent.get("area", 0) or 0)
            if parent_area < max(8, area * 4):
                continue
            parent_bbox = dict(parent.get("bbox", {}) or {})
            if not _bbox_contains(parent_bbox, bbox, margin=1):
                continue
            parent_descriptor = descriptors.get(parent_id, {})
            if repeated_texture_marker(parent_descriptor) and parent_area <= 12:
                continue
            child_row = dict(component)
            child_row["suppressed_as_decorative"] = True
            child_row["parent_component_id"] = parent_id
            child_row["pattern_descriptor"] = descriptor
            suppressed_children[parent_id].append(child_row)
            suppressed_ids.add(component_id)
            break
    for component in sorted_components:
        if str(component.get("component_id") or "") not in suppressed_ids:
            kept.append(component)
    return kept, dict(suppressed_children)


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
    # Avatar candidates should be compact, local, and plausibly mobile.
    # The previous threshold scaled with the full map and mislabeled many small
    # world objects as avatar-like.
    if 1 <= area <= 12 and bbox_width <= 4 and bbox_height <= 4 and not touches_border:
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
    if not touches_border and 3 <= area <= 64 and bbox_width <= max(10, width // 3) and bbox_height <= max(10, height // 3):
        if density >= 0.35:
            hints.append("compact_structure_candidate")
    if not touches_border and area >= 3 and bbox_width <= max(8, width // 4) and bbox_height <= max(8, height // 4):
        hints.append("structural_candidate")
    if not touches_border and 3 <= area <= 20 and density >= 0.45:
        hints.append("symbol_candidate")
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
    components, suppressed_children = _suppress_decorative_components(components, observation)
    objects: list[dict] = []
    for index, component in enumerate(sorted(components, key=lambda row: (-int(row["area"]), row["component_id"]))):
        histogram = component["color_histogram"]
        color_mass = sum(histogram.values())
        mean_color = sum(color * count for color, count in histogram.items()) / float(max(1, color_mass))
        variance = sum(((float(color) - mean_color) ** 2) * count for color, count in histogram.items()) / float(max(1, color_mass))
        bbox = component["bbox"]
        patch = patch_crop(observation, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        descriptor = stable_descriptor(patch) if patch else {}
        pattern_id = stable_pattern_id(patch) if patch else None
        aspect_ratio = component["bbox_width"] / float(max(1, component["bbox_height"]))
        symbolic_structure_score = 0.0
        if component["kind"] in {"world_object", "structure"}:
            symbolic_structure_score += 0.2
        if "structural_candidate" in component["type_hints"] or "compact_structure_candidate" in component["type_hints"]:
            symbolic_structure_score += 0.25
        if "symbol_candidate" in component["type_hints"]:
            symbolic_structure_score += 0.2
        if pattern_id:
            symbolic_structure_score += 0.15
        if component["touches_border"]:
            symbolic_structure_score -= 0.1
        if "candidate_avatar" in component["type_hints"] or component["kind"] == "mobile_candidate":
            symbolic_structure_score -= 0.35
        if repeated_texture_marker(descriptor):
            symbolic_structure_score -= 0.25
        decorative_children = list(suppressed_children.get(str(component["component_id"]), []))
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
            "pattern_descriptor": descriptor,
            "pattern_id": pattern_id,
            "signature": component["signature"],
            "type_hints": list(component["type_hints"]),
            "symbolic_structure_score": max(0.0, min(1.0, symbolic_structure_score)),
            "decorative_child_count": len(decorative_children),
            "decorative_children": [
                {
                    "component_id": str(child.get("component_id") or ""),
                    "bbox": dict(child.get("bbox", {}) or {}),
                    "area": int(child.get("area", 0) or 0),
                    "pattern_descriptor": dict(child.get("pattern_descriptor", {}) or {}),
                }
                for child in decorative_children
            ],
            "texture_suppressed": repeated_texture_marker(descriptor),
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
        row["raw_signature_persistence"] = row["count"] / float(max(1, len(step_objects)))
        row["persistence"] = row["raw_signature_persistence"]
        row["mean_area"] = row["total_area"] / float(max(1, row["count"]))
        row["type_hints"] = sorted(row["type_hints"])

    tracks: list[dict] = []
    for step_idx, objects in enumerate(step_objects):
        assigned_track_ids: set[str] = set()
        for obj in sorted((dict(row) for row in objects), key=lambda row: (-int(row.get("area", 0) or 0), str(row.get("object_id") or ""))):
            best_track = None
            best_score = 0.0
            for track in tracks:
                track_id = str(track.get("canonical_track_id") or "")
                if track_id in assigned_track_ids:
                    continue
                score = _canonical_track_match_score(track, obj, step_idx=step_idx)
                if score > best_score:
                    best_score = score
                    best_track = track
            if best_track is None or best_score < 0.44:
                track = {
                    "canonical_track_id": f"track:{stable_digest({'seed_object_id': obj.get('object_id'), 'bbox': obj.get('bbox'), 'primary_color': obj.get('primary_color')})}",
                    "first_seen_step": step_idx,
                    "last_seen_step": step_idx,
                    "count": 1,
                    "step_indices": [step_idx],
                    "object_rows": [dict(obj)],
                    "raw_signatures": {str(obj.get('signature') or '')},
                    "raw_source_ids": {str(obj.get('object_id') or '')},
                    "total_area": int(obj.get("area", 0) or 0),
                    "type_hints": Counter(list(obj.get("type_hints", []) or [])),
                    "last_object": dict(obj),
                    "exemplar": dict(obj),
                }
                tracks.append(track)
                assigned_track_ids.add(str(track["canonical_track_id"]))
                continue
            best_track["last_seen_step"] = step_idx
            best_track["count"] = int(best_track.get("count", 0) or 0) + 1
            best_track.setdefault("step_indices", []).append(step_idx)
            best_track.setdefault("object_rows", []).append(dict(obj))
            best_track.setdefault("raw_signatures", set()).add(str(obj.get("signature") or ""))
            best_track.setdefault("raw_source_ids", set()).add(str(obj.get("object_id") or ""))
            best_track["total_area"] = int(best_track.get("total_area", 0) or 0) + int(obj.get("area", 0) or 0)
            best_track.setdefault("type_hints", Counter()).update(list(obj.get("type_hints", []) or []))
            best_track["last_object"] = dict(obj)
            current_exemplar = dict(best_track.get("exemplar", {}) or {})
            if int(obj.get("area", 0) or 0) >= int(current_exemplar.get("area", 0) or 0):
                best_track["exemplar"] = dict(obj)
            assigned_track_ids.add(str(best_track.get("canonical_track_id") or ""))

    canonical_by_id: dict[str, dict] = {}
    for track in tracks:
        canonical_track_id = str(track.get("canonical_track_id") or "")
        track_count = int(track.get("count", 0) or 0)
        canonical_persistence = track_count / float(max(1, len(step_objects)))
        exemplar = dict(track.get("exemplar", {}) or {})
        canonical_row = {
            "canonical_track_id": canonical_track_id,
            "kind": str(exemplar.get("kind") or "world_object"),
            "signature": str(exemplar.get("signature") or ""),
            "first_seen_step": int(track.get("first_seen_step", 0) or 0),
            "last_seen_step": int(track.get("last_seen_step", 0) or 0),
            "step_indices": list(track.get("step_indices", []) or []),
            "count": track_count,
            "canonical_track_count": track_count,
            "canonical_track_persistence": canonical_persistence,
            "persistence": canonical_persistence,
            "mean_area": float(track.get("total_area", 0) or 0) / float(max(1, track_count)),
            "primary_color": exemplar.get("primary_color"),
            "type_hints": sorted(track.get("type_hints", Counter())),
            "exemplar": exemplar,
            "object_rows": list(track.get("object_rows", []) or []),
            "raw_signatures": sorted(track.get("raw_signatures", set()) or []),
            "raw_source_ids": sorted(track.get("raw_source_ids", set()) or []),
        }
        exemplar_signature = str(exemplar.get("signature") or "")
        raw_stats = dict(by_signature.get(exemplar_signature, {}) or {})
        canonical_row["raw_signature_count"] = int(raw_stats.get("count", 0) or 0)
        canonical_row["raw_signature_persistence"] = float(raw_stats.get("raw_signature_persistence", 0.0) or 0.0)
        canonical_by_id[canonical_track_id] = canonical_row
    return canonical_by_id
