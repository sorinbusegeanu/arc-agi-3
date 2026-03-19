from __future__ import annotations

from typing import Any

from v3_1.utils.ids import stable_digest


def patch_crop(observation: list[list[int]], bbox: list[int]) -> list[list[int]]:
    if not isinstance(observation, list) or not isinstance(bbox, list) or len(bbox) != 4:
        return []
    x0, y0, x1, y1 = [int(value) for value in bbox]
    rows = []
    for y in range(max(0, y0), min(len(observation), y1 + 1)):
        row = observation[y]
        if not isinstance(row, list):
            continue
        rows.append([int(value) for value in row[max(0, x0): min(len(row), x1 + 1)]])
    return rows


def canonicalize_patch(patch: list[list[int]]) -> list[list[int]]:
    rows = [list(row) for row in list(patch or []) if isinstance(row, list)]
    while rows and not any(value != 0 for value in rows[0]):
        rows.pop(0)
    while rows and not any(value != 0 for value in rows[-1]):
        rows.pop()
    if not rows:
        return []
    width = max(len(row) for row in rows)
    normalized = [row + ([0] * (width - len(row))) for row in rows]
    left = 0
    right = width
    while left < right and all(row[left] == 0 for row in normalized):
        left += 1
    while right > left and all(row[right - 1] == 0 for row in normalized):
        right -= 1
    return [row[left:right] for row in normalized]


def stable_descriptor(patch: list[list[int]]) -> dict[str, Any]:
    canonical = canonicalize_patch(patch)
    palette = sorted({int(value) for row in canonical for value in row})
    filled = sum(1 for row in canonical for value in row if int(value) != 0)
    width = max((len(row) for row in canonical), default=0)
    height = len(canonical)
    bbox_area = max(1, width * height)
    density = filled / float(bbox_area)
    row_signatures = [tuple(int(value) for value in row) for row in canonical]
    col_signatures = [
        tuple(int(canonical[y][x]) for y in range(height))
        for x in range(width)
    ] if width and height else []
    palette_counts: dict[int, int] = {}
    for row in canonical:
        for value in row:
            value = int(value)
            if value == 0:
                continue
            palette_counts[value] = palette_counts.get(value, 0) + 1
    edge_profile = {
        "top": tuple(int(value) for value in canonical[0]) if canonical else (),
        "bottom": tuple(int(value) for value in canonical[-1]) if canonical else (),
        "left": tuple(int(canonical[y][0]) for y in range(height)) if width and height else (),
        "right": tuple(int(canonical[y][width - 1]) for y in range(height)) if width and height else (),
    }
    horizontal_reflection = [tuple(reversed(row)) for row in canonical]
    vertical_reflection = list(reversed(canonical))
    symmetry_score = 0.0
    if canonical:
        if canonical == horizontal_reflection:
            symmetry_score += 0.5
        if canonical == vertical_reflection:
            symmetry_score += 0.5
    return {
        "width": width,
        "height": height,
        "palette": palette,
        "filled_cells": filled,
        "density": density,
        "texture_like": bool(filled > 0 and density <= 0.22 and len(palette_counts) <= 2),
        "micro_pattern": bool(width <= 2 and height <= 2),
        "canonical_rows": canonical,
        "row_signatures": row_signatures,
        "col_signatures": col_signatures,
        "palette_counts": palette_counts,
        "edge_profile": edge_profile,
        "symmetry_score": symmetry_score,
    }


def repeated_texture_marker(descriptor: dict[str, Any] | None) -> bool:
    row = dict(descriptor or {})
    width = int(row.get("width", 0) or 0)
    height = int(row.get("height", 0) or 0)
    density = float(row.get("density", 0.0) or 0.0)
    palette_counts = dict(row.get("palette_counts", {}) or {})
    non_zero_colors = [int(color) for color, count in palette_counts.items() if int(count or 0) > 0]
    return bool(
        row.get("texture_like")
        or row.get("micro_pattern")
        or (width <= 2 and height <= 2)
        or (density <= 0.22 and len(non_zero_colors) <= 2)
    )


def stable_pattern_id(patch: list[list[int]]) -> str:
    descriptor = stable_descriptor(patch)
    return f"pattern:{stable_digest(descriptor)}"


def pattern_similarity(left: list[list[int]], right: list[list[int]]) -> float:
    left_desc = stable_descriptor(left)
    right_desc = stable_descriptor(right)
    if not left_desc["canonical_rows"] or not right_desc["canonical_rows"]:
        return 0.0
    if left_desc["canonical_rows"] == right_desc["canonical_rows"]:
        return 1.0
    left_set = {(x, y, value) for y, row in enumerate(left_desc["canonical_rows"]) for x, value in enumerate(row) if value != 0}
    right_set = {(x, y, value) for y, row in enumerate(right_desc["canonical_rows"]) for x, value in enumerate(row) if value != 0}
    union = len(left_set | right_set)
    if union <= 0:
        return 0.0
    overlap = float(len(left_set & right_set)) / float(union)
    palette_overlap = 0.0
    left_palette = set(left_desc["palette"])
    right_palette = set(right_desc["palette"])
    if left_palette or right_palette:
        palette_overlap = float(len(left_palette & right_palette)) / float(max(1, len(left_palette | right_palette)))
    shape_similarity = 1.0
    if left_desc["width"] or right_desc["width"] or left_desc["height"] or right_desc["height"]:
        shape_similarity = 1.0 - min(
            1.0,
            (
                abs(int(left_desc["width"]) - int(right_desc["width"]))
                + abs(int(left_desc["height"]) - int(right_desc["height"]))
            ) / float(max(1, max(int(left_desc["width"]), int(right_desc["width"])) + max(int(left_desc["height"]), int(right_desc["height"])))),
        )
    density_similarity = 1.0 - min(
        1.0,
        abs(float(left_desc.get("density", 0.0) or 0.0) - float(right_desc.get("density", 0.0) or 0.0)),
    )
    edge_similarity = 0.0
    left_edges = dict(left_desc.get("edge_profile", {}) or {})
    right_edges = dict(right_desc.get("edge_profile", {}) or {})
    comparable_edges = [key for key in ("top", "bottom", "left", "right") if key in left_edges or key in right_edges]
    if comparable_edges:
        edge_matches = sum(1 for key in comparable_edges if tuple(left_edges.get(key, ())) == tuple(right_edges.get(key, ())))
        edge_similarity = edge_matches / float(len(comparable_edges))
    return min(
        1.0,
        (0.50 * overlap)
        + (0.15 * palette_overlap)
        + (0.12 * shape_similarity)
        + (0.13 * density_similarity)
        + (0.10 * edge_similarity),
    )


def patterns_match(left: list[list[int]], right: list[list[int]], *, threshold: float = 0.9) -> bool:
    return pattern_similarity(left, right) >= float(threshold)


def pattern_equality_decision(left: list[list[int]], right: list[list[int]], *, threshold: float = 0.9) -> dict[str, Any]:
    confidence = pattern_similarity(left, right)
    return {
        "matches": confidence >= float(threshold),
        "confidence": confidence,
        "threshold": float(threshold),
        "left_pattern_id": stable_pattern_id(left) if left else None,
        "right_pattern_id": stable_pattern_id(right) if right else None,
    }
