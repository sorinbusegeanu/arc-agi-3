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
    return {
        "width": width,
        "height": height,
        "palette": palette,
        "filled_cells": filled,
        "canonical_rows": canonical,
    }


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
    return float(len(left_set & right_set)) / float(union)


def patterns_match(left: list[list[int]], right: list[list[int]], *, threshold: float = 0.9) -> bool:
    return pattern_similarity(left, right) >= float(threshold)
