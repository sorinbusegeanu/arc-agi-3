"""Exact frame measurement: pixel deltas, delta visualisations, candidate displacement.

Tools report measurements only. They never label a region as player, goal or HUD
(DESIGN.md section 3, "Python vision tools").
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..frames import png_bytes, to_frame_array

DIFF_SCHEMA = "v10.pixel_diff/1"
DISPLACEMENT_SCHEMA = "v10.candidate_displacement/1"
APPEARED_RGB = (0, 224, 0)
DISAPPEARED_RGB = (224, 0, 0)
CHANGED_RGB = (224, 224, 0)
GRAY_SCALE = 16
DEFAULT_MAX_TRANSITIONS = 32
DEFAULT_MAX_SHIFT = 16


def _match_pair(before: np.ndarray, after: np.ndarray) -> None:
    if before.shape != after.shape:
        raise ValueError(f"frame shape mismatch: {before.shape} vs {after.shape}")


def _clip_region(region: Sequence[int] | None, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return a half-open (y0, x0, y1, x1) range clipped into the frame."""
    height, width = shape
    if region is None:
        return (0, 0, height, width)
    if len(region) != 4:
        raise ValueError(f"region must be [y0, x0, y1, x1], got {list(region)}")
    y0, x0, y1, x1 = (int(value) for value in region)
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"region {list(region)} does not intersect the {width}x{height} frame")
    return (y0, x0, y1, x1)


def _changed_box(mask: np.ndarray, offset_y: int, offset_x: int) -> list[int] | None:
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return [int(offset_y + rows[0]), int(offset_x + cols[0]), int(offset_y + rows[-1] + 1), int(offset_x + cols[-1] + 1)]


def _mask_in_region(mask_before: np.ndarray, mask_after: np.ndarray, region: Sequence[int] | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    _match_pair(mask_before, mask_after)
    y0, x0, y1, x1 = _clip_region(region, mask_before.shape)
    changed = mask_before[y0:y1, x0:x1] != mask_after[y0:y1, x0:x1]
    full = np.zeros(mask_before.shape, dtype=bool)
    full[y0:y1, x0:x1] = changed
    return full, (y0, x0, y1, x1)


def _background_value(before: np.ndarray, box: Sequence[int] | None) -> int:
    """Modal pixel value outside the changed box; ties resolve to the smaller value."""
    if box is None:
        values = before.reshape(-1)
    else:
        y0, x0, y1, x1 = box
        outside = np.ones(before.shape, dtype=bool)
        outside[y0:y1, x0:x1] = False
        values = before[outside] if outside.any() else before.reshape(-1)
    if values.size == 0:
        return 0
    unique, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    top = counts.max()
    candidates = [int(value) for value, count in zip(unique, counts) if count == top]
    del best
    return min(candidates)


def pixel_diff(
    before: Any,
    after: Any,
    region: Sequence[int] | None = None,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
) -> dict[str, Any]:
    """Exact changed-pixel measurement between two frames, optionally inside a region."""
    before_array = to_frame_array(before, "before")
    after_array = to_frame_array(after, "after")
    changed, (y0, x0, y1, x1) = _mask_in_region(before_array, after_array, region)
    changed_pixels = int(changed.sum())
    total_pixels = int(changed.size)
    box = _changed_box(changed, 0, 0)

    pairs: list[dict[str, int]] = []
    per_color: dict[str, dict[str, int]] = {}
    truncated = False
    if changed_pixels:
        from_values = before_array[changed]
        to_values = after_array[changed]
        keys, counts = np.unique(np.stack([from_values, to_values], axis=1), axis=0, return_counts=True)
        ordered = sorted(
            ((int(key[0]), int(key[1]), int(count)) for key, count in zip(keys, counts)),
            key=lambda item: (-item[2], item[0], item[1]),
        )
        pairs = [{"from": frm, "to": to, "pixels": count} for frm, to, count in ordered]
        truncated = len(pairs) > int(max_transitions)
        if truncated:
            pairs = pairs[: int(max_transitions)]
        for value in sorted(set(from_values.tolist()) | set(to_values.tolist())):
            key = str(int(value))
            per_color[key] = {
                "before": int((from_values == value).sum()),
                "after": int((to_values == value).sum()),
            }

    return {
        "schema": DIFF_SCHEMA,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": round(changed_pixels / total_pixels, 6) if total_pixels else 0.0,
        "region": [y0, x0, y1, x1],
        "changed_box": box,
        "color_transitions": pairs,
        "transitions_truncated": truncated,
        "per_color": per_color,
    }


def delta_image(
    before: Any,
    after: Any,
    region: Sequence[int] | None = None,
    background: int | None = None,
) -> np.ndarray:
    """HxWx3 uint8 overlay: grey base with appeared/disappeared/other-changed pixels."""
    before_array = to_frame_array(before, "before")
    after_array = to_frame_array(after, "after")
    changed, _ = _mask_in_region(before_array, after_array, region)
    box = _changed_box(changed, 0, 0)
    base = np.clip(before_array.astype(np.int64) * GRAY_SCALE, 0, 255)
    image = np.repeat(base[:, :, None], 3, axis=2).astype(np.uint8)
    if not changed.any():
        return image
    background_value = int(background) if background is not None else _background_value(before_array, box)
    appeared, disappeared, other = appearance_masks(before_array, after_array, changed, background_value)
    image[appeared] = APPEARED_RGB
    image[disappeared] = DISAPPEARED_RGB
    image[other] = CHANGED_RGB
    return image


def appearance_masks(
    before: np.ndarray,
    after: np.ndarray,
    changed: np.ndarray,
    background: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split changed pixels into appeared, disappeared and recoloured-in-place masks."""
    appeared = changed & (before == background) & (after != background)
    disappeared = changed & (before != background) & (after == background)
    other = changed & ~appeared & ~disappeared
    return appeared, disappeared, other


def _shift_overlap(needle: np.ndarray, haystack: np.ndarray, dy: int, dx: int) -> int:
    """Count needle cells that land on haystack cells when needle is shifted by (dy, dx)."""
    height, width = needle.shape
    if abs(dy) >= height or abs(dx) >= width:
        return 0
    if dy >= 0:
        needle_rows, haystack_rows = slice(0, height - dy), slice(dy, height)
    else:
        needle_rows, haystack_rows = slice(-dy, height), slice(0, height + dy)
    if dx >= 0:
        needle_cols, haystack_cols = slice(0, width - dx), slice(dx, width)
    else:
        needle_cols, haystack_cols = slice(-dx, width), slice(0, width + dx)
    if needle_rows.stop <= needle_rows.start or haystack_rows.stop <= haystack_rows.start:
        return 0
    if needle_cols.stop <= needle_cols.start or haystack_cols.stop <= haystack_cols.start:
        return 0
    return int(np.count_nonzero(needle[needle_rows, needle_cols] & haystack[haystack_rows, haystack_cols]))


def _crop_pair_to_union(
    first: np.ndarray,
    second: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop two masks to one shared padded box so shift comparisons align."""
    combined = first | second
    rows = np.flatnonzero(combined.any(axis=1))
    cols = np.flatnonzero(combined.any(axis=0))
    if not len(rows) or not len(cols):
        return first, second
    height, width = combined.shape
    y0 = max(0, int(rows[0]) - padding)
    y1 = min(height, int(rows[-1]) + padding + 1)
    x0 = max(0, int(cols[0]) - padding)
    x1 = min(width, int(cols[-1]) + padding + 1)
    return first[y0:y1, x0:x1], second[y0:y1, x0:x1]


def candidate_displacement(
    before: Any,
    after: Any,
    region: Sequence[int] | None = None,
    max_shift: int = DEFAULT_MAX_SHIFT,
) -> dict[str, Any]:
    """Best rigid shift between disappeared and appeared pixel masks, with uncertainty.

    A shift is only reported when both masks are non-empty; otherwise the measurement
    is `best: null` because no displacement candidate exists to track.
    """
    before_array = to_frame_array(before, "before")
    after_array = to_frame_array(after, "after")
    changed, _ = _mask_in_region(before_array, after_array, region)
    box = _changed_box(changed, 0, 0)
    background_value = _background_value(before_array, box)
    appeared, disappeared, _ = appearance_masks(before_array, after_array, changed, background_value)
    radius = max(0, int(max_shift))

    result: dict[str, Any] = {
        "schema": DISPLACEMENT_SCHEMA,
        "changed_pixels": int(changed.sum()),
        "appeared_pixels": int(appeared.sum()),
        "disappeared_pixels": int(disappeared.sum()),
        "background": background_value,
        "best": None,
        "second": None,
        "margin": None,
        "search_radius": radius,
    }
    if not appeared.any() or not disappeared.any():
        return result

    disappeared_crop, appeared_crop = _crop_pair_to_union(disappeared, appeared, radius)
    denominator = max(int(appeared.sum()), int(disappeared.sum()))
    scored: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            overlap = _shift_overlap(disappeared_crop, appeared_crop, dy, dx)
            if overlap:
                scored.append((overlap / denominator, dy, dx))
    scored.sort(key=lambda item: (-item[0], abs(item[1]) + abs(item[2]), item[1], item[2]))
    if scored:
        best_score, best_dy, best_dx = scored[0]
        result["best"] = {"dy": best_dy, "dx": best_dx, "score": round(best_score, 6)}
        if len(scored) > 1:
            second_score, second_dy, second_dx = scored[1]
            result["second"] = {"dy": second_dy, "dx": second_dx, "score": round(second_score, 6)}
            result["margin"] = round(best_score - second_score, 6)
        else:
            result["margin"] = round(best_score, 6)
    else:
        result["best"] = {"dy": 0, "dx": 0, "score": 0.0}
        result["margin"] = 0.0
    return result


__all__ = [
    "APPEARED_RGB",
    "CHANGED_RGB",
    "DISAPPEARED_RGB",
    "DIFF_SCHEMA",
    "DISPLACEMENT_SCHEMA",
    "appearance_masks",
    "candidate_displacement",
    "delta_image",
    "pixel_diff",
    "png_bytes",
]
