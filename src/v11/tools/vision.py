"""Exact frame measurement: pixel deltas, delta visualisations, candidate displacement.

Tools report measurements only. They never label a region as player, goal or HUD
(DESIGN.md section 3, "Python vision tools").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

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
CLICK_CANDIDATES_SCHEMA = "v11.click_candidates/1"
CLICK_EFFECT_SCHEMA = "v11.click_effect/1"
MAX_SPATIAL_DETAILS = 64
MAX_COMPARE_REGION_CELLS = 4096


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


def changed_cell_summary(
    before: Any, after: Any, *, ui_mask: Mapping[str, Any] | None = None,
    max_cells: int = MAX_SPATIAL_DETAILS,
) -> dict[str, Any]:
    """Return bounded native-grid deltas, optionally separated by a valid UI mask."""
    first, second = to_frame_array(before, "before"), to_frame_array(after, "after")
    _match_pair(first, second)
    changed = first != second
    height, width = first.shape
    excluded = np.zeros(first.shape, dtype=bool)
    mask_valid = False
    if isinstance(ui_mask, Mapping) and ui_mask.get("frame_shape") == [height, width]:
        regions = ui_mask.get("excluded_regions")
        if isinstance(regions, list):
            mask_valid = True
            for item in regions:
                bounds = item.get("bounds") if isinstance(item, Mapping) else None
                if not isinstance(bounds, list) or len(bounds) != 4:
                    mask_valid = False
                    break
                y0, x0, y1, x1 = bounds
                if any(type(v) is not int for v in bounds) or not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                    mask_valid = False
                    break
                excluded[y0:y1, x0:x1] = True
    if not mask_valid:
        excluded.fill(False)
    board_changed = changed & ~excluded
    ui_changed = changed & excluded

    def box(mask: np.ndarray) -> list[int] | None:
        return _changed_box(mask, 0, 0)

    rows, cols = np.nonzero(changed)
    details = []
    limit = min(MAX_SPATIAL_DETAILS, max(0, int(max_cells)))
    for y, x in zip(rows[:limit], cols[:limit]):
        details.append({
            "x": int(x), "y": int(y), "before": int(first[y, x]), "after": int(second[y, x]),
            "region": "ui" if excluded[y, x] else "board" if mask_valid else "unclassified",
        })
    return {
        "changed_cells": int(changed.sum()), "changed_box_yxyx": box(changed),
        "board_changed_cells": int(board_changed.sum()) if mask_valid else None,
        "board_changed_box_yxyx": box(board_changed) if mask_valid else None,
        "ui_changed_cells": int(ui_changed.sum()) if mask_valid else None,
        "ui_changed_box_yxyx": box(ui_changed) if mask_valid else None,
        "ui_mask_status": "available" if mask_valid else "unavailable",
        "cell_details": details, "cell_details_limit": limit,
        "cell_details_truncated": int(changed.sum()) > limit,
    }


def compare_regions(
    frame: Any,
    first_region: Sequence[int],
    second_region: Sequence[int],
    transformation: str = "identity",
    *,
    max_mismatches: int = MAX_SPATIAL_DETAILS,
) -> dict[str, Any]:
    """Compare two bounded native-grid regions after transforming the first.

    Rectangles use half-open [y0, x0, y1, x1] coordinates. Rotations are
    clockwise for 90 degrees; 270 degrees is counterclockwise.
    """
    array = to_frame_array(frame, "frame")
    height, width = array.shape
    allowed = {"identity", "mirror_horizontal", "mirror_vertical", "rotate90", "rotate180", "rotate270"}

    def bounds(value: Sequence[int], label: str) -> tuple[int, int, int, int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4 or any(type(v) is not int for v in value):
            raise ValueError(f"{label} must be four integer half-open [y0,x0,y1,x1] coordinates")
        y0, x0, y1, x1 = value
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"{label} must be a non-empty in-frame rectangle")
        if (y1 - y0) * (x1 - x0) > MAX_COMPARE_REGION_CELLS:
            raise ValueError(f"{label} exceeds {MAX_COMPARE_REGION_CELLS} cells")
        return y0, x0, y1, x1

    if not isinstance(transformation, str) or transformation not in allowed:
        raise ValueError(f"transformation must be one of {sorted(allowed)}")
    y0, x0, y1, x1 = bounds(first_region, "first_region")
    ty0, tx0, ty1, tx1 = bounds(second_region, "second_region")
    source = array[y0:y1, x0:x1]
    target = array[ty0:ty1, tx0:tx1]
    if transformation == "identity":
        transformed = source
        inverse = lambda row, col: (row, col)
    elif transformation == "mirror_horizontal":
        transformed = np.fliplr(source)
        inverse = lambda row, col: (row, source.shape[1] - 1 - col)
    elif transformation == "mirror_vertical":
        transformed = np.flipud(source)
        inverse = lambda row, col: (source.shape[0] - 1 - row, col)
    elif transformation == "rotate90":
        transformed = np.rot90(source, k=3)
        inverse = lambda row, col: (source.shape[0] - 1 - col, row)
    elif transformation == "rotate180":
        transformed = np.rot90(source, k=2)
        inverse = lambda row, col: (source.shape[0] - 1 - row, source.shape[1] - 1 - col)
    else:
        transformed = np.rot90(source, k=1)
        inverse = lambda row, col: (col, source.shape[1] - 1 - row)
    compatible = transformed.shape == target.shape
    if not compatible:
        return {
            "transformed_match": False, "shape_compatible": False,
            "transformed_source_shape": list(transformed.shape), "target_shape": list(target.shape),
            "mismatched_cells": None, "mismatch_box_yxyx": None,
            "mismatches": [], "mismatches_truncated": False,
            "difference_runs": [], "difference_runs_truncated": False,
            "transformation": transformation,
        }
    mismatch = transformed != target
    mismatch_count = int(np.count_nonzero(mismatch))
    mismatch_rows, mismatch_cols = np.nonzero(mismatch)
    limit = min(MAX_SPATIAL_DETAILS, max(0, int(max_mismatches)))
    details = []
    for row, col in zip(mismatch_rows[:limit], mismatch_cols[:limit]):
        source_row, source_col = inverse(int(row), int(col))
        details.append({
            "source_xy": [int(x0 + source_col), int(y0 + source_row)],
            "target_xy": [int(tx0 + col), int(ty0 + row)],
            "transformed_source_value": int(transformed[row, col]),
            "target_value": int(target[row, col]),
        })
    mismatch_box = _changed_box(mismatch, ty0, tx0)
    runs = []
    for local_y in range(mismatch.shape[0]):
        xs = np.flatnonzero(mismatch[local_y])
        if not len(xs):
            continue
        starts = [int(xs[0])]
        ends = []
        for left, right in zip(xs, xs[1:]):
            if right != left + 1:
                ends.append(int(left + 1))
                starts.append(int(right))
        ends.append(int(xs[-1] + 1))
        for start, end in zip(starts, ends):
            if len(runs) < limit:
                runs.append({"y": int(ty0 + local_y), "x0": int(tx0 + start), "x1": int(tx0 + end)})
    run_count = sum(int(np.count_nonzero(np.diff(np.r_[False, row, False].astype(np.int8)) == 1)) for row in mismatch)
    return {
        "transformed_match": mismatch_count == 0, "shape_compatible": True,
        "transformation": transformation,
        "source_shape": list(source.shape), "transformed_source_shape": list(transformed.shape),
        "target_shape": list(target.shape), "mismatched_cells": mismatch_count,
        "mismatch_box_yxyx": mismatch_box,
        "mismatches": details, "mismatches_truncated": mismatch_count > len(details),
        "difference_runs": runs, "difference_runs_truncated": run_count > len(runs),
        "coordinate_mapping": "Each mismatch maps the indicated source_xy cell through the requested transform to target_xy.",
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


def propose_click_candidates(
    frame: Any, frame_id: str, *, previous_attempts: Sequence[tuple[int, int]] = (),
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Return deterministic native-grid click suggestions without semantic labels."""
    grid = to_frame_array(frame, "click candidate frame")
    height, width = grid.shape
    limit = max(0, min(12, int(max_candidates)))
    if limit == 0:
        return {
            "schema": CLICK_CANDIDATES_SCHEMA, "source_frame_id": str(frame_id),
            "frame_size": {"width": int(width), "height": int(height)},
            "coordinate_system": "native_zero_based_xy", "items": [],
        }
    attempted = {(int(x), int(y)) for x, y in previous_attempts}
    sites: list[tuple[int, int, list[int], str, int]] = []

    # Connected components per non-background value retain tiny and narrow sites.
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    seen: set[tuple[int, int]] = set()
    for value in values:
        if int(value) == background:
            continue
        mask = grid == value
        for y0 in range(height):
            for x0 in range(width):
                if not mask[y0, x0] or (x0, y0) in seen:
                    continue
                stack = [(x0, y0)]
                seen.add((x0, y0))
                points = []
                while stack:
                    x, y = stack.pop()
                    points.append((x, y))
                    for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and (nx, ny) not in seen:
                            seen.add((nx, ny)); stack.append((nx, ny))
                points.sort(key=lambda p: (p[1], p[0]))
                # Choose the component cell nearest its centroid, stable tie break.
                cx = sum(x for x, _ in points) / len(points)
                cy = sum(y for _, y in points) / len(points)
                x, y = min(points, key=lambda p: ((p[0]-cx)**2 + (p[1]-cy)**2, p[1], p[0]))
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                sites.append((x, y, [min(ys), min(xs), max(ys)+1, max(xs)+1], "distinct_pixel_region", len(points)))

    # Coarse spatial coverage includes empty/background cells and remains bounded.
    step = max(1, int(np.ceil(max(height, width) / 6)))
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            sites.append((x, y, [y, x, y+1, x+1], "coarse_spatial_sample", 0))
    sites.sort(key=lambda item: (item[4] == 0, -item[4], item[1], item[0]))
    selected: list[tuple[int, int, list[int], str, int]] = []
    for site in sites:
        x, y = site[:2]
        if (x, y) in attempted or any(max(abs(x-s[0]), abs(y-s[1])) <= 1 for s in selected):
            continue
        selected.append(site)
        if len(selected) >= limit:
            break
    return {
        "schema": CLICK_CANDIDATES_SCHEMA, "source_frame_id": str(frame_id),
        "frame_size": {"width": int(width), "height": int(height)},
        "coordinate_system": "native_zero_based_xy",
        "items": [
            {"id": f"C{index}", "x": x, "y": y, "bounds_yxyx": bounds, "basis": basis}
            for index, (x, y, bounds, basis, _size) in enumerate(selected, start=1)
        ],
    }


def click_effect(
    before: Any, after: Any, x: int, y: int, *, radius: int = 2,
    level_changed: bool = False, terminal_changed: bool = False,
    ui_mask: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure pixel changes near and away from a native-grid click point."""
    first, second = to_frame_array(before, "before"), to_frame_array(after, "after")
    _match_pair(first, second)
    changed = first != second
    height, width = first.shape
    r = max(0, min(16, int(radius)))
    x, y = int(x), int(y)
    local = np.zeros(first.shape, dtype=bool)
    local[max(0, y-r):min(height, y+r+1), max(0, x-r):min(width, x+r+1)] = True
    local_count = int(np.count_nonzero(changed & local))
    total = int(np.count_nonzero(changed))
    board_changed_pixels: int | None = None
    excluded_changed_pixels: int | None = None
    board_measurement_status = "mask_unavailable"
    if isinstance(ui_mask, dict):
        expected_shape = ui_mask.get("frame_shape")
        regions = ui_mask.get("excluded_regions")
        if expected_shape != [height, width]:
            board_measurement_status = "frame_shape_mismatch"
        elif isinstance(regions, list):
            excluded = np.zeros(first.shape, dtype=bool)
            valid = True
            for item in regions:
                bounds = item.get("bounds") if isinstance(item, dict) else None
                if (
                    not isinstance(bounds, list) or len(bounds) != 4
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in bounds)
                ):
                    valid = False
                    break
                y0, x0, y1, x1 = bounds
                if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                    valid = False
                    break
                excluded[y0:y1, x0:x1] = True
            if valid:
                board_changed_pixels = int(np.count_nonzero(changed & ~excluded))
                excluded_changed_pixels = int(np.count_nonzero(changed & excluded))
                board_measurement_status = "available"
            else:
                board_measurement_status = "invalid_mask"
        else:
            board_measurement_status = "invalid_mask"
    return {
        "schema": CLICK_EFFECT_SCHEMA, "x": x, "y": y, "radius": r,
        "local_changed_pixels": local_count, "remote_changed_pixels": total-local_count,
        "total_changed_pixels": total, "level_changed": bool(level_changed),
        "terminal_changed": bool(terminal_changed),
        "board_measurement_status": board_measurement_status,
        "board_changed_pixels": board_changed_pixels,
        "excluded_changed_pixels": excluded_changed_pixels,
    }


def load_click_ui_mask(game_id: str | None) -> dict[str, Any] | None:
    """Load a versioned native-grid UI exclusion mask for one game, if authored."""
    if not isinstance(game_id, str) or not game_id.strip():
        return None
    safe_id = quote(game_id.strip(), safe="-_.")
    path = Path(__file__).resolve().parents[3] / "skills" / "v11" / "game_masks" / f"{safe_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"frame_shape": None, "excluded_regions": None}
    if not isinstance(payload, dict) or payload.get("schema") != "v11.click_ui_mask/1":
        return {"frame_shape": None, "excluded_regions": None}
    return {
        "schema": payload.get("schema"),
        "game_id": payload.get("game_id"),
        "frame_shape": payload.get("frame_shape"),
        "excluded_regions": payload.get("excluded_regions"),
    }


__all__ = [
    "APPEARED_RGB",
    "CHANGED_RGB",
    "DISAPPEARED_RGB",
    "DIFF_SCHEMA",
    "DISPLACEMENT_SCHEMA",
    "appearance_masks",
    "candidate_displacement",
    "changed_cell_summary",
    "click_effect",
    "compare_regions",
    "delta_image",
    "load_click_ui_mask",
    "pixel_diff",
    "png_bytes",
]
