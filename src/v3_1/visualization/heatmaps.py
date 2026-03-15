from __future__ import annotations

import struct
import zlib


ARC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


def build_visit_heatmap(episodes: list[dict], *, width: int, height: int) -> dict[str, object]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    sequence: list[list[int]] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            cell = step.get("avatar_cell")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                continue
            x, y = int(cell[0]), int(cell[1])
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] += 1
                sequence.append([x, y])
    return {
        "counts": grid,
        "sequence": sequence,
        "start": sequence[0] if sequence else None,
        "end": sequence[-1] if sequence else None,
    }


def build_poi_heatmap(
    blackboard_state: dict,
    *,
    width: int,
    height: int,
    confidence_threshold: float = 0.35,
    observation_threshold: int = 2,
    low_utility_threshold: float = 0.05,
) -> dict[str, object]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    stable: list[dict] = []
    active_stable: list[dict] = []
    stale_stable: list[dict] = []
    rejected: list[dict] = []
    trigger_related: list[dict] = []
    targetable: list[dict] = []
    entities = dict(blackboard_state.get("entities", {}))
    trigger_entities = {str(trigger.get("entity_id")) for trigger in blackboard_state.get("trigger_zones", {}).values() if trigger.get("entity_id")}
    for entity_id, poi in entities.items():
        if poi.get("kind") != "poi":
            continue
        confidence = float(poi.get("confidence", 0.0))
        observations = int(poi.get("observations", poi.get("evidence_count", 1)))
        utility = float(poi.get("utility", 0.0))
        lifecycle = str(poi.get("lifecycle_state", "active"))
        explicitly_rejected = bool(poi.get("noise_rejected")) or utility < low_utility_threshold
        stable_flag = confidence >= confidence_threshold and observations >= observation_threshold and not explicitly_rejected and lifecycle in {"active", "stale"}
        record = {
            "entity_id": entity_id,
            "confidence": confidence,
            "observations": observations,
            "utility": utility,
            "bbox": poi.get("bbox"),
            "centroid": poi.get("centroid"),
            "interaction_effect_score": float(poi.get("interaction_effect_score", 0.0)),
            "distance_from_avatar": float(poi.get("distance_from_avatar", 0.0)),
            "distance_score": float(poi.get("distance_score", 0.0)),
            "motion_variance": float(poi.get("motion_variance", 0.0)),
            "motion_score": float(poi.get("motion_score", 0.0)),
            "lifecycle_state": lifecycle,
            "stable_poi": stable_flag,
            "accepted_for_export": stable_flag,
            "explicitly_rejected": explicitly_rejected,
        }
        layer = stable if stable_flag else rejected
        layer.append(record)
        if stable_flag and lifecycle == "active":
            active_stable.append(record)
        if stable_flag and lifecycle == "stale":
            stale_stable.append(record)
        if entity_id in trigger_entities:
            trigger_related.append(record)
        if poi.get("reachable_now") or poi.get("reachable_later"):
            targetable.append(record)
        if not stable_flag:
            continue
        _paint_poi(grid, poi, width=width, height=height)
    stable_count = max(1, len(stable))
    return {
        "accepted_counts": grid,
        "stable_pois": stable,
        "active_stable_pois": active_stable,
        "stale_stable_pois": stale_stable,
        "rejected_pois": rejected,
        "trigger_related_pois": trigger_related,
        "targetable_pois": targetable,
        "avg_interaction_effect": sum(float(row.get("interaction_effect_score", 0.0)) for row in stable) / float(stable_count) if stable else 0.0,
        "avg_distance_score": sum(float(row.get("distance_score", 0.0)) for row in stable) / float(stable_count) if stable else 0.0,
        "avg_motion_score": sum(float(row.get("motion_score", 0.0)) for row in stable) / float(stable_count) if stable else 0.0,
    }


def render_observation_png(observation: list[list[int]] | None, *, width: int = 64, height: int = 64, scale: int = 15) -> bytes:
    base = _normalize_observation(observation, width=width, height=height)
    pixels = [[_arc_color(cell) for cell in row] for row in base]
    return _render_rgb_matrix(pixels, scale=scale)


def render_heatmap_png(matrix: list[list[int]], *, scale: int = 15) -> bytes:
    if not matrix or not matrix[0]:
        matrix = [[0]]
    max_value = max(max(row) for row in matrix) if matrix else 0
    pixels = [[_heat_color(int(value), max_value) for value in row] for row in matrix]
    return _render_rgb_matrix(pixels, scale=scale)


def render_overlay_png(observation: list[list[int]] | None, heatmap: list[list[int]], *, overlay_kind: str, width: int = 64, height: int = 64, scale: int = 15, start: list[int] | None = None, end: list[int] | None = None) -> bytes:
    base = _normalize_observation(observation, width=width, height=height)
    heat = _normalize_heatmap(heatmap, width=width, height=height)
    max_value = max(max(row) for row in heat) if heat else 0
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            base_color = _arc_color(base[y][x])
            value = heat[y][x]
            if value <= 0 or max_value <= 0:
                row.append(base_color)
                continue
            intensity = _normalized_intensity(value, max_value)
            overlay = _overlay_color(overlay_kind, intensity)
            row.append(_blend(base_color, overlay, 0.55))
        pixels.append(row)
    _mark_point(pixels, start, color=bytes((80, 255, 120)))
    _mark_point(pixels, end, color=bytes((255, 255, 255)))
    return _render_rgb_matrix(pixels, scale=scale)


def render_heatmap_debug_png(heatmap: list[list[int]], *, overlay_kind: str, width: int = 64, height: int = 64, scale: int = 15, start: list[int] | None = None, end: list[int] | None = None) -> bytes:
    heat = _normalize_heatmap(heatmap, width=width, height=height)
    max_value = max(max(row) for row in heat) if heat else 0
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            value = heat[y][x]
            if value <= 0 or max_value <= 0:
                row.append(bytes((0, 0, 0)))
                continue
            intensity = _normalized_intensity(value, max_value)
            row.append(_overlay_color(overlay_kind, intensity))
        pixels.append(row)
    _mark_point(pixels, start, color=bytes((80, 255, 120)))
    _mark_point(pixels, end, color=bytes((255, 255, 255)))
    return _render_rgb_matrix(pixels, scale=scale)


def _heat_color(value: int, max_value: int) -> bytes:
    if max_value <= 0 or value <= 0:
        return bytes((245, 247, 250))
    ratio = min(1.0, float(value) / float(max_value))
    red = 255
    green = max(32, int(235 - (170 * ratio)))
    blue = max(24, int(210 - (210 * ratio)))
    return bytes((red, green, blue))


def _arc_color(value: int) -> bytes:
    color = ARC_PALETTE.get(int(value), (255, 255, 255))
    return bytes(color)


def _overlay_color(kind: str, intensity: float) -> bytes:
    if kind == "poi":
        return bytes((255, max(24, int(120 - 70 * intensity)), 0))
    return bytes((255, max(12, int(220 - 205 * intensity)), max(0, int(80 - 80 * intensity))))


def _blend(base: bytes, overlay: bytes, alpha: float) -> bytes:
    return bytes(int((1.0 - alpha) * base[idx] + alpha * overlay[idx]) for idx in range(3))


def _normalize_observation(observation: list[list[int]] | None, *, width: int, height: int) -> list[list[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    if not observation:
        return grid
    for y, row in enumerate(observation[:height]):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row[:width]):
            grid[y][x] = int(value)
    return grid


def _normalize_heatmap(heatmap: list[list[int]] | None, *, width: int, height: int) -> list[list[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    if not heatmap:
        return grid
    for y, row in enumerate(heatmap[:height]):
        for x, value in enumerate(row[:width]):
            grid[y][x] = int(value)
    return grid


def _paint_poi(grid: list[list[int]], poi: dict, *, width: int, height: int) -> None:
    bbox = poi.get("bbox")
    confidence = float(poi.get("confidence", 0.0))
    observations = int(poi.get("observations", poi.get("evidence_count", 1)))
    utility = float(poi.get("utility", 0.0))
    weight = max(1, int(round(2.0 * confidence + 0.5 * utility + 0.25 * observations)))
    if isinstance(bbox, dict):
        x1 = max(0, min(width - 1, int(bbox.get("x1", 0))))
        y1 = max(0, min(height - 1, int(bbox.get("y1", 0))))
        x2 = max(0, min(width - 1, int(bbox.get("x2", x1))))
        y2 = max(0, min(height - 1, int(bbox.get("y2", y1))))
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                grid[y][x] += weight
        return
    centroid = poi.get("centroid")
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        cx = int(float(centroid[0]))
        cy = int(float(centroid[1]))
        if 0 <= cy < height and 0 <= cx < width:
            grid[cy][cx] += weight


def _normalized_intensity(value: int, max_value: int) -> float:
    if max_value <= 0 or value <= 0:
        return 0.0
    baseline = max(4.0, float(max_value))
    return min(1.0, (float(value) ** 0.75) / (baseline ** 0.75))


def _mark_point(pixels: list[list[bytes]], point: list[int] | None, *, color: bytes) -> None:
    if not isinstance(point, list) or len(point) != 2:
        return
    x, y = int(point[0]), int(point[1])
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    for yy in range(max(0, y - 1), min(height, y + 2)):
        for xx in range(max(0, x - 1), min(width, x + 2)):
            pixels[yy][xx] = color


def _render_rgb_matrix(pixels: list[list[bytes]], *, scale: int) -> bytes:
    scale = max(1, int(scale))
    src_height = len(pixels)
    src_width = len(pixels[0]) if pixels else 1
    width = src_width * scale
    height = src_height * scale
    rows = []
    for src_row in pixels:
        expanded_pixels = bytearray()
        for pixel in src_row:
            for _ in range(scale):
                expanded_pixels.extend(pixel)
        row_bytes = bytes(expanded_pixels)
        for _ in range(scale):
            rows.append(b"\x00" + row_bytes)
    return _encode_png(width=width, height=height, raw_data=b"".join(rows))


def _encode_png(*, width: int, height: int, raw_data: bytes) -> bytes:
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(raw_data, level=9)
    return header + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
