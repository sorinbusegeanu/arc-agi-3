from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


def _build_heatmap(coords: Iterable[Tuple[int, int]], grid_size: Tuple[int, int]) -> List[List[int]]:
    width, height = int(grid_size[0]), int(grid_size[1])
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for coord in coords:
        if coord is None or len(coord) != 2:
            continue
        x, y = int(coord[0]), int(coord[1])
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        matrix[y][x] += 1
    return matrix


def build_poi_heatmap_from_coordinates(poi_coords, grid_size=(64, 64)):
    return _build_heatmap(poi_coords, grid_size)


def build_avatar_visit_heatmap_from_coordinates(visit_coords, grid_size=(64, 64)):
    return _build_heatmap(visit_coords, grid_size)


def _normalize(value: float, max_value: float, normalize_mode: str) -> float:
    if max_value <= 0.0:
        return 0.0
    if normalize_mode == "log":
        return math.log1p(value) / math.log1p(max_value)
    return value / max_value


def _heat_color(normalized: float) -> Tuple[int, int, int]:
    normalized = max(0.0, min(1.0, float(normalized)))
    if normalized <= 0.0:
        return (10, 10, 18)
    r = int(255 * min(1.0, normalized * 1.8))
    g = int(255 * min(1.0, max(0.0, (normalized - 0.2) * 1.2)))
    b = int(255 * max(0.0, 1.0 - normalized * 1.4))
    return (r, g, b)


def save_heatmap_png(matrix, output_path, title, normalize_mode):
    if not matrix:
        matrix = [[0]]
    height = len(matrix)
    width = len(matrix[0]) if height else 1
    max_value = max((float(value) for row in matrix for value in row), default=0.0)
    cell_px = max(4, int(512 / max(width, height, 1)))
    title_height = 28
    image = Image.new("RGB", (width * cell_px, height * cell_px + title_height), color=(8, 8, 14))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), str(title), fill=(235, 235, 235))
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            normalized = _normalize(float(value), max_value, normalize_mode)
            color = _heat_color(normalized)
            x0 = x * cell_px
            y0 = title_height + y * cell_px
            draw.rectangle((x0, y0, x0 + cell_px - 1, y0 + cell_px - 1), fill=color)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format="PNG")
