from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .types import BBox, Component


def ascii_grid(grid: np.ndarray) -> str:
    h, w = grid.shape
    header = "   " + "".join(_coord_glyph(x) for x in range(w))
    lines = [header]
    for y in range(h):
        row = "".join(_color_glyph(int(v)) for v in grid[y])
        lines.append(f"{_coord_label(y)} {row}")
    return "\n".join(lines)


def ascii_overlay(
    base_grid: np.ndarray,
    overlay: np.ndarray,
) -> str:
    h, w = base_grid.shape
    header = "   " + "".join(_coord_glyph(x) for x in range(w))
    lines = [header]
    for y in range(h):
        row_chars = []
        for x in range(w):
            if overlay[y, x] != " ":
                row_chars.append(overlay[y, x])
            else:
                row_chars.append(_color_glyph(int(base_grid[y, x])))
        lines.append(f"{_coord_label(y)} {''.join(row_chars)}")
    return "\n".join(lines)


def bbox_overlay(grid: np.ndarray, components: List[Component]) -> np.ndarray:
    overlay = np.full(grid.shape, " ", dtype="<U1")
    for comp in components:
        y0, x0, y1, x1 = comp.bbox
        for x in range(x0, x1 + 1):
            overlay[y0, x] = "#"
            overlay[y1, x] = "#"
        for y in range(y0, y1 + 1):
            overlay[y, x0] = "#"
            overlay[y, x1] = "#"
    return overlay


def component_id_overlay(grid: np.ndarray, components: List[Component]) -> np.ndarray:
    overlay = np.full(grid.shape, " ", dtype="<U1")
    for idx, comp in enumerate(components):
        cy = int(round(comp.centroid[0]))
        cx = int(round(comp.centroid[1]))
        if 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]:
            overlay[cy, cx] = _id_glyph(idx)
    return overlay


def diff_mask_overlay(grid: np.ndarray, diff_mask: np.ndarray) -> np.ndarray:
    overlay = np.full(grid.shape, " ", dtype="<U1")
    ys, xs = np.where(diff_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        overlay[y, x] = "*"
    return overlay


def motion_overlay(grid: np.ndarray, motions: List[Tuple[float, float, float, float]]) -> np.ndarray:
    overlay = np.full(grid.shape, " ", dtype="<U1")
    for y0, x0, y1, x1 in motions:
        sy = int(round(y0))
        sx = int(round(x0))
        ty = int(round(y1))
        tx = int(round(x1))
        if 0 <= sy < grid.shape[0] and 0 <= sx < grid.shape[1]:
            overlay[sy, sx] = "+"
        if 0 <= ty < grid.shape[0] and 0 <= tx < grid.shape[1]:
            overlay[ty, tx] = ">"
    return overlay


def _coord_glyph(x: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return digits[x % len(digits)]


def _coord_label(y: int) -> str:
    return str(y).rjust(2, " ")


def _color_glyph(color: int) -> str:
    glyphs = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return glyphs[color % len(glyphs)]


def _id_glyph(idx: int) -> str:
    glyphs = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return glyphs[idx % len(glyphs)]


try:
    from arc_agi.rendering import COLOR_MAP as ARC_COLOR_MAP
except Exception:
    ARC_COLOR_MAP = {
        0: "#FFFFFFFF",
        1: "#CCCCCCFF",
        2: "#999999FF",
        3: "#666666FF",
        4: "#333333FF",
        5: "#000000FF",
        6: "#E53AA3FF",
        7: "#FF7BCCFF",
        8: "#F93C31FF",
        9: "#1E93FFFF",
        10: "#88D8F1FF",
        11: "#FFDC00FF",
        12: "#FF851BFF",
        13: "#921231FF",
        14: "#4FCC30FF",
        15: "#A356D6FF",
    }


def save_grid_image(path: str, grid: np.ndarray) -> Optional[str]:
    try:
        from PIL import Image
    except Exception:
        return None
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            color = ARC_COLOR_MAP.get(int(grid[y, x]), "#000000FF")
            r, g, b = _hex_to_rgb(color)
            rgb[y, x] = (r, g, b)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)
    return path


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b
