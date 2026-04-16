from __future__ import annotations

from typing import Mapping

try:
    from arc_agi.rendering import COLOR_MAP as _ENGINE_COLOR_MAP
except Exception:
    _ENGINE_COLOR_MAP = {
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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = str(hex_color).lstrip("#")
    if len(raw) < 6:
        return (0, 0, 0)
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def get_render_palette() -> dict[int, tuple[int, int, int]]:
    return {int(value): _hex_to_rgb(color) for value, color in dict(_ENGINE_COLOR_MAP).items()}


def render_value_to_rgb(value: int, palette: Mapping[int, tuple[int, int, int]] | None = None) -> tuple[int, int, int]:
    resolved = dict(palette) if palette is not None else get_render_palette()
    return tuple(int(channel) for channel in resolved.get(int(value), (0, 0, 0)))

