from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def write_frame_png(observation: Any, frame_dir: str, frame_index: int) -> str:
    image = _observation_to_image(observation)
    upscaled = image.resize((image.width * 10, image.height * 10), resample=Image.Resampling.NEAREST)
    frame_path = Path(frame_dir) / f"frame_{frame_index:06d}.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(frame_path)
    return str(frame_path)


def _observation_to_image(observation: Any) -> Image.Image:
    array = np.asarray(observation)
    if array.ndim == 2:
        return Image.fromarray(_palette_rgb(array), mode="RGB")
    if array.ndim == 3 and array.shape[2] in (3, 4):
        clipped = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(clipped[:, :, :3], mode="RGB")
    raise ValueError(f"unsupported observation shape: {tuple(array.shape)}")


def _palette_rgb(grid: np.ndarray) -> np.ndarray:
    clipped = np.asarray(grid, dtype=np.int64)
    height, width = clipped.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb[y, x] = _hex_to_rgb(ARC_COLOR_MAP.get(int(clipped[y, x]), "#000000FF"))
    return rgb


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    token = hex_color.lstrip("#")
    return int(token[0:2], 16), int(token[2:4], 16), int(token[4:6], 16)
