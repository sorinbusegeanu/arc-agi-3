from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from vlm_v2.video_builder import build_episode_video

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


def write_episode_video_from_observations(
    observations: Iterable[Any],
    *,
    output_root: Path,
    fps: int = 2,
) -> str:
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    for frame_index, obs in enumerate(observations):
        frame_paths.append(write_observation_png(obs, frames_dir, frame_index))
    if not frame_paths:
        raise RuntimeError("cannot build episode video: no frames")
    return build_episode_video(str(frames_dir), fps=int(fps), output_name="episode.mp4", frame_paths=frame_paths)


def write_observation_png(observation: Any, frame_dir: Path, frame_index: int) -> str:
    grid = _observation_to_palette_grid(observation)
    rgb = _palette_rgb(grid)
    image = Image.fromarray(rgb, mode="RGB")
    upscaled = image.resize((image.width * 10, image.height * 10), resample=Image.Resampling.NEAREST)
    frame_path = frame_dir / f"frame_{frame_index:06d}.png"
    upscaled.save(frame_path)
    return str(frame_path)


def _observation_to_palette_grid(observation: Any) -> np.ndarray:
    if not hasattr(observation, "current_frame"):
        raise ValueError("observation is missing current_frame")
    frame = observation.current_frame
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported frame shape: {tuple(arr.shape)}")
    grid = np.rint(arr * 255.0).astype(np.int64)
    grid = np.clip(grid, 0, 15)
    if hasattr(observation, "valid_pixel_mask") and observation.valid_pixel_mask is not None:
        valid = observation.valid_pixel_mask
        if isinstance(valid, torch.Tensor):
            valid_mask = valid.detach().cpu().numpy()
        else:
            valid_mask = np.asarray(valid)
        valid_mask = np.asarray(valid_mask)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0]
        if valid_mask.shape == grid.shape:
            grid = np.where(valid_mask > 0, grid, 0)
    return grid


def _palette_rgb(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb[y, x] = _hex_to_rgb(ARC_COLOR_MAP.get(int(grid[y, x]), "#000000FF"))
    return rgb


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    token = hex_color.lstrip("#")
    return int(token[0:2], 16), int(token[2:4], 16), int(token[4:6], 16)

