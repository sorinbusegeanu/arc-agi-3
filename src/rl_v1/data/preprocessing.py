from __future__ import annotations

from typing import Any

import numpy as np
import torch


def frame_to_model_tensor(
    frame_data: Any,
    canvas_height: int,
    canvas_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_layers = getattr(frame_data, "frame", None) or []
    if not frame_layers:
        image = np.zeros((1, canvas_height, canvas_width), dtype=np.float32)
        mask = np.zeros((1, canvas_height, canvas_width), dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)
    frame = np.asarray(frame_layers[0], dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"expected 2d frame layer, got shape={frame.shape}")
    height, width = frame.shape
    if height > canvas_height or width > canvas_width:
        raise ValueError(
            f"frame {height}x{width} exceeds canvas {canvas_height}x{canvas_width}"
        )
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    valid = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas[:height, :width] = frame / 255.0
    valid[:height, :width] = 1.0
    return torch.from_numpy(canvas[None, ...]), torch.from_numpy(valid[None, ...])
