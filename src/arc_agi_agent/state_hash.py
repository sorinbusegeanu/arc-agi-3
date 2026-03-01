from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


_FILTERED_SENTINEL = -1


def hash_state(grid: Any) -> str:
    arr = np.asarray(grid)
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def hash_state_filtered(grid: Any, ui_mask: Any) -> str:
    arr = np.asarray(grid)
    mask = np.asarray(ui_mask, dtype=bool)
    if mask.shape != arr.shape:
        raise ValueError(f"ui_mask shape {mask.shape} does not match grid shape {arr.shape}")
    # Upcast to ensure sentinel cannot collide with normal unsigned palette values.
    filtered = arr.astype(np.int16, copy=True)
    filtered[mask] = _FILTERED_SENTINEL
    hasher = hashlib.sha256()
    hasher.update(str(filtered.shape).encode("utf-8"))
    hasher.update(filtered.tobytes())
    return hasher.hexdigest()
