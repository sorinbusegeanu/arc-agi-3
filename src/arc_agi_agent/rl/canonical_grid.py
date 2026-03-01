from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np


def canonical_grid(obs_norm: Dict[str, Any], default_h: int = 64, default_w: int = 64) -> np.ndarray:
    grids = obs_norm.get("grids") if isinstance(obs_norm, dict) else None
    if isinstance(grids, list) and grids:
        g0 = grids[0]
        if isinstance(g0, dict) and "grid" in g0:
            arr = np.asarray(g0["grid"], dtype=np.int64)
            if arr.ndim >= 2:
                return np.asarray(arr[: arr.shape[0], : arr.shape[1]], dtype=np.int64)
    return np.zeros((default_h, default_w), dtype=np.int64)


def stable_hash_grid(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=np.int64)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()
