from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from arcengine import FrameDataRaw
except Exception:  # pragma: no cover
    FrameDataRaw = None  # type: ignore


@dataclass
class NormalizedObservation:
    grids: List[np.ndarray]
    grid_names: List[str]
    meta: Dict[str, Any]
    step_idx: int


def normalize_observation(
    observation: Any, *, schema_warnings: List[str]
) -> NormalizedObservation:
    if FrameDataRaw is not None and isinstance(observation, FrameDataRaw):
        grids = [np.array(layer) for layer in observation.frame]
        names = [f"frame_{i}" for i in range(len(grids))]
        meta = {
            "game_id": getattr(observation, "game_id", None),
            "state": getattr(observation, "state", None),
            "levels_completed": getattr(observation, "levels_completed", None),
            "win_levels": getattr(observation, "win_levels", None),
            "guid": getattr(observation, "guid", None),
            "available_actions": getattr(observation, "available_actions", None),
            "terminal": getattr(observation, "terminal", None),
            "reward": getattr(observation, "reward", None),
        }
        return NormalizedObservation(grids=grids, grid_names=names, meta=meta, step_idx=0)

    if isinstance(observation, dict):
        payload = observation
        if "data" in payload and isinstance(payload["data"], dict):
            payload = payload["data"]

        grids: List[np.ndarray] = []
        names: List[str] = []

        if "grids" in payload:
            raw_grids = payload.get("grids")
            if isinstance(raw_grids, dict):
                for name, grid in raw_grids.items():
                    names.append(str(name))
                    grids.append(np.array(grid))
            elif isinstance(raw_grids, list):
                grids = [np.array(layer) for layer in raw_grids]
                raw_names = payload.get("grid_names")
                if isinstance(raw_names, list) and len(raw_names) == len(grids):
                    names = [str(n) for n in raw_names]
                else:
                    names = [f"frame_{i}" for i in range(len(grids))]
            else:
                schema_warnings.append("Invalid 'grids' format; expected dict or list")
        else:
            frame = payload.get("frame")
            if frame is None:
                schema_warnings.append("Missing 'frame' in observation payload")
                grids = []
            else:
                grids = [np.array(layer) for layer in frame]
            names = [f"frame_{i}" for i in range(len(grids))]

        meta = {
            "game_id": payload.get("game_id"),
            "state": payload.get("state"),
            "levels_completed": payload.get("levels_completed"),
            "win_levels": payload.get("win_levels"),
            "guid": payload.get("guid"),
            "available_actions": payload.get("available_actions"),
            "terminal": payload.get("terminal"),
            "reward": payload.get("reward"),
        }
        step_candidate = payload.get("step_idx", payload.get("step", payload.get("steps", 0)))
        step_idx = int(step_candidate) if step_candidate is not None else 0
        return NormalizedObservation(grids=grids, grid_names=names, meta=meta, step_idx=step_idx)

    schema_warnings.append("Unsupported observation type; expected FrameDataRaw or dict")
    return NormalizedObservation(grids=[], grid_names=[], meta={}, step_idx=0)
