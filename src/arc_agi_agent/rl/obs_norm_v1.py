from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..normalize import normalize_observation


Scalar = float | int | bool | str | None


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    return {}


def _coerce_grid(name: str, grid_like: Any) -> Optional[Dict[str, Any]]:
    try:
        arr = np.asarray(grid_like)
    except Exception:
        return None
    if arr.ndim < 2:
        return None
    arr2 = arr.astype(int)
    h, w = int(arr2.shape[0]), int(arr2.shape[1])
    return {
        "name": str(name),
        "height": h,
        "width": w,
        "grid": arr2.tolist(),
    }


def _normalize_grids_from_obs(observation: Any) -> List[Dict[str, Any]]:
    warnings: List[str] = []
    norm = normalize_observation(observation, schema_warnings=warnings)
    grids: List[Dict[str, Any]] = []
    for i, grid in enumerate(norm.grids):
        name = norm.grid_names[i] if i < len(norm.grid_names) else f"frame_{i}"
        entry = _coerce_grid(str(name), grid)
        if entry is not None:
            grids.append(entry)
    grids.sort(key=lambda g: (g["name"], g["height"], g["width"]))
    return grids


def _meta_from_fp_report(fp_report: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if fp_report is None:
        return meta

    report_dict = _as_dict(fp_report)
    features = report_dict.get("features_v1", {}) if report_dict else {}
    if isinstance(features, dict):
        meta_features = features.get("meta_features", {})
        if isinstance(meta_features, dict):
            for k in sorted(meta_features.keys()):
                meta[k] = meta_features[k]

    state_summary = report_dict.get("state_summary", {}) if report_dict else {}
    if isinstance(state_summary, dict):
        step_idx = state_summary.get("step_idx")
        if isinstance(step_idx, int):
            meta["step_idx"] = step_idx

    return meta


def _stable_meta(meta: Dict[str, Any]) -> Tuple[List[str], List[float], Dict[str, Any]]:
    canonical: Dict[str, Any] = {}
    for key in sorted(meta.keys()):
        value = meta[key]
        if isinstance(value, bool):
            canonical[key] = bool(value)
        elif isinstance(value, (int, float)):
            canonical[key] = float(value)
        elif isinstance(value, str):
            canonical[key] = value
        elif isinstance(value, list):
            canonical[key] = [str(v) for v in value]
        elif value is None:
            canonical[key] = None
        else:
            canonical[key] = str(value)

    keys = sorted(canonical.keys())
    vec: List[float] = []
    for key in keys:
        value = canonical[key]
        if isinstance(value, bool):
            vec.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vec.append(float(value))
        elif isinstance(value, list):
            vec.append(float(len(value)))
        elif isinstance(value, str):
            vec.append(float(len(value)))
        else:
            vec.append(0.0)
    return keys, vec, canonical


def normalize_obs_v1(observation: Any, fp_report: Optional[Any] = None) -> Dict[str, Any]:
    """Normalize observation into deterministic OBS_NORM_V1.

    Reuses FP analyst-derived normalized meta fields when available.
    """
    if isinstance(observation, dict) and observation.get("schema_version") == "OBS_NORM_V1":
        grids = observation.get("grids") if isinstance(observation.get("grids"), list) else []
        meta = observation.get("meta") if isinstance(observation.get("meta"), dict) else {}
        sorted_grids = sorted(grids, key=lambda g: (str(g.get("name", "")), int(g.get("height", 0)), int(g.get("width", 0))))
        meta_keys, meta_vector, canonical_meta = _stable_meta(meta)
        return {
            "schema_version": "OBS_NORM_V1",
            "grids": sorted_grids,
            "meta": canonical_meta,
            "meta_keys": meta_keys,
            "meta_vector": meta_vector,
            "step_idx": int(observation.get("step_idx", canonical_meta.get("step_idx", 0) or 0)),
        }

    warnings: List[str] = []
    norm = normalize_observation(observation, schema_warnings=warnings)
    grids = _normalize_grids_from_obs(observation)

    meta: Dict[str, Any] = {}
    if isinstance(norm.meta, dict):
        meta.update(norm.meta)
    meta.update(_meta_from_fp_report(fp_report))

    available_actions = meta.get("available_actions")
    if isinstance(available_actions, list):
        meta["available_actions_sorted"] = sorted(str(a) for a in available_actions)
    elif isinstance(meta.get("available_actions_sorted"), list):
        meta["available_actions_sorted"] = sorted(str(a) for a in meta["available_actions_sorted"])
    else:
        meta["available_actions_sorted"] = []

    meta_keys, meta_vector, canonical_meta = _stable_meta(meta)

    return {
        "schema_version": "OBS_NORM_V1",
        "grids": grids,
        "meta": canonical_meta,
        "meta_keys": meta_keys,
        "meta_vector": meta_vector,
        "step_idx": int(norm.step_idx),
        "schema_warnings": warnings,
    }
