from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ActionV1:
    type: str
    id: int
    x: Optional[int] = None
    y: Optional[int] = None
    raw: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"type": self.type, "id": int(self.id)}
        if self.type == "coord":
            payload["x"] = int(self.x) if self.x is not None else None
            payload["y"] = int(self.y) if self.y is not None else None
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload


@dataclass
class AvailableActionsV1:
    discrete_mask: Optional[List[bool]] = None
    coord_enabled: bool = False
    coord_action_id: Optional[int] = None
    coord_bounds: Optional[Tuple[int, int, int, int]] = None
    allowed_coords: Optional[List[Tuple[int, int]]] = None
    tags: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "discrete_mask": self.discrete_mask,
            "coord_enabled": bool(self.coord_enabled),
            "coord_action_id": self.coord_action_id,
            "coord_bounds": self.coord_bounds,
            "allowed_coords": self.allowed_coords,
            "tags": self.tags,
        }
        return payload


@dataclass
class ObsV1:
    grid: List[List[int]]
    h: int
    w: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"grid": self.grid, "h": int(self.h), "w": int(self.w)}
        payload.update(self.meta)
        return payload


@dataclass
class StepRecordV1:
    t: int
    obs: Dict[str, Any]
    available_actions: Dict[str, Any]
    action: Optional[Dict[str, Any]]
    action_valid: Optional[bool]
    reward: Optional[float]
    done: bool
    info_passthrough: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeTraceV1:
    schema_version: str
    agent_name: str
    episode_id: str
    game_id: str
    seed: int
    trace_id: str
    timestamp_step: int
    probe_steps_requested: int
    probe_steps_executed: int
    terminated_early_reason: str
    steps: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapExplorerReportV1:
    schema_version: str
    agent_name: str
    episode_id: str
    game_id: str
    seed: int
    trace_id: str
    timestamp_step: int
    start_obs_shape: Tuple[int, int]
    probe_steps_executed: int
    ended_with_done: bool
    action_selection_mode: str
    errors: List[str]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_action(action: Any) -> Optional[ActionV1]:
    if action is None:
        return None
    if isinstance(action, ActionV1):
        return action
    if isinstance(action, dict):
        return ActionV1(
            type=str(action.get("type", "discrete")),
            id=int(action.get("id", 0)),
            x=action.get("x"),
            y=action.get("y"),
            raw=action.get("raw"),
        )
    raise TypeError(f"Unsupported action type: {type(action)}")


def normalize_available_actions(payload: Any) -> AvailableActionsV1:
    if isinstance(payload, AvailableActionsV1):
        return payload
    if isinstance(payload, dict):
        allowed_coords_raw = payload.get("allowed_coords")
        allowed_coords = None
        if isinstance(allowed_coords_raw, list):
            allowed_coords = [tuple(xy) for xy in allowed_coords_raw]
        return AvailableActionsV1(
            discrete_mask=list(payload.get("discrete_mask") or []) or None,
            coord_enabled=bool(payload.get("coord_enabled", False)),
            coord_action_id=payload.get("coord_action_id"),
            coord_bounds=tuple(payload.get("coord_bounds")) if payload.get("coord_bounds") else None,
            allowed_coords=allowed_coords,
            tags=payload.get("tags"),
        )
    raise TypeError(f"Unsupported available_actions type: {type(payload)}")


def normalize_obs(payload: Any) -> ObsV1:
    if isinstance(payload, ObsV1):
        return payload
    if isinstance(payload, dict):
        grid = _normalize_grid(payload.get("grid"))
        h = int(payload.get("h", len(grid)))
        w = int(payload.get("w", len(grid[0]) if grid else 0))
        meta = {k: v for k, v in payload.items() if k not in {"grid", "h", "w"}}
        return ObsV1(grid=grid, h=h, w=w, meta=meta)
    raise TypeError(f"Unsupported obs type: {type(payload)}")


def _normalize_grid(grid: Any) -> List[List[int]]:
    if grid is None:
        return []
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [list(map(int, row)) for row in grid]


def canonical_points(h: int, w: int) -> List[Tuple[int, int]]:
    if h <= 0 or w <= 0:
        return []
    center = (w // 2, h // 2)
    points = [
        center,
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, 0),
        (w // 2, h - 1),
        (0, h // 2),
        (w - 1, h // 2),
    ]
    seen = set()
    ordered: List[Tuple[int, int]] = []
    for x, y in points:
        x_clipped = max(0, min(w - 1, int(x)))
        y_clipped = max(0, min(h - 1, int(y)))
        pt = (x_clipped, y_clipped)
        if pt not in seen:
            seen.add(pt)
            ordered.append(pt)
    return ordered


def normalize_info(info: Any) -> Optional[Dict[str, Any]]:
    if info is None:
        return None
    if isinstance(info, dict):
        return info
    return {"repr": str(info)}


def count_true(mask: Optional[Sequence[bool]]) -> int:
    if not mask:
        return 0
    return sum(1 for v in mask if v)


def action_tags_for_id(tags: Optional[Dict[str, Any]], action_id: int) -> List[str]:
    if not tags:
        return []
    if isinstance(tags, dict):
        if "discrete" in tags and isinstance(tags["discrete"], dict):
            candidate = tags["discrete"].get(str(action_id)) or tags["discrete"].get(action_id)
            if candidate is None:
                return []
            return _normalize_tag_list(candidate)
        candidate = tags.get(str(action_id)) or tags.get(action_id)
        if candidate is not None:
            return _normalize_tag_list(candidate)
    return []


def _normalize_tag_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]
