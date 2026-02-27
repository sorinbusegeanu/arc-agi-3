from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    kind: str  # "simple" or "coord"


@dataclass(frozen=True)
class ActionSchema:
    version: str
    primary_grid: GridSpec
    actions: List[ActionSpec]


def build_action_schema_from_env(
    action_space: List[Any],
    *,
    width: int,
    height: int,
    version: str = "1.0",
) -> ActionSchema:
    actions: List[ActionSpec] = []
    for action in action_space:
        name = getattr(action, "name", None)
        if not name:
            continue
        kind = "coord" if getattr(action, "is_complex", lambda: False)() else "simple"
        actions.append(ActionSpec(action_id=name, kind=kind))
    if not actions:
        raise ValueError("action_space produced no actions")
    return ActionSchema(version=version, primary_grid=GridSpec(width=width, height=height), actions=actions)


def parse_action_schema_data(data: Dict[str, Any]) -> ActionSchema:
    if not isinstance(data, dict):
        raise ValueError("action_schema must be a JSON object")

    allowed_keys = {"version", "primary_grid", "actions"}
    extra = set(data.keys()) - allowed_keys
    if extra:
        raise ValueError(f"action_schema has unsupported keys: {sorted(extra)}")

    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("action_schema.version is required and must be a non-empty string")

    grid = data.get("primary_grid")
    if not isinstance(grid, dict):
        raise ValueError("action_schema.primary_grid is required and must be an object")
    grid_extra = set(grid.keys()) - {"width", "height"}
    if grid_extra:
        raise ValueError(f"primary_grid has unsupported keys: {sorted(grid_extra)}")
    width = grid.get("width")
    height = grid.get("height")
    if not isinstance(width, int) or width <= 0:
        raise ValueError("primary_grid.width is required and must be an int > 0")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("primary_grid.height is required and must be an int > 0")

    actions_raw = data.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise ValueError("action_schema.actions is required and must be a non-empty list")

    actions: List[ActionSpec] = []
    seen_ids = set()
    for idx, item in enumerate(actions_raw):
        if not isinstance(item, dict):
            raise ValueError(f"actions[{idx}] must be an object")
        extra_action = set(item.keys()) - {"action_id", "kind"}
        if extra_action:
            raise ValueError(f"actions[{idx}] has unsupported keys: {sorted(extra_action)}")
        action_id = item.get("action_id")
        kind = item.get("kind")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"actions[{idx}].action_id must be a non-empty string")
        if action_id in seen_ids:
            raise ValueError(f"duplicate action_id in action_schema: {action_id}")
        if kind not in ("simple", "coord"):
            raise ValueError(f"actions[{idx}].kind must be 'simple' or 'coord'")
        actions.append(ActionSpec(action_id=action_id, kind=kind))
        seen_ids.add(action_id)

    return ActionSchema(version=version, primary_grid=GridSpec(width=width, height=height), actions=actions)
