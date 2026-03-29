from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


COORDINATE_BOUNDS_ACTION_NAME = "ACTION6"


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable_scalar(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_jsonable_scalar(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable_scalar(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class V4EnvironmentMetadata:
    source_object_name: str
    raw_payload: dict[str, Any]
    game_id: str
    title: str | None = None
    description: str | None = None
    local_dir: str | None = None
    date_downloaded: str | None = None
    action_ids: tuple[int, ...] = ()
    action_names: tuple[str, ...] = ()
    coordinate_action_id: int | None = None
    coordinate_bounds: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_object_name, str) or not self.source_object_name:
            raise ValueError("source_object_name: must be a non-empty string")
        if not isinstance(self.raw_payload, dict):
            raise ValueError("raw_payload: must be a dict")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id: must be a non-empty string")
        for field_name in ("title", "description", "local_dir", "date_downloaded"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name}: must be a string or null")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("action_ids: must be a tuple")
        if not isinstance(self.action_names, tuple):
            raise ValueError("action_names: must be a tuple")
        if self.action_names and len(self.action_ids) != len(self.action_names):
            raise ValueError("action_names: must align 1:1 with action_ids")
        for index, action_id in enumerate(self.action_ids):
            if not isinstance(action_id, int):
                raise ValueError(f"action_ids[{index}]: must be an int")
        for index, action_name in enumerate(self.action_names):
            if not isinstance(action_name, str) or not action_name:
                raise ValueError(f"action_names[{index}]: must be a non-empty string")
        if self.coordinate_action_id is not None:
            if not isinstance(self.coordinate_action_id, int):
                raise ValueError("coordinate_action_id: must be an int or null")
            if self.coordinate_action_id not in self.action_ids:
                raise ValueError("coordinate_action_id: must be present in action_ids")
            if not self.coordinate_bounds:
                raise ValueError("coordinate_bounds: required when coordinate_action_id is present")
        if self.coordinate_bounds is not None:
            if not isinstance(self.coordinate_bounds, tuple) or len(self.coordinate_bounds) != 4:
                raise ValueError("coordinate_bounds: must be a 4-tuple or null")
            for index, value in enumerate(self.coordinate_bounds):
                if not isinstance(value, int):
                    raise ValueError(f"coordinate_bounds[{index}]: must be an int")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable_scalar(asdict(self))
