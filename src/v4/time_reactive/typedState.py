from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GridPos = tuple[int, int]


def _validate_pos(field_name: str, value: GridPos) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name}: must be a 2-tuple")
    if not all(isinstance(part, int) for part in value):
        raise ValueError(f"{field_name}: coordinates must be ints")


def _validate_pos_tuple(field_name: str, values: tuple[GridPos, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name}: must be a tuple")
    for index, value in enumerate(values):
        _validate_pos(f"{field_name}[{index}]", value)


@dataclass(frozen=True)
class TimeReactiveCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    avatar_position: GridPos
    walkable_cells: tuple[GridPos, ...]
    current_legal_actions: tuple[int, ...]
    terminal_status: str
    step_depth: int
    static_bounds: GridPos


@dataclass(frozen=True)
class TimeReactiveFamilyFieldsV4:
    food_cells: tuple[GridPos, ...] = ()
    warm_zone_cells: tuple[GridPos, ...] = ()
    hunger_value: int = 0
    warmth_value: int = 0
    survival_timer_remaining: int = 0
    hunger_decay_per_step: int = 2
    warmth_decay_per_step: int = 2
    food_restore_amount: int = 20
    wait_action_id: int | None = None

    def __post_init__(self) -> None:
        _validate_pos_tuple("food_cells", self.food_cells)
        _validate_pos_tuple("warm_zone_cells", self.warm_zone_cells)


@dataclass(frozen=True)
class TimeReactiveTypedStateV4:
    common: TimeReactiveCommonFieldsV4
    family: TimeReactiveFamilyFieldsV4 = field(default_factory=TimeReactiveFamilyFieldsV4)
    layout_evidence_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimeReactiveTypedStateV4:
        return cls(
            common=TimeReactiveCommonFieldsV4(**payload["common"]),
            family=TimeReactiveFamilyFieldsV4(**payload.get("family", {})),
            layout_evidence_source=payload.get("layout_evidence_source"),
        )

    def to_key(self) -> str:
        payload = self.to_dict()
        common = payload.get("common", {})
        if isinstance(common, dict):
            common.pop("step_depth", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
