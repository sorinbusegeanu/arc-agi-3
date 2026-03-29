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
class HybridConstructionCommonFieldsV4:
    game_family: str
    game_id: str
    level_index: int
    avatar_position: GridPos
    current_legal_actions: tuple[int, ...]
    terminal_status: str
    step_depth: int
    static_bounds: GridPos


@dataclass(frozen=True)
class HybridConstructionFamilyFieldsV4:
    land_cells: tuple[GridPos, ...] = ()
    water_cells: tuple[GridPos, ...] = ()
    reef_cells: tuple[GridPos, ...] = ()
    bridge_built_cells: tuple[GridPos, ...] = ()
    bridge_budget_remaining: int | None = None
    step_limit_remaining: int | None = None
    goal_cell: GridPos | None = None
    legal_movement_actions: tuple[int, ...] = ()
    legal_click_cells: tuple[GridPos, ...] = ()

    def __post_init__(self) -> None:
        _validate_pos_tuple("land_cells", self.land_cells)
        _validate_pos_tuple("water_cells", self.water_cells)
        _validate_pos_tuple("reef_cells", self.reef_cells)
        _validate_pos_tuple("bridge_built_cells", self.bridge_built_cells)
        if self.goal_cell is not None:
            _validate_pos("goal_cell", self.goal_cell)
        _validate_pos_tuple("legal_click_cells", self.legal_click_cells)


@dataclass(frozen=True)
class HybridConstructionTypedStateV4:
    common: HybridConstructionCommonFieldsV4
    family: HybridConstructionFamilyFieldsV4 = field(default_factory=HybridConstructionFamilyFieldsV4)
    layout_evidence_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HybridConstructionTypedStateV4:
        return cls(
            common=HybridConstructionCommonFieldsV4(**payload["common"]),
            family=HybridConstructionFamilyFieldsV4(**payload.get("family", {})),
            layout_evidence_source=payload.get("layout_evidence_source"),
        )

    def to_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
