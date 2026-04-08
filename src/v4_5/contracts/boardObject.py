from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardObject:
    object_id: str
    object_type: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    position_x: float
    position_y: float
    color: int

