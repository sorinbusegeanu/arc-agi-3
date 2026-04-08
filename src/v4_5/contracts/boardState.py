from __future__ import annotations

from dataclasses import dataclass

from v4_5.contracts.boardObject import BoardObject


@dataclass(frozen=True)
class BoardGeometrySummary:
    frame_width: int
    frame_height: int
    bbox_convention: str = "pixel_xyxy"
    center_convention: str = "pixel_center"
    position_convention: str = "pixel_anchor_center"
    pixel_origin: str = "top_left"


@dataclass(frozen=True)
class BoardState:
    schema_version: str
    round_id: str
    board_geometry_summary: BoardGeometrySummary
    source_observation_window_size: int
    avatar_object: BoardObject | None
    objects: tuple[BoardObject, ...]
    traversable_background: tuple[tuple[bool, ...], ...]
    blocking_background: tuple[tuple[bool, ...], ...]
    hazard_representation: tuple[tuple[bool, ...], ...] | None = None
    hud_region: tuple[int, int, int, int] | None = None
    progress_bar_region: tuple[int, int, int, int] | None = None
    lives_region: tuple[int, int, int, int] | None = None
    advisory_only: bool = True
    gaps: tuple[str, ...] = ()

