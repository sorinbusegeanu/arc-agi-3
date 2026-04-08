from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryRegion:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    colors: tuple[int, ...] = ()
    description: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class LevelMemoryRecord:
    game_id: str
    level_id: str
    memory_state: str
    avatar: MemoryRegion | None
    hud_regions: tuple[MemoryRegion, ...]
    life_regions: tuple[MemoryRegion, ...]
    progress_regions: tuple[MemoryRegion, ...]
    pois: tuple[MemoryRegion, ...]
    exit_regions: tuple[MemoryRegion, ...]
    created_at: str
    updated_at: str
    schema_version: str
