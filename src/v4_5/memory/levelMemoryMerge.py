from __future__ import annotations

from dataclasses import replace

from v4_5.memory.levelMemoryTypes import LevelMemoryRecord, MemoryRegion


def merge_level_memory(existing: LevelMemoryRecord | None, incoming: LevelMemoryRecord) -> LevelMemoryRecord:
    if existing is None:
        return incoming
    return LevelMemoryRecord(
        game_id=incoming.game_id,
        level_id=incoming.level_id,
        memory_state="validated" if "validated" in {existing.memory_state, incoming.memory_state} else "hypothesis",
        avatar=incoming.avatar if incoming.avatar is not None else existing.avatar,
        hud_regions=_merge_region_list(existing.hud_regions, incoming.hud_regions),
        life_regions=_merge_region_list(existing.life_regions, incoming.life_regions),
        progress_regions=_merge_region_list(existing.progress_regions, incoming.progress_regions),
        pois=_merge_region_list(existing.pois, incoming.pois),
        exit_regions=_merge_region_list(existing.exit_regions, incoming.exit_regions),
        created_at=existing.created_at,
        updated_at=incoming.updated_at,
        schema_version=incoming.schema_version,
    )


def _merge_region_list(existing: tuple[MemoryRegion, ...], incoming: tuple[MemoryRegion, ...]) -> tuple[MemoryRegion, ...]:
    if not incoming:
        return existing
    merged = []
    for idx, region in enumerate(incoming):
        current = existing[idx] if idx < len(existing) else None
        merged.append(_merge_region(current, region))
    return tuple(merged)


def _merge_region(existing: MemoryRegion | None, incoming: MemoryRegion) -> MemoryRegion:
    if existing is None:
        return incoming
    return replace(
        incoming,
        description=incoming.description if incoming.description is not None else existing.description,
        hint=incoming.hint if incoming.hint is not None else existing.hint,
    )
