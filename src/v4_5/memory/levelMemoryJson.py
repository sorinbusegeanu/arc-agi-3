from __future__ import annotations

import json

from v4_5.memory.levelMemoryTypes import LevelMemoryRecord, MemoryRegion


def serialize_memory_region(region: MemoryRegion | None) -> str | None:
    if region is None:
        return None
    return json.dumps(_region_dict(region), separators=(",", ":"), sort_keys=True)


def serialize_memory_regions(regions: tuple[MemoryRegion, ...]) -> str:
    return json.dumps([_region_dict(region) for region in regions], separators=(",", ":"), sort_keys=True)


def deserialize_memory_region(payload: str | None) -> MemoryRegion | None:
    if payload is None:
        return None
    return _region_from_dict(json.loads(payload))


def deserialize_memory_regions(payload: str | None) -> tuple[MemoryRegion, ...]:
    if payload is None:
        return ()
    decoded = json.loads(payload)
    return tuple(_region_from_dict(item) for item in decoded)


def _region_dict(region: MemoryRegion) -> dict[str, object]:
    return {
        "bbox": list(region.bbox),
        "center": [float(region.center[0]), float(region.center[1])],
        "colors": list(region.colors),
        "description": region.description,
        "hint": region.hint,
    }


def _region_from_dict(payload: dict[str, object]) -> MemoryRegion:
    return MemoryRegion(
        bbox=tuple(int(value) for value in payload["bbox"]),
        center=(float(payload["center"][0]), float(payload["center"][1])),
        colors=tuple(int(value) for value in payload.get("colors", ())),
        description=payload.get("description"),
        hint=payload.get("hint"),
    )
