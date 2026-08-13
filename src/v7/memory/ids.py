from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NewType

MemoryId = NewType("MemoryId", int)
_MAX_UINT64 = (1 << 64) - 1


class MemoryLevel(IntEnum):
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6


def validate_memory_id(value: int) -> MemoryId:
    value = int(value)
    if value <= 0 or value > _MAX_UINT64:
        raise ValueError(f"memory id must be in 1..{_MAX_UINT64}")
    return MemoryId(value)


@dataclass(slots=True)
class MemoryIdAllocator:
    """Deterministic monotonic uint64 allocator owned by the canonical writer."""

    next_value: int = 1

    def allocate(self) -> MemoryId:
        memory_id = validate_memory_id(self.next_value)
        self.next_value += 1
        return memory_id

    def allocate_many(self, count: int) -> tuple[MemoryId, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return tuple(self.allocate() for _ in range(count))
