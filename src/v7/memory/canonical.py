from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v7.memory.ids import MemoryId, MemoryIdAllocator, MemoryLevel


@dataclass(frozen=True, order=True, slots=True)
class CanonicalMemoryKey:
    """Stable semantic identity independent of worker completion order."""

    level: MemoryLevel
    type_id: int
    parts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.type_id < 0:
            raise ValueError("type_id must be non-negative")
        if not self.parts:
            raise ValueError("canonical key parts cannot be empty")


class CanonicalMemoryRegistry:
    """Single-writer deterministic key-to-MemoryId resolver."""

    def __init__(self, *, start_id: int = 1) -> None:
        self._allocator = MemoryIdAllocator(start_id)
        self._ids_by_key: dict[CanonicalMemoryKey, MemoryId] = {}
        self._keys_by_id: dict[MemoryId, CanonicalMemoryKey] = {}

    def get(self, key: CanonicalMemoryKey) -> MemoryId | None:
        return self._ids_by_key.get(key)

    def key_for(self, memory_id: MemoryId) -> CanonicalMemoryKey | None:
        return self._keys_by_id.get(memory_id)

    def observe_existing_id(self, memory_id: MemoryId) -> None:
        next_value = int(memory_id) + 1
        if self._allocator.next_value < next_value:
            self._allocator.next_value = next_value

    def bind_existing(self, key: CanonicalMemoryKey, memory_id: MemoryId) -> None:
        existing_id = self._ids_by_key.get(key)
        existing_key = self._keys_by_id.get(memory_id)
        if existing_id is not None and existing_id != memory_id:
            raise ValueError("canonical key is already bound to another memory ID")
        if existing_key is not None and existing_key != key:
            raise ValueError("memory ID is already bound to another canonical key")
        self._ids_by_key[key] = memory_id
        self._keys_by_id[memory_id] = key
        self.observe_existing_id(memory_id)

    def resolve_many(self, keys: Iterable[CanonicalMemoryKey]) -> dict[CanonicalMemoryKey, MemoryId]:
        unique = tuple(sorted(set(keys)))
        for key in unique:
            if key in self._ids_by_key:
                continue
            memory_id = self._allocator.allocate()
            self._ids_by_key[key] = memory_id
            self._keys_by_id[memory_id] = key
        return {key: self._ids_by_key[key] for key in unique}

    @property
    def count(self) -> int:
        return len(self._ids_by_key)
