from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from v7.memory.ids import MemoryId, MemoryLevel


@dataclass(frozen=True, slots=True)
class DependencyMutation:
    source_id: MemoryId
    source_level: MemoryLevel
    target_id: MemoryId
    target_level: MemoryLevel

    def __post_init__(self) -> None:
        if int(self.target_level) <= int(self.source_level):
            raise ValueError("dependency target must be at a higher memory level")


@dataclass(frozen=True, slots=True)
class DirtyDerivationPlan:
    by_level: Mapping[MemoryLevel, tuple[MemoryId, ...]]

    @classmethod
    def empty(cls) -> "DirtyDerivationPlan":
        return cls(by_level=MappingProxyType({}))

    def ids_for_level(self, level: MemoryLevel) -> tuple[MemoryId, ...]:
        return self.by_level.get(level, ())

    @property
    def total_count(self) -> int:
        return sum(len(values) for values in self.by_level.values())


class MemoryDependencyGraph:
    """Writer-owned reverse dependency graph for incremental M1-M6 derivation.

    An edge source -> target means a change in source can invalidate/recompute target.
    Only upward dependencies are allowed. Dirty propagation is deterministic and
    traverses only registered affected neighborhoods rather than whole populations.
    """

    def __init__(self) -> None:
        self._levels: dict[MemoryId, MemoryLevel] = {}
        self._upstream_to_dependents: dict[MemoryId, set[MemoryId]] = {}
        self._dirty: set[MemoryId] = set()

    def register_node(self, memory_id: MemoryId, level: MemoryLevel) -> None:
        existing = self._levels.get(memory_id)
        if existing is not None and existing != level:
            raise ValueError(f"dependency node level is immutable for memory_id={int(memory_id)}")
        self._levels[memory_id] = level

    def apply_dependency_batch(self, mutations: Iterable[DependencyMutation]) -> int:
        unique = {
            (
                mutation.source_id,
                mutation.source_level,
                mutation.target_id,
                mutation.target_level,
            )
            for mutation in mutations
        }
        for source_id, source_level, target_id, target_level in unique:
            self.register_node(source_id, source_level)
            self.register_node(target_id, target_level)
        for source_id, _, target_id, _ in unique:
            self._upstream_to_dependents.setdefault(source_id, set()).add(target_id)
        return len(unique)

    def mark_dirty(self, memory_ids: Iterable[MemoryId]) -> int:
        seeds = tuple(dict.fromkeys(memory_ids))
        unknown = [memory_id for memory_id in seeds if memory_id not in self._levels]
        if unknown:
            raise KeyError(f"unregistered dependency nodes: {[int(value) for value in unknown]}")

        queue: deque[MemoryId] = deque(seeds)
        newly_dirty: set[MemoryId] = set()
        while queue:
            memory_id = queue.popleft()
            if memory_id in self._dirty or memory_id in newly_dirty:
                continue
            newly_dirty.add(memory_id)
            for dependent in sorted(
                self._upstream_to_dependents.get(memory_id, ()), key=int
            ):
                if dependent not in self._dirty and dependent not in newly_dirty:
                    queue.append(dependent)
        self._dirty.update(newly_dirty)
        return len(newly_dirty)

    def snapshot_plan(self) -> DirtyDerivationPlan:
        grouped: dict[MemoryLevel, list[MemoryId]] = {}
        for memory_id in self._dirty:
            grouped.setdefault(self._levels[memory_id], []).append(memory_id)
        frozen = MappingProxyType(
            {
                level: tuple(sorted(values, key=int))
                for level, values in sorted(grouped.items(), key=lambda item: int(item[0]))
            }
        )
        return DirtyDerivationPlan(by_level=frozen)

    def consume_plan(self) -> DirtyDerivationPlan:
        plan = self.snapshot_plan()
        self._dirty.clear()
        return plan

    def clear_dirty(self) -> None:
        self._dirty.clear()

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)
