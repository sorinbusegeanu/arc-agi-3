from __future__ import annotations

from dataclasses import dataclass

from v9.memory.identity import MemoryUid


@dataclass(frozen=True, slots=True)
class PartitionMap:
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("partition count must be positive")

    def owner(self, uid: MemoryUid) -> int:
        return uid.shard(self.count)

