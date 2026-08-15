from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class OutcomeClass:
    uid: MemoryUid
    members: tuple[MemoryUid, ...]
    descriptor: tuple[int, int]
    version: int


@dataclass(frozen=True, slots=True)
class OutcomeRevision:
    kind: str
    sources: tuple[MemoryUid, ...]
    target: MemoryUid
    version: int


class OutcomeEquivalenceEstimator:
    """Persistent consequence-class index with versioned merge/split proposals."""

    def __init__(self) -> None:
        self._version = 0
        self._classes: dict[tuple[int, int], OutcomeClass] = {}

    def rebuild(self, rows: tuple[NodeRecord, ...]) -> tuple[OutcomeClass, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M6) or int(row.memory_type) != int(MemoryType.OUTCOME):
                continue
            if len(row.key_parts) < 2:
                continue
            grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)
        self._version += 1
        classes: dict[tuple[int, int], OutcomeClass] = {}
        for descriptor, members in grouped.items():
            uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptor)
            classes[descriptor] = OutcomeClass(uid, tuple(row.uid for row in members), descriptor, self._version)
        self._classes = classes
        return tuple(classes.values())

    def classes(self) -> tuple[OutcomeClass, ...]:
        return tuple(self._classes.values())

    def revision_for_merge(self, a: OutcomeClass, b: OutcomeClass) -> OutcomeRevision | None:
        if a.descriptor != b.descriptor or a.uid == b.uid:
            return None
        self._version += 1
        target = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, a.descriptor)
        return OutcomeRevision("MERGE", (a.uid, b.uid), target, self._version)

    def revision_for_split(self, source: OutcomeClass, descriptors: tuple[tuple[int, int], ...]) -> tuple[OutcomeRevision, ...]:
        if len(set(descriptors)) < 2:
            return ()
        result = []
        for descriptor in sorted(set(descriptors)):
            self._version += 1
            target = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptor)
            result.append(OutcomeRevision("SPLIT", (source.uid,), target, self._version))
        return tuple(result)
