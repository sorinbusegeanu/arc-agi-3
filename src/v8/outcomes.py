from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class OutcomeClass:
    uid: MemoryUid
    members: tuple[MemoryUid, ...]
    descriptor: tuple[int, int]
    version: int
    support: int


@dataclass(frozen=True, slots=True)
class OutcomeRevision:
    kind: str
    sources: tuple[MemoryUid, ...]
    target: MemoryUid
    descriptor: tuple[int, ...]
    version: int


class OutcomeEquivalenceEstimator:
    """Persistent consequence-class index with versioned merge/split proposals."""

    def __init__(self) -> None:
        self._version = 0
        self._classes: dict[tuple[int, int], OutcomeClass] = {}
        self._signature: tuple[tuple[int, int, tuple[MemoryUid, ...]], ...] = ()

    def rebuild(self, rows: tuple[NodeRecord, ...]) -> tuple[OutcomeClass, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        admissible = {
            int(CognitiveState.CANDIDATE),
            int(CognitiveState.PROBATION),
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        for row in rows:
            if int(row.level) != int(MemoryLevel.M6) or int(row.memory_type) != int(MemoryType.OUTCOME):
                continue
            if len(row.key_parts) < 2 or int(row.cognitive_state) not in admissible:
                continue
            grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)
        signature = tuple(
            sorted(
                (descriptor[0], descriptor[1], tuple(sorted(row.uid for row in members)))
                for descriptor, members in grouped.items()
            )
        )
        if signature != self._signature:
            self._version += 1
            self._signature = signature
        classes: dict[tuple[int, int], OutcomeClass] = {}
        for descriptor, members in grouped.items():
            uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptor)
            classes[descriptor] = OutcomeClass(
                uid,
                tuple(sorted(row.uid for row in members)),
                descriptor,
                self._version,
                sum(max(0, int(row.support_count)) for row in members),
            )
        self._classes = classes
        return tuple(classes.values())

    def classes(self) -> tuple[OutcomeClass, ...]:
        return tuple(self._classes.values())

    def merge_revision(self, outcome: OutcomeClass) -> OutcomeRevision | None:
        sources = tuple(uid for uid in outcome.members if uid != outcome.uid)
        if len(sources) < 2:
            return None
        self._version += 1
        return OutcomeRevision(
            "MERGE",
            sources,
            outcome.uid,
            outcome.descriptor,
            self._version,
        )

    def split_revision(self, source: OutcomeClass) -> tuple[OutcomeRevision, ...]:
        """Restore fine members when a coarse class is explicitly invalidated."""
        if source.uid not in source.members:
            return ()
        members = tuple(uid for uid in source.members if uid != source.uid)
        result = []
        for member in members:
            self._version += 1
            result.append(
                OutcomeRevision("SPLIT", (source.uid,), member, source.descriptor, self._version)
            )
        return tuple(result)

    def state_dict(self) -> dict[str, object]:
        return {"version": self._version}

    def load_state(self, state: dict[str, object] | None) -> None:
        if state:
            self._version = max(self._version, int(state.get("version", 0)))
