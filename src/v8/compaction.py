from __future__ import annotations

from dataclasses import dataclass

from v8.model import CognitiveState, MemoryUid


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    retained: tuple[MemoryUid, ...]
    retired: tuple[MemoryUid, ...]


def plan_compaction(nodes) -> CompactionPlan:
    retained = []
    retired = []
    for row in nodes:
        if int(row.cognitive_state) == int(CognitiveState.RETIRED):
            retired.append(row.uid)
        else:
            retained.append(row.uid)
    return CompactionPlan(tuple(retained), tuple(retired))
