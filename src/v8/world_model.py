from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryLevel, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class WorldModelComponent:
    uid: MemoryUid
    explained_nodes: int
    explained_support: int
    coherence: float


class WorldModelEstimator:
    """Measure emergent M5 explanatory organization without assuming a primitive model."""

    def evaluate(self, nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> tuple[WorldModelComponent, ...]:
        nodes = tuple(nodes)
        by_uid = {row.uid: row for row in nodes}
        explained: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for edge in edges:
            if int(edge.relation_type) != int(RelationType.EXPLAINS):
                continue
            source = by_uid.get(edge.source_uid)
            target = by_uid.get(edge.target_uid)
            if source is None or target is None or int(source.level) != int(MemoryLevel.M5):
                continue
            explained[source.uid].add(target.uid)
        total_support = sum(max(1, row.support_count) for row in nodes if int(row.level) < int(MemoryLevel.M5))
        result: list[WorldModelComponent] = []
        for row in nodes:
            if int(row.level) != int(MemoryLevel.M5):
                continue
            targets = explained.get(row.uid, set())
            support = sum(max(1, by_uid[uid].support_count) for uid in targets if uid in by_uid)
            coherence = 0.0 if total_support <= 0 else min(1.0, support / total_support)
            result.append(WorldModelComponent(row.uid, len(targets), support, coherence))
        return tuple(result)
