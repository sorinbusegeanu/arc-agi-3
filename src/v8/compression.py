from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class CompressionEvidence:
    uid: MemoryUid
    explanatory_reach: int
    compression_benefit: float
    superseded: tuple[MemoryUid, ...]


class CompressionEstimator:
    """Measure explanatory reach and identify safely redundant lower-level structure."""

    def __init__(self, *, min_reach: int = 2) -> None:
        self.min_reach = int(min_reach)

    def evaluate(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
    ) -> tuple[CompressionEvidence, ...]:
        by_uid = {row.uid: row for row in nodes}
        children: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for edge in edges:
            if int(edge.relation_type) not in {int(RelationType.EXPLAINS), int(RelationType.SUPERSEDES)}:
                continue
            if edge.source_uid in by_uid and edge.target_uid in by_uid:
                children[edge.source_uid].add(edge.target_uid)
        result: list[CompressionEvidence] = []
        for uid, targets in children.items():
            if len(targets) < self.min_reach:
                continue
            parent = by_uid[uid]
            unique_cost = sum(max(1, by_uid[target].support_count) for target in targets)
            representation_cost = max(1, parent.support_count)
            benefit = float(max(0, unique_cost - representation_cost))
            result.append(CompressionEvidence(uid, len(targets), benefit, tuple(sorted(targets))))
        return tuple(result)
