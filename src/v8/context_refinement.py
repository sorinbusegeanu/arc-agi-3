from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, stable_u64


@dataclass(frozen=True, slots=True)
class ContextRefinement:
    source_uid: MemoryUid
    candidate_uid: MemoryUid
    key_parts: tuple[int, ...]
    contradiction_rate: float


class ContextRefiner:
    """Propose context partitions only when they reduce predictive contradiction."""

    def __init__(self, *, min_support: int = 4, contradiction_threshold: float = 0.20) -> None:
        self.min_support = int(min_support)
        self.contradiction_threshold = float(contradiction_threshold)

    @staticmethod
    def _error(rows: list[NodeRecord]) -> float:
        total = sum(max(0, int(row.support_count)) for row in rows)
        if total <= 0:
            return 0.0
        by_outcome: dict[int, int] = defaultdict(int)
        for row in rows:
            if len(row.key_parts) >= 3:
                by_outcome[int(row.key_parts[2])] += max(0, int(row.support_count))
        dominant = max(by_outcome.values(), default=0)
        return 1.0 - dominant / total

    def propose(self, rows: tuple[NodeRecord, ...]) -> tuple[ContextRefinement, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) == int(MemoryLevel.M1) and len(row.key_parts) >= 3:
                grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)
        result: list[ContextRefinement] = []
        for (context, action), variants in grouped.items():
            total = sum(int(row.support_count) for row in variants)
            if total < self.min_support or len(variants) < 2:
                continue
            broad_error = self._error(variants)
            if broad_error < self.contradiction_threshold:
                continue

            partitions: dict[int, list[NodeRecord]] = defaultdict(list)
            for row in variants:
                next_context = int(row.key_parts[3]) if len(row.key_parts) >= 4 else 0
                partitions[next_context].append(row)
            refined_error = 0.0
            for partition_rows in partitions.values():
                partition_support = sum(max(0, int(row.support_count)) for row in partition_rows)
                refined_error += (partition_support / max(1, total)) * self._error(partition_rows)
            gain = max(0.0, broad_error - refined_error)
            if gain <= 1e-12:
                continue

            for row in variants:
                outcome = int(row.key_parts[2])
                next_context = int(row.key_parts[3]) if len(row.key_parts) >= 4 else 0
                context_partition = stable_u64(context, next_context, person=b"v8-context-split")
                key = (context_partition, action, outcome)
                result.append(
                    ContextRefinement(
                        row.uid,
                        MemoryUid.from_key(MemoryLevel.M3, MemoryType.CONTEXTUAL_ROLE, key),
                        key,
                        gain,
                    )
                )
        return tuple(result)
