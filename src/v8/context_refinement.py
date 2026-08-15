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
    """Propose context splits only when one action has recurrent conflicting outcomes."""

    def __init__(self, *, min_support: int = 4, contradiction_threshold: float = 0.20) -> None:
        self.min_support = int(min_support)
        self.contradiction_threshold = float(contradiction_threshold)

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
            dominant = max(int(row.support_count) for row in variants)
            contradiction = 1.0 - dominant / max(1, total)
            if contradiction < self.contradiction_threshold:
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
                        contradiction,
                    )
                )
        return tuple(result)
