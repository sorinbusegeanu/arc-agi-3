from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, signed_u64


@dataclass(frozen=True, slots=True)
class StrategyEvidence:
    uid: MemoryUid
    outcome_uid: MemoryUid
    action_id: int
    context_bucket: int
    attempts: int
    reliability: float
    mean_cost: float


class StrategyEstimator:
    """Maintain alternative strategies separately from persistent outcome identity."""

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[StrategyEvidence, ...]:
        result = []
        for row in rows:
            if int(row.level) != int(MemoryLevel.M7) or int(row.memory_type) != int(MemoryType.STRATEGY):
                continue
            if len(row.key_parts) < 4:
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            attempts = max(0, int(row.support_count))
            reliability = min(1.0, attempts / 4.0)
            # Until explicit per-trajectory cost evidence is available, one emitted
            # strategy step is the declared minimal unit cost.
            mean_cost = 1.0
            result.append(
                StrategyEvidence(
                    row.uid,
                    outcome,
                    action,
                    int(row.key_parts[3]),
                    attempts,
                    reliability,
                    mean_cost,
                )
            )
        return tuple(result)

    def by_outcome(self, rows: tuple[NodeRecord, ...]) -> dict[MemoryUid, tuple[StrategyEvidence, ...]]:
        grouped: dict[MemoryUid, list[StrategyEvidence]] = defaultdict(list)
        for evidence in self.evaluate(rows):
            grouped[evidence.outcome_uid].append(evidence)
        return {uid: tuple(values) for uid, values in grouped.items()}
