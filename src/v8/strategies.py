from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, signed_u64


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
    """Maintain outcome-conditioned strategy reliability and observed cost."""

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[StrategyEvidence, ...]:
        result = []
        admissible = {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        for row in rows:
            if int(row.level) != int(MemoryLevel.M7) or int(row.memory_type) != int(MemoryType.STRATEGY):
                continue
            if len(row.key_parts) < 4 or int(row.cognitive_state) not in admissible:
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            attempts = max(0, int(round(row.attempt_weight)))
            if attempts > 0:
                reliability = max(0.0, min(1.0, row.strategy_reliability))
                mean_cost = max(1e-9, row.strategy_mean_cost)
            else:
                # Prior only; it can guide exploration but is not efficiency evidence.
                attempts = max(0, int(row.support_count))
                reliability = min(0.75, attempts / 8.0)
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
