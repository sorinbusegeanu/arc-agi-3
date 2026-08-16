from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.isf import (
    ISFScore,
    infer_developmental_stage,
    publish_developmental_stage,
    score_memories,
)
from v8.model import CognitiveState, MemoryUid


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    uid: MemoryUid
    priority: float
    score: ISFScore
    reason: str


class ReplayScheduler:
    """Bound developmental attention using the frozen v8.2 ISF snapshot."""

    def __init__(self, *, min_priority: float = 0.25) -> None:
        self.min_priority = float(min_priority)
        self.last_developmental_stage = 0

    def candidates(
        self,
        rows: tuple[NodeRecord, ...],
        *,
        budget: int,
    ) -> tuple[ReplayCandidate, ...]:
        stage = infer_developmental_stage(rows)
        self.last_developmental_stage = publish_developmental_stage(stage)
        scores = score_memories(rows, developmental_stage=stage)
        ranked: list[ReplayCandidate] = []
        for row in rows:
            if int(row.cognitive_state) in {
                int(CognitiveState.RETIRED),
                int(CognitiveState.RETIRE_PENDING),
            }:
                continue
            score = scores[row.uid]
            novelty = 1.0 / max(1.0, float(row.support_count))
            priority = score.total + 0.10 * novelty
            if priority < self.min_priority:
                continue
            if score.prediction_error >= 0.5:
                reason = "prediction_violation"
            elif score.transfer_potential >= 0.4:
                reason = "transfer_opportunity"
            elif score.explanatory_potential >= 0.4:
                reason = "explanatory_opportunity"
            elif score.future_option_value >= 0.4:
                reason = "future_option"
            else:
                reason = "developmental_importance"
            ranked.append(ReplayCandidate(row.uid, float(priority), score, reason))
        ranked.sort(key=lambda item: (-item.priority, item.uid))
        return tuple(ranked[: max(0, int(budget))])
