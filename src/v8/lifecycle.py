from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryUid, ValidationState


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    uid: MemoryUid
    cognitive_state: int
    validation_state: int
    fitness: float
    reason: str


class LifecycleController:
    """Hysteretic retention/quarantine/retirement decisions with provenance preserved."""

    def __init__(
        self,
        *,
        promotion_threshold: float = 0.55,
        demotion_threshold: float = 0.20,
        min_support: int = 3,
    ) -> None:
        if demotion_threshold >= promotion_threshold:
            raise ValueError("demotion threshold must be below promotion threshold")
        self.promotion_threshold = float(promotion_threshold)
        self.demotion_threshold = float(demotion_threshold)
        self.min_support = int(min_support)
        self._low_windows: dict[MemoryUid, int] = {}

    @staticmethod
    def fitness(row: NodeRecord) -> float:
        support = min(1.0, max(0, row.support_count) / 8.0)
        transfer = min(1.0, max(0.0, row.transfer_prior))
        explanatory = min(1.0, max(0.0, row.explanatory_reach) / 4.0)
        option = min(1.0, abs(float(row.future_option_delta)) / 4.0)
        learning = min(1.0, max(0.0, row.learning_value))
        return 0.25 * support + 0.20 * transfer + 0.20 * explanatory + 0.15 * option + 0.20 * learning

    def decide(self, row: NodeRecord) -> LifecycleDecision | None:
        if int(row.level) <= int(MemoryLevel.M1):
            return None
        fitness = self.fitness(row)
        current = int(row.cognitive_state)
        validation = int(row.validation_state)
        if row.support_count >= self.min_support and fitness >= self.promotion_threshold:
            self._low_windows.pop(row.uid, None)
            state = int(CognitiveState.VALIDATED if validation >= int(ValidationState.VALIDATED) else CognitiveState.ACTIVE)
            if state != current:
                return LifecycleDecision(row.uid, state, validation, fitness, "sustained promotion evidence")
            return None
        if fitness <= self.demotion_threshold:
            windows = self._low_windows.get(row.uid, 0) + 1
            self._low_windows[row.uid] = windows
            if windows >= 3 and current in {int(CognitiveState.ACTIVE), int(CognitiveState.VALIDATED)}:
                return LifecycleDecision(row.uid, int(CognitiveState.QUARANTINED), validation, fitness, "three low-fitness windows")
            if windows >= 6 and current == int(CognitiveState.QUARANTINED):
                return LifecycleDecision(row.uid, int(CognitiveState.RETIRE_PENDING), validation, fitness, "six low-fitness windows")
        else:
            self._low_windows.pop(row.uid, None)
            if current in {int(CognitiveState.QUARANTINED), int(CognitiveState.RETIRE_PENDING)}:
                return LifecycleDecision(row.uid, int(CognitiveState.REACTIVATED), validation, fitness, "new supporting evidence")
        return None
