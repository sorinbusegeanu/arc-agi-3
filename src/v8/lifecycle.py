from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.isf import score_memory
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState


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
        self.developmental_stage = 0
        self._low_windows: dict[MemoryUid, int] = {}

    def set_developmental_stage(self, stage: int) -> None:
        self.developmental_stage = max(0, min(6, int(stage)))

    def fitness(self, row: NodeRecord) -> float:
        base = score_memory(row, developmental_stage=self.developmental_stage).total
        validation_bonus = (
            0.10
            if int(row.validation_state) >= int(ValidationState.VALIDATED)
            else 0.0
        )
        reliability_bonus = 0.10 * max(
            0.0, min(1.0, row.strategy_reliability)
        )
        return min(1.0, base + validation_bonus + reliability_bonus)

    def decide(self, row: NodeRecord) -> LifecycleDecision | None:
        if int(row.level) <= int(MemoryLevel.M1):
            return None
        fitness = self.fitness(row)
        current = int(row.cognitive_state)
        validation = int(row.validation_state)

        if (
            int(row.level) == int(MemoryLevel.M6)
            and int(row.memory_type) == int(MemoryType.OUTCOME)
            and len(row.key_parts) == 2
            and validation == int(ValidationState.FAILED)
            and current
            in {
                int(CognitiveState.ACTIVE),
                int(CognitiveState.VALIDATED),
                int(CognitiveState.REACTIVATED),
            }
        ):
            self._low_windows[row.uid] = max(
                3, self._low_windows.get(row.uid, 0)
            )
            return LifecycleDecision(
                row.uid,
                int(CognitiveState.QUARANTINED),
                validation,
                fitness,
                "failed coarse outcome validation; split to persistent members",
            )

        if row.support_count >= self.min_support and fitness >= self.promotion_threshold:
            self._low_windows.pop(row.uid, None)
            state = int(
                CognitiveState.VALIDATED
                if validation >= int(ValidationState.VALIDATED)
                else CognitiveState.ACTIVE
            )
            if state != current:
                reason = (
                    "reactivated by renewed evidence"
                    if current == int(CognitiveState.RETIRED)
                    else "sustained promotion evidence"
                )
                return LifecycleDecision(row.uid, state, validation, fitness, reason)
            return None
        if fitness <= self.demotion_threshold:
            windows = self._low_windows.get(row.uid, 0) + 1
            self._low_windows[row.uid] = windows
            if windows >= 3 and current in {
                int(CognitiveState.ACTIVE),
                int(CognitiveState.VALIDATED),
                int(CognitiveState.REACTIVATED),
            }:
                return LifecycleDecision(
                    row.uid,
                    int(CognitiveState.QUARANTINED),
                    validation,
                    fitness,
                    "three low-fitness windows",
                )
            if windows >= 6 and current == int(CognitiveState.QUARANTINED):
                return LifecycleDecision(
                    row.uid,
                    int(CognitiveState.RETIRE_PENDING),
                    validation,
                    fitness,
                    "six low-fitness windows",
                )
        else:
            self._low_windows.pop(row.uid, None)
            if current in {
                int(CognitiveState.QUARANTINED),
                int(CognitiveState.RETIRE_PENDING),
            }:
                return LifecycleDecision(
                    row.uid,
                    int(CognitiveState.REACTIVATED),
                    validation,
                    fitness,
                    "new supporting evidence",
                )
        return None

    def finalize_retirement(
        self,
        row: NodeRecord,
        *,
        protected_by_dependencies: bool,
    ) -> LifecycleDecision | None:
        if int(row.cognitive_state) != int(CognitiveState.RETIRE_PENDING):
            return None
        if protected_by_dependencies:
            return None
        windows = self._low_windows.get(row.uid, 6)
        if windows < 6:
            return None
        return LifecycleDecision(
            row.uid,
            int(CognitiveState.RETIRED),
            int(row.validation_state),
            self.fitness(row),
            "retired after dependency-safe low-fitness probation",
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "developmental_stage": self.developmental_stage,
            "low_windows": [
                {"uid": [uid.hi, uid.lo], "windows": windows}
                for uid, windows in self._low_windows.items()
            ],
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.set_developmental_stage(int(state.get("developmental_stage", 0)))
        for raw in state.get("low_windows", []):
            if not isinstance(raw, dict):
                continue
            uid_raw = raw.get("uid", [0, 0])
            self._low_windows[
                MemoryUid(int(uid_raw[0]), int(uid_raw[1]))
            ] = int(raw.get("windows", 0))
