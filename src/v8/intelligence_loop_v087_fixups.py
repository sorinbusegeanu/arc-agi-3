from __future__ import annotations

from v8.model import CognitiveState, MemoryLevel, MemoryType, ValidationState


_INSTALLED = False


def install_intelligence_loop_v087_fixups() -> None:
    """Preserve pre-v8.7 specialized lifecycle semantics under the new failure gate."""
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import lifecycle as lifecycle_module

    current_decide = lifecycle_module.LifecycleController.decide

    def decide(self, row):
        # v8.2 outcome splitting is more specific than the generic v8.7 empirical
        # failure rule and must retain its split reason and hysteresis bookkeeping.
        if (
            int(row.level) == int(MemoryLevel.M6)
            and int(row.memory_type) == int(MemoryType.OUTCOME)
            and len(row.key_parts) == 2
            and int(row.validation_state) == int(ValidationState.FAILED)
            and int(row.cognitive_state)
            in {
                int(CognitiveState.ACTIVE),
                int(CognitiveState.VALIDATED),
                int(CognitiveState.REACTIVATED),
            }
        ):
            self._low_windows[row.uid] = max(3, self._low_windows.get(row.uid, 0))
            return lifecycle_module.LifecycleDecision(
                row.uid,
                int(CognitiveState.QUARANTINED),
                int(ValidationState.FAILED),
                self.fitness(row),
                "failed coarse outcome validation; split to persistent members",
            )
        return current_decide(self, row)

    lifecycle_module.LifecycleController.decide = decide
    _INSTALLED = True
