from __future__ import annotations

from .typedState import TimeReactiveTypedStateV4


def resource_risk_heuristic(state: TimeReactiveTypedStateV4) -> int:
    return max(0, 100 - state.family.hunger_value) + max(0, 100 - state.family.warmth_value)
