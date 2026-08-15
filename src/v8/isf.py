from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, ValidationState


@dataclass(frozen=True, slots=True)
class ISFScore:
    option_structure_impact: float
    prediction_error: float
    learning_value: float
    transfer_potential: float
    explanatory_potential: float
    future_option_value: float
    total: float
    developmental_stage: int


# Stage weights are runtime hypotheses, not universal coefficients. They are fixed
# for one peer update interval and selected from capabilities established before
# that interval. Channels that are not yet available naturally contribute zero.
_STAGE_WEIGHTS: dict[int, tuple[float, float, float, float, float, float]] = {
    0: (0.70, 0.00, 0.15, 0.00, 0.05, 0.10),
    1: (0.25, 0.30, 0.25, 0.05, 0.05, 0.10),
    2: (0.15, 0.15, 0.25, 0.20, 0.20, 0.05),
    3: (0.10, 0.10, 0.15, 0.30, 0.30, 0.05),
    4: (0.15, 0.10, 0.15, 0.15, 0.20, 0.25),
    5: (0.15, 0.05, 0.10, 0.15, 0.20, 0.35),
    6: (0.15, 0.05, 0.10, 0.10, 0.15, 0.45),
}

_ADMISSIBLE = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def infer_developmental_stage(rows: Iterable[NodeRecord]) -> int:
    """Infer Stage_t only from capabilities already present in the published cut.

    A positive stored prediction error is itself evidence that a supported expectation
    existed before the corresponding transition, so it implies at least Stage 1 even
    when the row carrying that evidence is not yet an active higher abstraction.
    """
    rows = tuple(rows)
    minimum_stage = 1 if any(float(row.prediction_error) > 0.0 for row in rows) else 0
    active = tuple(row for row in rows if int(row.cognitive_state) in _ADMISSIBLE)
    if any(
        int(row.level) == int(MemoryLevel.M7)
        and int(row.memory_type) == int(MemoryType.STRATEGY)
        and row.attempt_weight > 0
        and row.strategy_reliability > 0
        for row in active
    ):
        return 6
    if any(
        int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        for row in active
    ):
        return 5
    if any(int(row.level) == int(MemoryLevel.M5) for row in active):
        return 4
    if any(
        int(row.level) == int(MemoryLevel.M4)
        and (
            int(row.validation_state) >= int(ValidationState.VALIDATED)
            or row.transfer_prior > 0
        )
        for row in active
    ):
        return 3
    if any(int(row.level) in {int(MemoryLevel.M2), int(MemoryLevel.M3)} for row in active):
        return 2
    if any(
        int(row.level) == int(MemoryLevel.M1) and int(row.support_count) >= 3
        for row in rows
    ):
        return 1
    return minimum_stage


def _fallback_stage(row: NodeRecord) -> int:
    level = int(row.level)
    if level <= int(MemoryLevel.M0):
        return 0
    if level == int(MemoryLevel.M1):
        return 1
    if level in {int(MemoryLevel.M2), int(MemoryLevel.M3)}:
        return 2
    if level == int(MemoryLevel.M4):
        return 3
    if level == int(MemoryLevel.M5):
        return 4
    if level == int(MemoryLevel.M6):
        return 5
    return 6


def score_memory(row: NodeRecord, *, developmental_stage: int | None = None) -> ISFScore:
    """Bounded causally-available interaction significance / fitness input."""
    stage = _fallback_stage(row) if developmental_stage is None else max(0, min(6, int(developmental_stage)))
    option_impact = _clip(max(row.significance, row.strategy_reliability))
    prediction = _clip(row.prediction_error)
    learning = _clip(row.learning_value)
    transfer = _clip(row.transfer_prior)
    explanatory = _clip(row.explanatory_reach / 4.0)
    future = _clip(abs(row.future_option_delta) / 4.0)
    weights = _STAGE_WEIGHTS[stage]
    components = (
        option_impact,
        prediction,
        learning,
        transfer,
        explanatory,
        future,
    )
    total = sum(
        weight * component
        for weight, component in zip(weights, components, strict=True)
    )
    return ISFScore(
        option_impact,
        prediction,
        learning,
        transfer,
        explanatory,
        future,
        float(total),
        stage,
    )
