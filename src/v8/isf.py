from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel


@dataclass(frozen=True, slots=True)
class ISFScore:
    option_structure_impact: float
    prediction_error: float
    learning_value: float
    transfer_potential: float
    explanatory_potential: float
    future_option_value: float
    total: float


_STAGE_WEIGHTS: dict[int, tuple[float, float, float, float, float, float]] = {
    int(MemoryLevel.M0): (0.25, 0.30, 0.25, 0.05, 0.05, 0.10),
    int(MemoryLevel.M1): (0.20, 0.30, 0.25, 0.05, 0.05, 0.15),
    int(MemoryLevel.M2): (0.15, 0.20, 0.20, 0.15, 0.20, 0.10),
    int(MemoryLevel.M3): (0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
    int(MemoryLevel.M4): (0.10, 0.10, 0.15, 0.25, 0.25, 0.15),
    int(MemoryLevel.M5): (0.10, 0.10, 0.15, 0.15, 0.30, 0.20),
    int(MemoryLevel.M6): (0.15, 0.05, 0.10, 0.15, 0.20, 0.35),
    int(MemoryLevel.M7): (0.25, 0.05, 0.10, 0.10, 0.15, 0.35),
}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_memory(row: NodeRecord) -> ISFScore:
    """Stage-dependent normalized interaction significance / memory fitness input.

    The first component is option-structure impact, not survival or terminal utility.
    It is estimated from causally available structural significance and, for mature
    strategy memories, observed reliability. Transfer and explanatory evidence gain
    weight progressively at higher developmental memory levels.
    """
    option_impact = _clip(max(row.significance, row.strategy_reliability))
    prediction = _clip(row.prediction_error)
    learning = _clip(row.learning_value)
    transfer = _clip(row.transfer_prior)
    explanatory = _clip(row.explanatory_reach / 4.0)
    future = _clip(abs(row.future_option_delta) / 4.0)
    weights = _STAGE_WEIGHTS.get(
        int(row.level), _STAGE_WEIGHTS[int(MemoryLevel.M4)]
    )
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
    )
