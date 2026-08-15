from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryUid


@dataclass(frozen=True, slots=True)
class SignificanceVector:
    uid: MemoryUid
    stage: int
    osi: float
    prediction_error: float
    learning_value: float
    transfer_prior: float
    explanatory_potential: float
    isf: float


@dataclass(frozen=True, slots=True)
class ReplayPriority:
    uid: MemoryUid
    probability: float
    fitness: float


_STAGE_WEIGHTS: tuple[tuple[float, float, float, float, float], ...] = (
    (1.00, 0.00, 0.00, 0.00, 0.00),
    (0.35, 0.30, 0.35, 0.00, 0.00),
    (0.10, 0.10, 0.35, 0.20, 0.25),
    (0.05, 0.05, 0.20, 0.40, 0.30),
    (0.25, 0.05, 0.15, 0.25, 0.30),
    (0.20, 0.05, 0.15, 0.25, 0.35),
    (0.20, 0.05, 0.10, 0.25, 0.40),
)


def _bound(value: float, scale: float = 1.0) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return max(0.0, min(1.0, float(value) / float(scale)))


def infer_developmental_stage(rows: Iterable[NodeRecord]) -> int:
    """Infer the next resource-allocation stage only from already established state."""
    rows = tuple(rows)
    levels = {int(row.level) for row in rows if row.support_count > 0}
    validated = {int(row.level) for row in rows if int(row.validation_state) >= 3}
    if int(MemoryLevel.M7) in levels:
        return 6
    if int(MemoryLevel.M6) in levels:
        return 5
    if int(MemoryLevel.M5) in levels:
        return 4
    if int(MemoryLevel.M4) in validated:
        return 3
    if int(MemoryLevel.M3) in levels or int(MemoryLevel.M2) in levels:
        return 2
    if int(MemoryLevel.M1) in levels:
        return 1
    return 0


class InteractionSignificanceEstimator:
    """Bounded, auditable ISF over causally available sufficient statistics."""

    def evaluate(self, rows: Iterable[NodeRecord], *, stage: int | None = None) -> tuple[SignificanceVector, ...]:
        rows = tuple(rows)
        active_stage = infer_developmental_stage(rows) if stage is None else max(0, min(6, int(stage)))
        weights = _STAGE_WEIGHTS[active_stage]
        result: list[SignificanceVector] = []
        for row in rows:
            osi = _bound(abs(row.future_option_delta), 4.0)
            pe = _bound(row.prediction_error, 1.0)
            lv = _bound(row.learning_value, 1.0)
            tp = _bound(row.transfer_prior, 1.0)
            ep = _bound(row.explanatory_reach, 4.0)
            values = (osi, pe, lv, tp, ep)
            active = [(w, v) for w, v in zip(weights, values, strict=True) if w > 0]
            denom = sum(weight for weight, _ in active)
            score = 0.0 if denom <= 0 else sum(weight * value for weight, value in active) / denom
            result.append(SignificanceVector(row.uid, active_stage, osi, pe, lv, tp, ep, score))
        return tuple(result)


class ReplayAllocator:
    """Allocate bounded replay probability from current memory fitness; replay is not validation."""

    def __init__(self, *, temperature: float = 3.0, max_items: int = 64) -> None:
        self.temperature = float(temperature)
        self.max_items = int(max_items)

    def allocate(self, scores: Iterable[SignificanceVector]) -> tuple[ReplayPriority, ...]:
        rows = tuple(sorted(scores, key=lambda row: (-row.isf, row.uid))[: max(0, self.max_items)])
        if not rows:
            return ()
        logits = [exp(max(-30.0, min(30.0, self.temperature * row.isf))) for row in rows]
        total = sum(logits)
        return tuple(
            ReplayPriority(row.uid, weight / total, row.isf)
            for row, weight in zip(rows, logits, strict=True)
        )
