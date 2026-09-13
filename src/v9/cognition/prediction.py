from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionComparison:
    baseline: float
    conditioned: float
    actual: float

    @property
    def delta(self) -> float:
        return abs(self.baseline - self.actual) - abs(self.conditioned - self.actual)

