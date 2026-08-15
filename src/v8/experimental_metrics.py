from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterventionMetric:
    metric_on: float
    metric_off: float
    higher_is_better: bool = True

    @property
    def effect(self) -> float:
        delta = float(self.metric_on) - float(self.metric_off)
        return delta if self.higher_is_better else -delta
