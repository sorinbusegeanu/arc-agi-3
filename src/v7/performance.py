from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class PerformanceSample:
    metric: str
    value: float
    unit: str
    lower_is_better: bool = True

@dataclass(frozen=True, slots=True)
class PerformanceComparison:
    metric: str
    baseline_value: float
    candidate_value: float
    ratio: float
    improved: bool

REQUIRED_PERFORMANCE_METRICS = (
    "memory_bytes",
    "generation_commit_seconds",
    "derivation_items_per_second",
    "action_selection_seconds",
    "mmap_attach_seconds",
    "parallel_derivation_items_per_second",
)

class PerformanceValidationSuite:
    def validate_complete(self, samples: Mapping[str, PerformanceSample]) -> tuple[str, ...]:
        return tuple(metric for metric in REQUIRED_PERFORMANCE_METRICS if metric not in samples)

    def compare(self, baseline: Mapping[str, PerformanceSample], candidate: Mapping[str, PerformanceSample]) -> tuple[PerformanceComparison, ...]:
        missing = self.validate_complete(baseline) + self.validate_complete(candidate)
        if missing:
            raise ValueError(f"missing required performance metrics: {sorted(set(missing))}")
        rows = []
        for metric in REQUIRED_PERFORMANCE_METRICS:
            left, right = baseline[metric], candidate[metric]
            if left.unit != right.unit or left.lower_is_better != right.lower_is_better:
                raise ValueError(f"incompatible performance sample for {metric}")
            ratio = float("inf") if left.value == 0 else right.value / left.value
            improved = right.value <= left.value if left.lower_is_better else right.value >= left.value
            rows.append(PerformanceComparison(metric, left.value, right.value, ratio, improved))
        return tuple(rows)
