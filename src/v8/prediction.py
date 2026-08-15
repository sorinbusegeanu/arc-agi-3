from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    uid_hi: int
    uid_lo: int
    error: float
    support: int
    stable: bool


class PredictionEstimator:
    """Expose prediction violation only after a supported expectation existed.

    Actor events store error against the outcome distribution visible before the
    transition. This estimator only admits those causal errors after the corresponding
    context/action contingency has enough support and a stable dominant expectation.
    """

    def __init__(self, *, min_support: int = 3, stability_threshold: float = 0.60) -> None:
        self.min_support = int(min_support)
        self.stability_threshold = float(stability_threshold)

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[PredictionEvidence, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M1) or len(row.key_parts) < 3:
                continue
            grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)

        result: list[PredictionEvidence] = []
        for variants in grouped.values():
            total = sum(max(0, int(row.support_count)) for row in variants)
            if total < self.min_support:
                continue
            dominant = max(max(0, int(row.support_count)) for row in variants)
            stability = dominant / total if total else 0.0
            if stability < self.stability_threshold:
                continue
            for row in variants:
                result.append(
                    PredictionEvidence(
                        int(row.uid.hi),
                        int(row.uid.lo),
                        max(0.0, float(row.prediction_error)),
                        int(row.support_count),
                        True,
                    )
                )
        return tuple(result)
