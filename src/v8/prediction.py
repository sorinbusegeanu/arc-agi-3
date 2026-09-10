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
    expectation_key: tuple[int, int, int]
    observation_uid_hi: int = 0
    observation_uid_lo: int = 0

    @property
    def expectation_uid(self):
        from v8.model import MemoryUid

        return MemoryUid(self.uid_hi, self.uid_lo)

    @property
    def observation_uid(self):
        from v8.model import MemoryUid

        return MemoryUid(self.observation_uid_hi, self.observation_uid_lo)

    @property
    def violated(self) -> bool:
        return self.error > 0.0 and not self.observation_uid.is_zero


class PredictionEstimator:
    """Expose prediction violation only after a supported expectation existed.

    Actor events store error against the outcome distribution visible before the
    transition. This estimator only admits those causal errors after the corresponding
    context/action contingency has enough support and a stable dominant expectation.
    """

    def __init__(self, *, min_support: int = 3, stability_threshold: float = 0.60) -> None:
        self.min_support = int(min_support)
        self.stability_threshold = float(stability_threshold)
        self.last_rejections: dict[str, int] = {}

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[PredictionEvidence, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M1) or len(row.key_parts) < 3:
                continue
            grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)

        result: list[PredictionEvidence] = []
        rejected: dict[str, int] = {}
        for (context, action), variants in grouped.items():
            total = sum(max(0, int(row.support_count)) for row in variants)
            if not variants or total < self.min_support:
                rejected["insufficient_expectation_support"] = rejected.get(
                    "insufficient_expectation_support", 0
                ) + 1
                continue
            expectation = min(
                variants,
                key=lambda row: (-max(0, int(row.support_count)), row.uid),
            )
            dominant = max(0, int(expectation.support_count))
            stability = dominant / total if total else 0.0
            if dominant < self.min_support:
                rejected["insufficient_dominant_expectation_support"] = rejected.get(
                    "insufficient_dominant_expectation_support", 0
                ) + 1
                continue
            if stability < self.stability_threshold:
                rejected["expectation_not_stable"] = rejected.get(
                    "expectation_not_stable", 0
                ) + 1
                continue
            outcome = int(expectation.key_parts[2])
            contradictory = [
                row
                for row in variants
                if int(row.key_parts[2]) != outcome
                and float(row.prediction_error) > 0.0
            ]
            observation = (
                min(
                    contradictory,
                    key=lambda row: (-float(row.prediction_error), row.updated_watermark, row.uid),
                )
                if contradictory
                else None
            )
            result.append(
                PredictionEvidence(
                    int(expectation.uid.hi),
                    int(expectation.uid.lo),
                    0.0 if observation is None else max(0.0, float(observation.prediction_error)),
                    dominant,
                    True,
                    (int(context), int(action), outcome),
                    0 if observation is None else int(observation.uid.hi),
                    0 if observation is None else int(observation.uid.lo),
                )
            )
        self.last_rejections = rejected
        return tuple(result)
