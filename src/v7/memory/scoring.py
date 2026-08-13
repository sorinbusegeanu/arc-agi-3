from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from v7.memory.indexes.packed import PackedCognitionIndexes


@dataclass(frozen=True, slots=True)
class ActionScoringWeights:
    future_option: float = 0.25
    positive: float = 0.30
    negative: float = 0.20
    failure: float = 0.15
    contradiction: float = 0.10


@dataclass(frozen=True, slots=True)
class ActionScoreBatch:
    action_ids: np.ndarray
    scores: np.ndarray
    evidence_counts: np.ndarray

    def best_action(self) -> int | None:
        if self.action_ids.size == 0:
            return None
        return int(self.action_ids[int(np.argmax(self.scores))])


class VectorizedActionScorer:
    """Fully batched O(actions log A) scoring over packed action aggregates."""

    def __init__(self, weights: ActionScoringWeights | None = None) -> None:
        self.weights = weights or ActionScoringWeights()

    def score(self, indexes: PackedCognitionIndexes, action_ids: Iterable[int]) -> ActionScoreBatch:
        ids = np.asarray(tuple(int(v) for v in action_ids), dtype=np.int64)
        if ids.size == 0:
            empty = np.asarray([], dtype=np.float64)
            return ActionScoreBatch(ids, empty, np.asarray([], dtype=np.int64))

        packed_ids = np.asarray(indexes.action_aggregates.action_ids, dtype=np.int64)
        matrix = np.zeros((ids.size, 6), dtype=np.float64)
        if packed_ids.size:
            positions = np.searchsorted(packed_ids, ids)
            present = positions < packed_ids.size
            present_indices = np.flatnonzero(present)
            if present_indices.size:
                matched = packed_ids[positions[present_indices]] == ids[present_indices]
                present_indices = present_indices[matched]
            if present_indices.size:
                positions = positions[present_indices]
                values = np.asarray(indexes.action_aggregates.values, dtype=np.float64).reshape((-1, 6))
                matrix[present_indices] = values[positions]

        future_option_count = matrix[:, 1]
        future_option_mean = np.divide(matrix[:, 0], future_option_count, out=np.zeros(ids.size, dtype=np.float64), where=future_option_count > 0)
        evidence_counts = matrix[:, 2:6].sum(axis=1).astype(np.int64)
        denom = np.maximum(1.0, evidence_counts.astype(np.float64))
        normalized = matrix[:, 2:6] / denom[:, None]
        w = self.weights
        scores = (
            w.future_option * future_option_mean
            + w.positive * normalized[:, 0]
            - w.negative * normalized[:, 1]
            - w.failure * normalized[:, 2]
            - w.contradiction * normalized[:, 3]
        )
        ids.setflags(write=False)
        scores.setflags(write=False)
        evidence_counts.setflags(write=False)
        return ActionScoreBatch(ids, scores, evidence_counts)


@dataclass(frozen=True, slots=True)
class PerformanceMeasurement:
    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    operations_per_second: float


class PerformanceProbe:
    """Small deterministic timing harness for v7 regression/performance validation."""

    @staticmethod
    def measure(name: str, operation: Callable[[], object], *, iterations: int = 1) -> PerformanceMeasurement:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = perf_counter()
        for _ in range(iterations):
            operation()
        total = perf_counter() - started
        mean = total / iterations
        return PerformanceMeasurement(name, iterations, total, mean, float("inf") if total == 0 else iterations / total)

    @staticmethod
    def within_budget(measurement: PerformanceMeasurement, *, max_mean_seconds: float) -> bool:
        if max_mean_seconds < 0:
            raise ValueError("max_mean_seconds must be non-negative")
        return measurement.mean_seconds <= max_mean_seconds
