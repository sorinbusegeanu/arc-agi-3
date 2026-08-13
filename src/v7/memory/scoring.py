from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    """O(actions) aggregate scorer over bounded cognition indexes."""

    def __init__(self, weights: ActionScoringWeights | None = None) -> None:
        self.weights = weights or ActionScoringWeights()

    def score(self, indexes: PackedCognitionIndexes, action_ids: Iterable[int]) -> ActionScoreBatch:
        ids = np.asarray(tuple(int(v) for v in action_ids), dtype=np.int64)
        if ids.size == 0:
            empty = np.asarray([], dtype=np.float64)
            return ActionScoreBatch(ids, empty, np.asarray([], dtype=np.int64))
        matrix = np.zeros((ids.size, 6), dtype=np.float64)
        for row, action_id in enumerate(ids):
            agg = indexes.action_aggregates.get(int(action_id))
            matrix[row] = (
                agg.future_option_mean,
                agg.positive_count,
                agg.negative_count,
                agg.failure_count,
                agg.contradiction_count,
                agg.positive_count + agg.negative_count + agg.failure_count + agg.contradiction_count,
            )
        counts = matrix[:, 5].astype(np.int64)
        denom = np.maximum(1.0, counts.astype(np.float64))
        normalized = matrix[:, 1:5] / denom[:, None]
        w = self.weights
        scores = (
            w.future_option * matrix[:, 0]
            + w.positive * normalized[:, 0]
            - w.negative * normalized[:, 1]
            - w.failure * normalized[:, 2]
            - w.contradiction * normalized[:, 3]
        )
        ids.setflags(write=False); scores.setflags(write=False); counts.setflags(write=False)
        return ActionScoreBatch(ids, scores, counts)
