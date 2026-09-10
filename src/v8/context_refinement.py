from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, stable_u64


@dataclass(frozen=True, slots=True)
class ContextRefinement:
    source_uid: MemoryUid
    candidate_uid: MemoryUid
    key_parts: tuple[int, ...]
    contradiction_rate: float
    broad_prediction_error: float
    refined_prediction_error: float
    matched_holdout_count: int = 0

    @property
    def matched_prediction_error_gain(self) -> float:
        return max(
            0.0,
            float(self.broad_prediction_error) - float(self.refined_prediction_error),
        )


class ContextRefiner:
    """Propose context partitions only when they reduce predictive contradiction."""

    def __init__(self, *, min_support: int = 4, contradiction_threshold: float = 0.20) -> None:
        self.min_support = int(min_support)
        self.contradiction_threshold = float(contradiction_threshold)

    @staticmethod
    def _error(rows: list[NodeRecord]) -> float:
        total = sum(max(0, int(row.support_count)) for row in rows)
        if total <= 0:
            return 0.0
        by_outcome: dict[int, int] = defaultdict(int)
        for row in rows:
            if len(row.key_parts) >= 3:
                by_outcome[int(row.key_parts[2])] += max(0, int(row.support_count))
        dominant = max(by_outcome.values(), default=0)
        return 1.0 - dominant / total

    @staticmethod
    def _leave_one_out_error(rows: list[NodeRecord]) -> tuple[float, int]:
        """Evaluate each occurrence against a predictor fit without that occurrence."""
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            if len(row.key_parts) >= 3:
                counts[int(row.key_parts[2])] += max(0, int(row.support_count))
        total = sum(counts.values())
        if total < 2:
            return 0.0, 0
        errors = evaluated = 0
        for actual, occurrences in counts.items():
            if occurrences <= 0:
                continue
            training = dict(counts)
            training[actual] -= 1
            if sum(training.values()) <= 0:
                continue
            predicted = min(training, key=lambda value: (-training[value], value))
            errors += occurrences * int(predicted != actual)
            evaluated += occurrences
        return (errors / evaluated if evaluated else 0.0), evaluated

    def propose(self, rows: tuple[NodeRecord, ...]) -> tuple[ContextRefinement, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) == int(MemoryLevel.M1) and len(row.key_parts) >= 3:
                grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)
        result: list[ContextRefinement] = []
        for (context, action), variants in grouped.items():
            total = sum(int(row.support_count) for row in variants)
            if total < self.min_support or len(variants) < 2:
                continue
            broad_error, broad_holdouts = self._leave_one_out_error(variants)
            if broad_error < self.contradiction_threshold:
                continue

            partitions: dict[int, list[NodeRecord]] = defaultdict(list)
            for row in variants:
                next_context = int(row.key_parts[3]) if len(row.key_parts) >= 4 else 0
                partitions[next_context].append(row)
            if len(partitions) < 2 or any(
                sum(max(0, int(row.support_count)) for row in part) < 2
                for part in partitions.values()
            ):
                continue
            refined_errors = 0.0
            refined_holdouts = 0
            for partition_rows in partitions.values():
                error, holdouts = self._leave_one_out_error(partition_rows)
                refined_errors += error * holdouts
                refined_holdouts += holdouts
            if broad_holdouts <= 0 or refined_holdouts != broad_holdouts:
                continue
            refined_error = refined_errors / refined_holdouts
            gain = max(0.0, broad_error - refined_error)
            if gain <= 1e-12:
                continue

            for row in variants:
                outcome = int(row.key_parts[2])
                next_context = int(row.key_parts[3]) if len(row.key_parts) >= 4 else 0
                context_partition = stable_u64(context, next_context, person=b"v8-context-split")
                key = (context_partition, action, outcome)
                result.append(
                    ContextRefinement(
                        row.uid,
                        MemoryUid.from_key(MemoryLevel.M3, MemoryType.CONTEXTUAL_ROLE, key),
                        key,
                        broad_error,
                        broad_error,
                        refined_error,
                        refined_holdouts,
                    )
                )
        return tuple(result)
