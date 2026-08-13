from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from v7.derivation.batches import DerivedMutationBatch
from v7.derivation.workers import DerivationTask, DerivationTaskResult
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryLevel
from v7.memory.read_view import MemoryReadView


@dataclass(frozen=True, slots=True)
class VectorizedDerivationInput:
    generation_id: GenerationId
    level: MemoryLevel
    memory_ids: np.ndarray
    node_rows: np.ndarray
    support_counts: np.ndarray
    status_flags: np.ndarray
    score_present: np.ndarray
    score_matrix: np.ndarray

    @property
    def count(self) -> int:
        return int(self.memory_ids.size)


VectorizedKernel = Callable[[VectorizedDerivationInput], DerivedMutationBatch]


class VectorizedDerivationEngine:
    """Prepare one dense numeric batch per dirty derivation task."""

    @staticmethod
    def _numeric(values, dtype) -> np.ndarray:
        return np.asarray(values, dtype=dtype)

    def build_input(self, view: MemoryReadView, task: DerivationTask) -> VectorizedDerivationInput:
        if task.generation_id != view.generation_id:
            raise ValueError("derivation task targets a different read generation")
        arena = view.compact_arena
        node_ids = self._numeric(arena.nodes.memory_ids, np.uint64)
        requested = np.asarray([int(value) for value in task.memory_ids], dtype=np.uint64)
        rows = np.searchsorted(node_ids, requested)
        if np.any(rows >= node_ids.size):
            raise KeyError("derivation task contains memory IDs absent from read view")
        if np.any(node_ids[rows] != requested):
            raise KeyError("derivation task contains memory IDs absent from read view")
        levels = self._numeric(arena.nodes.levels, np.uint8)[rows]
        if np.any(levels != int(task.level)):
            raise ValueError("derivation task contains IDs from a different memory level")

        support_counts = self._numeric(arena.nodes.support_counts, np.int64)[rows].copy()
        status_flags = self._numeric(arena.nodes.status_flags, np.uint64)[rows].copy()

        score_ids = self._numeric(arena.scores.memory_ids, np.uint64)
        score_rows = np.searchsorted(score_ids, requested) if score_ids.size else np.zeros(requested.size, dtype=np.int64)
        present = np.zeros(requested.size, dtype=np.bool_)
        if score_ids.size:
            valid = score_rows < score_ids.size
            present[valid] = score_ids[score_rows[valid]] == requested[valid]

        score_matrix = np.zeros((requested.size, 6), dtype=np.float64)
        if np.any(present):
            selected = score_rows[present]
            columns = (
                arena.scores.significance,
                arena.scores.prediction_error,
                arena.scores.learning_value,
                arena.scores.transfer_prior,
                arena.scores.explanatory_potential,
                arena.scores.future_option_delta,
            )
            for column_index, values in enumerate(columns):
                score_matrix[present, column_index] = self._numeric(values, np.float64)[selected]

        for values in (requested, rows, support_counts, status_flags, present, score_matrix):
            values.setflags(write=False)
        return VectorizedDerivationInput(
            generation_id=view.generation_id,
            level=task.level,
            memory_ids=requested,
            node_rows=rows,
            support_counts=support_counts,
            status_flags=status_flags,
            score_present=present,
            score_matrix=score_matrix,
        )

    def run(self, view: MemoryReadView, task: DerivationTask, kernel: VectorizedKernel) -> DerivationTaskResult[DerivedMutationBatch]:
        output = kernel(self.build_input(view, task))
        if output.generation_id != task.generation_id:
            raise ValueError("vectorized kernel returned wrong generation")
        if output.source_level != task.level:
            raise ValueError("vectorized kernel returned wrong source level")
        if output.source_ids != task.memory_ids:
            raise ValueError("vectorized kernel returned wrong source IDs")
        return DerivationTaskResult(task=task, output=output)
