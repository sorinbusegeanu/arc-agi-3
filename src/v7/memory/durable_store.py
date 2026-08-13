from __future__ import annotations

import sqlite3
from pathlib import Path

from v7.memory.delta import GenerationDelta
from v7.memory.generation import GenerationState
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.schema import ensure_v7_schema


class DurableGenerationStore:
    """Single-writer SQLite durability adapter for committed v7 generation deltas."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        ensure_v7_schema(self.connection)

    def close(self) -> None:
        self.connection.close()

    def persist_generation_delta(self, state: GenerationState, delta: GenerationDelta, *, batch_id: int = 0) -> None:
        generation_id = int(state.generation_id)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO generations(generation_id, parent_generation_id, first_global_step, last_global_step, committed) VALUES (?, ?, ?, ?, 0)",
                (
                    generation_id,
                    None if state.parent_generation_id is None else int(state.parent_generation_id),
                    state.first_global_step,
                    state.last_global_step,
                ),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO generation_batches(generation_id, batch_id, mutation_count) VALUES (?, ?, ?)",
                (generation_id, int(batch_id), delta.mutation_count),
            )
            if delta.nodes:
                self.connection.executemany(
                    """
                    INSERT INTO memory_nodes(memory_id, level_id, type_id, created_generation, updated_generation, status_flags, support_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        level_id=excluded.level_id,
                        type_id=excluded.type_id,
                        updated_generation=excluded.updated_generation,
                        status_flags=excluded.status_flags,
                        support_count=excluded.support_count
                    """,
                    [self._node_row(node) for node in delta.nodes],
                )
            if delta.scores:
                self.connection.executemany(
                    """
                    INSERT INTO memory_scores(memory_id, significance, prediction_error, learning_value, transfer_prior, explanatory_potential, future_option_delta, updated_generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        significance=excluded.significance,
                        prediction_error=excluded.prediction_error,
                        learning_value=excluded.learning_value,
                        transfer_prior=excluded.transfer_prior,
                        explanatory_potential=excluded.explanatory_potential,
                        future_option_delta=excluded.future_option_delta,
                        updated_generation=excluded.updated_generation
                    """,
                    [self._score_row(score, generation_id) for score in delta.scores],
                )
            for edge in delta.edges:
                key = (int(edge.source_id), int(edge.relation_type), int(edge.target_id))
                if edge.support_count <= 0:
                    self.connection.execute(
                        "DELETE FROM memory_edges WHERE source_id=? AND relation_type=? AND target_id=?",
                        key,
                    )
                else:
                    self.connection.execute(
                        """
                        INSERT INTO memory_edges(source_id, relation_type, target_id, support_count, updated_generation)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(source_id, relation_type, target_id) DO UPDATE SET
                            support_count=excluded.support_count,
                            updated_generation=excluded.updated_generation
                        """,
                        (*key, int(edge.support_count), generation_id),
                    )
            self.connection.execute(
                "UPDATE generations SET committed=1, committed_at=CURRENT_TIMESTAMP WHERE generation_id=?",
                (generation_id,),
            )

    @staticmethod
    def _node_row(node: MemoryNode) -> tuple[int, ...]:
        return (
            int(node.memory_id),
            int(node.level),
            int(node.type_id),
            int(node.created_generation),
            int(node.updated_generation),
            int(node.status_flags),
            int(node.support_count),
        )

    @staticmethod
    def _score_row(score: MemoryScore, generation_id: int) -> tuple[int | float, ...]:
        return (
            int(score.memory_id),
            float(score.significance),
            float(score.prediction_error),
            float(score.learning_value),
            float(score.transfer_prior),
            float(score.explanatory_potential),
            float(score.future_option_delta),
            generation_id,
        )
