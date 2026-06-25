from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interaction:
    id: int
    timestamp: int
    observation_before: np.ndarray
    action: int
    observation_after: np.ndarray
    delta_id: int
    global_step: int | None = None
    isf_version: str | None = None
    isf_total: float | None = None
    isf_survival_impact: float | None = None
    isf_prediction_error: float | None = None
    isf_learning_value: float | None = None
    isf_transfer_potential: float | None = None
    isf_explanatory_potential: float | None = None
    isf_weights_json: str | None = None
    carrier_signature: str | None = None
    carrier_source: str = "unknown"
    carrier_event_recorded: bool = False
    carrier_support_count: int | None = None
    carrier_distinct_family_count: int | None = None
    carrier_distinct_context_count: int | None = None
    memory_status: str | None = None
    memory_retention_reason: str | None = None
    memory_replay_priority: float = 0.0
    memory_replay_candidate: bool = False
    memory_replay_count: int = 0
    context_depth_used: int | None = None
    adaptive_context_expansion_applied: bool = False
    adaptive_context_depth_after: int | None = None
    efficiency_action_cost: float | None = None
    efficiency_cumulative_cost: float | None = None
    efficiency_repeated_state: bool = False
    efficiency_repeated_context_action: bool = False
    efficiency_no_effect_action: bool = False
    efficiency_terminal_outcome: bool = False
    efficiency_outcome_signature: str | None = None
    efficiency_best_known_cost_for_outcome: float | None = None
    efficiency_normalized_solve_efficiency: float | None = None
    efficiency_equivalent_outcome_cost_gap: float | None = None
    efficiency_future_option_gain_per_cost: float | None = None


def encode_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=int), allow_pickle=False)
    return buffer.getvalue()


def decode_array(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


class InteractionStore:
    def __init__(self, connection: sqlite3.Connection, *, auto_commit: bool = True) -> None:
        self.connection = connection
        self.auto_commit = bool(auto_commit)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                global_step INTEGER,
                observation_before BLOB NOT NULL,
                action INTEGER NOT NULL,
                observation_after BLOB NOT NULL,
                delta_id INTEGER NOT NULL,
                isf_version TEXT,
                isf_total REAL,
                isf_survival_impact REAL,
                isf_prediction_error REAL,
                isf_learning_value REAL,
                isf_transfer_potential REAL,
                isf_explanatory_potential REAL,
                isf_weights_json TEXT,
                carrier_signature TEXT,
                carrier_source TEXT,
                carrier_event_recorded INTEGER,
                carrier_support_count INTEGER,
                carrier_distinct_family_count INTEGER,
                carrier_distinct_context_count INTEGER,
                memory_status TEXT,
                memory_retention_reason TEXT,
                memory_replay_priority REAL,
                memory_replay_candidate INTEGER,
                memory_replay_count INTEGER,
                context_depth_used INTEGER,
                adaptive_context_expansion_applied INTEGER,
                adaptive_context_depth_after INTEGER,
                efficiency_action_cost REAL,
                efficiency_cumulative_cost REAL,
                efficiency_repeated_state INTEGER,
                efficiency_repeated_context_action INTEGER,
                efficiency_no_effect_action INTEGER,
                efficiency_terminal_outcome INTEGER,
                efficiency_outcome_signature TEXT,
                efficiency_best_known_cost_for_outcome REAL,
                efficiency_normalized_solve_efficiency REAL,
                efficiency_equivalent_outcome_cost_gap REAL,
                efficiency_future_option_gain_per_cost REAL
            )
            """
        )
        self._ensure_column("isf_version", "TEXT")
        self._ensure_column("global_step", "INTEGER")
        self._ensure_column("isf_total", "REAL")
        self._ensure_column("isf_survival_impact", "REAL")
        self._ensure_column("isf_prediction_error", "REAL")
        self._ensure_column("isf_learning_value", "REAL")
        self._ensure_column("isf_transfer_potential", "REAL")
        self._ensure_column("isf_explanatory_potential", "REAL")
        self._ensure_column("isf_weights_json", "TEXT")
        self._ensure_column("carrier_signature", "TEXT")
        self._ensure_column("carrier_source", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("carrier_event_recorded", "INTEGER")
        self._ensure_column("carrier_support_count", "INTEGER")
        self._ensure_column("carrier_distinct_family_count", "INTEGER")
        self._ensure_column("carrier_distinct_context_count", "INTEGER")
        self._ensure_column("memory_status", "TEXT")
        self._ensure_column("memory_retention_reason", "TEXT")
        self._ensure_column("memory_replay_priority", "REAL")
        self._ensure_column("memory_replay_candidate", "INTEGER")
        self._ensure_column("memory_replay_count", "INTEGER")
        self._ensure_column("context_depth_used", "INTEGER")
        self._ensure_column("adaptive_context_expansion_applied", "INTEGER")
        self._ensure_column("adaptive_context_depth_after", "INTEGER")
        self._ensure_column("efficiency_action_cost", "REAL")
        self._ensure_column("efficiency_cumulative_cost", "REAL")
        self._ensure_column("efficiency_repeated_state", "INTEGER")
        self._ensure_column("efficiency_repeated_context_action", "INTEGER")
        self._ensure_column("efficiency_no_effect_action", "INTEGER")
        self._ensure_column("efficiency_terminal_outcome", "INTEGER")
        self._ensure_column("efficiency_outcome_signature", "TEXT")
        self._ensure_column("efficiency_best_known_cost_for_outcome", "REAL")
        self._ensure_column("efficiency_normalized_solve_efficiency", "REAL")
        self._ensure_column("efficiency_equivalent_outcome_cost_gap", "REAL")
        self._ensure_column("efficiency_future_option_gain_per_cost", "REAL")
        self.connection.commit()

    def _ensure_column(self, column: str, declaration: str) -> None:
        rows = self.connection.execute("PRAGMA table_info(interactions)").fetchall()
        if any(str(row[1]) == column for row in rows):
            return
        self.connection.execute(f"ALTER TABLE interactions ADD COLUMN {column} {declaration}")

    def next_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM interactions").fetchone()
        return int(row[0])

    def add(self, interaction: Interaction) -> None:
        self.connection.execute(
            """
            INSERT INTO interactions (
                id,
                timestamp,
                global_step,
                observation_before,
                action,
                observation_after,
                delta_id,
                isf_version,
                isf_total,
                isf_survival_impact,
                isf_prediction_error,
                isf_learning_value,
                isf_transfer_potential,
                isf_explanatory_potential,
                isf_weights_json,
                carrier_signature,
                carrier_source,
                carrier_event_recorded,
                carrier_support_count,
                carrier_distinct_family_count,
                carrier_distinct_context_count,
                memory_status,
                memory_retention_reason,
                memory_replay_priority,
                memory_replay_candidate,
                memory_replay_count,
                context_depth_used,
                adaptive_context_expansion_applied,
                adaptive_context_depth_after,
                efficiency_action_cost,
                efficiency_cumulative_cost,
                efficiency_repeated_state,
                efficiency_repeated_context_action,
                efficiency_no_effect_action,
                efficiency_terminal_outcome,
                efficiency_outcome_signature,
                efficiency_best_known_cost_for_outcome,
                efficiency_normalized_solve_efficiency,
                efficiency_equivalent_outcome_cost_gap,
                efficiency_future_option_gain_per_cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(interaction.id),
                int(interaction.timestamp),
                None if interaction.global_step is None else int(interaction.global_step),
                encode_array(interaction.observation_before),
                int(interaction.action),
                encode_array(interaction.observation_after),
                int(interaction.delta_id),
                interaction.isf_version,
                None if interaction.isf_total is None else float(interaction.isf_total),
                None if interaction.isf_survival_impact is None else float(interaction.isf_survival_impact),
                None if interaction.isf_prediction_error is None else float(interaction.isf_prediction_error),
                None if interaction.isf_learning_value is None else float(interaction.isf_learning_value),
                None if interaction.isf_transfer_potential is None else float(interaction.isf_transfer_potential),
                None if interaction.isf_explanatory_potential is None else float(interaction.isf_explanatory_potential),
                interaction.isf_weights_json,
                interaction.carrier_signature,
                str(interaction.carrier_source or "unknown"),
                int(bool(interaction.carrier_event_recorded)),
                None if interaction.carrier_support_count is None else int(interaction.carrier_support_count),
                None if interaction.carrier_distinct_family_count is None else int(interaction.carrier_distinct_family_count),
                None if interaction.carrier_distinct_context_count is None else int(interaction.carrier_distinct_context_count),
                interaction.memory_status,
                interaction.memory_retention_reason,
                float(interaction.memory_replay_priority),
                int(bool(interaction.memory_replay_candidate)),
                int(interaction.memory_replay_count),
                None if interaction.context_depth_used is None else int(interaction.context_depth_used),
                int(bool(interaction.adaptive_context_expansion_applied)),
                None if interaction.adaptive_context_depth_after is None else int(interaction.adaptive_context_depth_after),
                None if interaction.efficiency_action_cost is None else float(interaction.efficiency_action_cost),
                None if interaction.efficiency_cumulative_cost is None else float(interaction.efficiency_cumulative_cost),
                int(bool(interaction.efficiency_repeated_state)),
                int(bool(interaction.efficiency_repeated_context_action)),
                int(bool(interaction.efficiency_no_effect_action)),
                int(bool(interaction.efficiency_terminal_outcome)),
                interaction.efficiency_outcome_signature,
                None if interaction.efficiency_best_known_cost_for_outcome is None else float(interaction.efficiency_best_known_cost_for_outcome),
                None if interaction.efficiency_normalized_solve_efficiency is None else float(interaction.efficiency_normalized_solve_efficiency),
                None if interaction.efficiency_equivalent_outcome_cost_gap is None else float(interaction.efficiency_equivalent_outcome_cost_gap),
                None if interaction.efficiency_future_option_gain_per_cost is None else float(interaction.efficiency_future_option_gain_per_cost),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def get(self, interaction_id: int) -> Interaction | None:
        row = self.connection.execute(
            """
            SELECT
                id,
                timestamp,
                global_step,
                observation_before,
                action,
                observation_after,
                delta_id,
                isf_version,
                isf_total,
                isf_survival_impact,
                isf_prediction_error,
                isf_learning_value,
                isf_transfer_potential,
                isf_explanatory_potential,
                isf_weights_json,
                carrier_signature,
                carrier_source,
                carrier_event_recorded,
                carrier_support_count,
                carrier_distinct_family_count,
                carrier_distinct_context_count,
                memory_status,
                memory_retention_reason,
                memory_replay_priority,
                memory_replay_candidate,
                memory_replay_count,
                context_depth_used,
                adaptive_context_expansion_applied,
                adaptive_context_depth_after,
                efficiency_action_cost,
                efficiency_cumulative_cost,
                efficiency_repeated_state,
                efficiency_repeated_context_action,
                efficiency_no_effect_action,
                efficiency_terminal_outcome,
                efficiency_outcome_signature,
                efficiency_best_known_cost_for_outcome,
                efficiency_normalized_solve_efficiency,
                efficiency_equivalent_outcome_cost_gap,
                efficiency_future_option_gain_per_cost
            FROM interactions
            WHERE id = ?
            """,
            (int(interaction_id),),
        ).fetchone()
        if row is None:
            return None
        return Interaction(
            id=int(row[0]),
            timestamp=int(row[1]),
            global_step=None if row[2] is None else int(row[2]),
            observation_before=decode_array(row[3]),
            action=int(row[4]),
            observation_after=decode_array(row[5]),
            delta_id=int(row[6]),
            isf_version=None if row[7] is None else str(row[7]),
            isf_total=None if row[8] is None else float(row[8]),
            isf_survival_impact=None if row[9] is None else float(row[9]),
            isf_prediction_error=None if row[10] is None else float(row[10]),
            isf_learning_value=None if row[11] is None else float(row[11]),
            isf_transfer_potential=None if row[12] is None else float(row[12]),
            isf_explanatory_potential=None if row[13] is None else float(row[13]),
            isf_weights_json=None if row[14] is None else str(row[14]),
            carrier_signature=None if row[15] is None else str(row[15]),
            carrier_source="unknown" if row[16] is None else str(row[16]),
            carrier_event_recorded=bool(row[17]) if row[17] is not None else False,
            carrier_support_count=None if row[18] is None else int(row[18]),
            carrier_distinct_family_count=None if row[19] is None else int(row[19]),
            carrier_distinct_context_count=None if row[20] is None else int(row[20]),
            memory_status=None if row[21] is None else str(row[21]),
            memory_retention_reason=None if row[22] is None else str(row[22]),
            memory_replay_priority=0.0 if row[23] is None else float(row[23]),
            memory_replay_candidate=bool(row[24]) if row[24] is not None else False,
            memory_replay_count=0 if row[25] is None else int(row[25]),
            context_depth_used=None if row[26] is None else int(row[26]),
            adaptive_context_expansion_applied=bool(row[27]) if row[27] is not None else False,
            adaptive_context_depth_after=None if row[28] is None else int(row[28]),
            efficiency_action_cost=None if row[29] is None else float(row[29]),
            efficiency_cumulative_cost=None if row[30] is None else float(row[30]),
            efficiency_repeated_state=bool(row[31]) if row[31] is not None else False,
            efficiency_repeated_context_action=bool(row[32]) if row[32] is not None else False,
            efficiency_no_effect_action=bool(row[33]) if row[33] is not None else False,
            efficiency_terminal_outcome=bool(row[34]) if row[34] is not None else False,
            efficiency_outcome_signature=None if row[35] is None else str(row[35]),
            efficiency_best_known_cost_for_outcome=None if row[36] is None else float(row[36]),
            efficiency_normalized_solve_efficiency=None if row[37] is None else float(row[37]),
            efficiency_equivalent_outcome_cost_gap=None if row[38] is None else float(row[38]),
            efficiency_future_option_gain_per_cost=None if row[39] is None else float(row[39]),
        )

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM interactions").fetchone()
        return int(row[0])
