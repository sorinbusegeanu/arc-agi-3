from __future__ import annotations

import json
import sqlite3
from hashlib import sha1

from v6.contingency.contingency_learner import Contingency


class ContingencyStore:
    def __init__(self, connection: sqlite3.Connection, *, auto_commit: bool = True) -> None:
        self.connection = connection
        self.auto_commit = bool(auto_commit)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contingencies (
                id INTEGER PRIMARY KEY,
                context_level INTEGER NOT NULL DEFAULT 0,
                context_signature TEXT NOT NULL,
                action INTEGER NOT NULL,
                transformation_family INTEGER NOT NULL,
                support_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                prediction_attempt_count INTEGER DEFAULT 0,
                prediction_success_count INTEGER DEFAULT 0,
                prediction_accuracy REAL,
                prediction_error_before REAL,
                prediction_error_after REAL,
                normalized_contingency_key TEXT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                global_step INTEGER,
                context_level INTEGER,
                context_signature TEXT NOT NULL,
                action INTEGER NOT NULL,
                predicted_family INTEGER,
                actual_family INTEGER,
                prediction_error INTEGER,
                episode_id INTEGER NOT NULL DEFAULT 0,
                isf_version TEXT,
                isf_total REAL,
                isf_survival_impact REAL,
                isf_prediction_error REAL,
                isf_learning_value REAL,
                isf_transfer_potential REAL,
                isf_explanatory_potential REAL,
                isf_weights_json TEXT,
                context_contradiction INTEGER,
                context_contradiction_key TEXT,
                context_expansion_suggested INTEGER,
                suggested_context_depth INTEGER,
                context_contradiction_reason TEXT,
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
                efficiency_future_option_gain_per_cost REAL,
                outcome_state TEXT,
                outcome_polarity TEXT,
                level_completed_event INTEGER,
                post_factum_level_completion_credit REAL DEFAULT 0.0,
                post_factum_level_completion_decay REAL,
                post_factum_level_completion_step INTEGER,
                post_factum_credit_reason TEXT,
                post_factum_trajectory_credit REAL DEFAULT 0.0,
                post_factum_trajectory_credit_kind TEXT,
                post_factum_trajectory_credit_polarity TEXT,
                post_factum_trajectory_credit_step INTEGER,
                post_factum_trajectory_credit_reason TEXT,
                game_id TEXT,
                level_id TEXT,
                sampler_name TEXT,
                state_hash_before TEXT,
                state_hash_after TEXT,
                memory_fitness_base REAL,
                memory_fitness REAL,
                memory_replay_priority_base REAL,
                retention_score_base REAL,
                retention_score REAL,
                trajectory_efficiency_active INTEGER,
                trajectory_outcome_class TEXT,
                comparable_outcome_group_id TEXT,
                trajectory_efficiency_score REAL,
                efficiency_memory_bonus REAL,
                efficiency_replay_bonus REAL,
                efficiency_retention_bonus REAL,
                efficiency_promotion_bonus REAL
            )
            """
        )
        self._ensure_column("contingencies", "context_level", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("contingencies", "prediction_attempt_count", "INTEGER DEFAULT 0")
        self._ensure_column("contingencies", "prediction_success_count", "INTEGER DEFAULT 0")
        self._ensure_column("contingencies", "prediction_accuracy", "REAL")
        self._ensure_column("contingencies", "prediction_error_before", "REAL")
        self._ensure_column("contingencies", "prediction_error_after", "REAL")
        self._ensure_column("contingencies", "normalized_contingency_key", "TEXT")
        self._ensure_column("prediction_results", "global_step", "INTEGER")
        self._ensure_column("prediction_results", "context_level", "INTEGER")
        self._ensure_column("prediction_results", "episode_id", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("prediction_results", "isf_version", "TEXT")
        self._ensure_column("prediction_results", "isf_total", "REAL")
        self._ensure_column("prediction_results", "isf_survival_impact", "REAL")
        self._ensure_column("prediction_results", "isf_prediction_error", "REAL")
        self._ensure_column("prediction_results", "isf_learning_value", "REAL")
        self._ensure_column("prediction_results", "isf_transfer_potential", "REAL")
        self._ensure_column("prediction_results", "isf_explanatory_potential", "REAL")
        self._ensure_column("prediction_results", "isf_weights_json", "TEXT")
        self._ensure_column("prediction_results", "context_contradiction", "INTEGER")
        self._ensure_column("prediction_results", "context_contradiction_key", "TEXT")
        self._ensure_column("prediction_results", "context_expansion_suggested", "INTEGER")
        self._ensure_column("prediction_results", "suggested_context_depth", "INTEGER")
        self._ensure_column("prediction_results", "context_contradiction_reason", "TEXT")
        self._ensure_column("prediction_results", "carrier_signature", "TEXT")
        self._ensure_column("prediction_results", "carrier_source", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("prediction_results", "carrier_event_recorded", "INTEGER")
        self._ensure_column("prediction_results", "carrier_support_count", "INTEGER")
        self._ensure_column("prediction_results", "carrier_distinct_family_count", "INTEGER")
        self._ensure_column("prediction_results", "carrier_distinct_context_count", "INTEGER")
        self._ensure_column("prediction_results", "memory_status", "TEXT")
        self._ensure_column("prediction_results", "memory_retention_reason", "TEXT")
        self._ensure_column("prediction_results", "memory_replay_priority", "REAL")
        self._ensure_column("prediction_results", "memory_replay_candidate", "INTEGER")
        self._ensure_column("prediction_results", "memory_replay_count", "INTEGER")
        self._ensure_column("prediction_results", "context_depth_used", "INTEGER")
        self._ensure_column("prediction_results", "adaptive_context_expansion_applied", "INTEGER")
        self._ensure_column("prediction_results", "adaptive_context_depth_after", "INTEGER")
        self._ensure_column("prediction_results", "efficiency_action_cost", "REAL")
        self._ensure_column("prediction_results", "efficiency_cumulative_cost", "REAL")
        self._ensure_column("prediction_results", "efficiency_repeated_state", "INTEGER")
        self._ensure_column("prediction_results", "efficiency_repeated_context_action", "INTEGER")
        self._ensure_column("prediction_results", "efficiency_no_effect_action", "INTEGER")
        self._ensure_column("prediction_results", "efficiency_terminal_outcome", "INTEGER")
        self._ensure_column("prediction_results", "efficiency_outcome_signature", "TEXT")
        self._ensure_column("prediction_results", "efficiency_best_known_cost_for_outcome", "REAL")
        self._ensure_column("prediction_results", "efficiency_normalized_solve_efficiency", "REAL")
        self._ensure_column("prediction_results", "efficiency_equivalent_outcome_cost_gap", "REAL")
        self._ensure_column("prediction_results", "efficiency_future_option_gain_per_cost", "REAL")
        self._ensure_column("prediction_results", "outcome_state", "TEXT")
        self._ensure_column("prediction_results", "outcome_polarity", "TEXT")
        self._ensure_column("prediction_results", "level_completed_event", "INTEGER")
        self._ensure_column("prediction_results", "post_factum_level_completion_credit", "REAL DEFAULT 0.0")
        self._ensure_column("prediction_results", "post_factum_level_completion_decay", "REAL")
        self._ensure_column("prediction_results", "post_factum_level_completion_step", "INTEGER")
        self._ensure_column("prediction_results", "post_factum_credit_reason", "TEXT")
        self._ensure_column("prediction_results", "post_factum_trajectory_credit", "REAL DEFAULT 0.0")
        self._ensure_column("prediction_results", "post_factum_trajectory_credit_kind", "TEXT")
        self._ensure_column("prediction_results", "post_factum_trajectory_credit_polarity", "TEXT")
        self._ensure_column("prediction_results", "post_factum_trajectory_credit_step", "INTEGER")
        self._ensure_column("prediction_results", "post_factum_trajectory_credit_reason", "TEXT")
        self._ensure_column("prediction_results", "memory_state", "TEXT")
        self._ensure_column("prediction_results", "stored_epoch", "INTEGER")
        self._ensure_column("prediction_results", "last_replayed_epoch", "INTEGER")
        self._ensure_column("prediction_results", "last_promoted_epoch", "INTEGER")
        self._ensure_column("prediction_results", "retention_score", "REAL")
        self._ensure_column("prediction_results", "forgetting_score", "REAL")
        self._ensure_column("prediction_results", "compressed_into_id", "TEXT")
        self._ensure_column("prediction_results", "superseded_by_id", "TEXT")
        self._ensure_column("prediction_results", "forgetting_reason", "TEXT")
        self._ensure_column("prediction_results", "game_id", "TEXT")
        self._ensure_column("prediction_results", "level_id", "TEXT")
        self._ensure_column("prediction_results", "sampler_name", "TEXT")
        self._ensure_column("prediction_results", "state_hash_before", "TEXT")
        self._ensure_column("prediction_results", "state_hash_after", "TEXT")
        self._ensure_column("prediction_results", "memory_fitness_base", "REAL")
        self._ensure_column("prediction_results", "memory_fitness", "REAL")
        self._ensure_column("prediction_results", "memory_replay_priority_base", "REAL")
        self._ensure_column("prediction_results", "retention_score_base", "REAL")
        self._ensure_column("prediction_results", "retention_score", "REAL")
        self._ensure_column("prediction_results", "trajectory_efficiency_active", "INTEGER")
        self._ensure_column("prediction_results", "trajectory_outcome_class", "TEXT")
        self._ensure_column("prediction_results", "comparable_outcome_group_id", "TEXT")
        self._ensure_column("prediction_results", "trajectory_efficiency_score", "REAL")
        self._ensure_column("prediction_results", "efficiency_memory_bonus", "REAL")
        self._ensure_column("prediction_results", "efficiency_replay_bonus", "REAL")
        self._ensure_column("prediction_results", "efficiency_retention_bonus", "REAL")
        self._ensure_column("prediction_results", "efficiency_promotion_bonus", "REAL")
        self._ensure_column("prediction_results", "level_advanced", "INTEGER")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(row[1]) == column for row in rows):
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def upsert_contingency(self, contingency: Contingency) -> None:
        normalized_key = self._normalized_contingency_key(
            context_level=int(contingency.context_level),
            action=int(contingency.action),
            transformation_family=int(contingency.transformation_family),
        )
        self.connection.execute(
            """
            INSERT INTO contingencies (
                id,
                context_level,
                context_signature,
                action,
                transformation_family,
                support_count,
                confidence,
                normalized_contingency_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                context_level = excluded.context_level,
                context_signature = excluded.context_signature,
                action = excluded.action,
                transformation_family = excluded.transformation_family,
                support_count = excluded.support_count,
                confidence = excluded.confidence,
                normalized_contingency_key = excluded.normalized_contingency_key
            """,
            (
                int(contingency.id),
                int(contingency.context_level),
                json.dumps(contingency.context_signature),
                int(contingency.action),
                int(contingency.transformation_family),
                int(contingency.support_count),
                float(contingency.confidence),
                normalized_key,
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def record_contingency_prediction(
        self,
        *,
        contingency_id: int,
        prediction_success: bool,
        prediction_error_before: float | None = None,
        prediction_error_after: float | None = None,
        normalized_contingency_key: str | None = None,
    ) -> None:
        row = self.connection.execute(
            """
            SELECT prediction_attempt_count, prediction_success_count, normalized_contingency_key
            FROM contingencies
            WHERE id = ?
            """,
            (int(contingency_id),),
        ).fetchone()
        if row is None:
            return
        attempts = int(row[0] or 0) + 1
        successes = int(row[1] or 0) + int(bool(prediction_success))
        key = normalized_contingency_key or row[2]
        self.connection.execute(
            """
            UPDATE contingencies
            SET
                prediction_attempt_count = ?,
                prediction_success_count = ?,
                prediction_accuracy = ?,
                prediction_error_before = ?,
                prediction_error_after = ?,
                normalized_contingency_key = COALESCE(?, normalized_contingency_key)
            WHERE id = ?
            """,
            (
                attempts,
                successes,
                (float(successes) / float(attempts)) if attempts > 0 else None,
                None if prediction_error_before is None else float(prediction_error_before),
                None if prediction_error_after is None else float(prediction_error_after),
                key,
                int(contingency_id),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def _normalized_contingency_key(self, *, context_level: int, action: int, transformation_family: int) -> str:
        payload = {
            "action": int(action),
            "context_level": int(context_level),
            "family_bucket": int(transformation_family),
        }
        return "raw_contingency:" + sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]

    def add_prediction_result(
        self,
        *,
        interaction_id: int,
        global_step: int | None = None,
        context_level: int | None = None,
        context_signature: tuple,
        action: int,
        predicted_family: int | None,
        actual_family: int | None,
        episode_id: int = 0,
        isf_version: str | None = None,
        isf_total: float | None = None,
        isf_survival_impact: float | None = None,
        isf_prediction_error: float | None = None,
        isf_learning_value: float | None = None,
        isf_transfer_potential: float | None = None,
        isf_explanatory_potential: float | None = None,
        isf_weights_json: str | None = None,
        context_contradiction: bool = False,
        context_contradiction_key: str | None = None,
        context_expansion_suggested: bool = False,
        suggested_context_depth: int | None = None,
        context_contradiction_reason: str | None = None,
        carrier_signature: str | None = None,
        carrier_source: str = "unknown",
        carrier_event_recorded: bool = False,
        carrier_support_count: int | None = None,
        carrier_distinct_family_count: int | None = None,
        carrier_distinct_context_count: int | None = None,
        memory_status: str | None = None,
        memory_retention_reason: str | None = None,
        memory_replay_priority: float = 0.0,
        memory_replay_candidate: bool = False,
        memory_replay_count: int = 0,
        context_depth_used: int | None = None,
        adaptive_context_expansion_applied: bool = False,
        adaptive_context_depth_after: int | None = None,
        efficiency_action_cost: float | None = None,
        efficiency_cumulative_cost: float | None = None,
        efficiency_repeated_state: bool = False,
        efficiency_repeated_context_action: bool = False,
        efficiency_no_effect_action: bool = False,
        efficiency_terminal_outcome: bool = False,
        efficiency_outcome_signature: str | None = None,
        efficiency_best_known_cost_for_outcome: float | None = None,
        efficiency_normalized_solve_efficiency: float | None = None,
        efficiency_equivalent_outcome_cost_gap: float | None = None,
        efficiency_future_option_gain_per_cost: float | None = None,
        outcome_state: str | None = None,
        outcome_polarity: str | None = None,
        level_completed_event: bool = False,
        post_factum_level_completion_credit: float = 0.0,
        post_factum_level_completion_decay: float | None = None,
        post_factum_level_completion_step: int | None = None,
        post_factum_credit_reason: str | None = None,
        post_factum_trajectory_credit: float = 0.0,
        post_factum_trajectory_credit_kind: str | None = None,
        post_factum_trajectory_credit_polarity: str | None = None,
        post_factum_trajectory_credit_step: int | None = None,
        post_factum_trajectory_credit_reason: str | None = None,
        game_id: str | None = None,
        level_id: str | None = None,
        sampler_name: str | None = None,
        state_hash_before: str | None = None,
        state_hash_after: str | None = None,
        memory_fitness_base: float | None = None,
        memory_fitness: float | None = None,
        memory_replay_priority_base: float | None = None,
        retention_score_base: float | None = None,
        retention_score: float | None = None,
        memory_state: str | None = None,
        stored_epoch: int | None = None,
        last_replayed_epoch: int | None = None,
        last_promoted_epoch: int | None = None,
        forgetting_score: float | None = None,
        compressed_into_id: str | None = None,
        superseded_by_id: str | None = None,
        forgetting_reason: str | None = None,
    ) -> int | None:
        prediction_error: int | None
        if predicted_family is None or actual_family is None:
            prediction_error = None
        else:
            prediction_error = 0 if int(predicted_family) == int(actual_family) else 1
        row = {
            "interaction_id": int(interaction_id),
            "global_step": None if global_step is None else int(global_step),
            "context_level": None if context_level is None else int(context_level),
            "context_signature": json.dumps(context_signature),
            "action": int(action),
            "predicted_family": None if predicted_family is None else int(predicted_family),
            "actual_family": None if actual_family is None else int(actual_family),
            "prediction_error": prediction_error,
            "episode_id": int(episode_id),
            "isf_version": isf_version,
            "isf_total": None if isf_total is None else float(isf_total),
            "isf_survival_impact": None if isf_survival_impact is None else float(isf_survival_impact),
            "isf_prediction_error": None if isf_prediction_error is None else float(isf_prediction_error),
            "isf_learning_value": None if isf_learning_value is None else float(isf_learning_value),
            "isf_transfer_potential": None if isf_transfer_potential is None else float(isf_transfer_potential),
            "isf_explanatory_potential": None if isf_explanatory_potential is None else float(isf_explanatory_potential),
            "isf_weights_json": isf_weights_json,
            "context_contradiction": int(bool(context_contradiction)),
            "context_contradiction_key": context_contradiction_key,
            "context_expansion_suggested": int(bool(context_expansion_suggested)),
            "suggested_context_depth": None if suggested_context_depth is None else int(suggested_context_depth),
            "context_contradiction_reason": context_contradiction_reason,
            "carrier_signature": carrier_signature,
            "carrier_source": str(carrier_source or "unknown"),
            "carrier_event_recorded": int(bool(carrier_event_recorded)),
            "carrier_support_count": None if carrier_support_count is None else int(carrier_support_count),
            "carrier_distinct_family_count": None if carrier_distinct_family_count is None else int(carrier_distinct_family_count),
            "carrier_distinct_context_count": None if carrier_distinct_context_count is None else int(carrier_distinct_context_count),
            "memory_status": memory_status,
            "memory_retention_reason": memory_retention_reason,
            "memory_replay_priority": float(memory_replay_priority),
            "memory_replay_candidate": int(bool(memory_replay_candidate)),
            "memory_replay_count": int(memory_replay_count),
            "context_depth_used": None if context_depth_used is None else int(context_depth_used),
            "adaptive_context_expansion_applied": int(bool(adaptive_context_expansion_applied)),
            "adaptive_context_depth_after": None if adaptive_context_depth_after is None else int(adaptive_context_depth_after),
            "efficiency_action_cost": None if efficiency_action_cost is None else float(efficiency_action_cost),
            "efficiency_cumulative_cost": None if efficiency_cumulative_cost is None else float(efficiency_cumulative_cost),
            "efficiency_repeated_state": int(bool(efficiency_repeated_state)),
            "efficiency_repeated_context_action": int(bool(efficiency_repeated_context_action)),
            "efficiency_no_effect_action": int(bool(efficiency_no_effect_action)),
            "efficiency_terminal_outcome": int(bool(efficiency_terminal_outcome)),
            "efficiency_outcome_signature": efficiency_outcome_signature,
            "efficiency_best_known_cost_for_outcome": None if efficiency_best_known_cost_for_outcome is None else float(efficiency_best_known_cost_for_outcome),
            "efficiency_normalized_solve_efficiency": None if efficiency_normalized_solve_efficiency is None else float(efficiency_normalized_solve_efficiency),
            "efficiency_equivalent_outcome_cost_gap": None if efficiency_equivalent_outcome_cost_gap is None else float(efficiency_equivalent_outcome_cost_gap),
            "efficiency_future_option_gain_per_cost": None if efficiency_future_option_gain_per_cost is None else float(efficiency_future_option_gain_per_cost),
            "outcome_state": outcome_state,
            "outcome_polarity": outcome_polarity,
            "level_completed_event": int(bool(level_completed_event)),
            "post_factum_level_completion_credit": float(post_factum_level_completion_credit),
            "post_factum_level_completion_decay": None if post_factum_level_completion_decay is None else float(post_factum_level_completion_decay),
            "post_factum_level_completion_step": None if post_factum_level_completion_step is None else int(post_factum_level_completion_step),
            "post_factum_credit_reason": post_factum_credit_reason,
            "post_factum_trajectory_credit": float(post_factum_trajectory_credit),
            "post_factum_trajectory_credit_kind": post_factum_trajectory_credit_kind,
            "post_factum_trajectory_credit_polarity": post_factum_trajectory_credit_polarity,
            "post_factum_trajectory_credit_step": None if post_factum_trajectory_credit_step is None else int(post_factum_trajectory_credit_step),
            "post_factum_trajectory_credit_reason": post_factum_trajectory_credit_reason,
            "game_id": game_id,
            "level_id": level_id,
            "sampler_name": sampler_name,
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "memory_fitness_base": None if memory_fitness_base is None else float(memory_fitness_base),
            "memory_fitness": None if memory_fitness is None else float(memory_fitness),
            "memory_replay_priority_base": None if memory_replay_priority_base is None else float(memory_replay_priority_base),
            "retention_score_base": None if retention_score_base is None else float(retention_score_base),
            "retention_score": None if retention_score is None else float(retention_score),
            "memory_state": memory_state,
            "stored_epoch": None if stored_epoch is None else int(stored_epoch),
            "last_replayed_epoch": None if last_replayed_epoch is None else int(last_replayed_epoch),
            "last_promoted_epoch": None if last_promoted_epoch is None else int(last_promoted_epoch),
            "forgetting_score": None if forgetting_score is None else float(forgetting_score),
            "compressed_into_id": compressed_into_id,
            "superseded_by_id": superseded_by_id,
            "forgetting_reason": forgetting_reason,
        }
        columns = list(row.keys())
        values = tuple(row[column] for column in columns)
        self.connection.execute(
            f"""
            INSERT INTO prediction_results (
                {", ".join(columns)}
            )
            VALUES ({", ".join("?" for _ in values)})
            """,
            values,
        )
        if self.auto_commit:
            self.connection.commit()
        return prediction_error

    def all_contingencies(self) -> list[Contingency]:
        rows = self.connection.execute(
            """
            SELECT id, context_level, context_signature, action, transformation_family, support_count, confidence
            FROM contingencies
            ORDER BY context_level, id
            """
        ).fetchall()
        return [
            Contingency(
                id=int(row[0]),
                context_level=int(row[1]),
                context_signature=tuple(json.loads(row[2])),
                action=int(row[3]),
                transformation_family=int(row[4]),
                support_count=int(row[5]),
                confidence=float(row[6]),
            )
            for row in rows
        ]
