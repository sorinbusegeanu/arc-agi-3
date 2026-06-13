from __future__ import annotations

import json
import sqlite3

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
                confidence REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                context_level INTEGER,
                context_signature TEXT NOT NULL,
                action INTEGER NOT NULL,
                predicted_family INTEGER,
                actual_family INTEGER,
                prediction_error INTEGER,
                episode_id INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._ensure_column("contingencies", "context_level", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("prediction_results", "context_level", "INTEGER")
        self._ensure_column("prediction_results", "episode_id", "INTEGER NOT NULL DEFAULT 0")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(row[1]) == column for row in rows):
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def upsert_contingency(self, contingency: Contingency) -> None:
        self.connection.execute(
            """
            INSERT INTO contingencies (
                id,
                context_level,
                context_signature,
                action,
                transformation_family,
                support_count,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                context_level = excluded.context_level,
                context_signature = excluded.context_signature,
                action = excluded.action,
                transformation_family = excluded.transformation_family,
                support_count = excluded.support_count,
                confidence = excluded.confidence
            """,
            (
                int(contingency.id),
                int(contingency.context_level),
                json.dumps(contingency.context_signature),
                int(contingency.action),
                int(contingency.transformation_family),
                int(contingency.support_count),
                float(contingency.confidence),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def add_prediction_result(
        self,
        *,
        interaction_id: int,
        context_level: int | None = None,
        context_signature: tuple,
        action: int,
        predicted_family: int | None,
        actual_family: int | None,
        episode_id: int = 0,
    ) -> int | None:
        prediction_error: int | None
        if predicted_family is None or actual_family is None:
            prediction_error = None
        else:
            prediction_error = 0 if int(predicted_family) == int(actual_family) else 1
        self.connection.execute(
            """
            INSERT INTO prediction_results (
                interaction_id,
                context_level,
                context_signature,
                action,
                predicted_family,
                actual_family,
                prediction_error,
                episode_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(interaction_id),
                None if context_level is None else int(context_level),
                json.dumps(context_signature),
                int(action),
                None if predicted_family is None else int(predicted_family),
                None if actual_family is None else int(actual_family),
                prediction_error,
                int(episode_id),
            ),
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
