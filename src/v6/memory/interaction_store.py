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
                observation_before BLOB NOT NULL,
                action INTEGER NOT NULL,
                observation_after BLOB NOT NULL,
                delta_id INTEGER NOT NULL
            )
            """
        )
        self.connection.commit()

    def next_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM interactions").fetchone()
        return int(row[0])

    def add(self, interaction: Interaction) -> None:
        self.connection.execute(
            """
            INSERT INTO interactions (
                id,
                timestamp,
                observation_before,
                action,
                observation_after,
                delta_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(interaction.id),
                int(interaction.timestamp),
                encode_array(interaction.observation_before),
                int(interaction.action),
                encode_array(interaction.observation_after),
                int(interaction.delta_id),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def get(self, interaction_id: int) -> Interaction | None:
        row = self.connection.execute(
            """
            SELECT id, timestamp, observation_before, action, observation_after, delta_id
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
            observation_before=decode_array(row[2]),
            action=int(row[3]),
            observation_after=decode_array(row[4]),
            delta_id=int(row[5]),
        )

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM interactions").fetchone()
        return int(row[0])
