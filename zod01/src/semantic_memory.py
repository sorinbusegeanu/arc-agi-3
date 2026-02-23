from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class SemanticMemory:
    def __init__(self, path: str = "zod01/data/semantic_memory.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mechanics (
                signature TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def put(self, signature: str, payload: dict[str, float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO mechanics(signature, payload) VALUES(?, ?)",
            (signature, json.dumps(payload, sort_keys=True)),
        )
        self._conn.commit()

    def get(self, signature: str) -> dict[str, float] | None:
        row = self._conn.execute(
            "SELECT payload FROM mechanics WHERE signature = ?", (signature,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
