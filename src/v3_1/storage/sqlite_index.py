from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteIndex:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure()

    def _ensure(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("create table if not exists manifests (kind text, name text, location text)")
            conn.commit()

    def insert_manifest(self, *, kind: str, name: str, location: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("insert into manifests(kind, name, location) values (?, ?, ?)", (kind, name, location))
            conn.commit()

