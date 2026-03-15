from __future__ import annotations

import sqlite3
from pathlib import Path
import json


class SQLiteIndex:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure()

    def _ensure(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("create table if not exists manifests (kind text, name text, location text, metadata_json text)")
            conn.commit()

    def insert_manifest(self, *, kind: str, name: str, location: str, metadata: dict | None = None) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "insert into manifests(kind, name, location, metadata_json) values (?, ?, ?, ?)",
                (kind, name, location, json.dumps(metadata or {}, sort_keys=True)),
            )
            conn.commit()
