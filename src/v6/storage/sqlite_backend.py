from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.storage.backend import StorageBackend


class SQLiteStorageBackend(StorageBackend):
    def __init__(self, path: str | Path, *, batch_size: int = 1000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.batch_size = int(batch_size)
        self.buffers: dict[str, list[dict]] = {}

    def write_interactions(self, records: list[dict]) -> None:
        self._write("interactions", records)

    def write_deltas(self, records: list[dict]) -> None:
        self._write("deltas", records)

    def write_transformation_families(self, records: list[dict]) -> None:
        self._write("transformation_families", records)

    def write_contingencies(self, records: list[dict]) -> None:
        self._write("contingencies", records)

    def write_future_effects(self, records: list[dict]) -> None:
        self._write("future_effects", records)

    def write_role_candidates(self, records: list[dict]) -> None:
        self._write("role_candidates", records)

    def write_run_summary(self, record: dict) -> None:
        self._write("run_summary", [record])

    def finalize(self) -> None:
        for table_name in list(self.buffers):
            self._flush(table_name)
        self.connection.commit()
        self.connection.close()

    def _write(self, table_name: str, records: list[dict]) -> None:
        if not records:
            return
        self.buffers.setdefault(table_name, []).extend(_normalize_record(record) for record in records)
        if len(self.buffers[table_name]) >= self.batch_size:
            self._flush(table_name)

    def _flush(self, table_name: str) -> None:
        records = self.buffers.get(table_name, [])
        if not records:
            return
        fields = sorted({key for record in records for key in record})
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(f'{field} {_sqlite_type(records, field)}' for field in fields)})"
        )
        placeholders = ", ".join("?" for _ in fields)
        self.connection.executemany(
            f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({placeholders})",
            [tuple(record.get(field) for field in fields) for record in records],
        )
        self.connection.commit()
        self.buffers[table_name] = []


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in record.items():
        if isinstance(value, bool):
            output[key] = int(value)
        elif isinstance(value, (dict, list, tuple)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def _sqlite_type(records: list[dict], field: str) -> str:
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        return "TEXT"
    return "TEXT"
