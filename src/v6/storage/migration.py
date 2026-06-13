from __future__ import annotations

import sqlite3
from pathlib import Path

from v6.storage.parquet_backend import ParquetStorageBackend


MIGRATION_TABLES = (
    "interactions",
    "deltas",
    "transformation_families",
    "contingencies",
    "future_effects",
    "role_candidates",
)


def migrate_sqlite_to_parquet(
    *,
    sqlite_path: str | Path,
    parquet_root: str | Path,
    game: str,
    sampler: str,
    seed: int,
    steps: int,
    batch_size: int = 1000,
    compression: str = "zstd",
    run_summary: dict | None = None,
) -> Path:
    backend = ParquetStorageBackend(
        root=parquet_root,
        game=game,
        sampler=sampler,
        seed=int(seed),
        steps=int(steps),
        batch_size=batch_size,
        compression=compression,
    )
    with sqlite3.connect(sqlite_path) as connection:
        for table_name in MIGRATION_TABLES:
            if not _table_exists(connection, table_name):
                continue
            records = _read_table(connection, table_name)
            _write_backend_table(backend, table_name, records)
        summary = {
            "game": game,
            "sampler": sampler,
            "seed": int(seed),
            "steps": int(steps),
            "sqlite_path": str(sqlite_path),
        }
        if run_summary:
            summary.update(run_summary)
        backend.write_run_summary(summary)
    backend.finalize()
    return backend.base_path


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return row is not None


def _read_table(connection: sqlite3.Connection, table_name: str) -> list[dict]:
    cursor = connection.execute(f"SELECT * FROM {table_name}")
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _write_backend_table(backend: ParquetStorageBackend, table_name: str, records: list[dict]) -> None:
    if table_name == "interactions":
        backend.write_interactions(records)
    elif table_name == "deltas":
        backend.write_deltas(records)
    elif table_name == "transformation_families":
        backend.write_transformation_families(records)
    elif table_name == "contingencies":
        backend.write_contingencies(records)
    elif table_name == "future_effects":
        backend.write_future_effects(records)
    elif table_name == "role_candidates":
        backend.write_role_candidates(records)
