from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = "v6.2"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is None:
        return set()
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()}


def _ensure_column(connection: sqlite3.Connection, table: str, name: str, declaration: str) -> None:
    if name not in _columns(connection, table):
        connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}')


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    text = str(value).strip().lower().lstrip("v")
    try:
        return tuple(int(part) for part in text.split("."))
    except ValueError:
        return ()


def _schema_version_before(connection: sqlite3.Connection) -> str | None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_versions'"
    ).fetchone() is None:
        return None
    row = connection.execute(
        "SELECT value FROM memory_versions WHERE key='memory_substrate_schema'"
    ).fetchone()
    return None if row is None or row[0] is None else str(row[0])


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    existing_schema_version = _schema_version_before(connection)
    # This migration must be safe on a brand-new compact-memory database.
    # MemorySubstrate normally creates memory_versions first, but compact-memory
    # initialization may invoke migrations before MemorySubstrate is constructed.
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_versions (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS memory_development_state (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_step INTEGER,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_promotion_evidence_v62 (
            node_id TEXT PRIMARY KEY,
            memory_level TEXT NOT NULL,
            node_type TEXT NOT NULL,
            evidence_dimensions_json TEXT NOT NULL,
            evidence_dimension_count INTEGER NOT NULL,
            required_dimension_count INTEGER NOT NULL,
            validation_status TEXT NOT NULL,
            validation_reason TEXT,
            updated_step INTEGER,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_promotion_evidence_v62_status
        ON memory_promotion_evidence_v62(validation_status, memory_level, node_type);
        """
    )
    if _columns(connection, "memory_scores"):
        for name, declaration in {
            "hierarchical_score": "REAL",
            "developmental_stage": "TEXT",
            "source_score_count": "INTEGER",
            "score_version": "TEXT",
        }.items():
            _ensure_column(connection, "memory_scores", name, declaration)
    if _columns(connection, "memory_promotions"):
        for name, declaration in {
            "evidence_dimension_count": "INTEGER",
            "validation_status": "TEXT",
            "validation_reason": "TEXT",
            "policy_version": "TEXT",
        }.items():
            _ensure_column(connection, "memory_promotions", name, declaration)
    effective_version = (
        existing_schema_version
        if _version_tuple(existing_schema_version) > _version_tuple(SCHEMA_VERSION)
        else SCHEMA_VERSION
    )
    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES ('memory_substrate_schema', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (effective_version,),
    )
    connection.commit()
    return {"schema_version": effective_version, "migration_applied": True}


def migrate_memory_dir(memory_dir: str | Path) -> dict[str, object]:
    root = Path(memory_dir)
    database = root / "current_state.sqlite"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database, timeout=60.0) as connection:
        result = migrate_connection(connection)
    manifest = root / "v62_schema_migration.json"
    manifest.write_text(
        json.dumps({**result, "database": str(database), "migrated_at": time.time()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
