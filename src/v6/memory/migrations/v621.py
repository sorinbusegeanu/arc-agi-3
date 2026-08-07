from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from v6.memory.migrations.v62 import migrate_connection as migrate_v62_connection

SCHEMA_VERSION = "v6.2.1"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if exists is None:
        return set()
    return {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    name: str,
    declaration: str,
) -> None:
    if not _columns(connection, table):
        return
    if name in _columns(connection, table):
        return
    connection.execute(
        f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}'
    )


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    migrate_v62_connection(connection)

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS concept_transfer_attempts_v621 (
            attempt_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            game TEXT,
            context_key TEXT,
            action INTEGER,
            predicted_family TEXT,
            actual_family TEXT,
            success INTEGER NOT NULL,
            evidence_source TEXT NOT NULL,
            global_step INTEGER,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS world_model_relations_v621 (
            relation_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            source_concept_id TEXT NOT NULL,
            target_concept_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            confidence REAL NOT NULL,
            evidence_json TEXT,
            updated_step INTEGER,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_level_lifecycle_v621 (
            memory_id TEXT PRIMARY KEY,
            memory_level TEXT NOT NULL,
            memory_state TEXT NOT NULL,
            replay_priority REAL NOT NULL,
            retention_score REAL NOT NULL,
            forgetting_score REAL NOT NULL,
            last_replayed_step INTEGER,
            last_transition_step INTEGER,
            reason TEXT,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_runtime_audit_v621 (
            audit_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            owner_id TEXT,
            payload_json TEXT,
            global_step INTEGER,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_concept_transfer_v621_concept
        ON concept_transfer_attempts_v621(concept_id, success);

        CREATE INDEX IF NOT EXISTS idx_concept_transfer_v621_game
        ON concept_transfer_attempts_v621(game, success);

        CREATE INDEX IF NOT EXISTS idx_world_model_relations_v621_model
        ON world_model_relations_v621(model_id, relation_type, confidence);

        CREATE INDEX IF NOT EXISTS idx_memory_level_lifecycle_v621_state
        ON memory_level_lifecycle_v621(memory_level, memory_state, replay_priority);

        CREATE INDEX IF NOT EXISTS idx_memory_runtime_audit_v621_event
        ON memory_runtime_audit_v621(event_type, global_step);
        """
    )

    _ensure_column(
        connection,
        "memory_scores",
        "lifecycle_version",
        "TEXT",
    )
    _ensure_column(
        connection,
        "memory_promotion_evidence_v62",
        "policy_version",
        "TEXT",
    )
    _ensure_column(
        connection,
        "memory_promotion_evidence_v62",
        "validation_source",
        "TEXT",
    )

    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES ('memory_substrate_schema', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (SCHEMA_VERSION,),
    )
    connection.commit()

    return {
        "schema_version": SCHEMA_VERSION,
        "migration_applied": True,
        "tables": [
            "concept_transfer_attempts_v621",
            "world_model_relations_v621",
            "memory_level_lifecycle_v621",
            "memory_runtime_audit_v621",
        ],
    }


def migrate_memory_dir(memory_dir: str | Path) -> dict[str, object]:
    root = Path(memory_dir)
    database = root / "current_state.sqlite"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database, timeout=60.0) as connection:
        result = migrate_connection(connection)
    manifest = root / "v621_schema_migration.json"
    manifest.write_text(
        json.dumps(
            {
                **result,
                "database": str(database),
                "migrated_at": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
