from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


SCHEMA_VERSION = "v6.1"
MIGRATION_MARKER_KEY = "v61_additive_schema"
MIGRATION_MARKER_VERSION = "v1"


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
    columns = _columns(connection, table)
    if not columns or name in columns:
        return
    connection.execute(
        f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}'
    )


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    text = str(value).strip().lower().lstrip("v")
    try:
        return tuple(int(part) for part in text.split("."))
    except ValueError:
        return ()


def _memory_version(connection: sqlite3.Connection, key: str) -> str | None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_versions'"
    ).fetchone() is None:
        return None
    row = connection.execute(
        "SELECT value FROM memory_versions WHERE key=?",
        (str(key),),
    ).fetchone()
    return None if row is None or row[0] is None else str(row[0])


def _install_current_runtime_performance(schema_version: str | None) -> None:
    if _version_tuple(schema_version) < (6, 3):
        return
    from v6.memory.v63_performance import install_v63_validation_performance

    install_v63_validation_performance()


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS memory_versions (key TEXT PRIMARY KEY, value TEXT)"
    )
    existing_schema_version = _memory_version(connection, "memory_substrate_schema")
    _install_current_runtime_performance(existing_schema_version)
    marker = _memory_version(connection, MIGRATION_MARKER_KEY)
    if marker == MIGRATION_MARKER_VERSION:
        return {
            "schema_version": existing_schema_version or SCHEMA_VERSION,
            "migration_applied": False,
        }

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_lifecycle_events (
            lifecycle_event_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            reason TEXT,
            previous_status TEXT,
            new_status TEXT,
            score REAL,
            epoch INTEGER,
            global_step INTEGER,
            payload_json TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS context_split_lineage (
            split_id TEXT PRIMARY KEY,
            parent_context_key TEXT NOT NULL,
            child_context_key TEXT NOT NULL,
            action_key TEXT,
            contradiction_key TEXT,
            differentiating_features_json TEXT,
            prediction_lift_before REAL,
            prediction_lift_after REAL,
            validation_status TEXT NOT NULL DEFAULT 'candidate',
            epoch INTEGER,
            global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS strategy_reuse_events (
            reuse_event_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            game TEXT,
            level_key TEXT,
            context_key TEXT,
            success INTEGER,
            cost REAL,
            epoch INTEGER,
            global_step INTEGER,
            payload_json TEXT
        );
        """
    )

    node_columns = {
        "schema_version": "TEXT NOT NULL DEFAULT 'v6.1'",
        "evidence_version": "TEXT NOT NULL DEFAULT 'v1'",
        "created_epoch": "INTEGER",
        "updated_epoch": "INTEGER",
        "status": "TEXT NOT NULL DEFAULT 'active'",
    }
    edge_columns = {
        "edge_status": "TEXT NOT NULL DEFAULT 'accepted'",
        "edge_confidence": "REAL",
        "edge_source": "TEXT",
        "specificity_score": "REAL",
        "last_validated_epoch": "INTEGER",
    }
    promotion_columns = {
        "source_memory_ids_json": "TEXT",
        "compression_gain": "REAL",
        "prediction_lift": "REAL",
        "transfer_score": "REAL",
        "explanatory_reach": "REAL",
        "epoch": "INTEGER",
        "global_step": "INTEGER",
    }

    for name, declaration in node_columns.items():
        _ensure_column(connection, "memory_nodes", name, declaration)
    for name, declaration in edge_columns.items():
        _ensure_column(connection, "memory_edges", name, declaration)
    for name, declaration in promotion_columns.items():
        _ensure_column(connection, "memory_promotions", name, declaration)

    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_nodes_identity_v61
        ON memory_nodes(memory_level, node_type, canonical_key, status);
        CREATE INDEX IF NOT EXISTS idx_memory_edges_lifecycle_v61
        ON memory_edges(edge_status, edge_confidence, edge_source);
        CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_memory_v61
        ON memory_lifecycle_events(memory_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_context_split_parent_v61
        ON context_split_lineage(parent_context_key, action_key);
        CREATE INDEX IF NOT EXISTS idx_strategy_reuse_strategy_v61
        ON strategy_reuse_events(strategy_id, success);
        """
    )

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
    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (MIGRATION_MARKER_KEY, MIGRATION_MARKER_VERSION),
    )

    if _columns(connection, "memory_nodes"):
        connection.execute(
            """
            UPDATE memory_nodes
            SET
                schema_version=COALESCE(NULLIF(schema_version, ''), 'v6.1'),
                evidence_version=COALESCE(NULLIF(evidence_version, ''), 'v1'),
                status=COALESCE(NULLIF(status, ''), 'active')
            """
        )
    connection.commit()
    return {
        "schema_version": effective_version,
        "memory_node_columns": sorted(node_columns),
        "memory_edge_columns": sorted(edge_columns),
        "memory_promotion_columns": sorted(promotion_columns),
        "migration_applied": True,
    }


def _ensure_latest_performance_indexes(
    connection: sqlite3.Connection,
    schema_version: str | None,
) -> list[str]:
    if _version_tuple(schema_version) < (6, 3):
        return []
    from v6.memory.migrations.v63 import (
        PERFORMANCE_SCHEMA_KEY,
        PERFORMANCE_SCHEMA_VERSION,
        _ensure_performance_indexes,
    )

    if _memory_version(connection, PERFORMANCE_SCHEMA_KEY) == PERFORMANCE_SCHEMA_VERSION:
        return []
    indexes = _ensure_performance_indexes(connection)
    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (PERFORMANCE_SCHEMA_KEY, PERFORMANCE_SCHEMA_VERSION),
    )
    connection.commit()
    return indexes


def migrate_memory_dir(memory_dir: str | Path) -> dict[str, object]:
    root = Path(memory_dir)
    database = root / "current_state.sqlite"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database, timeout=60.0) as connection:
        result = migrate_connection(connection)
        performance_indexes = _ensure_latest_performance_indexes(
            connection,
            str(result.get("schema_version") or ""),
        )
        if performance_indexes:
            result["performance_indexes"] = performance_indexes
    manifest = root / "v61_schema_migration.json"
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
