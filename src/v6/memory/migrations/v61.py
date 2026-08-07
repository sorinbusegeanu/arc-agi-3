from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "v6.1"


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
    if name in _columns(connection, table):
        return
    connection.execute(
        f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}'
    )


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_versions (
            key TEXT PRIMARY KEY,
            value TEXT
        );
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
        _ensure_column(
            connection, "memory_nodes", name, declaration
        )
    for name, declaration in edge_columns.items():
        _ensure_column(
            connection, "memory_edges", name, declaration
        )
    for name, declaration in promotion_columns.items():
        _ensure_column(
            connection, "memory_promotions", name, declaration
        )

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

    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES ('memory_substrate_schema', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (SCHEMA_VERSION,),
    )

    # Existing nodes are valid v6.1 identities after additive migration.
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
        "schema_version": SCHEMA_VERSION,
        "memory_node_columns": sorted(node_columns),
        "memory_edge_columns": sorted(edge_columns),
        "memory_promotion_columns": sorted(promotion_columns),
        "migration_applied": True,
    }


def migrate_memory_dir(memory_dir: str | Path) -> dict[str, object]:
    root = Path(memory_dir)
    database = root / "current_state.sqlite"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database, timeout=60.0) as connection:
        result = migrate_connection(connection)
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
