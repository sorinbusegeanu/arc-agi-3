from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from v6.memory.migrations.v621 import migrate_connection as migrate_v621_connection
from v6.memory.v63_transfer import install_v63_transfer_policy

SCHEMA_VERSION = "v6.3"


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


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    install_v63_transfer_policy()
    connection.execute(
        "CREATE TABLE IF NOT EXISTS memory_versions (key TEXT PRIMARY KEY, value TEXT)"
    )
    migrate_v621_connection(connection)

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_evidence_revisions_v63 (
            revision_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            evidence_kind TEXT NOT NULL,
            prospective_value REAL,
            realized_value REAL,
            evidence_status TEXT NOT NULL,
            evidence_source TEXT NOT NULL,
            global_step INTEGER,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS abstraction_frontier_audit_v63 (
            audit_id TEXT PRIMARY KEY,
            frontier_kind TEXT NOT NULL,
            candidates_seen INTEGER NOT NULL,
            candidates_retained INTEGER NOT NULL,
            comparisons_attempted INTEGER NOT NULL,
            comparison_budget INTEGER NOT NULL,
            global_step INTEGER,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_evidence_revisions_v63_node
        ON memory_evidence_revisions_v63(node_id, evidence_kind, created_at);

        CREATE INDEX IF NOT EXISTS idx_abstraction_frontier_v63_kind
        ON abstraction_frontier_audit_v63(frontier_kind, global_step);
        """
    )

    for name, declaration in {
        "transfer_prior": "REAL",
        "transfer_empirical_rate": "REAL",
        "transfer_evidence_status": "TEXT",
        "memory_fitness": "REAL",
        "recurrence_score": "REAL",
        "efficiency_score": "REAL",
        "score_components_json": "TEXT",
        "prospective_learning_value": "REAL",
        "realized_learning_value": "REAL",
        "prospective_explanatory_potential": "REAL",
        "realized_explanatory_reach": "REAL",
        "score_policy_version": "TEXT",
    }.items():
        _ensure_column(connection, "memory_scores", name, declaration)

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
            "memory_evidence_revisions_v63",
            "abstraction_frontier_audit_v63",
        ],
    }


def migrate_memory_dir(memory_dir: str | Path) -> dict[str, object]:
    root = Path(memory_dir)
    database = root / "current_state.sqlite"
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database, timeout=60.0) as connection:
        result = migrate_connection(connection)
    manifest = root / "v63_schema_migration.json"
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
