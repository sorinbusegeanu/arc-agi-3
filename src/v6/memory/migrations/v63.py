from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from v6.memory.migrations.v621 import migrate_connection as migrate_v621_connection
from v6.memory.v63_performance import install_v63_sampling_performance
from v6.memory.v63_transfer import install_v63_transfer_policy

SCHEMA_VERSION = "v6.3"
PERFORMANCE_SCHEMA_KEY = "v63_performance_schema"
PERFORMANCE_SCHEMA_VERSION = "v1"


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


def _ensure_index(
    connection: sqlite3.Connection,
    table: str,
    index_name: str,
    columns: tuple[str, ...],
) -> bool:
    available = _columns(connection, table)
    if not available or not set(columns).issubset(available):
        return False
    quoted = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({quoted})'
    )
    return True


def _ensure_performance_indexes(connection: sqlite3.Connection) -> list[str]:
    specs = (
        ("concept_transfer_attempts_v621", "idx_v63_concept_transfer_step_game_created", ("global_step", "game", "created_at")),
        ("concept_transfer_attempts_v621", "idx_v63_concept_transfer_evidence", ("concept_id", "evidence_source", "success", "game")),
        ("role_transfer_attempts", "idx_v63_role_transfer_role_step_success", ("role_signature", "last_seen_global_step", "reuse_success")),
        ("role_transfer_attempts", "idx_v63_role_transfer_source_success", ("source_role_signature", "reuse_success")),
        ("role_transfer_attempts", "idx_v63_role_transfer_attempt", ("attempt_id",)),
        ("role_links", "idx_v63_role_links_signature_type", ("role_signature", "linked_type")),
        ("concept_links", "idx_v63_concept_links_signature_type", ("concept_signature", "linked_type")),
        ("future_option_events", "idx_v63_future_option_family", ("source_family_id",)),
        ("future_option_events", "idx_v63_future_option_role_step", ("source_role_id", "last_seen_global_step")),
        ("future_option_motifs", "idx_v63_future_option_motif_step", ("last_seen_global_step",)),
        ("prediction_results", "idx_v63_prediction_results_global_step", ("global_step",)),
        ("world_model_prediction_events", "idx_v63_world_model_prediction_component", ("component_signature", "observed_global_step", "prediction_global_step")),
        ("family_members", "idx_v63_family_members_family", ("family_signature",)),
        ("family_members", "idx_v63_family_members_contingency", ("contingency_key", "family_signature")),
        ("transformation_families", "idx_v63_transformation_family_signature", ("canonical_signature",)),
        ("stable_contingencies", "idx_v63_stable_contingency_scope", ("canonical_key", "game", "sampler")),
        ("concept_promotion_validation_diagnostics", "idx_v63_concept_validation_signature", ("concept_signature",)),
    )
    created: list[str] = []
    for table, name, columns in specs:
        if _ensure_index(connection, table, name, columns):
            created.append(name)
    return created


def migrate_connection(connection: sqlite3.Connection) -> dict[str, object]:
    install_v63_transfer_policy()
    install_v63_sampling_performance()
    connection.execute(
        "CREATE TABLE IF NOT EXISTS memory_versions (key TEXT PRIMARY KEY, value TEXT)"
    )

    existing_schema = _memory_version(connection, "memory_substrate_schema")
    performance_version = _memory_version(connection, PERFORMANCE_SCHEMA_KEY)
    if (
        _version_tuple(existing_schema) >= _version_tuple(SCHEMA_VERSION)
        and performance_version == PERFORMANCE_SCHEMA_VERSION
    ):
        return {
            "schema_version": existing_schema or SCHEMA_VERSION,
            "migration_applied": False,
            "performance_schema_version": PERFORMANCE_SCHEMA_VERSION,
        }

    if _version_tuple(existing_schema) < _version_tuple(SCHEMA_VERSION):
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

    performance_indexes = _ensure_performance_indexes(connection)

    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES ('memory_substrate_schema', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (SCHEMA_VERSION,),
    )
    connection.execute(
        """
        INSERT INTO memory_versions(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (PERFORMANCE_SCHEMA_KEY, PERFORMANCE_SCHEMA_VERSION),
    )
    connection.commit()
    return {
        "schema_version": SCHEMA_VERSION,
        "migration_applied": True,
        "performance_schema_version": PERFORMANCE_SCHEMA_VERSION,
        "performance_indexes": performance_indexes,
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
