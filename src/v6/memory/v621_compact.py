from __future__ import annotations

import sqlite3
from pathlib import Path

from v6.memory.migrations.v63 import migrate_connection as migrate_v63_connection


V621_NODE_COLUMNS = (
    "schema_version",
    "evidence_version",
    "created_epoch",
    "updated_epoch",
    "status",
)

V621_EDGE_COLUMNS = (
    "edge_status",
    "edge_confidence",
    "edge_source",
    "specificity_score",
    "last_validated_epoch",
)

# The compact-memory entry points retain their historic v621 names because they
# are used widely by the fold pipeline. On the v6.3 branch they preserve both
# v6.2.1 and v6.3 score extensions.
V621_SCORE_COLUMNS = (
    "hierarchical_score",
    "developmental_stage",
    "source_score_count",
    "score_version",
    "lifecycle_version",
    "transfer_prior",
    "transfer_empirical_rate",
    "transfer_evidence_status",
    "memory_fitness",
    "recurrence_score",
    "efficiency_score",
    "score_components_json",
    "prospective_learning_value",
    "realized_learning_value",
    "prospective_explanatory_potential",
    "realized_explanatory_reach",
    "score_policy_version",
)

V621_PROMOTION_COLUMNS = (
    "source_memory_ids_json",
    "compression_gain",
    "prediction_lift",
    "transfer_score",
    "explanatory_reach",
    "epoch",
    "global_step",
    "evidence_dimension_count",
    "validation_status",
    "validation_reason",
    "policy_version",
)

V621_AUX_TABLES = (
    "memory_lifecycle_events",
    "context_split_lineage",
    "strategy_reuse_events",
    "memory_development_state",
    "memory_promotion_evidence_v62",
    "concept_transfer_attempts_v621",
    "world_model_relations_v621",
    "memory_level_lifecycle_v621",
    "memory_runtime_audit_v621",
    "memory_evidence_revisions_v63",
    "abstraction_frontier_audit_v63",
)


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if exists is None:
        return []
    return [
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    ]


def ensure_v621_state_connection(
    connection: sqlite3.Connection,
) -> None:
    migrate_v63_connection(connection)


def ensure_v621_state_path(path: str | Path) -> None:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=60.0) as connection:
        migrate_v63_connection(connection)


def _copy_extension_columns(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    key_column: str,
    extension_columns: tuple[str, ...],
) -> int:
    source_columns = set(_columns(source, table))
    target_columns = set(_columns(target, table))
    selected = [
        column
        for column in extension_columns
        if column in source_columns and column in target_columns
    ]
    if (
        key_column not in source_columns
        or key_column not in target_columns
        or not selected
    ):
        return 0

    query_columns = [key_column, *selected]
    rows = source.execute(
        f'SELECT {", ".join(query_columns)} '
        f'FROM "{table}" ORDER BY "{key_column}"'
    ).fetchall()
    updated = 0
    assignments = ", ".join(
        f'"{column}"=?' for column in selected
    )
    for row in rows:
        key = row[0]
        values = list(row[1:])
        if target.execute(
            f'SELECT 1 FROM "{table}" WHERE "{key_column}"=?',
            (key,),
        ).fetchone() is None:
            continue
        target.execute(
            f'UPDATE "{table}" SET {assignments} '
            f'WHERE "{key_column}"=?',
            (*values, key),
        )
        updated += 1
    return updated


def _copy_composite_extension_columns(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    key_columns: tuple[str, ...],
    extension_columns: tuple[str, ...],
) -> int:
    source_columns = set(_columns(source, table))
    target_columns = set(_columns(target, table))
    selected = [
        column
        for column in extension_columns
        if column in source_columns and column in target_columns
    ]
    if (
        not set(key_columns).issubset(source_columns)
        or not set(key_columns).issubset(target_columns)
        or not selected
    ):
        return 0

    query_columns = [*key_columns, *selected]
    rows = source.execute(
        f'SELECT {", ".join(query_columns)} FROM "{table}"'
    ).fetchall()
    assignments = ", ".join(
        f'"{column}"=?' for column in selected
    )
    where = " AND ".join(
        f'"{column}"=?' for column in key_columns
    )
    updated = 0
    for row in rows:
        key_values = tuple(row[: len(key_columns)])
        values = tuple(row[len(key_columns) :])
        if target.execute(
            f'SELECT 1 FROM "{table}" WHERE {where}',
            key_values,
        ).fetchone() is None:
            continue
        target.execute(
            f'UPDATE "{table}" SET {assignments} WHERE {where}',
            (*values, *key_values),
        )
        updated += 1
    return updated


def _copy_table_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
) -> int:
    source_columns = _columns(source, table)
    target_columns = set(_columns(target, table))
    common = [
        column
        for column in source_columns
        if column in target_columns
    ]
    if not common:
        return 0

    quoted = ", ".join(f'"{column}"' for column in common)
    placeholders = ", ".join("?" for _ in common)
    rows = source.execute(
        f'SELECT {quoted} FROM "{table}"'
    ).fetchall()
    for row in rows:
        target.execute(
            f'INSERT OR REPLACE INTO "{table}" ({quoted}) '
            f'VALUES ({placeholders})',
            tuple(row),
        )
    return len(rows)


def merge_v621_state_connections(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
) -> dict[str, int]:
    migrate_v63_connection(target)

    summary = {
        "memory_nodes_extended": _copy_extension_columns(
            source,
            target,
            "memory_nodes",
            "node_id",
            V621_NODE_COLUMNS,
        ),
        "memory_edges_extended": _copy_composite_extension_columns(
            source,
            target,
            "memory_edges",
            ("source_node_id", "target_node_id", "edge_type"),
            V621_EDGE_COLUMNS,
        ),
        "memory_scores_extended": _copy_extension_columns(
            source,
            target,
            "memory_scores",
            "node_id",
            V621_SCORE_COLUMNS,
        ),
        "memory_promotions_extended": _copy_extension_columns(
            source,
            target,
            "memory_promotions",
            "promotion_id",
            V621_PROMOTION_COLUMNS,
        ),
    }

    for table in V621_AUX_TABLES:
        summary[table] = _copy_table_rows(
            source,
            target,
            table,
        )

    target.commit()
    return summary


def merge_v621_state_file_into_connection(
    source_path: str | Path,
    target: sqlite3.Connection,
) -> dict[str, int]:
    source_database = Path(source_path)
    if not source_database.exists():
        return {}
    with sqlite3.connect(
        f"file:{source_database.resolve()}?mode=ro",
        uri=True,
        timeout=10.0,
    ) as source:
        return merge_v621_state_connections(source, target)
