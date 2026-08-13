from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS v7_schema_meta (
    schema_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_instances (
    memory_instance_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY,
    parent_generation_id INTEGER,
    first_global_step INTEGER,
    last_global_step INTEGER,
    committed INTEGER NOT NULL DEFAULT 0,
    committed_at TEXT,
    FOREIGN KEY(parent_generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS generation_batches (
    generation_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    mutation_count INTEGER NOT NULL,
    PRIMARY KEY(generation_id, batch_id),
    FOREIGN KEY(generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS memory_nodes (
    memory_id INTEGER PRIMARY KEY,
    level_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    created_generation INTEGER NOT NULL,
    updated_generation INTEGER NOT NULL,
    status_flags INTEGER NOT NULL DEFAULT 0,
    support_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_edges (
    source_id INTEGER NOT NULL,
    relation_type INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    support_count INTEGER NOT NULL DEFAULT 1,
    updated_generation INTEGER NOT NULL,
    PRIMARY KEY(source_id, relation_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_edges_target
ON memory_edges(target_id, relation_type, source_id);

CREATE TABLE IF NOT EXISTS memory_scores (
    memory_id INTEGER PRIMARY KEY,
    significance REAL NOT NULL DEFAULT 0,
    prediction_error REAL NOT NULL DEFAULT 0,
    learning_value REAL NOT NULL DEFAULT 0,
    transfer_prior REAL NOT NULL DEFAULT 0,
    explanatory_potential REAL NOT NULL DEFAULT 0,
    future_option_delta REAL NOT NULL DEFAULT 0,
    updated_generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER,
    evidence_type INTEGER NOT NULL,
    source_game TEXT,
    source_context TEXT,
    source_global_step INTEGER,
    payload_json TEXT NOT NULL,
    generation_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_memory_generation
ON evidence_records(memory_id, generation_id);
"""


def ensure_v7_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(DDL)
    row = connection.execute("SELECT schema_version FROM v7_schema_meta LIMIT 1").fetchone()
    if row is None:
        connection.execute("INSERT INTO v7_schema_meta(schema_version) VALUES (?)", (SCHEMA_VERSION,))
    elif int(row[0]) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported v7 schema version {row[0]}; expected {SCHEMA_VERSION}")
    connection.commit()
