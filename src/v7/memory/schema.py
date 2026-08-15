from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

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
    support_count INTEGER NOT NULL DEFAULT 0,
    cognitive_state INTEGER NOT NULL DEFAULT 1,
    validation_state INTEGER NOT NULL DEFAULT 4,
    gate_id INTEGER NOT NULL DEFAULT 0
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

CREATE INDEX IF NOT EXISTS idx_evidence_type_id
ON evidence_records(evidence_type, evidence_id);

CREATE TABLE IF NOT EXISTS provenance_records (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    parent_memory_id INTEGER,
    relation_type INTEGER NOT NULL DEFAULT 0,
    source_game TEXT,
    source_context TEXT,
    source_global_step INTEGER,
    generation_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provenance_memory_generation
ON provenance_records(memory_id, generation_id, parent_memory_id);

CREATE TABLE IF NOT EXISTS transfer_trials (
    transfer_trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    source_game TEXT NOT NULL,
    target_game TEXT NOT NULL,
    success INTEGER NOT NULL CHECK(success IN (0, 1)),
    score REAL NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    generation_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transfer_memory_generation
ON transfer_trials(memory_id, generation_id, target_game);

CREATE TABLE IF NOT EXISTS contradiction_records (
    contradiction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    severity REAL NOT NULL,
    source_game TEXT,
    source_context TEXT,
    source_global_step INTEGER,
    payload_json TEXT NOT NULL DEFAULT '{}',
    generation_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contradiction_memory_generation
ON contradiction_records(memory_id, generation_id);

CREATE TABLE IF NOT EXISTS candidate_provenance (
    memory_id INTEGER PRIMARY KEY,
    candidate_generation INTEGER NOT NULL,
    provenance_games_json TEXT NOT NULL DEFAULT '[]',
    provenance_contexts_json TEXT NOT NULL DEFAULT '[]',
    scope_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gate_trials (
    gate_trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    gate_id INTEGER NOT NULL,
    candidate_generation INTEGER NOT NULL,
    target_game TEXT,
    target_context TEXT,
    participated INTEGER NOT NULL CHECK(participated IN (0,1)),
    contribution REAL NOT NULL DEFAULT 0,
    causal_gain REAL NOT NULL DEFAULT 0,
    prediction_gain REAL NOT NULL DEFAULT 0,
    planning_gain REAL NOT NULL DEFAULT 0,
    future_option_gain REAL NOT NULL DEFAULT 0,
    terminal_gain REAL NOT NULL DEFAULT 0,
    efficiency_gain REAL NOT NULL DEFAULT 0,
    intervention_type TEXT NOT NULL,
    paired_trial_id TEXT NOT NULL,
    genuine INTEGER NOT NULL DEFAULT 0 CHECK(genuine IN (0,1)),
    success INTEGER NOT NULL DEFAULT 0 CHECK(success IN (0,1)),
    transfer_score REAL NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    generation_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gate_trials_memory_gate
ON gate_trials(memory_id, gate_id, genuine, generation_id, target_game, target_context);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gate_trials_pair
ON gate_trials(memory_id, gate_id, paired_trial_id);

CREATE TABLE IF NOT EXISTS lifecycle_windows (
    memory_id INTEGER PRIMARY KEY,
    consecutive_low_windows INTEGER NOT NULL DEFAULT 0,
    consecutive_harm_windows INTEGER NOT NULL DEFAULT 0,
    consecutive_positive_windows INTEGER NOT NULL DEFAULT 0,
    last_utility REAL NOT NULL DEFAULT 0,
    last_generation INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_tombstones (
    memory_id INTEGER PRIMARY KEY,
    level_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    canonical_key TEXT,
    retired_generation INTEGER NOT NULL,
    reason TEXT NOT NULL,
    replacement_memory_id INTEGER,
    provenance_pointer TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    columns = _columns(connection, "memory_nodes")
    additions = (
        ("cognitive_state", "INTEGER NOT NULL DEFAULT 1"),
        ("validation_state", "INTEGER NOT NULL DEFAULT 4"),
        ("gate_id", "INTEGER NOT NULL DEFAULT 0"),
    )
    for name, declaration in additions:
        if name not in columns:
            connection.execute(f"ALTER TABLE memory_nodes ADD COLUMN {name} {declaration}")
    connection.executescript(DDL)
    connection.execute("UPDATE v7_schema_meta SET schema_version=?", (SCHEMA_VERSION,))


def ensure_v7_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS v7_schema_meta (schema_version INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT schema_version FROM v7_schema_meta LIMIT 1").fetchone()
    if row is None:
        connection.executescript(DDL)
        connection.execute(
            "INSERT INTO v7_schema_meta(schema_version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
    else:
        version = int(row[0])
        if version == 1:
            _migrate_v1_to_v2(connection)
        elif version == SCHEMA_VERSION:
            connection.executescript(DDL)
        else:
            raise RuntimeError(
                f"unsupported v7 schema version {version}; expected 1 or {SCHEMA_VERSION}"
            )
    connection.commit()
