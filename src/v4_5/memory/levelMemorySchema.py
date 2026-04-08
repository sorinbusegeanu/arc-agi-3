from __future__ import annotations

import sqlite3


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS level_memory (
        game_id TEXT NOT NULL,
        level_id TEXT NOT NULL,
        avatar_json TEXT NULL,
        hud_regions_json TEXT NOT NULL,
        life_regions_json TEXT NOT NULL,
        progress_regions_json TEXT NOT NULL,
        pois_json TEXT NOT NULL,
        exit_regions_json TEXT NOT NULL,
        memory_state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        PRIMARY KEY (game_id, level_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_level_memory_game_id ON level_memory(game_id)",
)


def create_level_memory_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(level_memory)").fetchall()
    }
    if "memory_state" not in columns:
        connection.execute(
            "ALTER TABLE level_memory ADD COLUMN memory_state TEXT NOT NULL DEFAULT 'hypothesis'"
        )
    connection.commit()
