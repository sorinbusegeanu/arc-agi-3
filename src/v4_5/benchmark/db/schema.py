from __future__ import annotations

import sqlite3


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        family TEXT NOT NULL,
        in_benchmark INTEGER NOT NULL DEFAULT 0,
        notes TEXT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        run_id TEXT PRIMARY KEY,
        run_label TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT NULL,
        solver_version TEXT NOT NULL,
        runtime_mode TEXT NOT NULL,
        notes TEXT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_game_results (
        run_id TEXT NOT NULL,
        game_id TEXT NOT NULL,
        attempted INTEGER NOT NULL,
        levels_seen INTEGER NOT NULL,
        levels_solved INTEGER NOT NULL,
        total_steps_executed INTEGER NOT NULL,
        solved_levels_total_steps INTEGER NOT NULL,
        unsolved_levels_total_steps INTEGER NOT NULL,
        terminal_success INTEGER NOT NULL,
        terminal_failure INTEGER NOT NULL,
        status TEXT NOT NULL,
        failure_reason TEXT NULL,
        worker_pid INTEGER NULL,
        worker_status TEXT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, game_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_level_results (
        run_id TEXT NOT NULL,
        game_id TEXT NOT NULL,
        level_index INTEGER NOT NULL,
        attempted INTEGER NOT NULL,
        solved INTEGER NOT NULL,
        steps_executed INTEGER NOT NULL,
        terminal_status TEXT NOT NULL,
        failure_reason TEXT NULL,
        solution_action_count INTEGER NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, game_id, level_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS game_best_results (
        game_id TEXT PRIMARY KEY,
        best_levels_solved INTEGER NOT NULL,
        best_solved_levels_total_steps INTEGER NULL,
        best_total_steps_for_best_solved INTEGER NULL,
        best_run_id TEXT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS level_best_results (
        game_id TEXT NOT NULL,
        level_index INTEGER NOT NULL,
        best_steps_executed INTEGER NOT NULL,
        best_run_id TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (game_id, level_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_benchmark_game_results_game_id ON benchmark_game_results(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_level_results_game_id ON benchmark_level_results(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_benchmark_level_results_game_level ON benchmark_level_results(game_id, level_index)",
    "CREATE INDEX IF NOT EXISTS idx_games_in_benchmark ON games(in_benchmark)",
)


def create_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    _ensure_optional_columns(connection)
    connection.commit()


def _ensure_optional_columns(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(benchmark_game_results)").fetchall()
    columns = {row[1] for row in rows}
    if "worker_pid" not in columns:
        connection.execute("ALTER TABLE benchmark_game_results ADD COLUMN worker_pid INTEGER NULL")
    if "worker_status" not in columns:
        connection.execute("ALTER TABLE benchmark_game_results ADD COLUMN worker_status TEXT NULL")
