from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator
from uuid import uuid4

from v4_5.benchmark.db.schema import create_schema


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "artifacts" / "v4_5" / "benchmark"
DEFAULT_DB_PATH = DEFAULT_BENCHMARK_DIR / "benchmark.sqlite"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "output"


def utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_run_id() -> str:
    return uuid4().hex


def default_db_path() -> Path:
    return DEFAULT_DB_PATH


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


@dataclass
class BenchmarkStore:
    db_path: Path | str | None = None

    def __post_init__(self) -> None:
        resolved = Path(self.db_path) if self.db_path is not None else default_db_path()
        self.db_path = resolved
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            create_schema(connection)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def is_catalog_empty(self) -> bool:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM games").fetchone()
        return int(row["count"]) == 0

    def upsert_games(self, rows: Iterable[dict[str, Any]]) -> None:
        payload = [
            (
                row["game_id"],
                row["title"],
                row["description"],
                row["family"],
                int(bool(row.get("in_benchmark", False))),
                row.get("notes"),
                row["created_at"],
                row["updated_at"],
            )
            for row in rows
        ]
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO games (
                    game_id, title, description, family, in_benchmark, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    family = excluded.family,
                    in_benchmark = excluded.in_benchmark,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                payload,
            )

    def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_dict(row) for row in rows if row is not None]

    def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(sql, params).fetchone()
        return _row_dict(row)

    def update_game_benchmark_flag(self, game_id: str, in_benchmark: bool) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE games SET in_benchmark = ?, updated_at = ? WHERE game_id = ?",
                (int(bool(in_benchmark)), utc_now_text(), game_id),
            )

    def update_game_metadata(
        self,
        game_id: str,
        *,
        family: str | None = None,
        description: str | None = None,
        notes: str | None = None,
    ) -> None:
        current = self.fetch_one("SELECT * FROM games WHERE game_id = ?", (game_id,))
        if current is None:
            raise KeyError(f"unknown game_id: {game_id}")
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE games
                SET family = ?, description = ?, notes = ?, updated_at = ?
                WHERE game_id = ?
                """,
                (
                    current["family"] if family is None else family,
                    current["description"] if description is None else description,
                    current.get("notes") if notes is None else notes,
                    utc_now_text(),
                    game_id,
                ),
            )

    def insert_benchmark_run(self, row: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO benchmark_runs (
                    run_id, run_label, started_at, finished_at, solver_version, runtime_mode, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["run_id"],
                    row["run_label"],
                    row["started_at"],
                    row.get("finished_at"),
                    row["solver_version"],
                    row["runtime_mode"],
                    row.get("notes"),
                ),
            )

    def finalize_benchmark_run(self, run_id: str, finished_at: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE benchmark_runs SET finished_at = ? WHERE run_id = ?",
                (finished_at, run_id),
            )

    def insert_game_result(self, row: dict[str, Any]) -> None:
        self.insert_many_game_results([row])

    def insert_many_game_results(self, rows: Iterable[dict[str, Any]]) -> None:
        payload = [
            (
                row["run_id"],
                row["game_id"],
                int(row["attempted"]),
                row["levels_seen"],
                row["levels_solved"],
                row["total_steps_executed"],
                row["solved_levels_total_steps"],
                row["unsolved_levels_total_steps"],
                int(row["terminal_success"]),
                int(row["terminal_failure"]),
                row["status"],
                row.get("failure_reason"),
                row.get("worker_pid"),
                row.get("worker_status"),
                row["created_at"],
            )
            for row in rows
        ]
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO benchmark_game_results (
                    run_id, game_id, attempted, levels_seen, levels_solved, total_steps_executed,
                    solved_levels_total_steps, unsolved_levels_total_steps, terminal_success,
                    terminal_failure, status, failure_reason, worker_pid, worker_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def insert_level_results(self, rows: Iterable[dict[str, Any]]) -> None:
        self.insert_level_result_batch(rows)

    def insert_level_result_batch(self, rows: Iterable[dict[str, Any]]) -> None:
        payload = [
            (
                row["run_id"],
                row["game_id"],
                row["level_index"],
                int(row["attempted"]),
                int(row["solved"]),
                row["steps_executed"],
                row["terminal_status"],
                row.get("failure_reason"),
                row.get("solution_action_count"),
                row["created_at"],
            )
            for row in rows
        ]
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO benchmark_level_results (
                    run_id, game_id, level_index, attempted, solved, steps_executed,
                    terminal_status, failure_reason, solution_action_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def replace_game_best_result(self, row: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO game_best_results (
                    game_id, best_levels_solved, best_solved_levels_total_steps,
                    best_total_steps_for_best_solved, best_run_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    best_levels_solved = excluded.best_levels_solved,
                    best_solved_levels_total_steps = excluded.best_solved_levels_total_steps,
                    best_total_steps_for_best_solved = excluded.best_total_steps_for_best_solved,
                    best_run_id = excluded.best_run_id,
                    updated_at = excluded.updated_at
                """,
                (
                    row["game_id"],
                    row["best_levels_solved"],
                    row.get("best_solved_levels_total_steps"),
                    row.get("best_total_steps_for_best_solved"),
                    row.get("best_run_id"),
                    row["updated_at"],
                ),
            )

    def replace_level_best_result(self, row: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO level_best_results (
                    game_id, level_index, best_steps_executed, best_run_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(game_id, level_index) DO UPDATE SET
                    best_steps_executed = excluded.best_steps_executed,
                    best_run_id = excluded.best_run_id,
                    updated_at = excluded.updated_at
                """,
                (
                    row["game_id"],
                    row["level_index"],
                    row["best_steps_executed"],
                    row["best_run_id"],
                    row["updated_at"],
                ),
            )
