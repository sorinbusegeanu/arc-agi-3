from __future__ import annotations

from typing import Any

from v4_5.benchmark.db.store import BenchmarkStore


def list_all_games(store: BenchmarkStore) -> list[dict[str, Any]]:
    return store.fetch_all("SELECT * FROM games ORDER BY game_id")


def list_active_benchmark_games(store: BenchmarkStore) -> list[dict[str, Any]]:
    return store.fetch_all("SELECT * FROM games WHERE in_benchmark = 1 ORDER BY game_id")


def get_latest_run(store: BenchmarkStore, game_id: str) -> dict[str, Any] | None:
    return store.fetch_one(
        """
        SELECT
            r.run_id,
            r.run_label,
            r.started_at,
            r.finished_at,
            r.solver_version,
            r.runtime_mode,
            r.notes,
            g.game_id,
            g.attempted,
            g.levels_seen,
            g.levels_solved,
            g.total_steps_executed,
            g.solved_levels_total_steps,
            g.unsolved_levels_total_steps,
            g.terminal_success,
            g.terminal_failure,
            g.status,
            g.failure_reason,
            g.created_at
        FROM benchmark_runs r
        JOIN benchmark_game_results g ON g.run_id = r.run_id
        WHERE g.game_id = ?
        ORDER BY r.started_at DESC, r.run_id DESC
        LIMIT 1
        """,
        (game_id,),
    )


def get_one_run(store: BenchmarkStore, run_id: str) -> dict[str, Any] | None:
    return store.fetch_one("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))


def get_game_history(store: BenchmarkStore, game_id: str) -> list[dict[str, Any]]:
    return store.fetch_all(
        """
        SELECT
            r.run_id,
            r.run_label,
            r.started_at,
            r.finished_at,
            r.solver_version,
            r.runtime_mode,
            g.game_id,
            g.attempted,
            g.levels_seen,
            g.levels_solved,
            g.total_steps_executed,
            g.solved_levels_total_steps,
            g.unsolved_levels_total_steps,
            g.terminal_success,
            g.terminal_failure,
            g.status,
            g.failure_reason,
            g.created_at
        FROM benchmark_game_results g
        JOIN benchmark_runs r ON r.run_id = g.run_id
        WHERE g.game_id = ?
        ORDER BY r.started_at ASC, r.run_id ASC
        """,
        (game_id,),
    )


def get_level_history(store: BenchmarkStore, game_id: str, level_index: int) -> list[dict[str, Any]]:
    return store.fetch_all(
        """
        SELECT
            r.run_id,
            r.run_label,
            r.started_at,
            l.game_id,
            l.level_index,
            l.attempted,
            l.solved,
            l.steps_executed,
            l.terminal_status,
            l.failure_reason,
            l.solution_action_count,
            l.created_at
        FROM benchmark_level_results l
        JOIN benchmark_runs r ON r.run_id = l.run_id
        WHERE l.game_id = ? AND l.level_index = ?
        ORDER BY r.started_at ASC, r.run_id ASC
        """,
        (game_id, int(level_index)),
    )


def get_current_game_best_result(store: BenchmarkStore, game_id: str) -> dict[str, Any] | None:
    return store.fetch_one("SELECT * FROM game_best_results WHERE game_id = ?", (game_id,))


def get_current_level_best_result(store: BenchmarkStore, game_id: str, level_index: int) -> dict[str, Any] | None:
    return store.fetch_one(
        "SELECT * FROM level_best_results WHERE game_id = ? AND level_index = ?",
        (game_id, int(level_index)),
    )


def get_leaderboard_by_solved_levels(store: BenchmarkStore) -> list[dict[str, Any]]:
    return store.fetch_all(
        """
        SELECT
            games.game_id,
            games.title,
            games.family,
            best.best_levels_solved,
            best.best_solved_levels_total_steps,
            best.best_total_steps_for_best_solved,
            best.best_run_id,
            best.updated_at
        FROM game_best_results best
        JOIN games ON games.game_id = best.game_id
        ORDER BY
            best.best_levels_solved DESC,
            best.best_solved_levels_total_steps ASC,
            best.best_total_steps_for_best_solved ASC,
            games.game_id ASC
        """
    )


def get_leaderboard_by_best_level_step_records(store: BenchmarkStore) -> list[dict[str, Any]]:
    return store.fetch_all(
        """
        SELECT
            levels.game_id,
            games.title,
            games.family,
            levels.level_index,
            levels.best_steps_executed,
            levels.best_run_id,
            levels.updated_at
        FROM level_best_results levels
        JOIN games ON games.game_id = levels.game_id
        ORDER BY
            levels.best_steps_executed ASC,
            levels.game_id ASC,
            levels.level_index ASC
        """
    )


def compare_runs(store: BenchmarkStore, run_id_a: str, run_id_b: str) -> list[dict[str, Any]]:
    return store.fetch_all(
        """
        SELECT
            a.game_id AS game_id,
            a.run_id AS run_id_a,
            b.run_id AS run_id_b,
            a.levels_solved AS levels_solved_a,
            b.levels_solved AS levels_solved_b,
            a.solved_levels_total_steps AS solved_levels_total_steps_a,
            b.solved_levels_total_steps AS solved_levels_total_steps_b,
            a.total_steps_executed AS total_steps_executed_a,
            b.total_steps_executed AS total_steps_executed_b
        FROM benchmark_game_results a
        JOIN benchmark_game_results b ON a.game_id = b.game_id
        WHERE a.run_id = ? AND b.run_id = ?
        ORDER BY a.game_id
        """,
        (run_id_a, run_id_b),
    )
