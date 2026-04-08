from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from v4_5.benchmark.db import query
from v4_5.benchmark.db.store import BenchmarkStore, default_output_dir, utc_now_text


def _game_candidate_tuple(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("levels_solved", 0) or 0),
        -int(row.get("solved_levels_total_steps") or 10**12),
        -int(row.get("total_steps_executed") or 10**12),
    )


def is_better_game_result(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    candidate_levels = int(candidate.get("levels_solved", 0) or 0)
    incumbent_levels = int(incumbent.get("levels_solved", incumbent.get("best_levels_solved", 0)) or 0)
    if candidate_levels != incumbent_levels:
        return candidate_levels > incumbent_levels
    candidate_solved_steps = int(candidate.get("solved_levels_total_steps") or candidate.get("best_solved_levels_total_steps") or 10**12)
    incumbent_solved_steps = int(incumbent.get("solved_levels_total_steps") or incumbent.get("best_solved_levels_total_steps") or 10**12)
    if candidate_solved_steps != incumbent_solved_steps:
        return candidate_solved_steps < incumbent_solved_steps
    candidate_total_steps = int(candidate.get("total_steps_executed") or candidate.get("best_total_steps_for_best_solved") or 10**12)
    incumbent_total_steps = int(incumbent.get("total_steps_executed") or incumbent.get("best_total_steps_for_best_solved") or 10**12)
    if candidate_total_steps != incumbent_total_steps:
        return candidate_total_steps < incumbent_total_steps
    return False


def is_better_level_result(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return bool(candidate.get("solved", False))
    candidate_solved = bool(candidate.get("solved", False))
    incumbent_solved = bool(incumbent.get("solved", True))
    if candidate_solved != incumbent_solved:
        return candidate_solved and not incumbent_solved
    if not candidate_solved:
        return False
    candidate_steps = int(candidate.get("steps_executed", candidate.get("best_steps_executed", 10**12)) or 10**12)
    incumbent_steps = int(incumbent.get("steps_executed", incumbent.get("best_steps_executed", 10**12)) or 10**12)
    if candidate_steps != incumbent_steps:
        return candidate_steps < incumbent_steps
    return False


def refresh_best_results(store: BenchmarkStore, run_id: str) -> None:
    updated_at = utc_now_text()
    game_rows = store.fetch_all("SELECT * FROM benchmark_game_results WHERE run_id = ? ORDER BY game_id", (run_id,))
    for game_row in game_rows:
        current_best = store.fetch_one("SELECT * FROM game_best_results WHERE game_id = ?", (game_row["game_id"],))
        if is_better_game_result(game_row, current_best):
            store.replace_game_best_result(
                {
                    "game_id": game_row["game_id"],
                    "best_levels_solved": game_row["levels_solved"],
                    "best_solved_levels_total_steps": game_row["solved_levels_total_steps"],
                    "best_total_steps_for_best_solved": game_row["total_steps_executed"],
                    "best_run_id": run_id,
                    "updated_at": updated_at,
                }
            )

    level_rows = store.fetch_all("SELECT * FROM benchmark_level_results WHERE run_id = ? ORDER BY game_id, level_index", (run_id,))
    for level_row in level_rows:
        current_best = store.fetch_one(
            "SELECT * FROM level_best_results WHERE game_id = ? AND level_index = ?",
            (level_row["game_id"], level_row["level_index"]),
        )
        if is_better_level_result(level_row, current_best):
            store.replace_level_best_result(
                {
                    "game_id": level_row["game_id"],
                    "level_index": level_row["level_index"],
                    "best_steps_executed": level_row["steps_executed"],
                    "best_run_id": run_id,
                    "updated_at": updated_at,
                }
            )


def build_summary_payload(store: BenchmarkStore, run_id: str, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    run_row = query.get_one_run(store, run_id)
    if run_row is None:
        raise KeyError(f"unknown run_id: {run_id}")
    game_rows = store.fetch_all(
        """
        SELECT * FROM benchmark_game_results
        WHERE run_id = ?
        ORDER BY game_id
        """,
        (run_id,),
    )
    level_rows = store.fetch_all(
        """
        SELECT * FROM benchmark_level_results
        WHERE run_id = ?
        ORDER BY game_id, level_index
        """,
        (run_id,),
    )
    games: list[dict[str, Any]] = []
    for game_row in game_rows:
        game_levels = [row for row in level_rows if row["game_id"] == game_row["game_id"]]
        games.append({**game_row, "levels": game_levels})
    payload = {
        "run": run_row,
        "games": games,
        "game_leaderboard": query.get_leaderboard_by_solved_levels(store),
        "level_leaderboard": query.get_leaderboard_by_best_level_step_records(store),
        "multiprocessing": {
            "use_multiprocessing": False,
            "max_workers": 1,
            "completed_game_count": len([row for row in game_rows if row.get("status") == "completed"]),
            "failed_game_count": len([row for row in game_rows if row.get("status") not in {"completed", "timed_out"}]),
            "timed_out_game_count": len([row for row in game_rows if row.get("status") == "timed_out"]),
            "merge_failed_game_count": 0,
            "worker_status_summary": [{ "game_id": row["game_id"], "worker_status": row.get("worker_status"), "worker_pid": row.get("worker_pid") } for row in game_rows],
        },
    }
    if extra_summary:
        payload["multiprocessing"].update(extra_summary)
    return payload


def write_summary_json(payload: dict[str, Any], *, output_dir: Path | str | None = None, run_id: str) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{run_id}_summary.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def write_game_summary_csv(payload: dict[str, Any], *, output_dir: Path | str | None = None, run_id: str) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{run_id}_games.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "game_id",
                "attempted",
                "levels_seen",
                "levels_solved",
                "total_steps_executed",
                "solved_levels_total_steps",
                "unsolved_levels_total_steps",
                "terminal_success",
                "terminal_failure",
                "status",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for game_row in payload["games"]:
            writer.writerow(
                {
                    "run_id": payload["run"]["run_id"],
                    "game_id": game_row["game_id"],
                    "attempted": game_row["attempted"],
                    "levels_seen": game_row["levels_seen"],
                    "levels_solved": game_row["levels_solved"],
                    "total_steps_executed": game_row["total_steps_executed"],
                    "solved_levels_total_steps": game_row["solved_levels_total_steps"],
                    "unsolved_levels_total_steps": game_row["unsolved_levels_total_steps"],
                    "terminal_success": game_row["terminal_success"],
                    "terminal_failure": game_row["terminal_failure"],
                    "status": game_row["status"],
                    "failure_reason": game_row["failure_reason"],
                }
            )
    return path
