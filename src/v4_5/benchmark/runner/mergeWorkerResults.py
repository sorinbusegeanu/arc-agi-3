from __future__ import annotations

from v4_5.benchmark.db.store import BenchmarkStore
from v4_5.benchmark.reporting.summaryBuilder import refresh_best_results
from v4_5.benchmark.runner.processTypes import BenchmarkMergeResult, BenchmarkProcessResult
from v4_5.benchmark.runner.resultFileStore import load_result_file, validate_result_payload


def merge_worker_results(store: BenchmarkStore, run_id: str, process_results: tuple[BenchmarkProcessResult, ...]) -> BenchmarkMergeResult:
    merged = []
    invalid = []
    failed = []
    timed_out = []
    missing = []
    merge_failed = []
    game_rows = []
    level_rows = []
    for result in sorted(process_results, key=lambda item: item.game_id):
        if result.status == "timed_out":
            timed_out.append(result.game_id)
            continue
        if result.status != "completed":
            failed.append(result.game_id)
        if not result.result_json_path:
            missing.append(result.game_id)
            continue
        try:
            payload = load_result_file(result.result_json_path)
        except Exception:
            missing.append(result.game_id)
            continue
        if not validate_result_payload(payload):
            invalid.append(result.game_id)
            continue
        game_result = payload.get("game_result")
        if not isinstance(game_result, dict):
            invalid.append(result.game_id)
            continue
        try:
            game_rows.append(
                {
                    "run_id": run_id,
                    "game_id": game_result["game_id"],
                    "attempted": game_result["attempted"],
                    "levels_seen": game_result["levels_seen"],
                    "levels_solved": game_result["levels_solved"],
                    "total_steps_executed": game_result["total_steps_executed"],
                    "solved_levels_total_steps": game_result["solved_levels_total_steps"],
                    "unsolved_levels_total_steps": game_result["unsolved_levels_total_steps"],
                    "terminal_success": game_result["terminal_success"],
                    "terminal_failure": game_result["terminal_failure"],
                    "status": payload["status"] if payload["status"] != "completed" else game_result["status"],
                    "failure_reason": payload.get("error_message") or game_result.get("failure_reason"),
                    "created_at": game_result["created_at"],
                    "worker_pid": payload.get("worker_pid"),
                    "worker_status": payload.get("status"),
                }
            )
            for level in payload.get("level_results", []):
                level_rows.append(
                    {
                        "run_id": run_id,
                        "game_id": level["game_id"],
                        "level_index": level["level_index"],
                        "attempted": level["attempted"],
                        "solved": level["solved"],
                        "steps_executed": level["steps_executed"],
                        "terminal_status": level["terminal_status"],
                        "failure_reason": level.get("failure_reason"),
                        "solution_action_count": level.get("solution_action_count"),
                        "created_at": level["created_at"],
                    }
                )
            merged.append(result.game_id)
        except Exception:
            merge_failed.append(result.game_id)
    if game_rows:
        store.insert_many_game_results(game_rows)
    if level_rows:
        store.insert_level_result_batch(level_rows)
    refresh_best_results(store, run_id)
    return BenchmarkMergeResult(
        run_id=run_id,
        merged_game_ids=tuple(sorted(merged)),
        invalid_game_ids=tuple(sorted(set(invalid))),
        failed_game_ids=tuple(sorted(set(failed))),
        timed_out_game_ids=tuple(sorted(set(timed_out))),
        missing_result_files=tuple(sorted(set(missing))),
        merge_failed_game_ids=tuple(sorted(set(merge_failed))),
        inserted_game_count=len(game_rows),
    )
