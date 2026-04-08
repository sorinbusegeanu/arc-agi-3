from __future__ import annotations

from typing import Any

from v4_5.benchmark.db.store import utc_now_text
from v4_5.benchmark.runner.benchmarkTypes import NormalizedGameResult, NormalizedLevelResult


def _status_from_stop_reason(stop_reason: str, attempted: bool) -> str:
    if not attempted:
        return "not_attempted"
    if stop_reason in {"terminal_win", "terminal_fail"}:
        return "completed"
    if stop_reason in {"invalid_state_abort", "step_budget_exhausted"}:
        return "partial"
    return "partial"


def _terminal_status_for_unsolved(raw_result: dict[str, Any]) -> str:
    stop_reason = str(raw_result.get("stop_reason", ""))
    if stop_reason == "terminal_fail":
        return "failure"
    if stop_reason == "terminal_win":
        return "success"
    return "non_terminal"


def normalize_runner_output(raw_result: dict[str, Any], *, created_at: str | None = None) -> NormalizedGameResult:
    timestamp = created_at or utc_now_text()
    game_id = str(raw_result["game_id"])
    attempted = bool(raw_result.get("attempted", False))
    step_records = tuple(raw_result.get("step_records", ()))
    current_level = int(raw_result.get("levels_completed_start", 0) or 0)
    level_steps: dict[int, int] = {}
    solved_levels: set[int] = set()

    for record in step_records:
        if not bool(record.get("action_executed", False)):
            continue
        step_level = int(record.get("pre_levels_completed", current_level) or current_level)
        current_level = step_level
        level_steps[current_level] = level_steps.get(current_level, 0) + 1
        delta = int(record.get("levels_completed_delta", 0) or 0)
        if delta > 0:
            solved_levels.add(current_level)
            current_level = int(record.get("post_levels_completed", current_level + delta) or (current_level + delta))

    final_level = int(raw_result.get("levels_completed_end", current_level) or current_level)
    stop_reason = str(raw_result.get("stop_reason", ""))
    level_rows: list[NormalizedLevelResult] = []

    for level_index in sorted(solved_levels):
        steps = int(level_steps.get(level_index, 0))
        level_rows.append(
            NormalizedLevelResult(
                game_id=game_id,
                level_index=level_index,
                attempted=True,
                solved=True,
                steps_executed=steps,
                terminal_status="success",
                failure_reason=None,
                solution_action_count=steps,
                created_at=timestamp,
            )
        )

    unsolved_level_index = final_level
    unsolved_steps = int(level_steps.get(unsolved_level_index, 0))
    should_add_unsolved_row = attempted and stop_reason != "terminal_win"
    if should_add_unsolved_row and (step_records or unsolved_steps > 0 or not level_rows):
        if unsolved_level_index not in solved_levels:
            level_rows.append(
                NormalizedLevelResult(
                    game_id=game_id,
                    level_index=unsolved_level_index,
                    attempted=True,
                    solved=False,
                    steps_executed=unsolved_steps,
                    terminal_status=_terminal_status_for_unsolved(raw_result),
                    failure_reason=raw_result.get("failure_reason"),
                    solution_action_count=None,
                    created_at=timestamp,
                )
            )

    level_rows.sort(key=lambda item: item.level_index)
    total_steps = int(raw_result.get("steps_executed", 0) or 0)
    solved_steps = sum(item.steps_executed for item in level_rows if item.solved)
    unsolved_steps_total = sum(item.steps_executed for item in level_rows if not item.solved)
    if total_steps < solved_steps + unsolved_steps_total:
        total_steps = solved_steps + unsolved_steps_total

    return NormalizedGameResult(
        game_id=game_id,
        attempted=attempted,
        levels_seen=len(level_rows),
        levels_solved=sum(1 for item in level_rows if item.solved),
        total_steps_executed=total_steps,
        solved_levels_total_steps=solved_steps,
        unsolved_levels_total_steps=unsolved_steps_total,
        terminal_success=stop_reason == "terminal_win",
        terminal_failure=stop_reason == "terminal_fail",
        status=_status_from_stop_reason(stop_reason, attempted),
        failure_reason=raw_result.get("failure_reason"),
        level_results=tuple(level_rows),
        created_at=timestamp,
    )
