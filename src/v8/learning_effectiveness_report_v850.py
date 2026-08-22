from __future__ import annotations

"""v8.50 dedicated learning-effectiveness summary reporting.

The report is observational.  Every live stdout metric and JSONL metric now comes
from the same current-run snapshot.  Strategy and transfer success rates use
validated frontier attempts/successes; they are not causal-lift claims.  Causal
attribution remains explicitly unavailable until a matched ablation is measured.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path


_INSTALLED = False
_BASE_WRITE_ALLOCATION_LOG = None
_BASE_REPORTER_EMIT_LINE = None
_BASE_PERIODIC_PROGRESS_LINE = None
_BASE_RUN_ACTOR_JOBS = None
_REPORT_FILE = "learning_effectiveness.log"
_SCHEMA_VERSION = 3
_CAUSAL_STATUS = "NOT_MEASURED_NO_ABLATION"

_OUTCOME_FIELDS = (
    "steps",
    "wins",
    "failures",
    "levels_completed",
    "replans",
    "planned_steps",
    "resets",
)


def _empty_outcome() -> dict[str, int]:
    row = {field: 0 for field in _OUTCOME_FIELDS}
    row["first_win_step"] = 0
    row["max_level_reached"] = 0
    return row


def _live_outcomes(completed_by_game, active_progress, active_leases) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for game_id, raw in completed_by_game.items():
        row = _empty_outcome()
        for field in _OUTCOME_FIELDS:
            row[field] = max(0, int(raw.get(field, 0)))
        row["first_win_step"] = max(0, int(raw.get("first_win_step", 0)))
        row["max_level_reached"] = max(0, int(raw.get("max_level_reached", 0)))
        result[str(game_id)] = row

    for worker_id, progress in active_progress.items():
        lease = active_leases.get(worker_id)
        if lease is None or progress is None:
            continue
        game = str(lease.game_id)
        row = result.setdefault(game, _empty_outcome())
        base_steps = int(row["steps"])
        for field in (
            "steps",
            "wins",
            "failures",
            "levels_completed",
            "replans",
            "planned_steps",
        ):
            row[field] += max(0, int(getattr(progress, field, 0)))
        row["max_level_reached"] = max(
            int(row.get("max_level_reached", 0)),
            max(0, int(getattr(progress, "max_level_reached", 0))),
        )
        local_first = max(0, int(getattr(progress, "first_win_step", 0) or 0))
        if int(row.get("first_win_step", 0)) <= 0 and local_first > 0:
            row["first_win_step"] = base_steps + local_first
    return result


def _known_levels_solved(coordinator, game_id: str) -> int:
    """Legacy helper retained for callers; current-run reporting uses observations."""
    game = str(game_id)
    with coordinator._lock:
        if bool(coordinator._game_won.get(game, False)):
            return 5
        solved = [
            int(level)
            for (owner, level), record in coordinator._records.items()
            if owner == game and str(getattr(record.state, "value", record.state)) != "UNSOLVED"
        ]
    return max(0, min(5, max(solved, default=0)))


def _current_levels_solved(outcomes, games) -> int:
    """Distinct current-run competence: deepest level reached once per game."""
    from v8 import plateau_progress_v846 as progress

    total = 0
    for game_id in games:
        game = str(game_id)
        row = outcomes.get(game, {})
        if int(row.get("wins", 0)) > 0:
            total += 5
            continue
        deepest = max(
            max(0, int(row.get("max_level_reached", 0))),
            max(0, int(progress._MAX_LEVEL_REACHED.get(game, 0))),
        )
        if deepest <= 0:
            deepest = max(0, int(row.get("levels_completed", 0)))
        total += min(5, deepest)
    return total


def _action_totals(game_ids) -> tuple[int, int]:
    from v8 import action_learning_report_v849 as action_report

    observed = productive = 0
    for game_id in game_ids:
        raw = action_report._RUN.get(str(game_id), {})
        observed += max(0, int(raw.get("click_actions_executed", 0)))
        observed += max(0, int(raw.get("movement_actions_executed", 0)))
        productive += max(0, int(raw.get("click_productive", 0)))
        productive += max(0, int(raw.get("movement_productive", 0)))
    return observed, productive


def _telemetry_by_game(runtime, coordinator) -> dict[str, dict[str, object]]:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    return {
        str(row.game_id): asdict(row)
        for row in coordinator.telemetry(optimizer_service=service)
    }


def _frontier_validation_totals(
    coordinator,
    game_ids,
    *,
    source: str | None = None,
) -> tuple[int, int]:
    """Return attempts/successes of current selectable frontier winners.

    This measures empirical validation reliability, not causal improvement over a
    memory-off policy.  Restricting to one winner per scope avoids counting stale
    dominated alternatives as current intelligence.
    """
    selected = {str(game_id) for game_id in game_ids}
    attempts = successes = 0
    with coordinator._lock:
        scopes = tuple(coordinator.frontier.scopes())
        for scope in scopes:
            if str(scope.game_id) not in selected:
                continue
            winner = coordinator.frontier.winner(scope)
            if winner is None:
                continue
            winner_source = str(getattr(winner.source, "value", winner.source))
            if source is not None and winner_source != str(source):
                continue
            attempts += max(0, int(getattr(winner, "attempts", 0)))
            successes += max(0, int(getattr(winner, "successes", 0)))
    return attempts, min(successes, attempts)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) <= 0.0 else 100.0 * float(numerator) / float(denominator)


def _pct_or_none(numerator: int | float, denominator: int | float) -> float | None:
    return None if float(denominator) <= 0.0 else _pct(numerator, denominator)


def _run_actor_jobs_v850(
    runtime,
    jobs,
    *,
    timeout: float | None = None,
    progress_interval_seconds: float = 60.0,
    progress_callback=None,
    reporting_queue=None,
):
    jobs = tuple(jobs)
    runtime._v850_total_step_budget = sum(
        max(0, int(getattr(job, "steps", 0))) for job in jobs
    )
    return _BASE_RUN_ACTOR_JOBS(
        runtime,
        jobs,
        timeout=timeout,
        progress_interval_seconds=progress_interval_seconds,
        progress_callback=progress_callback,
        reporting_queue=reporting_queue,
    )


def learning_effectiveness_snapshot_v850(
    runtime,
    coordinator,
    completed_by_game,
    active_progress,
    active_leases,
) -> dict[str, object]:
    from v8 import action_learning_report_v849 as action_report

    action_report._refresh_events(force=True)
    outcomes = _live_outcomes(completed_by_game, active_progress, active_leases)
    telemetry = _telemetry_by_game(runtime, coordinator)
    games = tuple(
        sorted(set(getattr(coordinator, "_games", ())) | set(outcomes) | set(telemetry))
    )

    total_steps = sum(
        max(0, int(outcomes.get(game, {}).get("steps", 0))) for game in games
    )
    planned_steps = sum(
        max(0, int(outcomes.get(game, {}).get("planned_steps", 0))) for game in games
    )
    level_advance_events = sum(
        max(0, int(outcomes.get(game, {}).get("levels_completed", 0))) for game in games
    )
    current_run_games_won = sum(
        1 for game in games if int(outcomes.get(game, {}).get("wins", 0)) > 0
    )
    current_levels_solved = _current_levels_solved(outcomes, games)

    m7_attempts, m7_successes = _frontier_validation_totals(coordinator, games)
    transfer_validation_attempts, transfer_validation_successes = (
        _frontier_validation_totals(coordinator, games, source="TRANSFER")
    )

    optimizer_validations = optimizer_successes = optimizer_saved_actions = 0
    transfer_attempts = 0
    for game in games:
        learning = telemetry.get(game, {})
        transfer_attempts += max(0, int(learning.get("transfer_attempts", 0)))
        optimizer_validations += max(0, int(learning.get("optimizer_validations", 0)))
        optimizer_successes += max(0, int(learning.get("optimizer_successes", 0)))
        optimizer_saved_actions += max(0, int(learning.get("optimizer_saved_actions", 0)))

    observed_actions, productive_actions = _action_totals(games)
    first_win_steps = [
        int(outcomes.get(game, {}).get("first_win_step", 0))
        for game in games
        if int(outcomes.get(game, {}).get("first_win_step", 0)) > 0
    ]

    effectiveness = {
        "outcome_effectiveness": {
            "level_solve_rate_pct": _pct(current_levels_solved, 5 * len(games)),
            "game_solve_rate_pct": _pct(current_run_games_won, len(games)),
            "current_run_levels_solved": current_levels_solved,
            "current_run_level_advance_events": level_advance_events,
            "current_run_games_won": current_run_games_won,
        },
        "learning_application_effectiveness": {
            "m7_action_share_pct": _pct(planned_steps, total_steps),
            "m7_validation_success_rate_pct": _pct_or_none(m7_successes, m7_attempts),
            "m7_validation_attempts": m7_attempts,
            "m7_validation_successes": m7_successes,
            "measurement": "CURRENT_FRONTIER_VALIDATION_RELIABILITY",
        },
        "transfer_effectiveness": {
            "transfer_validation_success_rate_pct": _pct_or_none(
                transfer_validation_successes,
                transfer_validation_attempts,
            ),
            "transfer_validation_attempts": transfer_validation_attempts,
            "transfer_validation_successes": transfer_validation_successes,
            "transfer_attempts": transfer_attempts,
            "measurement": "TRANSFER_FRONTIER_VALIDATION_RELIABILITY",
        },
        "optimizer_effectiveness": {
            "optimizer_success_rate_pct": _pct_or_none(
                optimizer_successes,
                optimizer_validations,
            ),
            "optimizer_validations": optimizer_validations,
            "optimizer_successes": optimizer_successes,
            "optimizer_saved_actions": optimizer_saved_actions,
        },
        "action_effectiveness": {
            "productive_action_rate_pct": _pct_or_none(
                productive_actions,
                observed_actions,
            ),
            "observed_actions": observed_actions,
            "productive_actions": productive_actions,
        },
        "efficiency": {
            "steps_per_solved_level": (
                float(total_steps) / float(current_levels_solved)
                if current_levels_solved > 0
                else None
            ),
            "mean_first_win_step": (
                float(sum(first_win_steps)) / float(len(first_win_steps))
                if first_win_steps
                else None
            ),
        },
        "causal_effectiveness": {
            "status": _CAUSAL_STATUS,
        },
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "time": time.time(),
        "generation": int(getattr(runtime, "generation", 0)),
        "watermark": int(getattr(runtime, "watermark", 0)),
        "scope": {
            "kind": "CURRENT_RUN",
            "games": len(games),
            "steps": total_steps,
            "planned_steps": planned_steps,
        },
        "effectiveness": effectiveness,
    }


def _compact_number(value) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.1f}"


def _compact_pct(value) -> str:
    return "-" if value is None else f"{float(value):.1f}%"


def format_learning_effectiveness_stdout_v850(
    payload: dict[str, object],
    *,
    budget_consumed_pct: float | None = None,
) -> str:
    effectiveness = payload.get("effectiveness", {})
    if not isinstance(effectiveness, dict):
        effectiveness = {}
    outcome = effectiveness.get("outcome_effectiveness", {})
    learning = effectiveness.get("learning_application_effectiveness", {})
    transfer = effectiveness.get("transfer_effectiveness", {})
    optimizer = effectiveness.get("optimizer_effectiveness", {})
    action = effectiveness.get("action_effectiveness", {})
    efficiency = effectiveness.get("efficiency", {})
    if not isinstance(outcome, dict):
        outcome = {}
    if not isinstance(learning, dict):
        learning = {}
    if not isinstance(transfer, dict):
        transfer = {}
    if not isinstance(optimizer, dict):
        optimizer = {}
    if not isinstance(action, dict):
        action = {}
    if not isinstance(efficiency, dict):
        efficiency = {}
    budget_prefix = (
        ""
        if budget_consumed_pct is None
        else f"{max(0.0, min(100.0, float(budget_consumed_pct))):.0f}% - "
    )
    return (
        f"{budget_prefix}effectiveness "
        f"L={float(outcome.get('level_solve_rate_pct', 0.0)):.1f}% "
        f"G={float(outcome.get('game_solve_rate_pct', 0.0)):.1f}% "
        f"M7={float(learning.get('m7_action_share_pct', 0.0)):.1f}% "
        f"M7val={_compact_pct(learning.get('m7_validation_success_rate_pct'))} "
        f"XferVal={_compact_pct(transfer.get('transfer_validation_success_rate_pct'))} "
        f"Opt={_compact_pct(optimizer.get('optimizer_success_rate_pct'))} "
        f"Prod={_compact_pct(action.get('productive_action_rate_pct'))} "
        f"step/L={_compact_number(efficiency.get('steps_per_solved_level'))} "
        f"firstWin={_compact_number(efficiency.get('mean_first_win_step'))}"
    )


def _emit_effectiveness_stdout(
    payload: dict[str, object],
    *,
    budget_consumed_pct: float | None = None,
) -> None:
    print(
        f'[{time.strftime("%H:%M")}] '
        f"{format_learning_effectiveness_stdout_v850(payload, budget_consumed_pct=budget_consumed_pct)}",
        flush=True,
    )


def _reporter_emit_line_v850(message: str, output_queue) -> None:
    # Periodic competence rows are intentionally suppressed: the parent allocator
    # emits the authoritative full effectiveness snapshot at the same 60 s cadence.
    if "current_run_wins=" in str(message):
        return
    _BASE_REPORTER_EMIT_LINE(message, output_queue)


def _periodic_progress_line_v850(rows, total_steps, baseline=None) -> str:
    """Keep the reporter's legacy row intact so _reporter_emit_line_v850 suppresses it."""
    return _BASE_PERIODIC_PROGRESS_LINE(tuple(rows), total_steps, baseline)


def _write_learning_effectiveness_log(
    runtime,
    coordinator,
    completed_by_game,
    active_progress,
    active_leases,
) -> None:
    payload = learning_effectiveness_snapshot_v850(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    target = Path(runtime.root) / _REPORT_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        pass

    scope = payload.get("scope", {})
    used_steps = int(scope.get("steps", 0)) if isinstance(scope, dict) else 0
    budget = max(0, int(getattr(runtime, "_v850_total_step_budget", 0)))
    budget_pct = None if budget <= 0 else _pct(used_steps, budget)

    # Avoid an immediate duplicate final line while retaining the normal minute
    # cadence even when progress stalls.
    now = time.monotonic()
    last_steps = int(getattr(runtime, "_v850_last_effectiveness_stdout_steps", -1))
    last_time = float(getattr(runtime, "_v850_last_effectiveness_stdout_time", -1.0))
    if used_steps != last_steps or last_time < 0.0 or now - last_time >= 30.0:
        _emit_effectiveness_stdout(payload, budget_consumed_pct=budget_pct)
        runtime._v850_last_effectiveness_stdout_steps = used_steps
        runtime._v850_last_effectiveness_stdout_time = now


def _write_allocation_log_v850(
    runtime,
    coordinator,
    completed_by_game,
    active_progress,
    active_leases,
) -> None:
    _BASE_WRITE_ALLOCATION_LOG(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    _write_learning_effectiveness_log(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )


def install_learning_effectiveness_report_v850() -> None:
    global _INSTALLED, _BASE_WRITE_ALLOCATION_LOG, _BASE_REPORTER_EMIT_LINE
    global _BASE_PERIODIC_PROGRESS_LINE, _BASE_RUN_ACTOR_JOBS
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import lease_dispatch_lifecycle_v843 as v843
    from v8 import reporter

    _BASE_WRITE_ALLOCATION_LOG = v843._BASE_WRITE_ALLOCATION_LOG
    v843._BASE_WRITE_ALLOCATION_LOG = _write_allocation_log_v850

    _BASE_REPORTER_EMIT_LINE = reporter._emit_line
    reporter._emit_line = _reporter_emit_line_v850
    _BASE_PERIODIC_PROGRESS_LINE = reporter.format_periodic_progress_line
    reporter.format_periodic_progress_line = _periodic_progress_line_v850

    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    actor_module.run_actor_jobs = _run_actor_jobs_v850
    _INSTALLED = True
