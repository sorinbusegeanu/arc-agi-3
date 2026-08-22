from __future__ import annotations

"""v8.50 dedicated learning-effectiveness summary reporting.

This layer is observational. It writes only high-level effectiveness metrics from
already-authoritative adaptive allocation, M7 planning/frontier, actor progress,
optimizer, transfer, and v8.49 action-quality telemetry. It does not emit per-game
details, change H01-H15 decisions, or claim causal lift without controlled ablation.
The same compact effectiveness summary replaces the older current-run stdout line.
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
_SCHEMA_VERSION = 2
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
    return result


def _known_levels_solved(coordinator, game_id: str) -> int:
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
            # Compatibility for callers that provide legacy aggregate rows.
            deepest = max(0, int(row.get("levels_completed", 0)))
        total += min(5, deepest)
    return total


def _action_totals(game_ids) -> tuple[int, int]:
    from v8 import action_learning_report_v849 as action_report

    observed = productive = 0
    for game_id in game_ids:
        raw = action_report._RUN.get(str(game_id), {})
        click_actions = max(0, int(raw.get("click_actions_executed", 0)))
        movement_actions = max(0, int(raw.get("movement_actions_executed", 0)))
        click_productive = max(0, int(raw.get("click_productive", 0)))
        movement_productive = max(0, int(raw.get("movement_productive", 0)))
        observed += click_actions + movement_actions
        productive += click_productive + movement_productive
    return observed, productive


def _telemetry_by_game(runtime, coordinator) -> dict[str, dict[str, object]]:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    return {
        str(row.game_id): asdict(row)
        for row in coordinator.telemetry(optimizer_service=service)
    }


def _pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) <= 0.0 else 100.0 * float(numerator) / float(denominator)


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
    games = tuple(sorted(set(getattr(coordinator, "_games", ())) | set(outcomes) | set(telemetry)))

    total_steps = sum(max(0, int(outcomes.get(game, {}).get("steps", 0))) for game in games)
    planned_steps = sum(max(0, int(outcomes.get(game, {}).get("planned_steps", 0))) for game in games)
    level_advances = sum(max(0, int(outcomes.get(game, {}).get("levels_completed", 0))) for game in games)
    current_run_games_won = sum(1 for game in games if int(outcomes.get(game, {}).get("wins", 0)) > 0)
    current_levels_solved = _current_levels_solved(outcomes, games)

    m7_applied_games = 0
    m7_progress_games = 0
    transfer_frontier_games = 0
    transfer_progress_games = 0
    optimizer_validations = 0
    optimizer_successes = 0
    optimizer_saved_actions = 0
    transfer_attempts = 0

    for game in games:
        outcome = outcomes.get(game, _empty_outcome())
        learning = telemetry.get(game, {})
        has_progress = int(outcome.get("levels_completed", 0)) > 0 or int(outcome.get("wins", 0)) > 0
        if int(outcome.get("planned_steps", 0)) > 0:
            m7_applied_games += 1
            if has_progress:
                m7_progress_games += 1
        if str(learning.get("frontier_source", "")) == "TRANSFER":
            transfer_frontier_games += 1
            if has_progress:
                transfer_progress_games += 1
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
            "current_run_level_advances": level_advances,
            "current_run_games_won": current_run_games_won,
        },
        "learning_application_effectiveness": {
            "m7_action_share_pct": _pct(planned_steps, total_steps),
            "m7_strategy_effectiveness_pct": _pct(m7_progress_games, m7_applied_games),
            "games_with_m7_applied": m7_applied_games,
        },
        "transfer_effectiveness": {
            "transfer_effectiveness_pct": _pct(transfer_progress_games, transfer_frontier_games),
            "transfer_attempts": transfer_attempts,
            "games_with_transfer_frontier": transfer_frontier_games,
        },
        "optimizer_effectiveness": {
            "optimizer_success_rate_pct": _pct(optimizer_successes, optimizer_validations),
            "optimizer_saved_actions": optimizer_saved_actions,
        },
        "action_effectiveness": {
            "productive_action_rate_pct": _pct(productive_actions, observed_actions),
        },
        "efficiency": {
            "steps_per_level_advance": (
                float(total_steps) / float(level_advances) if level_advances > 0 else None
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
        "effectiveness": effectiveness,
    }


def _compact_number(value) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.1f}"


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
        f"M7eff={float(learning.get('m7_strategy_effectiveness_pct', 0.0)):.1f}% "
        f"Xfer={float(transfer.get('transfer_effectiveness_pct', 0.0)):.1f}% "
        f"Opt={float(optimizer.get('optimizer_success_rate_pct', 0.0)):.1f}% "
        f"Prod={float(action.get('productive_action_rate_pct', 0.0)):.1f}% "
        f"step/L={_compact_number(efficiency.get('steps_per_level_advance'))} "
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
    # The dedicated reporter still owns lifecycle messages such as "sampling done",
    # but its old current_run_* competence line is replaced by the effectiveness line
    # emitted from the authoritative 60-second allocation snapshot above.
    if output_queue is None and "current_run_wins=" in str(message):
        return
    _BASE_REPORTER_EMIT_LINE(message, output_queue)


def _periodic_progress_line_v850(rows, total_steps, baseline=None) -> str:
    """Format live actor progress in the compact v8.50 form every interval."""
    from v8 import reporter

    values = tuple(rows)
    base = _BASE_PERIODIC_PROGRESS_LINE(values, total_steps, baseline)
    match = reporter._PROGRESS_PATTERN.search(base)
    if match is None:
        return base

    used_steps = sum(max(0, int(getattr(row, "steps", 0))) for row in values)
    planned_steps = sum(max(0, int(getattr(row, "planned_steps", 0))) for row in values)
    m7_share = _pct(planned_steps, used_steps)
    m7_games = [row for row in values if int(getattr(row, "planned_steps", 0)) > 0]
    m7_progress_games = sum(
        1
        for row in m7_games
        if int(getattr(row, "levels_completed", 0)) > 0
        or int(getattr(row, "wins", 0)) > 0
    )
    m7_effectiveness = _pct(m7_progress_games, len(m7_games))
    level_advances = sum(
        max(0, int(getattr(row, "levels_completed", 0))) for row in values
    )
    steps_per_level = (
        float(used_steps) / float(level_advances) if level_advances > 0 else None
    )
    return (
        f"{base[:match.start()]}effectiveness "
        f"L={float(match.group('levels')):.1f}% "
        f"G={float(match.group('wins')):.1f}% "
        f"M7={m7_share:.1f}% "
        f"M7eff={m7_effectiveness:.1f}% "
        "Xfer=- Opt=- Prod=- "
        f"step/L={_compact_number(steps_per_level)} firstWin=-"
    )


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

    # The independent reporter owns the exact one-minute stdout cadence. This
    # richer allocator snapshot remains in JSONL; emitting both produced duplicate,
    # contradictory terminal lines whenever the allocator view lagged actor progress.


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
    global _BASE_PERIODIC_PROGRESS_LINE
    global _BASE_RUN_ACTOR_JOBS
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import lease_dispatch_lifecycle_v843 as v843
    from v8 import reporter

    # Keep v8.43 as the public allocation-report authority. Insert this report under
    # that guard and above the v8.49 action-learning writer.
    _BASE_WRITE_ALLOCATION_LOG = v843._BASE_WRITE_ALLOCATION_LOG
    v843._BASE_WRITE_ALLOCATION_LOG = _write_allocation_log_v850

    # Preserve the reporter process and its completion semantics; suppress only the
    # old current_run_* periodic line.
    _BASE_REPORTER_EMIT_LINE = reporter._emit_line
    reporter._emit_line = _reporter_emit_line_v850
    _BASE_PERIODIC_PROGRESS_LINE = reporter.format_periodic_progress_line
    reporter.format_periodic_progress_line = _periodic_progress_line_v850

    # Capture the original requested interaction budget once at the final actor-job
    # authority so the compact effectiveness line can retain the old progress prefix.
    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    actor_module.run_actor_jobs = _run_actor_jobs_v850
    _INSTALLED = True
