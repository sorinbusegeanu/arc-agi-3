from __future__ import annotations

"""v8.50 dedicated learning-effectiveness summary reporting.

This layer is observational. It writes only high-level effectiveness metrics from
already-authoritative adaptive allocation, M7 planning/frontier, actor progress,
optimizer, transfer, and v8.49 action-quality telemetry. It does not emit per-game
details, change H01-H15 decisions, or claim causal lift without controlled ablation.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path


_INSTALLED = False
_BASE_WRITE_ALLOCATION_LOG = None
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
    known_levels_solved = sum(_known_levels_solved(coordinator, game) for game in games)
    known_games_solved = sum(
        1 for game in games if bool(getattr(coordinator, "_game_won", {}).get(game, False))
    )

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
            "level_solve_rate_pct": _pct(known_levels_solved, 5 * len(games)),
            "game_solve_rate_pct": _pct(known_games_solved, len(games)),
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
        return


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
    global _INSTALLED, _BASE_WRITE_ALLOCATION_LOG
    if _INSTALLED:
        return

    from v8 import lease_dispatch_lifecycle_v843 as v843

    # Keep v8.43 as the public allocation-report authority. Insert this report under
    # that guard and above the v8.49 action-learning writer.
    _BASE_WRITE_ALLOCATION_LOG = v843._BASE_WRITE_ALLOCATION_LOG
    v843._BASE_WRITE_ALLOCATION_LOG = _write_allocation_log_v850
    _INSTALLED = True
