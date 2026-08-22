from __future__ import annotations

"""v8.50 dedicated learning-effectiveness reporting.

This layer is observational. It combines already-authoritative adaptive allocation,
M7 planning/frontier, actor progress, optimizer, transfer, and v8.49 action-quality
telemetry into one per-game time series. It does not change H01-H15 decisions and
does not claim causal lift without a controlled ablation run.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path


_INSTALLED = False
_BASE_WRITE_ALLOCATION_LOG = None
_REPORT_FILE = "learning_effectiveness.log"
_SCHEMA_VERSION = 1
_CAUSAL_ATTRIBUTION = "OBSERVATIONAL_NOT_ABLATED"

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


def _action_metrics(game_id: str) -> dict[str, object]:
    from v8 import action_learning_report_v849 as action_report

    raw = action_report._RUN.get(str(game_id), {})
    click_actions = max(0, int(raw.get("click_actions_executed", 0)))
    movement_actions = max(0, int(raw.get("movement_actions_executed", 0)))
    click_productive = max(0, int(raw.get("click_productive", 0)))
    movement_productive = max(0, int(raw.get("movement_productive", 0)))
    action_count = click_actions + movement_actions
    productive = click_productive + movement_productive
    return {
        "observed_actions": action_count,
        "productive_actions": productive,
        "productive_action_rate": (
            float(productive) / float(action_count) if action_count > 0 else 0.0
        ),
        "click_noops": max(0, int(raw.get("click_noops", 0))),
    }


def _effectiveness_status(*, planned_steps: int, level_advances: int, wins: int) -> str:
    progress = int(level_advances) > 0 or int(wins) > 0
    if int(planned_steps) <= 0:
        return "PROGRESS_WITHOUT_M7_PLAN" if progress else "NO_M7_PLAN_APPLIED"
    return "M7_APPLIED_WITH_CURRENT_RUN_PROGRESS" if progress else "M7_APPLIED_NO_CURRENT_RUN_PROGRESS"


def _telemetry_by_game(runtime, coordinator) -> dict[str, dict[str, object]]:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    return {
        str(row.game_id): asdict(row)
        for row in coordinator.telemetry(optimizer_service=service)
    }


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
    total_steps = sum(int(outcomes.get(game, {}).get("steps", 0)) for game in games)

    rows: list[dict[str, object]] = []
    for game in games:
        outcome = outcomes.get(game, _empty_outcome())
        learning = telemetry.get(game, {})
        steps = max(0, int(outcome.get("steps", 0)))
        planned_steps = max(0, int(outcome.get("planned_steps", 0)))
        level_advances = max(0, int(outcome.get("levels_completed", 0)))
        wins = max(0, int(outcome.get("wins", 0)))
        failures = max(0, int(outcome.get("failures", 0)))
        known_levels = _known_levels_solved(coordinator, game)
        frontier_source = str(learning.get("frontier_source", ""))
        transfer_frontier = frontier_source == "TRANSFER"
        current_progress = level_advances > 0 or wins > 0
        action = _action_metrics(game)

        rows.append(
            {
                "game_id": game,
                "learning_state": str(learning.get("state", "UNKNOWN")),
                "effectiveness_status": _effectiveness_status(
                    planned_steps=planned_steps,
                    level_advances=level_advances,
                    wins=wins,
                ),
                "steps": steps,
                "sample_share": float(steps) / float(max(1, total_steps)),
                "known_levels_solved": known_levels,
                "current_run_level_advances": level_advances,
                "current_run_wins": wins,
                "current_run_failures": failures,
                "resets": max(0, int(outcome.get("resets", 0))),
                "first_win_step": max(0, int(outcome.get("first_win_step", 0))),
                "steps_per_level_advance": (
                    float(steps) / float(level_advances) if level_advances > 0 else None
                ),
                "planned_steps": planned_steps,
                "planned_step_share": (
                    float(planned_steps) / float(steps) if steps > 0 else 0.0
                ),
                "replans": max(0, int(outcome.get("replans", 0))),
                "frontier_source": frontier_source,
                "frontier_cost": max(0, int(learning.get("frontier_cost", 0))),
                "frontier_reliability": max(
                    0.0, min(1.0, float(learning.get("frontier_reliability", 0.0)))
                ),
                "frontier_version": max(0, int(learning.get("frontier_version", 0))),
                "sampling_mode": str(learning.get("sampling_mode", "")),
                "alternative_attempts": max(0, int(learning.get("alternative_attempts", 0))),
                "transfer_attempts": max(0, int(learning.get("transfer_attempts", 0))),
                "transfer_frontier": transfer_frontier,
                "transfer_frontier_with_current_run_progress": bool(
                    transfer_frontier and current_progress
                ),
                "optimizer_candidates": max(0, int(learning.get("optimizer_candidates", 0))),
                "optimizer_validations": max(0, int(learning.get("optimizer_validations", 0))),
                "optimizer_successes": max(0, int(learning.get("optimizer_successes", 0))),
                "optimizer_saved_actions": max(0, int(learning.get("optimizer_saved_actions", 0))),
                "optimizer_active": bool(learning.get("optimizer_active", False)),
                **action,
                "causal_attribution": _CAUSAL_ATTRIBUTION,
            }
        )

    total_planned = sum(int(row["planned_steps"]) for row in rows)
    total_advances = sum(int(row["current_run_level_advances"]) for row in rows)
    total_wins = sum(int(row["current_run_wins"]) for row in rows)
    observed_actions = sum(int(row["observed_actions"]) for row in rows)
    productive_actions = sum(int(row["productive_actions"]) for row in rows)
    optimizer_validations = sum(int(row["optimizer_validations"]) for row in rows)
    optimizer_successes = sum(int(row["optimizer_successes"]) for row in rows)
    m7_applied = [row for row in rows if int(row["planned_steps"]) > 0]
    m7_progress = [
        row
        for row in m7_applied
        if int(row["current_run_level_advances"]) > 0 or int(row["current_run_wins"]) > 0
    ]
    transfer_frontier_rows = [row for row in rows if bool(row["transfer_frontier"])]
    transfer_progress = [
        row for row in transfer_frontier_rows if bool(row["transfer_frontier_with_current_run_progress"])
    ]

    summary = {
        "games": len(rows),
        "sample_steps": total_steps,
        "known_levels_solved": sum(int(row["known_levels_solved"]) for row in rows),
        "known_levels_total": 5 * len(rows),
        "current_run_level_advances": total_advances,
        "current_run_wins": total_wins,
        "current_run_games_with_win": sum(1 for row in rows if int(row["current_run_wins"]) > 0),
        "planned_steps": total_planned,
        "planned_step_share": float(total_planned) / float(total_steps) if total_steps > 0 else 0.0,
        "games_with_m7_plan_applied": len(m7_applied),
        "games_with_m7_plan_and_current_run_progress": len(m7_progress),
        "games_with_m7_plan_without_current_run_progress": len(m7_applied) - len(m7_progress),
        "transfer_attempts": sum(int(row["transfer_attempts"]) for row in rows),
        "games_with_transfer_frontier": len(transfer_frontier_rows),
        "games_with_transfer_frontier_and_current_run_progress": len(transfer_progress),
        "optimizer_validations": optimizer_validations,
        "optimizer_successes": optimizer_successes,
        "optimizer_success_rate": (
            float(optimizer_successes) / float(optimizer_validations)
            if optimizer_validations > 0
            else 0.0
        ),
        "optimizer_saved_actions": sum(int(row["optimizer_saved_actions"]) for row in rows),
        "productive_action_rate": (
            float(productive_actions) / float(observed_actions) if observed_actions > 0 else 0.0
        ),
        "causal_attribution": _CAUSAL_ATTRIBUTION,
        "causal_ablation_executed": False,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "time": time.time(),
        "generation": int(getattr(runtime, "generation", 0)),
        "watermark": int(getattr(runtime, "watermark", 0)),
        "summary": summary,
        "games": rows,
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
