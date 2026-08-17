from __future__ import annotations

import queue
from dataclasses import dataclass

from v8.episode_progress_reporting_v821 import EpisodeActorProgress


_INSTALLED = False
_FIRST_WIN_AT: dict[tuple[int, str], int] = {}
_LAST_STEPS: dict[tuple[int, str], int] = {}
_LAST_WINS: dict[tuple[int, str], int] = {}


@dataclass(frozen=True, slots=True)
class V822ActorProgress(EpisodeActorProgress):
    """Run progress with both episode depth and true winning-path action counts."""

    best_win_steps: int = 0
    last_win_steps: int = 0


def _publish_progress_v822(
    progress_queue,
    reporting_queue=None,
    *,
    job,
    steps: int,
    wins: int,
    failures: int,
    levels_completed: int,
    replans: int,
    planned_steps: int,
) -> None:
    from v8 import episode_progress_reporting_v821 as episode
    from v8 import learning_fixes_v088 as learning

    key = (int(job.actor_id), str(job.game_id))
    current_steps = int(steps)
    current_wins = int(wins)
    prior_steps = _LAST_STEPS.get(key)
    prior_wins = _LAST_WINS.get(key)
    if (
        (prior_steps is not None and current_steps < int(prior_steps))
        or (prior_wins is not None and current_wins < int(prior_wins))
    ):
        _FIRST_WIN_AT.pop(key, None)
    if current_wins > 0 and key not in _FIRST_WIN_AT:
        _FIRST_WIN_AT[key] = current_steps
    _LAST_STEPS[key] = current_steps
    _LAST_WINS[key] = current_wins

    row = V822ActorProgress(
        int(job.actor_id),
        str(job.game_id),
        current_steps,
        current_wins,
        int(failures),
        int(levels_completed),
        int(replans),
        int(planned_steps),
        int(_FIRST_WIN_AT.get(key, 0)),
        int(episode._MAX_LEVEL_REACHED.get(str(job.game_id), -1)),
        int(getattr(learning, "_BEST_WIN_STEPS", 0)),
        int(getattr(learning, "_LAST_WIN_STEPS", 0)),
    )
    for target in (progress_queue, reporting_queue):
        if target is None:
            continue
        try:
            target.put_nowait(row)
        except queue.Full:
            pass


def _format_game_rate_line_v822(rows) -> str:
    from v8 import diagnostics

    rows = tuple(rows)
    win_rate, level_rate, solved_games, games = diagnostics.game_summary(rows)
    grouped = diagnostics._group_games(rows)
    details = []
    for game_id, lane_rows in sorted(grouped.items()):
        solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
        if not solved_rows:
            continue
        best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
        best = min((value for value in best_values if value > 0), default=0)
        # Actor-local step counters cannot establish temporal ordering between
        # concurrent lanes.  A cross-lane best is valid; a cross-lane "last" is not.
        last = 0
        if len(solved_rows) == 1:
            last = int(getattr(solved_rows[0], "last_win_steps", 0) or 0)
        if best > 0 and last > 0 and best != last:
            details.append(
                f"{game_id}:best_win_actions={best},last_win_actions={last}"
            )
        elif best > 0:
            details.append(f"{game_id}:best_win_actions={best}")
        else:
            details.append(f"{game_id}:win_observed")
    suffix = "" if not details else " (" + "; ".join(details) + ")"
    return (
        f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
        f"current_run_solved_games={solved_games}/{games}{suffix}"
    )


def install_progress_runtime_fix_v822() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import diagnostics

    actor_module.ActorProgress = V822ActorProgress
    actor_module._publish_progress = _publish_progress_v822
    diagnostics.format_game_rate_line = _format_game_rate_line_v822
    _INSTALLED = True
