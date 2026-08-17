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


def install_progress_runtime_fix_v822() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module

    actor_module.ActorProgress = V822ActorProgress
    actor_module._publish_progress = _publish_progress_v822
    _INSTALLED = True
