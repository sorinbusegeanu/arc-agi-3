from __future__ import annotations

import queue
from dataclasses import dataclass

from v8 import primary_valence as _primary
from v8 import progress_reporting_v054 as _progress_v054


_INSTALLED = False
_BASE_ACTOR_WORKER = None
_BASE_ENV_STEP = None

_ACTIVE_PROGRESS_KEY: tuple[int, str] | None = None
_MAX_LEVEL_REACHED: dict[tuple[int, str], int] = {}


@dataclass(frozen=True, slots=True)
class EpisodeActorProgress(_progress_v054.ActorProgress):
    """Actor progress plus deepest level reached in any single current-run episode."""

    max_level_reached: int = 0


def _record_level_progress(levels_completed: int) -> None:
    key = _ACTIVE_PROGRESS_KEY
    if key is None:
        return
    level = max(0, int(levels_completed))
    prior = int(_MAX_LEVEL_REACHED.get(key, 0))
    if level > prior:
        _MAX_LEVEL_REACHED[key] = level


def _tracked_env_step(self, action):
    result = _BASE_ENV_STEP(self, action)
    _record_level_progress(int(getattr(self, "last_levels_completed", 0)))
    return result


def _tracked_actor_worker(*, job, **kwargs):
    global _ACTIVE_PROGRESS_KEY
    prior = _ACTIVE_PROGRESS_KEY
    key = (int(job.actor_id), str(job.game_id))
    _ACTIVE_PROGRESS_KEY = key
    _MAX_LEVEL_REACHED.setdefault(key, 0)
    try:
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    finally:
        _ACTIVE_PROGRESS_KEY = prior


def _publish_episode_progress(
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
    capture_active = bool(getattr(_primary, "_CAPTURE_ACTIVE", False))
    first_win_step = int(_progress_v054._FIRST_WIN_STEP) if capture_active else 0
    if capture_active and int(wins) > 0 and first_win_step <= 0:
        first_win_step = int(steps)

    key = (int(job.actor_id), str(job.game_id))
    row = EpisodeActorProgress(
        int(job.actor_id),
        str(job.game_id),
        int(steps),
        int(wins),
        int(failures),
        int(levels_completed),
        int(replans),
        int(planned_steps),
        int(first_win_step),
        int(_MAX_LEVEL_REACHED.get(key, 0)),
    )
    for target in (progress_queue, reporting_queue):
        if target is None:
            continue
        try:
            target.put_nowait(row)
        except queue.Full:
            pass


def install_episode_progress_reporting_v821() -> None:
    global _INSTALLED, _BASE_ACTOR_WORKER, _BASE_ENV_STEP
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module

    _BASE_ACTOR_WORKER = actor_module.actor_worker
    _BASE_ENV_STEP = ArcGridEnvironment.step
    actor_module.ActorProgress = EpisodeActorProgress
    actor_module.actor_worker = _tracked_actor_worker
    actor_module._publish_progress = _publish_episode_progress
    ArcGridEnvironment.step = _tracked_env_step
    _INSTALLED = True
