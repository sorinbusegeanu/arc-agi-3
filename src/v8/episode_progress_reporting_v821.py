from __future__ import annotations

import queue
from dataclasses import dataclass

from v8 import primary_valence as _primary
from v8 import progress_reporting_v054 as _progress_v054


_INSTALLED = False
_BASE_ENV_INIT = None
_BASE_ENV_STEP = None

_ACTIVE_GAME_ID: str | None = None
_MAX_LEVEL_REACHED: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class EpisodeActorProgress(_progress_v054.ActorProgress):
    """Actor progress plus deepest level reached in any single current-run episode."""

    # -1 identifies callers/tests constructing legacy-style progress rows without
    # a real environment-backed episode-depth metric.
    max_level_reached: int = -1


def _record_level_progress(levels_completed: int) -> None:
    game_id = _ACTIVE_GAME_ID
    if game_id is None:
        return
    level = max(0, int(levels_completed))
    prior = int(_MAX_LEVEL_REACHED.get(game_id, 0))
    if level > prior:
        _MAX_LEVEL_REACHED[game_id] = level


def _tracked_env_init(self, *args, **kwargs) -> None:
    global _ACTIVE_GAME_ID
    _BASE_ENV_INIT(self, *args, **kwargs)
    game_id = kwargs.get("game_id")
    if game_id is None:
        game_id = getattr(self, "game_id", None)
    if game_id is not None:
        _ACTIVE_GAME_ID = str(game_id)
        _MAX_LEVEL_REACHED.setdefault(_ACTIVE_GAME_ID, 0)
        _record_level_progress(int(getattr(self, "last_levels_completed", 0)))


def _tracked_env_step(self, action):
    result = _BASE_ENV_STEP(self, action)
    _record_level_progress(int(getattr(self, "last_levels_completed", 0)))
    return result


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

    game_id = str(job.game_id)
    row = EpisodeActorProgress(
        int(job.actor_id),
        game_id,
        int(steps),
        int(wins),
        int(failures),
        int(levels_completed),
        int(replans),
        int(planned_steps),
        int(first_win_step),
        int(_MAX_LEVEL_REACHED.get(game_id, -1)),
    )
    for target in (progress_queue, reporting_queue):
        if target is None:
            continue
        try:
            target.put_nowait(row)
        except queue.Full:
            pass


def install_episode_progress_reporting_v821() -> None:
    global _INSTALLED, _BASE_ENV_INIT, _BASE_ENV_STEP
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module

    _BASE_ENV_INIT = ArcGridEnvironment.__init__
    _BASE_ENV_STEP = ArcGridEnvironment.step
    actor_module.ActorProgress = EpisodeActorProgress
    actor_module._publish_progress = _publish_episode_progress
    ArcGridEnvironment.__init__ = _tracked_env_init
    ArcGridEnvironment.step = _tracked_env_step
    _INSTALLED = True
