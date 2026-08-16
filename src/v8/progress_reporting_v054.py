from __future__ import annotations

import queue
from dataclasses import dataclass

from v8 import primary_valence as _primary


_INSTALLED = False
_BASE_EXPERIENCE = None
_BASE_RESET_CAPTURE = None
_LOCAL_STEP = 0
_FIRST_WIN_STEP = 0


@dataclass(frozen=True, slots=True)
class ActorProgress:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    replans: int = 0
    planned_steps: int = 0
    first_win_step: int = 0


def _reset_progress_capture() -> None:
    global _LOCAL_STEP, _FIRST_WIN_STEP
    if _BASE_RESET_CAPTURE is not None:
        _BASE_RESET_CAPTURE()
    _LOCAL_STEP = 0
    _FIRST_WIN_STEP = 0


def _experience_with_progress(*args, **kwargs):
    global _LOCAL_STEP, _FIRST_WIN_STEP
    event = _BASE_EXPERIENCE(*args, **kwargs)
    if bool(getattr(_primary, "_CAPTURE_ACTIVE", False)):
        _LOCAL_STEP += 1
        if int(getattr(event, "terminal_polarity", 0)) > 0 and _FIRST_WIN_STEP <= 0:
            _FIRST_WIN_STEP = int(_LOCAL_STEP)
    return event


def _publish_progress(
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
    first_win_step = int(_FIRST_WIN_STEP)
    if int(wins) > 0 and first_win_step <= 0:
        first_win_step = int(steps)
    row = ActorProgress(
        int(job.actor_id),
        str(job.game_id),
        int(steps),
        int(wins),
        int(failures),
        int(levels_completed),
        int(replans),
        int(planned_steps),
        int(first_win_step),
    )
    for target in (progress_queue, reporting_queue):
        if target is None:
            continue
        try:
            target.put_nowait(row)
        except queue.Full:
            pass


def install_progress_reporting_v054() -> None:
    global _INSTALLED, _BASE_EXPERIENCE, _BASE_RESET_CAPTURE
    if _INSTALLED:
        return

    from v8 import actor as actor_module

    _BASE_EXPERIENCE = actor_module.ExperienceEvent
    _BASE_RESET_CAPTURE = _primary._reset_actor_capture

    _primary._reset_actor_capture = _reset_progress_capture
    actor_module.ExperienceEvent = _experience_with_progress
    actor_module.ActorProgress = ActorProgress
    actor_module._publish_progress = _publish_progress
    _INSTALLED = True
