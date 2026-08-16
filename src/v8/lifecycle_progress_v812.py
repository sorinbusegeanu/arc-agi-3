from __future__ import annotations

import time

from v8.model import CognitiveState


_INSTALLED = False
_BASE_RUN_GENERATION_LIFECYCLE = None


def _bucket(uid, bucket_count: int) -> int:
    return (int(uid.hi) ^ int(uid.lo)) & (int(bucket_count) - 1)


def _state_counts(nodes) -> tuple[int, int, int, int]:
    quarantined = retire_pending = retired = reactivated = 0
    for row in nodes:
        state = int(row.cognitive_state)
        if state == int(CognitiveState.QUARANTINED):
            quarantined += 1
        elif state == int(CognitiveState.RETIRE_PENDING):
            retire_pending += 1
        elif state == int(CognitiveState.RETIRED):
            retired += 1
        elif state == int(CognitiveState.REACTIVATED):
            reactivated += 1
    return quarantined, retire_pending, retired, reactivated


def _run_generation_lifecycle_with_progress(supervisor, nodes) -> int:
    from v8 import final_save_lifecycle_v812 as base

    runner = _BASE_RUN_GENERATION_LIFECYCLE
    if runner is None:
        return 0

    lifecycle = supervisor.lifecycle
    global_window = max(0, int(supervisor.current_generation())) // int(
        base._LIFECYCLE_GENERATION_SPAN
    )
    active_before = int(getattr(lifecycle, "_v812_active_window", -1))
    last_before = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    next_bucket_before = int(getattr(lifecycle, "_v812_next_bucket", 0))
    target_window = (
        active_before
        if active_before >= 0
        else global_window if global_window > last_before else -1
    )

    if target_window >= 0 and int(
        getattr(lifecycle, "_v812_progress_window", -1)
    ) != target_window:
        lifecycle._v812_progress_window = target_window
        lifecycle._v812_progress_evaluated = sum(
            1
            for row in nodes
            if _bucket(row.uid, base._LIFECYCLE_BUCKETS) < next_bucket_before
        )

    evaluated = int(runner(supervisor, nodes))
    if target_window >= 0:
        lifecycle._v812_progress_evaluated = int(
            getattr(lifecycle, "_v812_progress_evaluated", 0)
        ) + evaluated

    active_after = int(getattr(lifecycle, "_v812_active_window", -1))
    last_after = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    completed = (
        target_window >= 0
        and active_after < 0
        and last_after == target_window
        and last_after != last_before
    )
    if completed:
        quarantined, retire_pending, retired, reactivated = _state_counts(nodes)
        print(
            f"[{time.strftime('%H:%M')}] lifecycle window={target_window} complete "
            f"generation={int(supervisor.current_generation())} "
            f"evaluated={int(getattr(lifecycle, '_v812_progress_evaluated', 0))} "
            f"quarantined={quarantined} retire_pending={retire_pending} "
            f"retired={retired} reactivated={reactivated}",
            flush=True,
        )
        lifecycle._v812_progress_window = -1
        lifecycle._v812_progress_evaluated = 0

    return evaluated


def install_lifecycle_progress_v812() -> None:
    global _INSTALLED, _BASE_RUN_GENERATION_LIFECYCLE
    if _INSTALLED:
        return
    from v8 import final_save_lifecycle_v812 as base

    _BASE_RUN_GENERATION_LIFECYCLE = base._run_generation_lifecycle
    base._run_generation_lifecycle = _run_generation_lifecycle_with_progress
    _INSTALLED = True
