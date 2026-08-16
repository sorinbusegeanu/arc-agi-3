from __future__ import annotations

from v8.model import CognitiveState


_INSTALLED = False


def _run_generation_lifecycle(supervisor, nodes) -> int:
    from v8 import final_save_lifecycle_v812 as base

    lifecycle = supervisor.lifecycle
    if not bool(getattr(lifecycle, "_v812_enforce_generation_sweep", False)):
        return 0
    global_window = max(0, int(supervisor.current_generation())) // int(
        base._LIFECYCLE_GENERATION_SPAN
    )
    active_window = int(getattr(lifecycle, "_v812_active_window", -1))
    last_completed = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    if active_window < 0:
        if global_window <= last_completed:
            return 0
        active_window = global_window
        lifecycle._v812_active_window = active_window
        lifecycle._v812_next_bucket = 0

    previous = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    delta = 1 if previous < 0 else max(1, active_window - previous)
    lifecycle._v812_window_delta = delta
    lifecycle._v812_sweep_mode = True
    evaluated = 0
    try:
        buckets_per_cycle = max(1, min(8, int(supervisor.candidate_budget) // 64))
        start = int(getattr(lifecycle, "_v812_next_bucket", 0))
        stop = min(int(base._LIFECYCLE_BUCKETS), start + buckets_per_cycle)
        for bucket in range(start, stop):
            for row in nodes:
                if (
                    (int(row.uid.hi) ^ int(row.uid.lo))
                    & (base._LIFECYCLE_BUCKETS - 1)
                ) != bucket:
                    continue
                evaluated += 1
                decision = lifecycle.decide(row)
                if decision is None:
                    continue
                freshness = f"lifecycle-window:{active_window}"
                if hasattr(supervisor, "_fresh") and not supervisor._fresh(
                    freshness, row.uid, active_window
                ):
                    continue
                supervisor._submit(
                    supervisor._existing_proposal(
                        row,
                        cognitive_state=int(decision.cognitive_state),
                        validation_state=int(decision.validation_state),
                    )
                )
        lifecycle._v812_next_bucket = stop
        if stop >= int(base._LIFECYCLE_BUCKETS):
            lifecycle._v812_last_completed_window = active_window
            lifecycle._v812_active_window = -1
            lifecycle._v812_next_bucket = 0
    finally:
        lifecycle._v812_sweep_mode = False
        lifecycle._v812_window_delta = 1
    return evaluated


def install_final_save_lifecycle_v812_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v8 import final_save_lifecycle_v812 as base

    base._run_generation_lifecycle = _run_generation_lifecycle
    _INSTALLED = True
