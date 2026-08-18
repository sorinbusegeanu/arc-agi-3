from __future__ import annotations

import os


_INSTALLED = False
_BASE_UNSOLVED_LEASE_STEPS = None
_ALLOCATION_LEASE_ENV = "ARC_AGI3_V8_ALLOCATION_LEASE_STEPS"
_DEFAULT_INITIAL_BREADTH_STEPS = 4096


def _initial_breadth_lease_steps_v840(
    *,
    available: int,
    base_steps: int,
    initial_probe: bool,
    worker_count: int,
    game_count: int,
) -> int:
    """Use a bounded first breadth lease without capping later long attempts.

    The first pass across games uses the configured adaptive allocation quantum.
    After that pass, v8.26's episode-aligned authority remains the ceiling, so a
    game that needs a long uninterrupted trajectory still receives a full-length
    subsequent attempt rather than being permanently cut at the breadth quantum.
    """

    limit = int(
        _BASE_UNSOLVED_LEASE_STEPS(
            available=int(available),
            base_steps=int(base_steps),
            initial_probe=bool(initial_probe),
            worker_count=int(worker_count),
            game_count=int(game_count),
        )
    )
    if not bool(initial_probe):
        return max(1, limit)
    raw = os.environ.get(_ALLOCATION_LEASE_ENV)
    try:
        quantum = int(raw) if raw is not None and str(raw).strip() else _DEFAULT_INITIAL_BREADTH_STEPS
    except ValueError:
        quantum = _DEFAULT_INITIAL_BREADTH_STEPS
    quantum = max(1, int(quantum))
    return max(1, min(int(limit), quantum))


def install_adaptive_allocator_breadth_v840() -> None:
    global _INSTALLED, _BASE_UNSOLVED_LEASE_STEPS
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf

    _BASE_UNSOLVED_LEASE_STEPS = perf._v823_initial_unsolved_lease_steps
    perf._v823_initial_unsolved_lease_steps = _initial_breadth_lease_steps_v840
    _INSTALLED = True
