from __future__ import annotations

"""v8.43: keep lifecycle graph rebuilds out of serial actor lease dispatch."""

import threading


_INSTALLED = False
_BASE_LIFECYCLE_INDEX = None
_BASE_REFILL_IDLE_WORKERS = None
_BASE_CHOOSE_MODE = None
_DISPATCH = threading.local()


def _dispatch_lifecycle_index(read_view):
    if not bool(getattr(_DISPATCH, "active", False)):
        return _BASE_LIFECYCLE_INDEX(read_view)
    cache = getattr(_DISPATCH, "cache", None)
    if cache is None:
        cache = {}
        _DISPATCH.cache = cache
    key = id(read_view)
    if key not in cache:
        cache[key] = _BASE_LIFECYCLE_INDEX(read_view)
    return cache[key]


def _refill_idle_workers_v843(idle_workers: set[int], assign) -> tuple[int, ...]:
    prior_active = bool(getattr(_DISPATCH, "active", False))
    prior_cache = getattr(_DISPATCH, "cache", None)
    if not prior_active:
        _DISPATCH.active = True
        _DISPATCH.cache = {}
    try:
        return _BASE_REFILL_IDLE_WORKERS(idle_workers, assign)
    finally:
        if not prior_active:
            _DISPATCH.active = False
            _DISPATCH.cache = prior_cache


def _choose_mode_v843(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import lifecycle_competence_integration_v827 as lifecycle

    # The raw learning state is authoritative for genuinely unsolved games and is
    # cheap to read. Do not inspect the lifecycle/frontier graph before this test.
    raw_state = lifecycle._BASE_GAME_STATE(self, str(game_id))
    if raw_state == v819.GameLearningState.UNSOLVED:
        return v819.SamplingMode.DISCOVERY
    return _BASE_CHOOSE_MODE(self, game_id)


def install_lease_dispatch_lifecycle_v843() -> None:
    global _INSTALLED, _BASE_LIFECYCLE_INDEX, _BASE_REFILL_IDLE_WORKERS
    global _BASE_CHOOSE_MODE
    if _INSTALLED:
        return

    from v8 import adaptive_allocator_occupancy_v840 as occupancy
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import lifecycle_competence_integration_v827 as lifecycle

    _BASE_LIFECYCLE_INDEX = lifecycle._lifecycle_index
    lifecycle._lifecycle_index = _dispatch_lifecycle_index

    _BASE_REFILL_IDLE_WORKERS = occupancy._refill_idle_workers
    occupancy._refill_idle_workers = _refill_idle_workers_v843

    _BASE_CHOOSE_MODE = v819.AdaptiveLearningCoordinator.choose_mode
    v819.AdaptiveLearningCoordinator.choose_mode = _choose_mode_v843

    _INSTALLED = True
