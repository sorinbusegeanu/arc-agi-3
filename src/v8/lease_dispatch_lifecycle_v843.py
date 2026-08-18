from __future__ import annotations

"""v8.43: keep lifecycle graph rebuilds out of serial actor lease dispatch."""

import threading
import weakref
from contextlib import contextmanager


_INSTALLED = False
_BASE_LIFECYCLE_INDEX = None
_BASE_REFILL_IDLE_WORKERS = None
_BASE_CHOOSE_MODE = None
_BASE_WRITE_ALLOCATION_LOG = None
_BASE_ALLOCATION_STDOUT = None
_DISPATCH = threading.local()
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE = weakref.WeakKeyDictionary()


@contextmanager
def _dispatch_lifecycle_cache():
    prior_active = bool(getattr(_DISPATCH, "active", False))
    prior_cache = getattr(_DISPATCH, "cache", None)
    if not prior_active:
        _DISPATCH.active = True
        _DISPATCH.cache = {}
    try:
        yield
    finally:
        if not prior_active:
            _DISPATCH.active = False
            _DISPATCH.cache = prior_cache


def _cached_lifecycle_index(read_view):
    if read_view is None:
        return None
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(read_view)
    if cached is not None:
        return cached
    existing = getattr(read_view, "_node_by_uid", None)
    if existing:
        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE[read_view] = existing
        return existing
    return None


def _dispatch_lifecycle_index(read_view):
    if not bool(getattr(_DISPATCH, "active", False)):
        index = _BASE_LIFECYCLE_INDEX(read_view)
        if read_view is not None and index is not None:
            with _INDEX_CACHE_LOCK:
                _INDEX_CACHE[read_view] = index
        return index
    cache = getattr(_DISPATCH, "cache", None)
    if cache is None:
        cache = {}
        _DISPATCH.cache = cache
    key = id(read_view)
    if key not in cache:
        # Lease dispatch is latency-sensitive. Never rebuild the restored graph
        # here; use the latest index published by lifecycle/validator work. An
        # absent index yields UNKNOWN and preserves the raw learning state until
        # a non-dispatch lifecycle refresh publishes one.
        cache[key] = _cached_lifecycle_index(read_view)
    return cache[key]


def _refill_idle_workers_v843(idle_workers: set[int], assign) -> tuple[int, ...]:
    with _dispatch_lifecycle_cache():
        return _BASE_REFILL_IDLE_WORKERS(idle_workers, assign)


def _write_allocation_log_v843(*args, **kwargs):
    with _dispatch_lifecycle_cache():
        return _BASE_WRITE_ALLOCATION_LOG(*args, **kwargs)


def _allocation_stdout_v843(*args, **kwargs):
    with _dispatch_lifecycle_cache():
        return _BASE_ALLOCATION_STDOUT(*args, **kwargs)


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
    global _BASE_CHOOSE_MODE, _BASE_WRITE_ALLOCATION_LOG, _BASE_ALLOCATION_STDOUT
    if _INSTALLED:
        return

    from v8 import adaptive_allocator_occupancy_v840 as occupancy
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import lifecycle_competence_integration_v827 as lifecycle

    _BASE_LIFECYCLE_INDEX = lifecycle._lifecycle_index
    lifecycle._lifecycle_index = _dispatch_lifecycle_index

    _BASE_REFILL_IDLE_WORKERS = occupancy._refill_idle_workers
    occupancy._refill_idle_workers = _refill_idle_workers_v843

    _BASE_CHOOSE_MODE = v819.AdaptiveLearningCoordinator.choose_mode
    v819.AdaptiveLearningCoordinator.choose_mode = _choose_mode_v843

    _BASE_WRITE_ALLOCATION_LOG = perf._write_allocation_log_live
    _BASE_ALLOCATION_STDOUT = perf._allocation_stdout_live
    perf._write_allocation_log_live = _write_allocation_log_v843
    perf._allocation_stdout_live = _allocation_stdout_v843

    _INSTALLED = True
