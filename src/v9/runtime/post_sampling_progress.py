from __future__ import annotations

import time
from threading import Event, RLock, Thread, current_thread
from typing import Any

from .bounded_transfer_validation import run_transfer_validation_interval
from .progress import InlineProgress


_status: InlineProgress | None = None
_suppress_nested = False
_active = False
_current_runtime: Any | None = None
_progress_interval_seconds = 60.0
_phase_detail = ""
_phase_started = 0.0
_heartbeat_event: Event | None = None
_heartbeat_thread: Thread | None = None
_state_lock = RLock()


def _phase_state() -> tuple[str, float]:
    with _state_lock:
        detail = str(_phase_detail)
        started = float(_phase_started)
    elapsed = max(0.0, time.monotonic() - started) if detail and started else 0.0
    return detail, elapsed


def _heartbeat_loop(stop: Event) -> None:
    while not stop.wait(max(0.1, float(_progress_interval_seconds))):
        detail, elapsed = _phase_state()
        if not _active or not detail:
            continue
        with _state_lock:
            if _status is not None:
                _status.update(f"{detail} elapsed={elapsed:.0f}s")


def _start_heartbeat() -> None:
    global _heartbeat_event, _heartbeat_thread
    with _state_lock:
        if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
            return
        stop = Event()
        thread = Thread(
            target=_heartbeat_loop,
            args=(stop,),
            name="v9-post-sampling-progress",
            daemon=True,
        )
        _heartbeat_event = stop
        _heartbeat_thread = thread
        thread.start()


def _stop_heartbeat() -> None:
    global _heartbeat_event, _heartbeat_thread
    with _state_lock:
        stop = _heartbeat_event
        thread = _heartbeat_thread
        _heartbeat_event = None
        _heartbeat_thread = None
    if stop is not None:
        stop.set()
    if thread is not None and thread is not current_thread():
        thread.join(timeout=0.2)


def _phase(detail: str) -> None:
    global _status, _phase_detail, _phase_started
    if not _active:
        return
    now = time.monotonic()
    with _state_lock:
        if _status is None:
            _status = InlineProgress(f"{time.strftime('[%H:%M]')} post-sampling")
        if str(detail) != _phase_detail:
            _phase_detail = str(detail)
            _phase_started = now
        _status.update(f"{_phase_detail} elapsed={max(0.0, now - _phase_started):.0f}s")
    _start_heartbeat()


def _finish(detail: str = "complete") -> None:
    global _status, _phase_detail, _phase_started
    _stop_heartbeat()
    with _state_lock:
        if _status is not None:
            _status.finish(detail)
            _status = None
        _phase_detail = ""
        _phase_started = 0.0


def _dashboard_with_phase(snapshot: dict[str, Any], *, busy: bool = False) -> dict[str, Any]:
    result = dict(snapshot)
    primary = dict(result.get("primary_dashboard", {}) or {})
    detail, elapsed = _phase_state()
    if detail:
        primary["post_sampling_phase"] = detail
        primary["post_sampling_elapsed_seconds"] = round(elapsed, 1)
        result["post_sampling_phase"] = detail
        result["post_sampling_elapsed_seconds"] = round(elapsed, 1)
    if busy:
        primary["dashboard_snapshot"] = "cached"
        result["dashboard_snapshot"] = "cached"
    result["primary_dashboard"] = primary
    return result


def install(epoch_runner_module: Any, runtime_cls: type) -> None:
    """Install live visibility for long post-sampling work and dashboard reads."""
    if getattr(epoch_runner_module, "_post_sampling_progress_installed", False):
        return
    epoch_runner_module._post_sampling_progress_installed = True

    # The epoch helper resolves this module-global at call time.
    epoch_runner_module.run_transfer_validation_interval = run_transfer_validation_interval

    original_run_epochs = epoch_runner_module.run_epochs

    def run_epochs_with_progress(*args: Any, **kwargs: Any):
        global _active, _current_runtime, _progress_interval_seconds
        previous_active = _active
        previous_runtime = _current_runtime
        previous_interval = _progress_interval_seconds
        runtime = args[0] if args else kwargs.get("runtime")
        run_args = args[2] if len(args) > 2 else kwargs.get("args")
        _current_runtime = runtime
        _progress_interval_seconds = max(
            0.1,
            float(getattr(run_args, "progress_interval_seconds", 60.0)),
        )
        _active = True
        try:
            return original_run_epochs(*args, **kwargs)
        finally:
            _finish()
            _active = previous_active
            _current_runtime = previous_runtime
            _progress_interval_seconds = previous_interval

    epoch_runner_module.run_epochs = run_epochs_with_progress

    original_train = epoch_runner_module.train_hgt_epoch

    def train_with_progress(*args: Any, **kwargs: Any):
        _phase("HGT training")
        result = original_train(*args, **kwargs)
        _phase(f"HGT trained model={getattr(result, 'model_version', 'unknown')}")
        return result

    epoch_runner_module.train_hgt_epoch = train_with_progress

    original_lifecycle = epoch_runner_module.run_lifecycle_maintenance

    def lifecycle_with_progress(*args: Any, **kwargs: Any):
        _phase("lifecycle/compaction")
        result = original_lifecycle(*args, **kwargs)
        _finish("lifecycle complete")
        return result

    epoch_runner_module.run_lifecycle_maintenance = lifecycle_with_progress

    original_wait = runtime_cls.wait_quiescent

    def wait_with_progress(self: Any, *args: Any, **kwargs: Any):
        if not _suppress_nested:
            _phase("quiescent/drain")
        return original_wait(self, *args, **kwargs)

    runtime_cls.wait_quiescent = wait_with_progress

    original_flush = runtime_cls.flush_deferred_memory_updates

    def flush_with_progress(self: Any, *args: Any, **kwargs: Any):
        if not _suppress_nested:
            _phase("memory flush")
        return original_flush(self, *args, **kwargs)

    runtime_cls.flush_deferred_memory_updates = flush_with_progress

    original_replay = runtime_cls.replay_once

    def replay_with_progress(self: Any, *args: Any, **kwargs: Any):
        _phase("replay")
        return original_replay(self, *args, **kwargs)

    runtime_cls.replay_once = replay_with_progress

    original_snapshot = runtime_cls.snapshot

    def snapshot_with_progress(self: Any, *args: Any, **kwargs: Any):
        global _suppress_nested
        _phase("snapshot")
        previous = _suppress_nested
        _suppress_nested = True
        try:
            return original_snapshot(self, *args, **kwargs)
        finally:
            _suppress_nested = previous
            _finish("snapshot complete")

    runtime_cls.snapshot = snapshot_with_progress

    original_dashboard = getattr(runtime_cls, "dashboard_metrics", None)
    if callable(original_dashboard):
        def dashboard_with_progress(self: Any) -> dict[str, Any]:
            # Long post-sampling operations can own the runtime RLock for minutes.
            # Dashboard reads must remain non-blocking and serve the last good cut.
            lock = getattr(self, "_lock", None)
            acquired = lock is None
            if lock is not None:
                try:
                    acquired = bool(lock.acquire(blocking=False))
                except TypeError:
                    acquired = bool(lock.acquire(False))
            if acquired:
                try:
                    snapshot = dict(original_dashboard(self))
                    snapshot = _dashboard_with_phase(snapshot)
                    self.__dict__["_post_sampling_dashboard_cache"] = snapshot
                    return snapshot
                finally:
                    if lock is not None:
                        lock.release()

            cached = self.__dict__.get("_post_sampling_dashboard_cache")
            if cached is None:
                cached = dict(getattr(self, "_metrics_cache", {}) or {})
                cached.setdefault("primary_dashboard", {})
            return _dashboard_with_phase(dict(cached), busy=True)

        runtime_cls.dashboard_metrics = dashboard_with_progress
