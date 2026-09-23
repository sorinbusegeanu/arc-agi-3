from __future__ import annotations

import time
from threading import Event, RLock, Thread, current_thread
from typing import Any

from .bounded_transfer_validation import run_transfer_validation_interval


_suppress_nested = False
_active = False
_current_runtime: Any | None = None
_progress_interval_seconds = 60.0
_phase_detail = ""
_phase_started = 0.0
_sampling_active = False
_sampling_started = 0.0
_sampling_label = "sampling"
_heartbeat_event: Event | None = None
_heartbeat_thread: Thread | None = None
_state_lock = RLock()


def _phase_state() -> tuple[str, float]:
    with _state_lock:
        detail = str(_phase_detail)
        started = float(_phase_started)
    elapsed = max(0.0, time.monotonic() - started) if detail and started else 0.0
    return detail, elapsed


def _set_run_phase(value: str, runtime: Any | None = None) -> None:
    selected = runtime if runtime is not None else _current_runtime
    if selected is None:
        return
    try:
        selected.set_telemetry_gauge("run_phase", str(value))
    except Exception:
        return


def _heartbeat_loop(stop: Event) -> None:
    while not stop.wait(max(0.1, float(_progress_interval_seconds))):
        runtime = _current_runtime
        with _state_lock:
            sampling_active = bool(_sampling_active)
            sampling_started = float(_sampling_started)
            sampling_label = str(_sampling_label)
        if sampling_active:
            elapsed = max(0.0, time.monotonic() - sampling_started)
            _set_run_phase(f"{sampling_label}/pipeline", runtime)
            print(
                f"{time.strftime('[%H:%M]')} {sampling_label}/pipeline still active elapsed={elapsed:.0f}s",
                flush=True,
            )
            continue
        detail, elapsed = _phase_state()
        if not _active or not detail:
            continue
        _set_run_phase(detail, runtime)
        print(
            f"{time.strftime('[%H:%M]')} post-sampling phase={detail} elapsed={elapsed:.0f}s",
            flush=True,
        )


def _start_heartbeat() -> None:
    global _heartbeat_event, _heartbeat_thread
    with _state_lock:
        if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
            return
        stop = Event()
        thread = Thread(
            target=_heartbeat_loop,
            args=(stop,),
            name="v9-live-progress",
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
    global _phase_detail, _phase_started
    if not _active:
        return
    now = time.monotonic()
    changed = False
    with _state_lock:
        if str(detail) != _phase_detail:
            _phase_detail = str(detail)
            _phase_started = now
            changed = True
        elapsed = max(0.0, now - _phase_started)
    _set_run_phase(str(detail))
    if changed:
        print(
            f"{time.strftime('[%H:%M]')} post-sampling phase={detail} elapsed={elapsed:.0f}s",
            flush=True,
        )


def _finish(detail: str = "complete") -> None:
    global _phase_detail, _phase_started
    with _state_lock:
        had_phase = bool(_phase_detail)
        _phase_detail = ""
        _phase_started = 0.0
    if had_phase:
        print(f"{time.strftime('[%H:%M]')} post-sampling {detail}", flush=True)


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
    """Install live visibility for sampling drains, post-sampling work, and dashboard reads."""
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
        _start_heartbeat()
        try:
            return original_run_epochs(*args, **kwargs)
        finally:
            _finish()
            _stop_heartbeat()
            _set_run_phase("complete", runtime)
            _active = previous_active
            _current_runtime = previous_runtime
            _progress_interval_seconds = previous_interval

    epoch_runner_module.run_epochs = run_epochs_with_progress

    original_parallel = getattr(epoch_runner_module, "run_parallel_memory_jobs", None)
    if callable(original_parallel):
        def parallel_with_progress(runtime: Any, jobs: Any, *args: Any, **kwargs: Any):
            global _sampling_active, _sampling_started, _sampling_label
            evaluation_only = bool(kwargs.get("evaluation_only", False))
            label = "evaluation" if evaluation_only else "sampling"
            requested = sum(int(row[2]) for row in jobs)
            with _state_lock:
                _sampling_active = True
                _sampling_started = time.monotonic()
                _sampling_label = label
            _set_run_phase(label, runtime)
            try:
                result = original_parallel(runtime, jobs, *args, **kwargs)
            finally:
                with _state_lock:
                    _sampling_active = False
            produced = sum(int(getattr(row, "steps", 0)) for row in result)
            pct = 100.0 * produced / requested if requested else 100.0
            print(
                f"{time.strftime('[%H:%M]')} {pct:5.1f}% sampled={produced}/{requested} "
                f"games_finished={len(result)}/{len(jobs)} {label} complete",
                flush=True,
            )
            if not evaluation_only:
                _set_run_phase("post-sampling", runtime)
            return result

        epoch_runner_module.run_parallel_memory_jobs = parallel_with_progress

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
            # Long sampling/post-sampling operations can own the runtime RLock for minutes.
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
                finally:
                    if lock is not None:
                        lock.release()
            else:
                cached = self.__dict__.get("_post_sampling_dashboard_cache")
                if cached is None:
                    cached = dict(getattr(self, "_metrics_cache", {}) or {})
                    cached.setdefault("primary_dashboard", {})
                snapshot = _dashboard_with_phase(dict(cached), busy=True)

            # These gauges are lock-independent and stay live even when the graph
            # snapshot itself must come from cache.
            try:
                diagnostic = dict(self.unified_telemetry.diagnostic_metrics())
            except Exception:
                diagnostic = {}
            primary = dict(snapshot.get("primary_dashboard", {}) or {})
            run_phase = diagnostic.get("run_phase")
            if run_phase is not None:
                primary["run_phase"] = str(run_phase)
                snapshot["run_phase"] = str(run_phase)
            for key in (
                "sampled_steps",
                "ingested_steps",
                "sampling_backlog",
                "publication_backlog",
                "canonical_ingest_backlog",
                "active_actor_processes",
                "pending_environment_jobs",
            ):
                if key in diagnostic:
                    primary[key] = diagnostic[key]
            snapshot["primary_dashboard"] = primary
            return snapshot

        runtime_cls.dashboard_metrics = dashboard_with_progress
