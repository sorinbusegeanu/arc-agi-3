from __future__ import annotations

"""v8.39: keep adaptive lease dispatch independent from graph-heavy maintenance."""

import queue
import threading
import time
from dataclasses import dataclass


_INSTALLED = False
_BASE_RECORD_ACTOR_RESULTS = None
_BASE_RETRY_DEFERRED = None
_BASE_RUN_ACTOR_JOBS = None
_BASE_RUNTIME_CLOSE = None

_DEFERRED_BATCH_SIZE = 4
_DEFERRED_INTERVAL_SECONDS = 0.50
_DRAIN_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class LeaseDispatchMetrics:
    feedback_submitted: int
    feedback_completed: int
    feedback_pending: int
    feedback_max_pending: int
    deferred_passes: int
    deferred_examined: int
    deferred_resolved: int
    deferred_pending: int
    deferred_last_seconds: float
    deferred_max_seconds: float


class _AsyncQueueWorker:
    """Single ordered maintenance worker; submit() never runs work inline."""

    def __init__(
        self,
        callback,
        *,
        name: str,
        error_queue=None,
        coalesce_pending: bool = False,
    ) -> None:
        self.callback = callback
        self.error_queue = error_queue
        self._coalesce_pending = bool(coalesce_pending)
        self._queue: queue.Queue[object | None] = queue.Queue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._max_pending = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def submit(self, value) -> None:
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"{self._thread.name} failed: {type(error).__name__}: {error}"
            ) from error
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self._thread.name} is closed")
            if self._coalesce_pending:
                while True:
                    try:
                        stale = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if stale is None:
                        self._queue.put_nowait(None)
                        self._queue.task_done()
                        break
                    self._queue.task_done()
            self._submitted += 1
            self._queue.put_nowait(value)
            self._max_pending = max(
                self._max_pending, int(self._queue.unfinished_tasks)
            )

    def _run(self) -> None:
        while True:
            value = self._queue.get()
            try:
                if value is None:
                    return
                self.callback(value)
                with self._lock:
                    self._completed += 1
            except BaseException as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                if self.error_queue is not None:
                    try:
                        self.error_queue.put_nowait(
                            (self._thread.name, type(exc).__name__, str(exc))
                        )
                    except BaseException:
                        pass
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = _DRAIN_TIMEOUT_SECONDS) -> None:
        # Queue's condition owns the authoritative unfinished-task count. Waiting
        # here avoids coupling maintenance cleanup to the actor scheduler's mocked
        # or otherwise externally controlled monotonic clock.
        with self._queue.all_tasks_done:
            if self._queue.unfinished_tasks:
                self._queue.all_tasks_done.wait_for(
                    lambda: self._queue.unfinished_tasks == 0,
                    timeout=max(0.01, float(timeout)),
                )
            pending = int(self._queue.unfinished_tasks)
        if pending:
            raise TimeoutError(
                f"{self._thread.name} did not drain (pending={pending})"
            )
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"{self._thread.name} failed: {type(error).__name__}: {error}"
            ) from error

    def close(self, timeout: float = _DRAIN_TIMEOUT_SECONDS) -> None:
        with self._lock:
            if self._closed:
                return
        self.flush(timeout)
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put_nowait(None)
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError(f"{self._thread.name} did not stop")

    def abort(self) -> None:
        """Stop accepting work without repeating a failed drain on error cleanup."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        self._queue.put_nowait(None)

    def metrics(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                int(self._submitted),
                int(self._completed),
                int(self._queue.unfinished_tasks),
                int(self._max_pending),
            )


class _DeferredRetryWorker:
    """Retry only a bounded deferred batch per cadence, outside dispatch."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._request = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._passes = self._examined = self._resolved = 0
        self._last_seconds = self._max_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run, name="v8-deferred-target-retry", daemon=True
        )
        self._thread.start()

    def request(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"v8 deferred retry failed: {type(error).__name__}: {error}"
            ) from error
        self._request.set()

    def _run(self) -> None:
        while not self._stop.wait(_DEFERRED_INTERVAL_SECONDS):
            if not self._request.is_set():
                continue
            self._request.clear()
            started = time.perf_counter()
            try:
                examined, resolved = _retry_deferred_batch(
                    self.runtime, limit=_DEFERRED_BATCH_SIZE
                )
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                try:
                    self.runtime._error_queue.put_nowait(
                        ("v8-deferred-target-retry", type(exc).__name__, str(exc))
                    )
                except BaseException:
                    pass
                return
            elapsed = time.perf_counter() - started
            with self._lock:
                self._passes += 1
                self._examined += examined
                self._resolved += resolved
                self._last_seconds = elapsed
                self._max_seconds = max(self._max_seconds, elapsed)
            if getattr(self.runtime, "_v819_deferred_sources", ()):
                self._request.set()

    def close(self, timeout: float = _DRAIN_TIMEOUT_SECONDS) -> None:
        self._stop.set()
        self._request.set()
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError("v8 deferred retry worker did not stop")
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"v8 deferred retry failed: {type(error).__name__}: {error}"
            ) from error

    def metrics(self) -> tuple[int, int, int, float, float]:
        with self._lock:
            return (
                self._passes,
                self._examined,
                self._resolved,
                self._last_seconds,
                self._max_seconds,
            )


def _feedback_worker(runtime) -> _AsyncQueueWorker:
    worker = getattr(runtime, "_v839_actor_feedback", None)
    if worker is None or bool(getattr(worker, "_closed", False)):
        worker = _AsyncQueueWorker(
            lambda rows: _BASE_RECORD_ACTOR_RESULTS(runtime, tuple(rows)),
            name="v8-actor-feedback",
            error_queue=getattr(runtime, "_error_queue", None),
        )
        runtime._v839_actor_feedback = worker
    return worker


def _deferred_worker(runtime) -> _DeferredRetryWorker:
    worker = getattr(runtime, "_v839_deferred_retry", None)
    if worker is None or not worker._thread.is_alive():
        worker = _DeferredRetryWorker(runtime)
        runtime._v839_deferred_retry = worker
    return worker


def _retry_deferred_batch(runtime, *, limit: int) -> tuple[int, int]:
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v818 as v818

    pending = getattr(runtime, "_v819_deferred_sources", None)
    if not isinstance(pending, list) or not pending:
        return 0, 0
    count = min(len(pending), max(1, int(limit)))
    batch = list(pending[:count])
    del pending[:count]

    unresolved = []
    processed = resolved = 0
    try:
        for candidate, result in batch:
            target_uid = v818._resolve_target_outcome(runtime, candidate, result)
            if target_uid.is_zero:
                unresolved.append((candidate, result))
            else:
                v819._publish_validated_source(runtime, candidate, result, target_uid)
                resolved += 1
            processed += 1
    except BaseException:
        pending.extend(unresolved)
        pending.extend(batch[processed:])
        raise
    pending.extend(unresolved)
    return len(batch), resolved


def _record_actor_results_v839(self, results) -> None:
    rows = tuple(results)
    if not rows:
        return
    if not bool(getattr(self, "_v839_sampling_active", False)):
        return _BASE_RECORD_ACTOR_RESULTS(self, rows)
    _feedback_worker(self).submit(rows)


def _retry_deferred_v839(runtime) -> None:
    if not bool(getattr(runtime, "_v839_sampling_active", False)):
        return _BASE_RETRY_DEFERRED(runtime)
    _deferred_worker(runtime).request()


def _run_actor_jobs_v839(runtime, jobs, **kwargs):
    prior = bool(getattr(runtime, "_v839_sampling_active", False))
    runtime._v839_sampling_active = True

    try:
        result = _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)
    finally:
        runtime._v839_sampling_active = prior

    reporting_queue = kwargs.get("reporting_queue")
    if reporting_queue is not None:
        from v8.reporter import SAMPLING_COMPLETE

        reporting_queue.put_nowait(SAMPLING_COMPLETE)
        runtime._v839_sampling_done_reported = True

    # Late adaptive allocator replacements bypass v8.9's actor wrapper, so make
    # the final-drain state authoritative here as well.  Peers must stay paused
    # after the feedback barrier: resuming them in the small gap before the CLI's
    # wait_quiescent() call can start another large graph cycle and refill every
    # canonical queue during shutdown.
    _request_final_peer_drain(runtime)

    feedback = getattr(runtime, "_v839_actor_feedback", None)
    if feedback is not None:
        peers = getattr(runtime, "peers", None)
        if peers is not None and not peers.wait_idle(_DRAIN_TIMEOUT_SECONDS):
            raise TimeoutError("v8 peers did not become idle before feedback drain")
        feedback.flush()

    deferred = getattr(runtime, "_v839_deferred_retry", None)
    if deferred is not None:
        deferred.close()
        runtime._v839_deferred_retry = None

    # Reporting/final save still sees all resolvable deferred validation evidence.
    _BASE_RETRY_DEFERRED(runtime)
    return result


def _request_final_peer_drain(runtime) -> None:
    """Stop autonomous semantic work once the actor interaction budget is spent."""
    runtime._sampling_complete = True
    optimizer = getattr(runtime, "_v814_trajectory_optimizer", None)
    if optimizer is not None:
        # Successful trajectories are already durable JSON inbox records. Stop
        # admitting that potentially unbounded backlog once actor sampling ends;
        # the optimizer still drains every item already admitted to RAM, while
        # residual inbox files remain available to the next continuous run.
        optimizer._v841_preserve_inbox_on_shutdown = True

    peers = getattr(runtime, "peers", None)
    if peers is not None:
        cancel = getattr(peers, "_v841_peer_cancel", None)
        if cancel is not None:
            cancel.set()
        peers.pause()


def _runtime_close_v839(self, *args, **kwargs):
    normal = bool(kwargs.get("normal", args[0] if args else True))
    feedback = getattr(self, "_v839_actor_feedback", None)
    if feedback is not None:
        if normal:
            feedback.close()
        else:
            abort = getattr(feedback, "abort", None)
            if callable(abort):
                abort()
        self._v839_actor_feedback = None
    deferred = getattr(self, "_v839_deferred_retry", None)
    if deferred is not None:
        deferred.close()
        self._v839_deferred_retry = None
    return _BASE_RUNTIME_CLOSE(self, *args, **kwargs)


def lease_dispatch_metrics(runtime) -> LeaseDispatchMetrics:
    feedback = getattr(runtime, "_v839_actor_feedback", None)
    f = (0, 0, 0, 0) if feedback is None else feedback.metrics()
    deferred = getattr(runtime, "_v839_deferred_retry", None)
    d = (0, 0, 0, 0.0, 0.0) if deferred is None else deferred.metrics()
    return LeaseDispatchMetrics(
        feedback_submitted=f[0],
        feedback_completed=f[1],
        feedback_pending=f[2],
        feedback_max_pending=f[3],
        deferred_passes=d[0],
        deferred_examined=d[1],
        deferred_resolved=d[2],
        deferred_pending=len(getattr(runtime, "_v819_deferred_sources", ())),
        deferred_last_seconds=d[3],
        deferred_max_seconds=d[4],
    )


def install_lease_dispatch_continuity_v839() -> None:
    global _INSTALLED
    global _BASE_RECORD_ACTOR_RESULTS, _BASE_RETRY_DEFERRED
    global _BASE_RUN_ACTOR_JOBS, _BASE_RUNTIME_CLOSE
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RECORD_ACTOR_RESULTS = V82ContinuousMemoryRuntime.record_actor_results
    _BASE_RETRY_DEFERRED = v819._retry_deferred_sources
    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    _BASE_RUNTIME_CLOSE = V82ContinuousMemoryRuntime.close

    V82ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v839
    v819._retry_deferred_sources = _retry_deferred_v839
    actor_module.run_actor_jobs = _run_actor_jobs_v839
    V82ContinuousMemoryRuntime.close = _runtime_close_v839
    _INSTALLED = True
