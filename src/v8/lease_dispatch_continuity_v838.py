from __future__ import annotations

"""v8.38 lease-dispatch continuity.

The adaptive allocator is the single authority that consumes actor completion
events and assigns replacement leases.  Graph-heavy learning assimilation and
deferred target reconciliation must never run inline on that dispatch path.

This layer keeps scientific semantics unchanged while moving those two
maintenance workloads to dedicated parent-process threads:

* Actor learning batches are queued for ordered asynchronous assimilation.
* Deferred optimizer target resolution is retried in bounded background batches.
* Periodic progress/experiment callbacks run outside the lease dispatcher.
* The final actor-run wrapper drains learning feedback and performs one canonical
  deferred retry after sampling, so reports/final snapshots still see completed
  feedback.

No interaction credits, allocation weights, memory schemas or validation gates
are changed.
"""

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
_FEEDBACK_FLUSH_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class LeaseDispatchContinuityMetrics:
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


class _ActorFeedbackService:
    """Serialize actor feedback away from the lease-dispatch thread."""

    def __init__(self, runtime, consume) -> None:
        self.runtime = runtime
        self.consume = consume
        self._queue: queue.Queue[tuple[object, ...] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._max_pending = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="v8-actor-feedback",
            daemon=True,
        )
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def submit(self, rows) -> None:
        values = tuple(rows)
        if not values:
            return
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"v8 actor feedback worker failed: {type(error).__name__}: {error}"
            ) from error
        with self._lock:
            if self._closed:
                raise RuntimeError("v8 actor feedback service is closed")
            self._submitted += 1
        self._queue.put_nowait(values)
        with self._lock:
            self._max_pending = max(
                self._max_pending,
                int(self._queue.unfinished_tasks),
            )

    def _run(self) -> None:
        while True:
            rows = self._queue.get()
            try:
                if rows is None:
                    return
                self.consume(self.runtime, rows)
                with self._lock:
                    self._completed += 1
            except BaseException as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                try:
                    self.runtime._error_queue.put_nowait(
                        (
                            "v8-actor-feedback",
                            type(exc).__name__,
                            str(exc),
                        )
                    )
                except BaseException:
                    pass
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = _FEEDBACK_FLUSH_SECONDS) -> None:
        deadline = time.monotonic() + max(0.01, float(timeout))
        while int(self._queue.unfinished_tasks) > 0:
            error = self.error
            if error is not None:
                raise RuntimeError(
                    f"v8 actor feedback worker failed: {type(error).__name__}: {error}"
                ) from error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "v8 actor feedback did not drain before timeout "
                    f"(pending={int(self._queue.unfinished_tasks)})"
                )
            time.sleep(0.01)

    def close(self, timeout: float = _FEEDBACK_FLUSH_SECONDS) -> None:
        with self._lock:
            if self._closed:
                return
        self.flush(timeout=timeout)
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put_nowait(None)
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError("v8 actor feedback worker did not stop")

    def metrics(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                int(self._submitted),
                int(self._completed),
                int(self._queue.unfinished_tasks),
                int(self._max_pending),
            )


class _ProgressCallbackService:
    """Run periodic maintenance callbacks without occupying lease dispatch."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self._queue: queue.Queue[tuple[object, ...] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="v8-progress-maintenance",
            daemon=True,
        )
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def submit(self, rows) -> None:
        values = tuple(rows)
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"v8 progress maintenance failed: {type(error).__name__}: {error}"
            ) from error
        self._queue.put_nowait(values)

    def _run(self) -> None:
        while True:
            rows = self._queue.get()
            try:
                if rows is None:
                    return
                self.callback(rows)
            except BaseException as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = _FEEDBACK_FLUSH_SECONDS) -> None:
        deadline = time.monotonic() + max(0.01, float(timeout))
        while int(self._queue.unfinished_tasks) > 0:
            error = self.error
            if error is not None:
                raise RuntimeError(
                    f"v8 progress maintenance failed: {type(error).__name__}: {error}"
                ) from error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "v8 progress maintenance did not drain before timeout "
                    f"(pending={int(self._queue.unfinished_tasks)})"
                )
            time.sleep(0.01)

    def close(self, timeout: float = _FEEDBACK_FLUSH_SECONDS) -> None:
        with self._lock:
            if self._closed:
                return
        self.flush(timeout=timeout)
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put_nowait(None)
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError("v8 progress maintenance worker did not stop")


class _DeferredRetryService:
    """Bound deferred graph reconciliation so it cannot starve lease dispatch."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._request = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._passes = 0
        self._examined = 0
        self._resolved = 0
        self._last_seconds = 0.0
        self._max_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="v8-deferred-target-retry",
            daemon=True,
        )
        self._thread.start()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def request(self) -> None:
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"v8 deferred retry worker failed: {type(error).__name__}: {error}"
            ) from error
        self._request.set()

    def _run(self) -> None:
        while not self._stop.wait(_DEFERRED_INTERVAL_SECONDS):
            if not self._request.is_set():
                continue
            self._request.clear()
            started = time.monotonic()
            try:
                examined, resolved = _retry_deferred_batch(
                    self.runtime,
                    limit=_DEFERRED_BATCH_SIZE,
                )
                elapsed = time.monotonic() - started
                with self._lock:
                    self._passes += 1
                    self._examined += int(examined)
                    self._resolved += int(resolved)
                    self._last_seconds = float(elapsed)
                    self._max_seconds = max(self._max_seconds, float(elapsed))
                if getattr(self.runtime, "_v819_deferred_sources", ()):
                    # Keep making bounded progress even if no new callback arrives.
                    self._request.set()
            except BaseException as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                try:
                    self.runtime._error_queue.put_nowait(
                        (
                            "v8-deferred-target-retry",
                            type(exc).__name__,
                            str(exc),
                        )
                    )
                except BaseException:
                    pass
                return

    def close(self, timeout: float = _FEEDBACK_FLUSH_SECONDS) -> None:
        self._stop.set()
        self._request.set()
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError("v8 deferred retry worker did not stop")
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"v8 deferred retry worker failed: {type(error).__name__}: {error}"
            ) from error

    def metrics(self) -> tuple[int, int, int, float, float]:
        with self._lock:
            return (
                int(self._passes),
                int(self._examined),
                int(self._resolved),
                float(self._last_seconds),
                float(self._max_seconds),
            )


def _feedback_service(runtime) -> _ActorFeedbackService:
    service = getattr(runtime, "_v838_actor_feedback", None)
    if service is None or bool(getattr(service, "_closed", False)):
        service = _ActorFeedbackService(runtime, _BASE_RECORD_ACTOR_RESULTS)
        runtime._v838_actor_feedback = service
    return service


def _deferred_service(runtime) -> _DeferredRetryService:
    service = getattr(runtime, "_v838_deferred_retry", None)
    if service is None or not bool(
        getattr(service, "_thread", None) and service._thread.is_alive()
    ):
        service = _DeferredRetryService(runtime)
        runtime._v838_deferred_retry = service
    return service


def _retry_deferred_batch(runtime, *, limit: int) -> tuple[int, int]:
    """Resolve at most ``limit`` deferred sources from one live graph cut."""

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v818 as v818

    pending = getattr(runtime, "_v819_deferred_sources", None)
    if not isinstance(pending, list) or not pending:
        return 0, 0

    count = min(len(pending), max(1, int(limit)))
    batch = list(pending[:count])
    del pending[:count]

    unresolved: list[tuple[object, object]] = []
    resolved = 0
    try:
        for candidate, result in batch:
            target_uid = v818._resolve_target_outcome(runtime, candidate, result)
            if target_uid.is_zero:
                unresolved.append((candidate, result))
                continue
            v819._publish_validated_source(runtime, candidate, result, target_uid)
            resolved += 1
    except BaseException:
        # Preserve the failed item plus anything not yet examined.
        current = len(unresolved) + resolved
        pending.extend(unresolved)
        pending.extend(batch[current:])
        raise

    pending.extend(unresolved)
    return len(batch), int(resolved)


def _record_actor_results_v838(self, results) -> None:
    rows = tuple(results)
    if not rows:
        return
    if not bool(getattr(self, "_v838_sampling_active", False)):
        return _BASE_RECORD_ACTOR_RESULTS(self, rows)
    _feedback_service(self).submit(rows)


def _retry_deferred_v838(runtime) -> None:
    if not bool(getattr(runtime, "_v838_sampling_active", False)):
        return _BASE_RETRY_DEFERRED(runtime)
    _deferred_service(runtime).request()


def _run_actor_jobs_v838(runtime, jobs, **kwargs):
    prior = bool(getattr(runtime, "_v838_sampling_active", False))
    runtime._v838_sampling_active = True

    callback = kwargs.get("progress_callback")
    callback_service = None
    call_kwargs = kwargs
    if callback is not None:
        callback_service = _ProgressCallbackService(callback)
        call_kwargs = dict(kwargs)
        call_kwargs["progress_callback"] = callback_service.submit

    try:
        result = _BASE_RUN_ACTOR_JOBS(runtime, jobs, **call_kwargs)
    except BaseException:
        # Do not let maintenance cleanup replace the actor-run exception.
        if callback_service is not None:
            try:
                callback_service.close(timeout=_FEEDBACK_FLUSH_SECONDS)
            except BaseException:
                pass
        raise
    finally:
        runtime._v838_sampling_active = prior

    feedback = getattr(runtime, "_v838_actor_feedback", None)
    if feedback is not None:
        feedback.flush(timeout=_FEEDBACK_FLUSH_SECONDS)

    if callback_service is not None:
        callback_service.close(timeout=_FEEDBACK_FLUSH_SECONDS)

    deferred = getattr(runtime, "_v838_deferred_retry", None)
    if deferred is not None:
        deferred.close(timeout=_FEEDBACK_FLUSH_SECONDS)
        runtime._v838_deferred_retry = None

    # Sampling is finished, so one complete canonical retry can safely run inline
    # before reports/final snapshots are produced.
    _BASE_RETRY_DEFERRED(runtime)
    return result


def _runtime_close_v838(self, *args, **kwargs):
    feedback = getattr(self, "_v838_actor_feedback", None)
    if feedback is not None:
        feedback.close(timeout=_FEEDBACK_FLUSH_SECONDS)
        self._v838_actor_feedback = None
    deferred = getattr(self, "_v838_deferred_retry", None)
    if deferred is not None:
        deferred.close(timeout=_FEEDBACK_FLUSH_SECONDS)
        self._v838_deferred_retry = None
    return _BASE_RUNTIME_CLOSE(self, *args, **kwargs)


def lease_dispatch_continuity_metrics(runtime) -> LeaseDispatchContinuityMetrics:
    feedback = getattr(runtime, "_v838_actor_feedback", None)
    if feedback is None:
        feedback_values = (0, 0, 0, 0)
    else:
        feedback_values = feedback.metrics()

    deferred = getattr(runtime, "_v838_deferred_retry", None)
    if deferred is None:
        deferred_values = (0, 0, 0, 0.0, 0.0)
    else:
        deferred_values = deferred.metrics()

    pending = getattr(runtime, "_v819_deferred_sources", ())
    return LeaseDispatchContinuityMetrics(
        feedback_submitted=int(feedback_values[0]),
        feedback_completed=int(feedback_values[1]),
        feedback_pending=int(feedback_values[2]),
        feedback_max_pending=int(feedback_values[3]),
        deferred_passes=int(deferred_values[0]),
        deferred_examined=int(deferred_values[1]),
        deferred_resolved=int(deferred_values[2]),
        deferred_pending=len(pending),
        deferred_last_seconds=float(deferred_values[3]),
        deferred_max_seconds=float(deferred_values[4]),
    )


def install_lease_dispatch_continuity_v838() -> None:
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

    V82ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v838
    v819._retry_deferred_sources = _retry_deferred_v838
    actor_module.run_actor_jobs = _run_actor_jobs_v838
    V82ContinuousMemoryRuntime.close = _runtime_close_v838

    _INSTALLED = True
