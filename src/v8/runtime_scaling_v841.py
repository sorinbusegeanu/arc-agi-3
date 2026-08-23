from __future__ import annotations

"""v8.41: bound maintenance state and remove remaining control-plane stalls."""

import pickle
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_INSTALLED = False
_DEFERRED_INTERVAL_SECONDS = 0.05
_DEFERRED_TIME_BUDGET_SECONDS = 0.025
_DEFERRED_CHUNK_SIZE = 1
_OPTIMIZER_OVERFLOW_PER_GAME = 256
_OPTIMIZER_DISPATCH_INTERVAL_SECONDS = 0.02

_BASE_PEER_INIT = None
_BASE_PEER_CLOSE = None
_BASE_PEER_RUN_ONCE = None
_BASE_SERVICE_STOP = None
_BASE_QUEUE_DEPTH = None
_BASE_INGEST_INBOX = None
_BASE_RUNTIME_IS_QUIESCENT = None
_BASE_RUNTIME_CLOSE = None


class _SqliteBatchWorker:
    """Bound actor-feedback RAM by buffering batches in a small SQLite queue."""

    def __init__(self, callback, *, name: str, root: str | Path, error_queue=None) -> None:
        self.callback = callback
        self.error_queue = error_queue
        self.path = Path(root) / "feedback_queue.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._wake = threading.Event()
        self._closed = False
        self._error: BaseException | None = None
        self._submitted = self._completed = self._max_pending = 0
        self._writer = sqlite3.connect(self.path, timeout=0.1, check_same_thread=False)
        self._writer.execute("PRAGMA journal_mode=WAL")
        self._writer.execute("PRAGMA synchronous=NORMAL")
        self._writer.execute(
            "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
        )
        # This is a runtime queue, not durable evidence. Never replay an interrupted
        # callback from a previous process because actor feedback is not idempotent.
        self._writer.execute("DELETE FROM feedback")
        self._writer.commit()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, value) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            if self._error is not None:
                error = self._error
                raise RuntimeError(
                    f"{self._thread.name} failed: {type(error).__name__}: {error}"
                ) from error
            if self._closed:
                raise RuntimeError(f"{self._thread.name} is closed")
            self._writer.execute("INSERT INTO feedback(payload) VALUES (?)", (payload,))
            self._writer.commit()
            self._submitted += 1
            self._max_pending = max(
                self._max_pending, self._submitted - self._completed
            )
        self._wake.set()

    def _run(self) -> None:
        reader = sqlite3.connect(self.path, timeout=0.1)
        reader.execute("PRAGMA journal_mode=WAL")
        try:
            while True:
                with self._lock:
                    if self._closed:
                        return
                rows = reader.execute(
                    "SELECT id, payload FROM feedback ORDER BY id LIMIT 32"
                ).fetchall()
                if not rows:
                    self._wake.clear()
                    self._wake.wait(0.05)
                    continue
                decoded = tuple(
                    item
                    for _row_id, payload in rows
                    for item in tuple(pickle.loads(payload))
                )
                ids = tuple(int(row_id) for row_id, _payload in rows)
                # Match v8.39 failure semantics: dequeue before callback. A failed
                # callback is surfaced to runtime and terminates the maintenance lane.
                placeholders = ",".join("?" for _ in ids)
                reader.execute(f"DELETE FROM feedback WHERE id IN ({placeholders})", ids)
                reader.commit()
                try:
                    self.callback(decoded)
                except BaseException as exc:
                    with self._done:
                        self._error = exc
                        self._done.notify_all()
                    if self.error_queue is not None:
                        try:
                            self.error_queue.put_nowait(
                                (self._thread.name, type(exc).__name__, str(exc))
                            )
                        except BaseException:
                            pass
                    return
                with self._done:
                    self._completed += len(rows)
                    self._done.notify_all()
        finally:
            reader.close()

    def flush(self, timeout: float = 300.0) -> None:
        with self._done:
            finished = self._done.wait_for(
                lambda: self._completed >= self._submitted or self._error is not None,
                timeout=max(0.01, float(timeout)),
            )
            pending = max(0, self._submitted - self._completed)
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"{self._thread.name} failed: {type(error).__name__}: {error}"
            ) from error
        if not finished or pending:
            raise TimeoutError(f"{self._thread.name} did not drain (pending={pending})")

    def close(self, timeout: float = 300.0) -> None:
        with self._lock:
            if self._closed:
                return
        self.flush(timeout)
        with self._lock:
            self._closed = True
        self._wake.set()
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError(f"{self._thread.name} did not stop")
        with self._lock:
            self._writer.close()

    def abort(self) -> None:
        """Release the writer promptly during an already-failing runtime close."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._writer.close()
        self._wake.set()

    def metrics(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                int(self._submitted),
                int(self._completed),
                max(0, int(self._submitted) - int(self._completed)),
                int(self._max_pending),
            )


class _AdaptiveDeferredRetryWorker:
    """Drain deferred targets by time budget and retry only when inputs change."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._request = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._passes = self._examined = self._resolved = 0
        self._last_seconds = self._max_seconds = 0.0
        self._last_signature = None
        self._force_resume = False
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

    def _signature(self):
        pending = getattr(self.runtime, "_v819_deferred_sources", ())
        return (
            int(getattr(self.runtime, "generation", 0)),
            int(getattr(self.runtime, "watermark", 0)),
            len(pending),
        )

    def _drain_slice(self) -> tuple[int, int]:
        from v8 import lease_dispatch_continuity_v839 as v839

        pending = getattr(self.runtime, "_v819_deferred_sources", None)
        if not isinstance(pending, list) or not pending:
            self._force_resume = False
            return 0, 0
        remaining = len(pending)
        deadline = time.perf_counter() + _DEFERRED_TIME_BUDGET_SECONDS
        examined = resolved = 0
        while (
            remaining > 0
            and not self._stop.is_set()
            and time.perf_counter() < deadline
        ):
            count, good = v839._retry_deferred_batch(
                self.runtime, limit=min(_DEFERRED_CHUNK_SIZE, remaining)
            )
            if count <= 0:
                break
            examined += int(count)
            resolved += int(good)
            remaining -= int(count)
        self._force_resume = remaining > 0
        return examined, resolved

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._request.wait(_DEFERRED_INTERVAL_SECONDS):
                continue
            self._request.clear()
            signature = self._signature()
            if signature == self._last_signature and not self._force_resume:
                continue
            started = time.perf_counter()
            try:
                examined, resolved = self._drain_slice()
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
            self._last_signature = signature
            with self._lock:
                self._passes += 1
                self._examined += int(examined)
                self._resolved += int(resolved)
                self._last_seconds = float(elapsed)
                self._max_seconds = max(self._max_seconds, float(elapsed))
            if self._force_resume:
                self._request.set()

    def close(self, timeout: float = 300.0) -> None:
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


def _feedback_worker_v841(runtime):
    from v8 import lease_dispatch_continuity_v839 as v839

    worker = getattr(runtime, "_v839_actor_feedback", None)
    if worker is None or bool(getattr(worker, "_closed", False)):
        worker = _SqliteBatchWorker(
            lambda rows: v839._BASE_RECORD_ACTOR_RESULTS(runtime, tuple(rows)),
            name="v8-actor-feedback",
            root=Path(runtime.root) / "maintenance",
            error_queue=getattr(runtime, "_error_queue", None),
        )
        runtime._v839_actor_feedback = worker
    return worker


def _deferred_worker_v841(runtime):
    worker = getattr(runtime, "_v839_deferred_retry", None)
    if worker is None or not worker._thread.is_alive():
        worker = _AdaptiveDeferredRetryWorker(runtime)
        runtime._v839_deferred_retry = worker
    return worker


def _peer_input_token(supervisor) -> tuple[int, ...]:
    view = getattr(supervisor, "_v813_live_read_view", None) or getattr(
        supervisor, "read_view", None
    )
    arenas = tuple(getattr(view, "_nodes", ())) + tuple(getattr(view, "_edges", ()))
    if arenas:
        return tuple(int(arena.sequence) for arena in arenas)
    return (int(supervisor.current_generation()), int(supervisor.current_watermark()))


def _peer_init_v841(self, *args, **kwargs):
    _BASE_PEER_INIT(self, *args, **kwargs)
    self._v841_peer_executor = ThreadPoolExecutor(max_workers=9, thread_name_prefix="v8-peer")
    self._v841_last_input_token = None
    self._v841_peer_cancel = threading.Event()


def _parallel_analyses_v841(self, nodes, edges):
    pool = self._v841_peer_executor
    futures = {
        "prediction": pool.submit(self.prediction.evaluate, nodes),
        "context": pool.submit(self.context.propose, nodes),
        "roles": pool.submit(self.roles.propose, nodes),
        "future": pool.submit(self.future_options.evaluate, nodes),
        "compression": pool.submit(self.compression.evaluate, nodes, edges),
        "similarity": pool.submit(self.similarity.evaluate, nodes, edges),
        "transfer": pool.submit(
            self.transfer.candidates,
            nodes,
            provenance=self.read_view.source_games,
            cancel_event=self._v841_peer_cancel,
        ),
        "world": pool.submit(self.world_model.propose, nodes),
        "replay": pool.submit(
            self.replay.candidates,
            nodes,
            budget=self.candidate_budget,
            cancel_event=self._v841_peer_cancel,
        ),
    }
    return {name: future.result() for name, future in futures.items()}


def _peer_run_once_v841(self):
    cancel = getattr(self, "_v841_peer_cancel", None)
    if cancel is not None and cancel.is_set():
        return None
    token = _peer_input_token(self)
    if token == getattr(self, "_v841_last_input_token", None):
        return None
    before_cycles = int(getattr(self, "_cycles", 0))
    before_cut = getattr(self, "_last_developmental_cut", None)
    try:
        result = _BASE_PEER_RUN_ONCE(self)
    except BaseException:
        self._v841_last_input_token = None
        raise
    if int(getattr(self, "_cycles", 0)) != before_cycles or getattr(
        self, "_last_developmental_cut", None
    ) is not before_cut:
        self._v841_last_input_token = token
    return result


def _peer_close_v841(self):
    try:
        return _BASE_PEER_CLOSE(self)
    finally:
        executor = getattr(self, "_v841_peer_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._v841_peer_executor = None


class _CandidateOverflowDispatcher:
    """Bounded nonblocking overflow for per-game validator queues."""

    def __init__(self, service, *, per_game_capacity: int = _OPTIMIZER_OVERFLOW_PER_GAME):
        self.service = service
        self.per_game_capacity = max(8, int(per_game_capacity))
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._pending: dict[str, dict[str, object]] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="v8-optimizer-overflow", daemon=True)
        self._thread.start()

    @staticmethod
    def _priority(candidate) -> tuple[int, int, str]:
        edit = str(getattr(candidate, "edit_kind", ""))
        edit_rank = {
            "TARGET_MINIMIZE": 0,
            "VALIDATE_SOURCE": 0,
            "PROJECT_ACTION_KIND": 1,
            "REDUCE_REPEAT": 2,
            "DELETE_SEGMENT": 3,
            "DELETE_ACTION": 4,
        }.get(edit, 5)
        return (max(0, int(getattr(candidate, "cost", len(candidate.actions)))), edit_rank, str(candidate.candidate_id))

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(values) for values in self._pending.values())

    def submit(self, candidate) -> bool:
        game = str(candidate.source.anchor.source_id)
        candidate_id = str(candidate.candidate_id)
        with self._done:
            bucket = self._pending.setdefault(game, {})
            if candidate_id in bucket:
                return True
            if len(bucket) < self.per_game_capacity:
                bucket[candidate_id] = candidate
            else:
                worst_id, worst = max(bucket.items(), key=lambda item: self._priority(item[1]))
                if self._priority(candidate) < self._priority(worst):
                    del bucket[worst_id]
                    bucket[candidate_id] = candidate
            self._done.notify_all()
        self._wake.set()
        return True

    def _drain_once(self) -> int:
        from v8 import trajectory_optimizer_v818 as v818

        with self._lock:
            games = tuple(sorted(self._pending))
        moved = 0
        for game in games:
            if self._stop.is_set() or self.service._stop.is_set():
                break
            v818._ensure_validator(self.service, game)
            with self.service._v818_validator_lock:
                target = self.service._v818_game_queues.setdefault(
                    game, queue.Queue(maxsize=v818._PER_GAME_QUEUE_CAPACITY)
                )
            while True:
                with self._lock:
                    bucket = self._pending.get(game)
                    if not bucket:
                        break
                    candidate_id, candidate = min(
                        bucket.items(), key=lambda item: self._priority(item[1])
                    )
                try:
                    target.put_nowait(candidate)
                except queue.Full:
                    break
                with self._done:
                    bucket = self._pending.get(game)
                    if bucket is not None:
                        bucket.pop(candidate_id, None)
                        if not bucket:
                            self._pending.pop(game, None)
                    self._done.notify_all()
                moved += 1
        v818._start_waiting_validators(self.service)
        return moved

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.pending_count() <= 0:
                self._wake.clear()
                self._wake.wait(_OPTIMIZER_DISPATCH_INTERVAL_SECONDS)
                continue
            if self._drain_once() <= 0:
                self._wake.clear()
                self._wake.wait(_OPTIMIZER_DISPATCH_INTERVAL_SECONDS)

    def drain(self, timeout: float = 10.0) -> None:
        self._wake.set()
        with self._done:
            ok = self._done.wait_for(
                lambda: sum(len(values) for values in self._pending.values()) == 0,
                timeout=max(0.01, float(timeout)),
            )
            pending = sum(len(values) for values in self._pending.values())
        if not ok or pending:
            raise TimeoutError(f"v8 optimizer overflow did not drain (pending={pending})")

    def close(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        if drain and self.pending_count() > 0:
            self.drain(timeout)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(0.01, float(timeout)))
        if self._thread.is_alive():
            raise TimeoutError("v8 optimizer overflow dispatcher did not stop")


def _overflow_dispatcher(service) -> _CandidateOverflowDispatcher:
    worker = getattr(service, "_v841_candidate_overflow", None)
    if worker is None or not worker._thread.is_alive():
        worker = _CandidateOverflowDispatcher(service)
        service._v841_candidate_overflow = worker
    return worker


def _route_candidate_base_v841(service, candidate) -> bool:
    from v8 import trajectory_optimizer_v818 as v818
    from v8 import trajectory_target_minimization_v820 as v820

    if v820._validation_cancel_requested(service):
        v820._preserve_cancelled_source(service, candidate)
        return False

    game = str(candidate.source.anchor.source_id)
    with service._v818_validator_lock:
        target = service._v818_game_queues.setdefault(
            game, queue.Queue(maxsize=v818._PER_GAME_QUEUE_CAPACITY)
        )
    v818._ensure_validator(service, game)
    try:
        target.put_nowait(candidate)
        return True
    except queue.Full:
        return _overflow_dispatcher(service).submit(candidate)


def _queue_depth_v841(service) -> int:
    base = int(_BASE_QUEUE_DEPTH(service))
    worker = getattr(service, "_v841_candidate_overflow", None)
    return base + (0 if worker is None else int(worker.pending_count()))


def _ingest_inbox_v841(service) -> None:
    if bool(getattr(service, "_v841_preserve_inbox_on_shutdown", False)):
        return
    return _BASE_INGEST_INBOX(service)


def _cancel_optimizer_validations_v841(service) -> None:
    """Cancel volatile validation work while preserving every source durably."""

    from v8 import trajectory_target_minimization_v820 as v820

    cancel = getattr(service, "_v841_validation_cancel", None)
    if cancel is None:
        cancel = threading.Event()
        service._v841_validation_cancel = cancel
    cancel.set()

    if not hasattr(service, "_sources"):
        return

    worker = getattr(service, "_v841_candidate_overflow", None)
    if worker is not None:
        with worker._lock:
            overflow = tuple(
                candidate
                for bucket in worker._pending.values()
                for candidate in bucket.values()
            )
        for candidate in overflow:
            v820._preserve_cancelled_source(service, candidate)
        worker.close(drain=False, timeout=5.0)
        service._v841_candidate_overflow = None

    while True:
        try:
            source = service._sources.get_nowait()
        except queue.Empty:
            break
        with service._lock:
            known = {
                str(row.trajectory_id)
                for row in service._v818_restored_sources
            }
            if str(source.trajectory_id) not in known:
                service._v818_restored_sources.append(source)
        service._sources.task_done()

    with service._v818_validator_lock:
        queues = tuple(service._v818_game_queues.values())
        service._v818_waiting_games.clear()
    for validation_queue in queues:
        while True:
            try:
                candidate = validation_queue.get_nowait()
            except queue.Empty:
                break
            v820._preserve_cancelled_source(service, candidate)
            validation_queue.task_done()


def _optimizer_idle_v841(service) -> bool:
    if service is None:
        return True
    worker = getattr(service, "_v841_candidate_overflow", None)
    if worker is not None and worker.pending_count() > 0:
        worker._wake.set()
        return False
    if service._sources.unfinished_tasks > 0:
        return False
    with service._lock:
        active = int(service._active_validations)
    with service._v818_validator_lock:
        queues = tuple(service._v818_game_queues.values())
    durable_inbox_idle = bool(
        getattr(service, "_v841_preserve_inbox_on_shutdown", False)
    ) or not any(service.inbox.glob("*.json"))
    return bool(
        active == 0
        and all(queue_.unfinished_tasks == 0 for queue_ in queues)
        and durable_inbox_idle
    )


def _runtime_is_quiescent_v841(self) -> bool:
    if not _BASE_RUNTIME_IS_QUIESCENT(self):
        return False
    if not (
        bool(getattr(self, "_sampling_complete", False))
        or bool(getattr(self, "_v841_optimizer_drain", False))
    ):
        return True
    return _optimizer_idle_v841(getattr(self, "_v814_trajectory_optimizer", None))


def _runtime_close_v841(self, *, normal: bool = True, timeout: float = 120.0):
    if normal and getattr(self, "_v814_trajectory_optimizer", None) is not None:
        self._v841_optimizer_drain = True
        try:
            self.wait_quiescent(
                timeout=max(0.1, float(timeout)),
                resume_peers=False,
                settle_peers=False,
            )
        finally:
            self._v841_optimizer_drain = False
    return _BASE_RUNTIME_CLOSE(self, normal=normal, timeout=timeout)


def _service_stop_v841(self, *, drain: bool = True, timeout: float = 10.0) -> None:
    worker = getattr(self, "_v841_candidate_overflow", None)
    deadline = time.monotonic() + max(0.1, float(timeout))
    if drain:
        while not _optimizer_idle_v841(self) and time.monotonic() < deadline:
            self.raise_if_failed()
            time.sleep(0.02)
        if not _optimizer_idle_v841(self):
            pending = 0 if worker is None else worker.pending_count()
            raise TimeoutError(
                "v8 optimizer did not drain "
                f"(overflow={pending}, queued={_queue_depth_v841(self)})"
            )
    try:
        return _BASE_SERVICE_STOP(
            self,
            # In preserve mode the wrapper has already drained all admitted RAM
            # work. The v8.18 drain also requires an empty durable inbox, which is
            # deliberately deferred to the next run and must not delay shutdown.
            drain=drain
            and not bool(getattr(self, "_v841_preserve_inbox_on_shutdown", False)),
            timeout=max(0.1, deadline - time.monotonic()),
        )
    finally:
        if worker is not None:
            worker.close(drain=False, timeout=max(0.1, deadline - time.monotonic()))
            self._v841_candidate_overflow = None


def install_runtime_scaling_v841() -> None:
    global _INSTALLED, _BASE_PEER_INIT, _BASE_PEER_CLOSE, _BASE_PEER_RUN_ONCE
    global _BASE_SERVICE_STOP, _BASE_QUEUE_DEPTH, _BASE_INGEST_INBOX
    global _BASE_RUNTIME_IS_QUIESCENT, _BASE_RUNTIME_CLOSE
    if _INSTALLED:
        return

    from v8 import lease_dispatch_continuity_v839 as v839
    from v8 import optimizer_budget_control_v830 as v830
    from v8 import trajectory_optimizer_v818 as v818
    from v8.peers_v82 import V82DevelopmentalPeerSupervisor
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.trajectory_optimizer_v814 import TrajectoryOptimizationService

    v839._feedback_worker = _feedback_worker_v841
    v839._deferred_worker = _deferred_worker_v841

    _BASE_PEER_INIT = V82DevelopmentalPeerSupervisor.__init__
    _BASE_PEER_CLOSE = V82DevelopmentalPeerSupervisor.close
    _BASE_PEER_RUN_ONCE = V82DevelopmentalPeerSupervisor.run_once
    V82DevelopmentalPeerSupervisor.__init__ = _peer_init_v841
    V82DevelopmentalPeerSupervisor._parallel_analyses = _parallel_analyses_v841
    V82DevelopmentalPeerSupervisor.run_once = _peer_run_once_v841
    V82DevelopmentalPeerSupervisor.close = _peer_close_v841

    # Keep v8.30 as the public route authority; replace only its blocking delegate.
    v830._BASE_ROUTE_CANDIDATE = _route_candidate_base_v841
    _BASE_QUEUE_DEPTH = v818._queue_depth
    v818._queue_depth = _queue_depth_v841

    _BASE_INGEST_INBOX = v818._ingest_inbox_v818
    v818._ingest_inbox_v818 = _ingest_inbox_v841
    TrajectoryOptimizationService._ingest_inbox = _ingest_inbox_v841

    _BASE_SERVICE_STOP = TrajectoryOptimizationService.stop
    TrajectoryOptimizationService.stop = _service_stop_v841

    _BASE_RUNTIME_IS_QUIESCENT = V82ContinuousMemoryRuntime._is_quiescent
    _BASE_RUNTIME_CLOSE = V82ContinuousMemoryRuntime.close
    V82ContinuousMemoryRuntime._is_quiescent = _runtime_is_quiescent_v841
    V82ContinuousMemoryRuntime.close = _runtime_close_v841
    _INSTALLED = True
