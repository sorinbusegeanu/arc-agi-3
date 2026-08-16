from __future__ import annotations

import threading
import time

from v8.developmental_cut import capture_developmental_cut
from v8.model import CognitiveState


_INSTALLED = False
_CONTEXT = threading.local()
_BASE_GENERATION_RUNNER = None
_BASE_PUBLIC_GENERATION_RUNNER = None
_BASE_FINALIZE_RETIREMENT = None


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


def _generation_dispatch(supervisor, nodes) -> int:
    """Keep direct/test calls working but remove lifecycle work from peer cognition."""
    if bool(getattr(_CONTEXT, "peer_cycle", False)):
        return 0
    runner = _BASE_PUBLIC_GENERATION_RUNNER
    return 0 if runner is None else int(runner(supervisor, nodes))


def _initialize_progress(supervisor, nodes, target_window: int, next_bucket: int) -> None:
    from v8 import final_save_lifecycle_v812 as base

    lifecycle = supervisor.lifecycle
    if int(getattr(lifecycle, "_v813_progress_window", -1)) == int(target_window):
        return
    lifecycle._v813_progress_window = int(target_window)
    lifecycle._v813_progress_evaluated = sum(
        1
        for row in nodes
        if _bucket(row.uid, base._LIFECYCLE_BUCKETS) < int(next_bucket)
    )


def _finalize_retirements(supervisor, *, window: int) -> tuple[object, ...]:
    """Run dependency-safe retirement on one coherent graph cut."""
    live_view = supervisor._v813_live_read_view
    cut = capture_developmental_cut(
        live_view,
        generation=int(supervisor.current_generation()),
        watermark=int(supervisor.current_watermark()),
    )
    protected = {
        candidate.uid: candidate.protected_by_dependencies
        for candidate in supervisor.pruning.candidates(cut.nodes, cut.edges)
    }
    for row in cut.nodes:
        if row.uid not in protected:
            continue
        decision = supervisor.lifecycle.finalize_retirement(
            row,
            protected_by_dependencies=protected[row.uid],
        )
        if decision is None or not supervisor._fresh(
            f"retire-window:{int(window)}",
            row.uid,
            int(row.updated_watermark),
        ):
            continue
        supervisor._submit(
            supervisor._existing_proposal(
                row,
                cognitive_state=int(decision.cognitive_state),
                validation_state=int(decision.validation_state),
            )
        )
        supervisor._append_evidence("memory_retired", row, 1.0)
    return cut.nodes


def _run_lifecycle_iteration(supervisor) -> None:
    from v8 import final_save_lifecycle_v812 as base

    runner = _BASE_GENERATION_RUNNER
    if runner is None:
        return
    nodes = tuple(supervisor._v813_live_read_view.node_records())
    if not nodes:
        return

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
    if target_window < 0:
        return

    _initialize_progress(supervisor, nodes, target_window, next_bucket_before)
    evaluated = int(runner(supervisor, nodes))
    lifecycle._v813_progress_evaluated = int(
        getattr(lifecycle, "_v813_progress_evaluated", 0)
    ) + evaluated

    active_after = int(getattr(lifecycle, "_v812_active_window", -1))
    last_after = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    completed = (
        active_after < 0
        and last_after == target_window
        and last_after != last_before
    )
    if not completed:
        return

    final_nodes = _finalize_retirements(supervisor, window=target_window)
    quarantined, retire_pending, retired, reactivated = _state_counts(final_nodes)
    print(
        f"[{time.strftime('%H:%M')}] lifecycle window={target_window} complete "
        f"generation={int(supervisor.current_generation())} "
        f"evaluated={int(getattr(lifecycle, '_v813_progress_evaluated', 0))} "
        f"quarantined={quarantined} retire_pending={retire_pending} "
        f"retired={retired} reactivated={reactivated}",
        flush=True,
    )
    lifecycle._v813_progress_window = -1
    lifecycle._v813_progress_evaluated = 0


def _lifecycle_worker(supervisor) -> None:
    """Lifecycle scheduler independent of prediction/promotion/replay peer latency."""
    try:
        while not supervisor._stop.is_set():
            if supervisor._pause.is_set():
                supervisor._stop.wait(0.05)
                continue
            acquired = supervisor._v813_lifecycle_run_lock.acquire(timeout=0.05)
            if not acquired:
                continue
            try:
                if not supervisor._pause.is_set() and not supervisor._stop.is_set():
                    _run_lifecycle_iteration(supervisor)
            finally:
                supervisor._v813_lifecycle_run_lock.release()
            supervisor._stop.wait(0.10)
    except BaseException as exc:
        supervisor._v813_lifecycle_error = f"{type(exc).__name__}: {exc}"
        supervisor._stop.set()


def install_dedicated_lifecycle_v813() -> None:
    global _INSTALLED, _BASE_GENERATION_RUNNER, _BASE_PUBLIC_GENERATION_RUNNER
    global _BASE_FINALIZE_RETIREMENT
    if _INSTALLED:
        return

    from v8 import final_save_lifecycle_v812 as lifecycle_runtime
    from v8 import final_save_lifecycle_v812_fixups as lifecycle_fixups
    from v8 import lifecycle_progress_v812 as lifecycle_progress
    from v8 import peers_v82
    from v8.lifecycle import LifecycleController

    cls = peers_v82.V82DevelopmentalPeerSupervisor
    base_init = cls.__init__
    base_start = cls.start
    base_close = cls.close
    base_wait_idle = cls.wait_idle
    base_raise_if_failed = cls.raise_if_failed
    base_run_once = cls.run_once
    base_event_id = cls._event_id
    base_fresh = cls._fresh
    base_submit = cls._submit
    _BASE_PUBLIC_GENERATION_RUNNER = lifecycle_runtime._run_generation_lifecycle
    _BASE_GENERATION_RUNNER = lifecycle_progress._BASE_RUN_GENERATION_LIFECYCLE
    if _BASE_GENERATION_RUNNER is None:
        _BASE_GENERATION_RUNNER = lifecycle_fixups._run_generation_lifecycle
    _BASE_FINALIZE_RETIREMENT = LifecycleController.finalize_retirement

    def supervisor_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self._v813_live_read_view = self.read_view
        self._v813_lifecycle_run_lock = threading.Lock()
        self._v813_event_lock = threading.Lock()
        self._v813_seen_lock = threading.Lock()
        self._v813_submit_lock = threading.Lock()
        self._v813_lifecycle_thread = None
        self._v813_lifecycle_error = None
        self.lifecycle._v813_progress_window = -1
        self.lifecycle._v813_progress_evaluated = 0

    def start(self):
        base_start(self)
        thread = getattr(self, "_v813_lifecycle_thread", None)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=_lifecycle_worker,
            args=(self,),
            name="v8-dedicated-lifecycle",
            daemon=True,
        )
        self._v813_lifecycle_thread = thread
        thread.start()

    def close(self):
        base_close(self)
        thread = getattr(self, "_v813_lifecycle_thread", None)
        if thread is not None:
            thread.join(timeout=3.0)

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        if not base_wait_idle(self, max(0.0, deadline - time.monotonic())):
            return False
        acquired = self._v813_lifecycle_run_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
        if not acquired:
            return False
        self._v813_lifecycle_run_lock.release()
        return True

    def raise_if_failed(self):
        base_raise_if_failed(self)
        error = getattr(self, "_v813_lifecycle_error", None)
        if error is not None:
            self._v813_lifecycle_error = None
            raise RuntimeError(f"v8 dedicated lifecycle failure: {error}")
        thread = getattr(self, "_v813_lifecycle_thread", None)
        if (
            thread is not None
            and not thread.is_alive()
            and not self._stop.is_set()
        ):
            raise RuntimeError("v8 dedicated lifecycle worker exited unexpectedly")

    def run_once(self):
        prior = bool(getattr(_CONTEXT, "peer_cycle", False))
        _CONTEXT.peer_cycle = True
        try:
            return base_run_once(self)
        finally:
            _CONTEXT.peer_cycle = prior

    def event_id(self):
        with self._v813_event_lock:
            return base_event_id(self)

    def fresh(self, kind, uid, watermark):
        with self._v813_seen_lock:
            return base_fresh(self, kind, uid, watermark)

    def submit(self, proposal):
        with self._v813_submit_lock:
            return base_submit(self, proposal)

    def finalize_retirement(self, row, *, protected_by_dependencies: bool):
        if bool(getattr(_CONTEXT, "peer_cycle", False)):
            return None
        return _BASE_FINALIZE_RETIREMENT(
            self,
            row,
            protected_by_dependencies=protected_by_dependencies,
        )

    cls.__init__ = supervisor_init
    cls.start = start
    cls.close = close
    cls.wait_idle = wait_idle
    cls.raise_if_failed = raise_if_failed
    cls.run_once = run_once
    cls._event_id = event_id
    cls._fresh = fresh
    cls._submit = submit
    LifecycleController.finalize_retirement = finalize_retirement
    lifecycle_runtime._run_generation_lifecycle = _generation_dispatch

    _INSTALLED = True
