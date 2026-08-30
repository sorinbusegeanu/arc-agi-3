from __future__ import annotations

"""v8.73 bounded, bucket-parallel lifecycle maintenance.

One immutable developmental cut is partitioned once per lifecycle window. Read-only
decisions for disjoint UID buckets run concurrently, while controller state and graph
proposals are committed in deterministic bucket/UID order. Both lifecycle decisions
and dependency-safe retirement advance persisted cursors only after a whole bucket is
committed, so pause/stop can discard unfinished analysis without losing graph data.
"""

import copy
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from v8.developmental_cut import DevelopmentalGenerationCut, capture_developmental_cut
from v8.model import CognitiveState


_INSTALLED = False
_MAX_WORKERS = 8
_MAX_BUCKETS_PER_SLICE = 8
_POLL_SECONDS = 0.01


class _LifecycleCancelEvent:
    def __init__(self, supervisor) -> None:
        self._supervisor = supervisor

    def is_set(self) -> bool:
        supervisor = self._supervisor
        return bool(supervisor._pause.is_set() or supervisor._stop.is_set())


@dataclass(frozen=True, slots=True)
class _PartitionResult:
    index: int
    buckets: tuple[tuple[object, ...], ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class _LifecycleBucketResult:
    bucket: int
    rows: tuple[object, ...]
    decisions: tuple[tuple[object, object], ...]
    low_window_updates: tuple[tuple[object, int | None], ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class _RetirementBucketResult:
    bucket: int
    decisions: tuple[tuple[object, object], ...]
    complete: bool


def _worker_count(item_count: int) -> int:
    configured = os.environ.get("ARC_AGI3_V8_LIFECYCLE_WORKERS", "").strip()
    requested = int(configured) if configured else int(os.cpu_count() or 1)
    return max(1, min(int(item_count), _MAX_WORKERS, requested))


def _executor(supervisor) -> ThreadPoolExecutor:
    executor = getattr(supervisor, "_v873_lifecycle_executor", None)
    if executor is None:
        executor = ThreadPoolExecutor(
            max_workers=_worker_count(_MAX_WORKERS),
            thread_name_prefix="v8-lifecycle",
        )
        supervisor._v873_lifecycle_executor = executor
    return executor


def _bucket(uid, bucket_count: int) -> int:
    return (int(uid.hi) ^ int(uid.lo)) & (int(bucket_count) - 1)


def _cancelled(supervisor) -> bool:
    return _LifecycleCancelEvent(supervisor).is_set()


def _partition_chunk(
    index: int,
    rows: tuple[object, ...],
    bucket_count: int,
    cancel_event: _LifecycleCancelEvent,
) -> _PartitionResult:
    buckets: list[list[object]] = [[] for _ in range(int(bucket_count))]
    for row_index, row in enumerate(rows):
        if row_index % 256 == 0 and cancel_event.is_set():
            return _PartitionResult(
                int(index),
                tuple(tuple(values) for values in buckets),
                False,
            )
        buckets[_bucket(row.uid, bucket_count)].append(row)
    return _PartitionResult(
        int(index),
        tuple(tuple(values) for values in buckets),
        not cancel_event.is_set(),
    )


def _partition_nodes(
    supervisor,
    nodes: tuple[object, ...],
    bucket_count: int,
) -> tuple[tuple[object, ...], ...] | None:
    workers = _worker_count(max(1, len(nodes)))
    if workers <= 1 or len(nodes) < 2048:
        result = _partition_chunk(
            0,
            nodes,
            bucket_count,
            _LifecycleCancelEvent(supervisor),
        )
        return result.buckets if result.complete else None

    chunk_size = max(1, (len(nodes) + workers - 1) // workers)
    chunks = tuple(
        nodes[start : start + chunk_size]
        for start in range(0, len(nodes), chunk_size)
    )
    cancel_event = _LifecycleCancelEvent(supervisor)
    futures = tuple(
        _executor(supervisor).submit(
            _partition_chunk,
            index,
            chunk,
            bucket_count,
            cancel_event,
        )
        for index, chunk in enumerate(chunks)
    )
    results = tuple(sorted((future.result() for future in futures), key=lambda row: row.index))
    if cancel_event.is_set() or not all(result.complete for result in results):
        return None
    merged: list[list[object]] = [[] for _ in range(int(bucket_count))]
    for result in results:
        for bucket_index, rows in enumerate(result.buckets):
            merged[bucket_index].extend(rows)
    return tuple(tuple(rows) for rows in merged)


def _cut_for_window(
    supervisor,
    target_window: int,
    bucket_count: int,
) -> tuple[DevelopmentalGenerationCut, tuple[tuple[object, ...], ...]] | None:
    cached_window = int(getattr(supervisor, "_v873_cut_window", -1))
    cached_cut = getattr(supervisor, "_v873_cut", None)
    cached_buckets = getattr(supervisor, "_v873_buckets", None)
    if (
        cached_window == int(target_window)
        and isinstance(cached_cut, DevelopmentalGenerationCut)
        and isinstance(cached_buckets, tuple)
        and len(cached_buckets) == int(bucket_count)
    ):
        return cached_cut, cached_buckets
    if _cancelled(supervisor):
        return None
    cut = capture_developmental_cut(
        supervisor._v813_live_read_view,
        generation=int(supervisor.current_generation()),
        watermark=int(supervisor.current_watermark()),
    )
    if _cancelled(supervisor):
        return None
    buckets = _partition_nodes(supervisor, cut.nodes, bucket_count)
    if buckets is None or _cancelled(supervisor):
        return None
    supervisor._v873_cut_window = int(target_window)
    supervisor._v873_cut = cut
    supervisor._v873_buckets = buckets
    supervisor._v873_prune_key = None
    supervisor._v873_prune_candidates = None
    supervisor._v873_prune_future = None
    return cut, buckets


def _analyze_lifecycle_bucket(
    controller,
    bucket: int,
    rows: tuple[object, ...],
    cancel_event: _LifecycleCancelEvent,
) -> _LifecycleBucketResult:
    clone = copy.copy(controller)
    original = controller._low_windows
    before = {row.uid: int(original[row.uid]) for row in rows if row.uid in original}
    clone._low_windows = dict(before)
    decisions = []
    for row_index, row in enumerate(rows):
        if row_index % 64 == 0 and cancel_event.is_set():
            return _LifecycleBucketResult(int(bucket), rows, (), (), False)
        decision = clone.decide(row)
        if decision is not None:
            decisions.append((row, decision))
    if cancel_event.is_set():
        return _LifecycleBucketResult(int(bucket), rows, (), (), False)
    updates = []
    for row in rows:
        prior = before.get(row.uid)
        current = clone._low_windows.get(row.uid)
        if prior != current or (row.uid in before) != (current is not None):
            updates.append((row.uid, None if current is None else int(current)))
    return _LifecycleBucketResult(
        int(bucket),
        rows,
        tuple(decisions),
        tuple(updates),
        True,
    )


def _commit_lifecycle_bucket(supervisor, result: _LifecycleBucketResult, window: int) -> None:
    lifecycle = supervisor.lifecycle
    for uid, value in result.low_window_updates:
        if value is None:
            lifecycle._low_windows.pop(uid, None)
        else:
            lifecycle._low_windows[uid] = int(value)
    for row, decision in result.decisions:
        if not supervisor._fresh(
            f"lifecycle-window:{int(window)}",
            row.uid,
            int(window),
        ):
            continue
        supervisor._submit(
            supervisor._existing_proposal(
                row,
                cognitive_state=int(decision.cognitive_state),
                validation_state=int(decision.validation_state),
            )
        )


def _analyze_lifecycle_slice(
    supervisor,
    buckets: tuple[tuple[object, ...], ...],
    *,
    window: int,
    start: int,
    stop: int,
) -> int:
    lifecycle = supervisor.lifecycle
    sync_stage = getattr(lifecycle, "_sync_published_stage", None)
    if callable(sync_stage):
        sync_stage()
    previous = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    lifecycle._v812_window_delta = 1 if previous < 0 else max(1, int(window) - previous)
    lifecycle._v812_sweep_mode = True
    cancel_event = _LifecycleCancelEvent(supervisor)
    try:
        futures = tuple(
            _executor(supervisor).submit(
                _analyze_lifecycle_bucket,
                lifecycle,
                bucket,
                buckets[bucket],
                cancel_event,
            )
            for bucket in range(int(start), int(stop))
        )
        results = tuple(sorted((future.result() for future in futures), key=lambda row: row.bucket))
    finally:
        lifecycle._v812_sweep_mode = False
        lifecycle._v812_window_delta = 1

    evaluated = 0
    expected = int(start)
    for result in results:
        if result.bucket != expected or not result.complete:
            break
        _commit_lifecycle_bucket(supervisor, result, window)
        evaluated += len(result.rows)
        lifecycle._v812_next_bucket = int(result.bucket) + 1
        lifecycle._v813_progress_evaluated = int(
            getattr(lifecycle, "_v813_progress_evaluated", 0)
        ) + len(result.rows)
        expected += 1
        if _cancelled(supervisor):
            break
    return evaluated


def _prune_candidates(supervisor, cut: DevelopmentalGenerationCut):
    return supervisor.pruning.candidates(
        cut.nodes,
        cut.edges,
        cancel_event=_LifecycleCancelEvent(supervisor),
    )


def _retirement_candidates(
    supervisor,
    cut: DevelopmentalGenerationCut,
    window: int,
) -> dict[object, bool] | None:
    key = (int(window), str(cut.graph_digest))
    if getattr(supervisor, "_v873_prune_key", None) == key:
        cached = getattr(supervisor, "_v873_prune_candidates", None)
        if isinstance(cached, dict):
            return cached

    future: Future | None = getattr(supervisor, "_v873_prune_future", None)
    future_key = getattr(supervisor, "_v873_prune_future_key", None)
    if future is None or future_key != key:
        future = _executor(supervisor).submit(_prune_candidates, supervisor, cut)
        supervisor._v873_prune_future = future
        supervisor._v873_prune_future_key = key
    while not future.done():
        if _cancelled(supervisor):
            supervisor._v873_prune_future = None
            supervisor._v873_prune_future_key = None
            return None
        time.sleep(_POLL_SECONDS)
    rows = future.result()
    supervisor._v873_prune_future = None
    supervisor._v873_prune_future_key = None
    if _cancelled(supervisor):
        return None
    protected = {
        candidate.uid: bool(candidate.protected_by_dependencies)
        for candidate in rows
    }
    supervisor._v873_prune_key = key
    supervisor._v873_prune_candidates = protected
    return protected


def _analyze_retirement_bucket(
    controller,
    bucket: int,
    rows: tuple[object, ...],
    protected: dict[object, bool],
    cancel_event: _LifecycleCancelEvent,
) -> _RetirementBucketResult:
    clone = copy.copy(controller)
    clone._low_windows = {
        row.uid: int(controller._low_windows[row.uid])
        for row in rows
        if row.uid in controller._low_windows
    }
    decisions = []
    for row_index, row in enumerate(rows):
        if row_index % 64 == 0 and cancel_event.is_set():
            return _RetirementBucketResult(int(bucket), (), False)
        if row.uid not in protected:
            continue
        decision = clone.finalize_retirement(
            row,
            protected_by_dependencies=bool(protected[row.uid]),
        )
        if decision is not None:
            decisions.append((row, decision))
    return _RetirementBucketResult(
        int(bucket),
        tuple(decisions),
        not cancel_event.is_set(),
    )


def _commit_retirement_bucket(
    supervisor,
    result: _RetirementBucketResult,
    window: int,
) -> None:
    for row, decision in result.decisions:
        if not supervisor._fresh(
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


def _state_counts(nodes) -> tuple[int, int, int, int]:
    counts = {int(state): 0 for state in CognitiveState}
    for row in nodes:
        counts[int(row.cognitive_state)] = counts.get(int(row.cognitive_state), 0) + 1
    return (
        counts.get(int(CognitiveState.QUARANTINED), 0),
        counts.get(int(CognitiveState.RETIRE_PENDING), 0),
        counts.get(int(CognitiveState.RETIRED), 0),
        counts.get(int(CognitiveState.REACTIVATED), 0),
    )


def _complete_window(supervisor, cut: DevelopmentalGenerationCut, window: int) -> None:
    lifecycle = supervisor.lifecycle
    lifecycle._v812_last_completed_window = int(window)
    lifecycle._v812_active_window = -1
    lifecycle._v812_next_bucket = 0
    lifecycle._v873_retirement_window = -1
    lifecycle._v873_retirement_next_bucket = 0
    quarantined, retire_pending, retired, reactivated = _state_counts(cut.nodes)
    print(
        f"[{time.strftime('%H:%M')}] lifecycle window={int(window)} complete "
        f"generation={int(supervisor.current_generation())} "
        f"evaluated={int(getattr(lifecycle, '_v813_progress_evaluated', 0))} "
        f"quarantined={quarantined} retire_pending={retire_pending} "
        f"retired={retired} reactivated={reactivated}",
        flush=True,
    )
    lifecycle._v813_progress_window = -1
    lifecycle._v813_progress_evaluated = 0
    supervisor._v873_cut_window = -1
    supervisor._v873_cut = None
    supervisor._v873_buckets = None
    supervisor._v873_prune_key = None
    supervisor._v873_prune_candidates = None


def _run_parallel_lifecycle_iteration_v873(supervisor) -> None:
    from v8 import final_save_lifecycle_v812 as base

    if _cancelled(supervisor):
        return
    lifecycle = supervisor.lifecycle
    bucket_count = int(base._LIFECYCLE_BUCKETS)
    active = int(getattr(lifecycle, "_v812_active_window", -1))
    last = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    pending_retirement = int(getattr(lifecycle, "_v873_retirement_window", -1))
    global_window = max(0, int(supervisor.current_generation())) // int(
        base._LIFECYCLE_GENERATION_SPAN
    )
    if pending_retirement >= 0:
        target_window = pending_retirement
        lifecycle._v812_active_window = target_window
        lifecycle._v812_next_bucket = bucket_count
    elif active >= 0:
        target_window = active
    elif global_window > last:
        target_window = global_window
        lifecycle._v812_active_window = target_window
        lifecycle._v812_next_bucket = 0
        lifecycle._v873_retirement_next_bucket = 0
    else:
        return

    cut_and_buckets = _cut_for_window(supervisor, target_window, bucket_count)
    if cut_and_buckets is None:
        return
    cut, buckets = cut_and_buckets
    if int(getattr(lifecycle, "_v813_progress_window", -1)) != target_window:
        lifecycle._v813_progress_window = target_window
        lifecycle._v813_progress_evaluated = sum(
            len(buckets[index])
            for index in range(
                min(bucket_count, int(getattr(lifecycle, "_v812_next_bucket", 0)))
            )
        )

    next_bucket = int(getattr(lifecycle, "_v812_next_bucket", 0))
    if pending_retirement < 0 and next_bucket < bucket_count:
        buckets_per_slice = max(
            1,
            min(
                _MAX_BUCKETS_PER_SLICE,
                max(1, int(supervisor.candidate_budget) // 64),
            ),
        )
        stop = min(bucket_count, next_bucket + buckets_per_slice)
        _analyze_lifecycle_slice(
            supervisor,
            buckets,
            window=target_window,
            start=next_bucket,
            stop=stop,
        )
        if int(getattr(lifecycle, "_v812_next_bucket", 0)) < bucket_count:
            return
        lifecycle._v873_retirement_window = target_window
        lifecycle._v873_retirement_next_bucket = 0
        return

    protected = _retirement_candidates(supervisor, cut, target_window)
    if protected is None:
        return
    start = max(0, int(getattr(lifecycle, "_v873_retirement_next_bucket", 0)))
    stop = min(bucket_count, start + _MAX_BUCKETS_PER_SLICE)
    cancel_event = _LifecycleCancelEvent(supervisor)
    futures = tuple(
        _executor(supervisor).submit(
            _analyze_retirement_bucket,
            lifecycle,
            bucket,
            buckets[bucket],
            protected,
            cancel_event,
        )
        for bucket in range(start, stop)
    )
    results = tuple(sorted((future.result() for future in futures), key=lambda row: row.bucket))
    expected = start
    for result in results:
        if result.bucket != expected or not result.complete:
            break
        _commit_retirement_bucket(supervisor, result, target_window)
        lifecycle._v873_retirement_next_bucket = int(result.bucket) + 1
        expected += 1
        if _cancelled(supervisor):
            break
    if int(getattr(lifecycle, "_v873_retirement_next_bucket", 0)) >= bucket_count:
        _complete_window(supervisor, cut, target_window)


def install_parallel_lifecycle_v873() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import lifecycle_competence_integration_v827_fixups as competence
    from v8 import peers_v82
    from v8.lifecycle import LifecycleController

    lifecycle_init = LifecycleController.__init__
    lifecycle_state_dict = LifecycleController.state_dict
    lifecycle_load_state = LifecycleController.load_state
    supervisor_init = peers_v82.V82DevelopmentalPeerSupervisor.__init__
    supervisor_close = peers_v82.V82DevelopmentalPeerSupervisor.close

    def controller_init(self, *args, **kwargs):
        lifecycle_init(self, *args, **kwargs)
        self._v873_retirement_window = -1
        self._v873_retirement_next_bucket = 0

    def state_dict(self):
        state = dict(lifecycle_state_dict(self))
        state["v873_parallel_lifecycle"] = {
            "retirement_window": int(getattr(self, "_v873_retirement_window", -1)),
            "retirement_next_bucket": int(
                getattr(self, "_v873_retirement_next_bucket", 0)
            ),
        }
        return state

    def load_state(self, state):
        lifecycle_load_state(self, state)
        raw = state.get("v873_parallel_lifecycle") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        window = int(raw.get("retirement_window", -1))
        self._v873_retirement_window = window
        self._v873_retirement_next_bucket = max(
            0,
            int(raw.get("retirement_next_bucket", 0)),
        )
        if window >= 0:
            from v8 import final_save_lifecycle_v812 as base

            self._v812_active_window = window
            self._v812_next_bucket = int(base._LIFECYCLE_BUCKETS)

    def peer_init(self, *args, **kwargs):
        supervisor_init(self, *args, **kwargs)
        self._v873_lifecycle_executor = None
        self._v873_cut_window = -1
        self._v873_cut = None
        self._v873_buckets = None
        self._v873_prune_key = None
        self._v873_prune_candidates = None
        self._v873_prune_future = None
        self._v873_prune_future_key = None

    def close(self):
        try:
            return supervisor_close(self)
        finally:
            executor = getattr(self, "_v873_lifecycle_executor", None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                self._v873_lifecycle_executor = None

    LifecycleController.__init__ = controller_init
    LifecycleController.state_dict = state_dict
    LifecycleController.load_state = load_state
    peers_v82.V82DevelopmentalPeerSupervisor.__init__ = peer_init
    peers_v82.V82DevelopmentalPeerSupervisor.close = close

    # Keep the public performance, snapshot-consistency, and competence-protection
    # wrappers intact. Replace only their innermost lifecycle implementation.
    competence._BASE_RUN_LIFECYCLE_ITERATION = _run_parallel_lifecycle_iteration_v873
    _INSTALLED = True
