from __future__ import annotations

"""v8.54 performance and long-run memory scaling fixes.

This layer keeps behavioral authorities intact while removing maintenance work that
scaled with historical state rather than current work:
- gate lifecycle scans before materializing the graph;
- reload validated trajectories only when the file changes and keep actor-local rows;
- reduce per-step action telemetry work and prune consumed dead-actor files;
- make memory-efficiency reporting streaming/direct instead of full-graph materialization;
- bound optimizer queue/deferred-binding RAM;
- make actor RAM telemetry lower-frequency;
- avoid actor graph invalidation when arena versions did not change;
- return hypothesis-report heap memory to the OS after large evidence cuts.
"""

import ctypes
import gc
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from v8.model import CognitiveState, MemoryLevel, ValidationState


_INSTALLED = False

_BASE_LIFECYCLE_ITERATION = None
_BASE_REFRESH_VARIANTS = None
_BASE_ENV_AVAILABLE = None
_BASE_REFRESH_EVENTS = None
_BASE_ACTOR_GRAPH_CHECK = None
_BASE_ADAPTIVE_WORKER = None
_BASE_OVERFLOW_INIT = None
_BASE_OVERFLOW_SUBMIT = None
_BASE_HYPOTHESIS_STATUS_LINE = None

_VARIANT_REFRESH_SECONDS = 1.0
_ACTION_TELEMETRY_FLUSH_STEPS = 256
_ACTOR_MEMORY_SAMPLE_SECONDS = 15.0
_MAX_OVERFLOW_TOTAL = 1024
_MAX_OVERFLOW_PER_GAME = 64
_MAX_DEFERRED_BINDINGS = 1024
_DEFERRED_RETRY_BATCH = 32

_HYPOTHESIS_CACHE_LOCK = threading.Lock()
_HYPOTHESIS_CACHE = {"signature": None, "line": None}


def _lifecycle_due_v854(supervisor) -> bool:
    from v8 import final_save_lifecycle_v812 as base

    lifecycle = supervisor.lifecycle
    active = int(getattr(lifecycle, "_v812_active_window", -1))
    if active >= 0:
        return True
    last = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    global_window = max(0, int(supervisor.current_generation())) // int(
        base._LIFECYCLE_GENERATION_SPAN
    )
    return global_window > last


def _run_lifecycle_iteration_v854(supervisor) -> None:
    if not _lifecycle_due_v854(supervisor):
        return
    return _BASE_LIFECYCLE_ITERATION(supervisor)


def _refresh_view_variants_v854(view) -> None:
    """Reload optimized variants only after validated.json actually changes."""
    from v8 import trajectory_optimizer_v814 as optimizer

    now = time.monotonic()
    if now < float(getattr(view, "_v814_next_refresh", 0.0)):
        return
    view._v814_next_refresh = now + _VARIANT_REFRESH_SECONDS

    root_raw = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
    source_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    if not root_raw or not source_id:
        view._v814_variants = ()
        view._v854_variant_file_signature = (source_id, None, None)
        return

    path = Path(root_raw) / "validated.json"
    try:
        stat = path.stat()
        signature = (source_id, int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        signature = (source_id, None, None)

    if signature == getattr(view, "_v854_variant_file_signature", None):
        return

    rows = optimizer._load_validated_rows(path)
    view._v814_variants = tuple(
        row for row in rows if str(row.anchor.source_id) == source_id
    )
    view._v854_variant_file_signature = signature


def _env_available_v854(env):
    """Avoid rescanning an unchanged action space on every available_actions() call."""
    from v8 import action_learning_report_v849 as report

    result = _BASE_ENV_AVAILABLE(env)
    signature = tuple(int(value) for value in result)
    if signature != getattr(env, "_v854_available_signature", None):
        report._observe_available(env, result)
        env._v854_available_signature = signature
    return result


def _changed_v854(before, after) -> bool:
    """Exact grid change test without allocating a full temporary boolean grid."""
    import numpy as np

    left = np.asarray(before)
    right = np.asarray(after)
    if left.shape != right.shape:
        return True
    if left.size == 0:
        return False
    return not bool(np.array_equal(left, right))


def _pid_from_event_path(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("actor-"):
        return None
    try:
        return int(stem.split("-", 1)[1])
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if int(pid) == os.getpid():
        return True
    proc = Path(f"/proc/{int(pid)}")
    if proc.exists():
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _prune_consumed_action_event_files_v854() -> int:
    from v8 import action_learning_report_v849 as report

    root = report._event_root()
    if root is None or not root.is_dir():
        return 0
    removed = 0
    for path in tuple(root.glob("actor-*.jsonl")):
        key = str(path)
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if int(report._FILE_OFFSETS.get(key, 0)) < size:
            continue
        pid = _pid_from_event_path(path)
        if pid is None or _pid_alive(pid):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        report._FILE_OFFSETS.pop(key, None)
        removed += 1
    return removed


def _refresh_events_v854(*, force: bool = False) -> None:
    _BASE_REFRESH_EVENTS(force=force)
    _prune_consumed_action_event_files_v854()


def _stream_node_metrics_v854(runtime):
    """Count node categories without retaining a full graph or any edge index."""
    categories = {
        "useful": 0,
        "scientifically_required": 0,
        "dormant": 0,
        "reclaimable": 0,
    }
    by_level = {f"M{level}": dict(categories) for level in range(8)}
    retained = tested = successful = 0

    arenas = tuple(getattr(runtime.read_view, "_nodes", ()))
    for arena in arenas:
        local = None
        for _attempt in range(8):
            before = int(arena.sequence)
            if before & 1:
                time.sleep(0.0005)
                continue
            local_categories = dict(categories)
            local_levels = {f"M{level}": dict(categories) for level in range(8)}
            local_retained = local_tested = local_successful = 0
            for row in arena.records():
                local_retained += 1
                level = int(row.level)
                is_strategy = (
                    level == int(MemoryLevel.M7)
                    and float(getattr(row, "attempt_weight", 0.0)) > 0.0
                )
                useful = is_strategy and float(getattr(row, "success_sum", 0.0)) > 0.0
                if useful:
                    category = "useful"
                elif int(row.cognitive_state) == int(CognitiveState.RETIRED):
                    category = "reclaimable"
                elif int(row.validation_state) == int(ValidationState.VALIDATED):
                    category = "scientifically_required"
                else:
                    category = "dormant"
                local_categories[category] += 1
                local_levels.setdefault(f"M{level}", {
                    "useful": 0,
                    "scientifically_required": 0,
                    "dormant": 0,
                    "reclaimable": 0,
                })[category] += 1
                if is_strategy:
                    local_tested += 1
                    if useful:
                        local_successful += 1
            after = int(arena.sequence)
            if before == after and not (after & 1):
                local = (
                    local_categories,
                    local_levels,
                    local_retained,
                    local_tested,
                    local_successful,
                )
                break
        if local is None:
            continue
        local_categories, local_levels, local_retained, local_tested, local_successful = local
        retained += int(local_retained)
        tested += int(local_tested)
        successful += int(local_successful)
        for key in categories:
            categories[key] += int(local_categories[key])
        for level, values in local_levels.items():
            bucket = by_level.setdefault(level, {
                "useful": 0,
                "scientifically_required": 0,
                "dormant": 0,
                "reclaimable": 0,
            })
            for key in categories:
                bucket[key] += int(values[key])

    return retained, categories, by_level, tested, successful


def _memory_efficiency_snapshot_v854(runtime) -> dict[str, object]:
    """Low-allocation periodic memory report; deep lifecycle safety stays elsewhere."""
    from v8 import actor_throughput_v853 as throughput
    from v8 import arena as arena_module
    from v8 import memory_efficiency_v851 as memory

    retained, categories, by_level, tested, successful = _stream_node_metrics_v854(runtime)

    node_size = int(arena_module.SharedNodeArena.record.size)
    edge_size = int(arena_module.SharedEdgeArena.record.size)
    action_size = int(arena_module.SharedActionArena.record.size)
    descriptors = tuple(runtime.shard_descriptors)
    edge_count = sum(
        int(getattr(arena, "count", 0))
        for arena in getattr(runtime.read_view, "_edges", ())
    )

    payload = {
        "schema_version": 2,
        "time": time.time(),
        "generation": int(runtime.generation),
        "watermark": int(runtime.watermark),
        "objects": {
            "retained": retained,
            **categories,
            "useful_ratio_pct": 0.0
            if retained <= 0
            else 100.0 * categories["useful"] / retained,
            "reclaimable_ratio_pct": 0.0
            if retained <= 0
            else 100.0 * categories["reclaimable"] / retained,
            "raw_node_bytes": {
                key: int(value) * node_size for key, value in categories.items()
            },
            "by_level": by_level,
            "behaviorally_tested_strategies": tested,
            "behaviorally_successful_strategies": successful,
            "classification_scope": "DIRECT_STREAMING_NO_EDGE_MATERIALIZATION",
        },
        "arena_bytes": {
            "node_used_bytes": retained * node_size,
            "edge_used_bytes": edge_count * edge_size,
            "node_capacity_bytes": sum(
                int(row.nodes.capacity) * node_size for row in descriptors
            ),
            "edge_capacity_bytes": sum(
                int(row.edges.capacity) * edge_size for row in descriptors
            ),
            "action_capacity_bytes": sum(
                int(row.actions.capacity) * action_size for row in descriptors
            ),
        },
        "process_memory": memory._process_tree_memory(),
        "system_memory": memory._system_memory(),
        "actors": memory._actor_memory_rows(runtime.root),
    }
    decision = getattr(
        getattr(runtime, "resource_controller", None),
        "_v853_last_decision",
        None,
    )
    if decision is not None:
        payload["resource_decision"] = throughput._decision_payload(decision)
    return payload


def _trim_heap_v854() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except (OSError, AttributeError):
        pass


def _disk_evidence_signature_v854(watermark: int):
    import sqlite3

    root = str(os.environ.get("ARC_AGI3_V8_ROOT", "")).strip()
    if not root:
        return None
    path = Path(root) / "maintenance" / "evidence.sqlite3"
    if not path.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        row = db.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM evidence "
            "WHERE available<=? AND decision<=?",
            (int(watermark), int(watermark)),
        ).fetchone()
        db.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return (int(row[0]), int(row[1]))


def _hypothesis_status_line_v854(evidence_rows, watermark_value: int) -> str:
    supplied = tuple(evidence_rows)
    if supplied:
        try:
            return _BASE_HYPOTHESIS_STATUS_LINE(supplied, watermark_value)
        finally:
            _trim_heap_v854()

    signature = _disk_evidence_signature_v854(int(watermark_value))
    if signature is not None:
        with _HYPOTHESIS_CACHE_LOCK:
            if (
                signature == _HYPOTHESIS_CACHE["signature"]
                and _HYPOTHESIS_CACHE["line"] is not None
            ):
                return str(_HYPOTHESIS_CACHE["line"])

    try:
        line = _BASE_HYPOTHESIS_STATUS_LINE((), watermark_value)
    finally:
        _trim_heap_v854()

    if signature is not None:
        with _HYPOTHESIS_CACHE_LOCK:
            _HYPOTHESIS_CACHE["signature"] = signature
            _HYPOTHESIS_CACHE["line"] = str(line)
    return str(line)


def _overflow_init_v854(self, service, *, per_game_capacity=_MAX_OVERFLOW_PER_GAME):
    _BASE_OVERFLOW_INIT(
        self,
        service,
        per_game_capacity=min(_MAX_OVERFLOW_PER_GAME, int(per_game_capacity)),
    )
    self.per_game_capacity = min(
        int(self.per_game_capacity),
        _MAX_OVERFLOW_PER_GAME,
    )


def _overflow_submit_v854(self, candidate) -> bool:
    candidate_id = str(candidate.candidate_id)
    game = str(candidate.source.anchor.source_id)
    with self._done:
        bucket = self._pending.setdefault(game, {})
        if candidate_id in bucket:
            return True

        total = sum(len(values) for values in self._pending.values())
        if total >= _MAX_OVERFLOW_TOTAL:
            worst_game = None
            worst_id = None
            worst = None
            for owner, values in self._pending.items():
                for row_id, row in values.items():
                    if worst is None or self._priority(row) > self._priority(worst):
                        worst_game, worst_id, worst = owner, row_id, row
            if worst is None or self._priority(candidate) >= self._priority(worst):
                return False
            self._pending[worst_game].pop(worst_id, None)
            if not self._pending[worst_game]:
                self._pending.pop(worst_game, None)
            bucket = self._pending.setdefault(game, {})

        if len(bucket) >= int(self.per_game_capacity):
            worst_id, worst = max(
                bucket.items(), key=lambda item: self._priority(item[1])
            )
            if self._priority(candidate) >= self._priority(worst):
                return False
            bucket.pop(worst_id, None)

        bucket[candidate_id] = candidate
        self._done.notify_all()
    self._wake.set()
    return True


def _retry_deferred_v854(runtime) -> None:
    from v8 import trajectory_optimizer_v818 as v818

    pending = list(getattr(runtime, "_v818_deferred_trajectory_bindings", ()))
    if not pending:
        return
    head = pending[:_DEFERRED_RETRY_BATCH]
    tail = pending[_DEFERRED_RETRY_BATCH:]
    unresolved = []
    for candidate, result, validated in head:
        target_uid = v818._resolve_target_outcome(runtime, candidate, result)
        if target_uid.is_zero:
            unresolved.append((candidate, result, validated))
            continue
        v818._publish_resolved_validation(runtime, candidate, result, validated, target_uid)
    runtime._v818_deferred_trajectory_bindings = tail + unresolved


def _enqueue_deferred_v854(runtime, candidate, result, validated) -> None:
    pending = list(getattr(runtime, "_v818_deferred_trajectory_bindings", ()))
    candidate_id = str(getattr(candidate, "candidate_id", ""))
    if candidate_id and any(
        str(getattr(row[0], "candidate_id", "")) == candidate_id for row in pending
    ):
        return
    if len(pending) >= _MAX_DEFERRED_BINDINGS:
        _retry_deferred_v854(runtime)
        pending = list(getattr(runtime, "_v818_deferred_trajectory_bindings", ()))
    if len(pending) >= _MAX_DEFERRED_BINDINGS:
        pending = pending[-(_MAX_DEFERRED_BINDINGS - 1):]
    pending.append((candidate, result, validated))
    runtime._v818_deferred_trajectory_bindings = pending


def _runtime_validation_callback_v854(runtime, candidate, result, validated) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    _retry_deferred_v854(runtime)
    if not bool(getattr(result, "success", False)):
        optimizer._runtime_validation_callback(runtime, candidate, result, None)
        return
    if validated is None:
        return
    target_uid = v818._resolve_target_outcome(runtime, candidate, result)
    if target_uid.is_zero:
        _enqueue_deferred_v854(runtime, candidate, result, validated)
        return
    v818._publish_resolved_validation(runtime, candidate, result, validated, target_uid)


def _adaptive_worker_v854(*, worker_id: int, trajectory_root: str, **kwargs) -> None:
    """Sample actor PSS/USS infrequently; do not read smaps every two seconds."""
    from v8 import memory_efficiency_v851 as memory
    from v8 import memory_efficiency_v852_review_fix as v852

    root = Path(trajectory_root).parent
    pseudo_job = SimpleNamespace(actor_id=int(worker_id), game_id="adaptive")
    stop = threading.Event()
    peak = [0, 0]

    def sample(finished: bool = False) -> None:
        try:
            peak[0], peak[1] = memory._write_actor_memory(
                root,
                pseudo_job,
                peak_pss=peak[0],
                peak_uss=peak[1],
                finished=finished,
            )
        except OSError:
            pass

    def monitor() -> None:
        while not stop.wait(_ACTOR_MEMORY_SAMPLE_SECONDS):
            sample(False)

    sample(False)
    thread = threading.Thread(target=monitor, name="v854-actor-memory", daemon=True)
    thread.start()
    try:
        return v852._BASE_ADAPTIVE_WORKER(
            worker_id=worker_id,
            trajectory_root=trajectory_root,
            **kwargs,
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sample(True)


def _actor_graph_check_v854(
    read_view,
    *,
    completed_steps: int,
    next_check_step: int,
    check_interval_steps: int,
) -> int:
    interval = int(check_interval_steps)
    if interval <= 0:
        raise ValueError("graph check interval must be positive")
    if int(completed_steps) < int(next_check_step):
        return int(next_check_step)

    current = tuple(
        int(arena.sequence)
        for arena in (
            *tuple(getattr(read_view, "_nodes", ())),
            *tuple(getattr(read_view, "_edges", ())),
        )
    )
    prior = tuple(getattr(read_view, "_strategy_version", ()))
    if not prior or current != prior or any(value & 1 for value in current):
        read_view.invalidate_strategy_cache()
    return (int(completed_steps) // interval + 1) * interval


def install_performance_memory_v854() -> None:
    global _INSTALLED
    global _BASE_LIFECYCLE_ITERATION, _BASE_REFRESH_VARIANTS
    global _BASE_ENV_AVAILABLE, _BASE_REFRESH_EVENTS, _BASE_ACTOR_GRAPH_CHECK
    global _BASE_ADAPTIVE_WORKER, _BASE_OVERFLOW_INIT, _BASE_OVERFLOW_SUBMIT
    global _BASE_HYPOTHESIS_STATUS_LINE
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import action_learning_report_v849 as report
    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819_performance_fix as adaptive
    from v8 import dedicated_lifecycle_v813 as lifecycle
    from v8 import memory_efficiency_v851 as memory
    from v8 import runtime_observability_v836 as observability
    from v8 import runtime_scaling_v841 as scaling
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    _BASE_LIFECYCLE_ITERATION = lifecycle._run_lifecycle_iteration
    lifecycle._run_lifecycle_iteration = _run_lifecycle_iteration_v854

    _BASE_REFRESH_VARIANTS = optimizer._refresh_view_variants
    optimizer._refresh_view_variants = _refresh_view_variants_v854

    _BASE_ENV_AVAILABLE = report._BASE_ENV_AVAILABLE
    ArcGridEnvironment.available_actions = _env_available_v854
    report._changed = _changed_v854
    report._FLUSH_STEPS = max(
        int(getattr(report, "_FLUSH_STEPS", 64)),
        _ACTION_TELEMETRY_FLUSH_STEPS,
    )
    _BASE_REFRESH_EVENTS = report._refresh_events
    report._refresh_events = _refresh_events_v854

    memory.memory_efficiency_snapshot_v851 = _memory_efficiency_snapshot_v854

    _BASE_HYPOTHESIS_STATUS_LINE = observability._hypothesis_status_line
    observability._hypothesis_status_line = _hypothesis_status_line_v854

    v818._PER_GAME_QUEUE_CAPACITY = min(
        int(getattr(v818, "_PER_GAME_QUEUE_CAPACITY", 256)),
        _MAX_OVERFLOW_PER_GAME,
    )
    _BASE_OVERFLOW_INIT = scaling._CandidateOverflowDispatcher.__init__
    _BASE_OVERFLOW_SUBMIT = scaling._CandidateOverflowDispatcher.submit
    scaling._CandidateOverflowDispatcher.__init__ = _overflow_init_v854
    scaling._CandidateOverflowDispatcher.submit = _overflow_submit_v854

    v818._retry_deferred = _retry_deferred_v854
    v818._runtime_validation_callback_v818 = _runtime_validation_callback_v854

    _BASE_ADAPTIVE_WORKER = adaptive._worker_until_win
    adaptive._worker_until_win = _adaptive_worker_v854

    _BASE_ACTOR_GRAPH_CHECK = actor_module._refresh_actor_graph_if_due
    actor_module._refresh_actor_graph_if_due = _actor_graph_check_v854

    _INSTALLED = True
