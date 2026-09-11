from __future__ import annotations

"""v8.62: bounded incremental peer processing and finite post-sampling drain.

The historical peer pass materializes the complete node/edge graph before doing any
work. That is acceptable for small graphs but makes one peer pass effectively
uninterruptible once the graph reaches millions of rows. v8.62 keeps the existing
peer semantics and authority chain while feeding the historical pass bounded,
coherent arena slices. Scan offsets are stored in the already-persisted ``_seen``
map, so interrupted maintenance resumes on the next run without a full rescan.

Higher-order formation is lineage-dependent. A bounded node/edge slice is not, in
general, a connected causal subgraph. Coherent full-cut checkpoints therefore run
on an independent bounded watermark/time cadence during active sampling, rather
than waiting for a complete incremental sweep. The slice-maintenance cursor remains
independent and continues from its current offset after each checkpoint.
"""

import time


_INSTALLED = False
_BASE_PEER_RUN_ONCE = None
_BASE_RUNTIME_WAIT = None

_NODE_SLICE = 1024
_EDGE_SLICE = 4096
_SLICE_READ_TIMEOUT = 0.25
_NODE_OFFSET_KEY = ("__v862_node_offset", 0, 0)
_EDGE_OFFSET_KEY = ("__v862_edge_offset", 0, 0)

# Full causal cuts are intentionally much less frequent than bounded maintenance.
# Watermark cadence is the primary production trigger; the time trigger prevents
# slow environments from waiting indefinitely while still requiring real new data.
_COHERENT_WATERMARK_INTERVAL = 10_000
_COHERENT_TIME_INTERVAL_SECONDS = 15.0
_COHERENT_MIN_WATERMARK_PROGRESS = 2_000
_FINAL_STABILIZATION_WINDOWS = 2


def _stable_arena_rows(arena, start: int, count: int, *, timeout: float = _SLICE_READ_TIMEOUT):
    if count <= 0:
        return ()
    deadline = time.monotonic() + max(0.01, float(timeout))
    while time.monotonic() < deadline:
        seq1 = int(arena.sequence)
        if seq1 & 1:
            time.sleep(0.0005)
            continue
        rows = tuple(arena.read(index) for index in range(int(start), int(start) + int(count)))
        seq2 = int(arena.sequence)
        if seq1 == seq2 and not (seq2 & 1):
            return rows
        time.sleep(0)
    return ()


def _bounded_arena_slice(arenas, absolute_offset: int, limit: int):
    arenas = tuple(arenas)
    counts = tuple(max(0, int(arena.count)) for arena in arenas)
    total = sum(counts)
    if total <= 0 or limit <= 0:
        return (), int(absolute_offset), True

    start = int(absolute_offset) % total
    remaining = min(int(limit), total)
    rows = []
    logical = start
    consumed = 0

    while remaining > 0:
        position = logical % total
        prefix = 0
        arena_index = 0
        row_index = 0
        for index, arena_count in enumerate(counts):
            if position < prefix + arena_count:
                arena_index = index
                row_index = position - prefix
                break
            prefix += arena_count
        available = counts[arena_index] - row_index
        take = min(remaining, available)
        part = _stable_arena_rows(arenas[arena_index], row_index, take)
        if len(part) != take:
            break
        rows.extend(part)
        consumed += take
        remaining -= take
        logical += take

    next_offset = int(absolute_offset) + int(consumed)
    wrapped = consumed >= total or (start + consumed) >= total
    return tuple(rows), next_offset, bool(wrapped)


def _saved_offset(supervisor, key) -> int:
    seen = getattr(supervisor, "_seen", None)
    if not isinstance(seen, dict):
        return 0
    return max(0, int(seen.get(key, 0)))


def _save_offset(supervisor, key, value: int) -> None:
    seen = getattr(supervisor, "_seen", None)
    if isinstance(seen, dict):
        seen[key] = max(int(seen.get(key, 0)), max(0, int(value)))


def _arena_backed(values) -> bool:
    values = tuple(values)
    return bool(values) and all(
        hasattr(value, "count")
        and hasattr(value, "read")
        and hasattr(value, "sequence")
        for value in values
    )


def _needs_coherent_checkpoint(node_arenas, edge_arenas) -> bool:
    node_total = sum(max(0, int(arena.count)) for arena in tuple(node_arenas))
    edge_total = sum(max(0, int(arena.count)) for arena in tuple(edge_arenas))
    return node_total > _NODE_SLICE or edge_total > _EDGE_SLICE


def _current_watermark(supervisor) -> int:
    callback = getattr(supervisor, "current_watermark", None)
    if callable(callback):
        return max(0, int(callback()))
    return max(0, int(getattr(supervisor, "watermark", 0)))


def _coherent_checkpoint_due(supervisor, *, watermark: int, now: float) -> bool:
    last_time = getattr(supervisor, "_v862_last_coherent_checkpoint_time", None)
    last_watermark = getattr(supervisor, "_v862_last_coherent_checkpoint_watermark", None)
    if last_time is None or last_watermark is None:
        # Do not anchor the first checkpoint to an already-advanced watermark. A
        # bounded peer pass can itself take long enough for sampling to advance by
        # tens of thousands of steps. Treat the run origin as the developmental
        # baseline so the first large-graph checkpoint can run immediately once
        # meaningful evidence exists.
        supervisor._v862_last_coherent_checkpoint_time = float(now)
        supervisor._v862_last_coherent_checkpoint_watermark = 0
        return int(watermark) >= _COHERENT_MIN_WATERMARK_PROGRESS

    progress = max(0, int(watermark) - int(last_watermark))
    elapsed = max(0.0, float(now) - float(last_time))
    return bool(
        progress >= _COHERENT_WATERMARK_INTERVAL
        or (
            progress >= _COHERENT_MIN_WATERMARK_PROGRESS
            and elapsed >= _COHERENT_TIME_INTERVAL_SECONDS
        )
    )


def _mark_coherent_checkpoint(supervisor, *, watermark: int, now: float) -> None:
    supervisor._v862_last_coherent_checkpoint_watermark = max(0, int(watermark))
    supervisor._v862_last_coherent_checkpoint_time = float(now)


def _coherent_checkpoint(supervisor, *, before_cycles: int, before_cut) -> None:
    """Run one complete immutable causal cut without resetting slice progress."""
    supervisor._cycles = before_cycles
    if hasattr(supervisor, "_last_developmental_cut"):
        supervisor._last_developmental_cut = before_cut
    prior_stabilizing = bool(getattr(supervisor, "_v82_stabilizing", False))
    supervisor._v82_stabilizing = True
    try:
        _BASE_PEER_RUN_ONCE(supervisor)
    finally:
        supervisor._v82_stabilizing = prior_stabilizing


def _peer_run_once_v862(self):
    if bool(getattr(self, "_v82_stabilizing", False)):
        return _BASE_PEER_RUN_ONCE(self)
    view = getattr(self, "_v813_live_read_view", None) or getattr(self, "read_view", None)
    if view is None:
        return _BASE_PEER_RUN_ONCE(self)

    node_arenas = tuple(getattr(view, "_nodes", ()))
    edge_arenas = tuple(getattr(view, "_edges", ()))
    if not _arena_backed(node_arenas) or (edge_arenas and not _arena_backed(edge_arenas)):
        return _BASE_PEER_RUN_ONCE(self)

    before_cycles = int(getattr(self, "_cycles", 0))
    before_cut = getattr(self, "_last_developmental_cut", None)
    large_graph = _needs_coherent_checkpoint(node_arenas, edge_arenas)
    watermark = _current_watermark(self)
    now = time.monotonic()

    # Developmental formation has priority over bounded maintenance. Previously the
    # cadence was evaluated only after the bounded pass; if that pass was slow, the
    # watermark could reach the end of sampling before the first full causal cut was
    # even considered. Evaluate and execute a due coherent checkpoint first.
    if large_graph and _coherent_checkpoint_due(self, watermark=watermark, now=now):
        _coherent_checkpoint(self, before_cycles=before_cycles, before_cut=before_cut)
        _mark_coherent_checkpoint(self, watermark=watermark, now=now)
        return None

    node_rows, node_offset, node_wrapped = _bounded_arena_slice(
        node_arenas, _saved_offset(self, _NODE_OFFSET_KEY), _NODE_SLICE
    )
    edge_rows, edge_offset, edge_wrapped = _bounded_arena_slice(
        edge_arenas, _saved_offset(self, _EDGE_OFFSET_KEY), _EDGE_SLICE
    )
    _save_offset(self, _NODE_OFFSET_KEY, node_offset)
    _save_offset(self, _EDGE_OFFSET_KEY, edge_offset)

    original_nodes = view.node_records
    original_edges = view.edge_records
    prior_edge_wrap = bool(getattr(self, "_v862_edge_wrapped_since_cycle", False))
    self._v862_edge_wrapped_since_cycle = prior_edge_wrap or bool(edge_wrapped)

    def sliced_nodes(*, level=None):
        if level is None:
            return tuple(node_rows)
        wanted = int(level)
        return tuple(row for row in node_rows if int(row.level) == wanted)

    def sliced_edges():
        return tuple(edge_rows)

    view.node_records = sliced_nodes
    view.edge_records = sliced_edges
    try:
        result = _BASE_PEER_RUN_ONCE(self)
    finally:
        view.node_records = original_nodes
        view.edge_records = original_edges

    completed_sweep = bool(node_wrapped and self._v862_edge_wrapped_since_cycle)
    if completed_sweep:
        self._v862_edge_wrapped_since_cycle = False

    # Re-evaluate after bounded maintenance because sampling may have advanced while
    # that pass was running. This catches long maintenance passes without waiting for
    # another supervisor interval.
    watermark_after = _current_watermark(self)
    now_after = time.monotonic()
    cadence_due_after = bool(
        large_graph
        and _coherent_checkpoint_due(self, watermark=watermark_after, now=now_after)
    )

    if large_graph and (completed_sweep or cadence_due_after):
        _coherent_checkpoint(self, before_cycles=before_cycles, before_cut=before_cut)
        _mark_coherent_checkpoint(self, watermark=watermark_after, now=now_after)
    elif not completed_sweep:
        self._cycles = before_cycles
        if hasattr(self, "_last_developmental_cut"):
            self._last_developmental_cut = before_cut
    return result


def _runtime_wait_quiescent_v862(
    self,
    *,
    timeout: float = 60.0,
    stable_checks: int = 5,
    resume_peers: bool = True,
    settle_peers: bool = True,
):
    final_drain = bool(getattr(self, "_sampling_complete", False)) or not bool(
        getattr(self, "_accepting", True)
    )
    if final_drain:
        settle_peers = False
        resume_peers = False
        peers = getattr(self, "peers", None)
        stabilize = getattr(peers, "run_until_stable", None)
        generation = int(getattr(self, "generation", -1))
        last_generation = getattr(self, "_v82_developmental_finalized_generation", None)
        if callable(stabilize) and (
            not getattr(self, "_v82_developmental_finalized", False)
            or last_generation != generation
        ):
            peers.pause()
            def commit_proposals() -> None:
                _BASE_RUNTIME_WAIT(
                    self, timeout=max(0.0, float(timeout)),
                    stable_checks=stable_checks, resume_peers=False, settle_peers=False,
                )

            reason = None
            for _window in range(_FINAL_STABILIZATION_WINDOWS):
                reason = stabilize(
                    max_cycles=8,
                    commit_proposals=commit_proposals,
                    timeout=timeout,
                )
                if reason != "max_cycles":
                    break
            if reason == "max_cycles":
                raise TimeoutError(
                    "v8 developmental finalization did not stabilize after "
                    f"{_FINAL_STABILIZATION_WINDOWS} bounded windows"
                )
            self._v82_developmental_finalized = True
            self._v82_developmental_finalized_generation = int(
                getattr(self, "generation", generation)
            )
            return
    return _BASE_RUNTIME_WAIT(
        self,
        timeout=timeout,
        stable_checks=stable_checks,
        resume_peers=resume_peers,
        settle_peers=settle_peers,
    )


def install_incremental_peer_drain_v862() -> None:
    global _INSTALLED, _BASE_PEER_RUN_ONCE, _BASE_RUNTIME_WAIT
    if _INSTALLED:
        return

    from v8 import runtime_scaling_v841 as v841
    from v8.runtime import ContinuousMemoryRuntime

    _BASE_PEER_RUN_ONCE = v841._BASE_PEER_RUN_ONCE
    v841._BASE_PEER_RUN_ONCE = _peer_run_once_v862

    _BASE_RUNTIME_WAIT = ContinuousMemoryRuntime.wait_quiescent
    ContinuousMemoryRuntime.wait_quiescent = _runtime_wait_quiescent_v862
    _INSTALLED = True
