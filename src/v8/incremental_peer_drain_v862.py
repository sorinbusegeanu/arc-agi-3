from __future__ import annotations

"""v8.62: bounded incremental peer processing and finite post-sampling drain.

The historical peer pass materializes the complete node/edge graph before doing any
work. That is acceptable for small graphs but makes one peer pass effectively
uninterruptible once the graph reaches millions of rows. v8.62 keeps the existing
peer semantics and authority chain while feeding the historical pass bounded,
coherent arena slices. Scan offsets are stored in the already-persisted ``_seen``
map, so interrupted maintenance resumes on the next run without a full rescan.
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
        # Peer state restoration merges _seen values with max(), therefore the
        # persisted cursor is an absolute monotonic offset rather than a row index.
        seen[key] = max(int(seen.get(key, 0)), max(0, int(value)))


def _arena_backed(values) -> bool:
    values = tuple(values)
    return bool(values) and all(
        hasattr(value, "count")
        and hasattr(value, "read")
        and hasattr(value, "sequence")
        for value in values
    )


def _peer_run_once_v862(self):
    if bool(getattr(self, "_v82_stabilizing", False)):
        return _BASE_PEER_RUN_ONCE(self)
    view = getattr(self, "_v813_live_read_view", None) or getattr(self, "read_view", None)
    if view is None:
        return _BASE_PEER_RUN_ONCE(self)

    node_arenas = tuple(getattr(view, "_nodes", ()))
    edge_arenas = tuple(getattr(view, "_edges", ()))
    # Several scientific unit fixtures intentionally expose already-materialized
    # NodeRecord/EdgeRecord tuples here. They are small and must retain historical
    # behavior; only production shared arenas need incremental slicing.
    if not _arena_backed(node_arenas) or (edge_arenas and not _arena_backed(edge_arenas)):
        return _BASE_PEER_RUN_ONCE(self)

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
    before_cycles = int(getattr(self, "_cycles", 0))
    before_cut = getattr(self, "_last_developmental_cut", None)
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
    else:
        # v8.41 marks an input token complete when the delegated cycle counter or
        # developmental cut advances. Keep both unchanged until one bounded sweep
        # has completed so the next interval consumes the next slice.
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
            deadline = time.monotonic() + max(0.0, float(timeout))

            def commit_proposals() -> None:
                _BASE_RUNTIME_WAIT(
                    self, timeout=max(0.0, deadline - time.monotonic()),
                    stable_checks=stable_checks, resume_peers=False, settle_peers=False,
                )

            stabilize(max_cycles=8, commit_proposals=commit_proposals, timeout=timeout)
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
