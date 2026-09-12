from __future__ import annotations

"""v8.82: bounded multi-generation coherent checkpoints during active sampling.

One coherent full cut can only propose the next developmental layer. Higher-order
formation therefore needs commit barriers between successive immutable cuts so that
M3 carrier -> M3 role -> M4 -> M5/M6 -> M7 can become visible while actors are
still sampling.

This layer keeps the existing v8.62 checkpoint cadence and all formation/evidence
gates unchanged. It expands one due coherent checkpoint into six cheap committed
formation generations followed by one full evidence-producing cut: enough for M1
to propagate through M2--M7 without spending the active sampling window repeating
the same expensive generic analyses at every intermediate generation.
"""

import time


_INSTALLED = False
_BASE_COHERENT_CHECKPOINT = None
_GENERATIONS = 7
_COMMIT_TIMEOUT_SECONDS = 5.0
_COMMIT_POLL_SECONDS = 0.005
_INGRESS_DRAIN_TIMEOUT_SECONDS = 5.0
_ACTIVE_M1_LIMIT = 2048
_ACTIVE_EDGE_LIMIT = 16384


def _recent_rows_per_arena(arenas, limit: int):
    """Take a bounded recent suffix from every shard arena."""
    from v8 import incremental_peer_drain_v862 as v862

    arenas = tuple(arenas)
    if not arenas or limit <= 0:
        return ()
    base, remainder = divmod(int(limit), len(arenas))
    rows = []
    for index, arena in enumerate(arenas):
        quota = base + (1 if index < remainder else 0)
        count = max(0, int(arena.count))
        take = min(count, quota)
        if take:
            rows.extend(v862._stable_arena_rows(arena, count - take, take))
    return tuple(rows)


def _materialize_active_rows(read_view):
    """Read live nodes once and only the newest bounded edge window."""
    from v8 import incremental_peer_drain_v862 as v862

    node_arenas = tuple(getattr(read_view, "_nodes", ()))
    edge_arenas = tuple(getattr(read_view, "_edges", ()))
    if v862._arena_backed(node_arenas) and (
        not edge_arenas or v862._arena_backed(edge_arenas)
    ):
        node_total = sum(max(0, int(arena.count)) for arena in node_arenas)
        edge_total = sum(max(0, int(arena.count)) for arena in edge_arenas)
        nodes, _node_offset, _node_wrapped = v862._bounded_arena_slice(
            node_arenas, 0, node_total
        )
        edges = _recent_rows_per_arena(edge_arenas, _ACTIVE_EDGE_LIMIT)
        return tuple(nodes), tuple(edges), edge_total
    edges = tuple(read_view.edge_records())
    return tuple(read_view.node_records()), edges[-_ACTIVE_EDGE_LIMIT:], len(edges)


def _active_checkpoint_view(read_view):
    """Materialize a bounded causal working set for an active checkpoint.

    Re-reading every raw M0/M1 row at each generation makes active formation
    super-linear as sampling advances.  Higher-order canonical state is small and
    must remain complete; M1 is therefore represented by a deterministic cohort.
    Edges are retained only when both live endpoints are present (or the target is
    a pseudo provenance world).  ``source_games`` continues to use the authoritative
    live view, so provenance is not inferred from the sample.
    """
    from v8.model import MemoryLevel

    nodes, live_edges, live_edge_count = _materialize_active_rows(read_view)
    higher = tuple(
        row for row in nodes if int(row.level) >= int(MemoryLevel.M2)
    )
    higher_uids = {row.uid for row in higher}
    all_m1 = {
        row.uid: row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M1)
    }
    # Formation edges point from each higher-order candidate to its causal parent.
    # Prioritize those referenced M1 parents; a UID-only sample can be disjoint from
    # the recent edge window and silently sever otherwise valid M6 -> M1 ancestry.
    referenced_m1_uids = {
        edge.target_uid
        for edge in live_edges
        if edge.source_uid in higher_uids and edge.target_uid in all_m1
    }
    prioritized_m1 = [all_m1[uid] for uid in sorted(referenced_m1_uids)]
    remaining_m1 = [
        row
        for uid, row in sorted(all_m1.items())
        if uid not in referenced_m1_uids
    ]
    m1 = tuple((prioritized_m1 + remaining_m1)[:_ACTIVE_M1_LIMIT])
    selected = tuple(higher + m1)
    selected_uids = {row.uid for row in selected}
    edges = tuple(
        edge
        for edge in live_edges
        if edge.source_uid in selected_uids
        and (edge.target_uid in selected_uids or int(edge.target_uid.hi) == 0)
    )

    class _ActiveCheckpointReadView:
        # Do not expose arena attributes: capture_developmental_cut must consume
        # this already-materialized bounded view rather than its live backing view.
        def node_records(self, *, level=None):
            if level is None:
                return selected
            wanted = int(level)
            return tuple(row for row in selected if int(row.level) == wanted)

        def edge_records(self):
            return edges

        def source_games(self, uid, *, max_depth=8):
            return read_view.source_games(uid, max_depth=max_depth)

    return (
        _ActiveCheckpointReadView(),
        len(nodes),
        len(selected),
        len(edges),
        live_edge_count,
    )


def _run_on_active_checkpoint_view(supervisor, callback) -> tuple[int, int, int, int]:
    original_view = getattr(supervisor, "read_view", None)
    # Dependency-injected supervisors used by specialist tests and tools may not
    # expose a production read view. Preserve their historical direct callback.
    if original_view is None:
        callback()
        return 0, 0, 0, 0
    (
        bounded_view,
        live_count,
        selected_count,
        edge_count,
        live_edge_count,
    ) = _active_checkpoint_view(original_view)
    supervisor.read_view = bounded_view
    try:
        callback()
    finally:
        supervisor.read_view = original_view
    from v8 import information_flow_diagnostics as flow

    flow.emit_bounded(
        "developmental",
        "active_checkpoint_working_view",
        input_count=live_count,
        output_count=selected_count,
        fields={
            "live_edge_count": live_edge_count,
            "selected_edge_count": edge_count,
            "m1_limit": _ACTIVE_M1_LIMIT,
            "edge_limit": _ACTIVE_EDGE_LIMIT,
        },
    )
    return live_count, selected_count, edge_count, live_edge_count


def _runtime_for(supervisor):
    submit = getattr(supervisor, "submit_proposal", None)
    runtime = getattr(submit, "__self__", None)
    if runtime is None:
        return None
    required = ("_shard_rings", "_node_arenas")
    if any(not hasattr(runtime, name) for name in required):
        return None
    return runtime


def _capture_fence(runtime):
    """Capture per-shard enqueue cursors and arena generations for one checkpoint.

    The ring tail is the number of packets enqueued so far.  A later barrier only
    needs each shard worker to consume through the tail produced by this checkpoint;
    unrelated packets arriving afterwards must not block the developmental chain.
    The node-arena seqlock confirms that the batch containing the fence has completed
    its canonical write, rather than merely having been dequeued.
    """
    rings = tuple(getattr(runtime, "_shard_rings", ()))
    arenas = tuple(getattr(runtime, "_node_arenas", ()))
    if not rings or len(rings) != len(arenas):
        return None
    tails = tuple(int(getattr(ring._tail, "value", 0)) for ring in rings)
    sequences = tuple(int(arena.sequence) for arena in arenas)
    return tails, sequences


def _fence_committed(runtime, before_fence, after_fence) -> bool:
    rings = tuple(getattr(runtime, "_shard_rings", ()))
    arenas = tuple(getattr(runtime, "_node_arenas", ()))
    if before_fence is None or after_fence is None:
        return False
    before_tails, before_sequences = before_fence
    target_tails, _after_sequences = after_fence
    if len(rings) != len(target_tails) or len(arenas) != len(target_tails):
        return False

    for index, (ring, arena) in enumerate(zip(rings, arenas, strict=True)):
        target = int(target_tails[index])
        if target <= int(before_tails[index]):
            continue
        head = int(getattr(ring._head, "value", 0))
        sequence = int(arena.sequence)
        if head < target:
            return False
        if sequence & 1:
            return False
        if sequence <= int(before_sequences[index]):
            return False
    return True


def _commit_barrier(supervisor, before_fence, after_fence) -> bool:
    """Wait for this checkpoint's proposal fence, never global shard emptiness."""
    runtime = _runtime_for(supervisor)
    if runtime is None:
        return False
    deadline = time.monotonic() + _COMMIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        raise_errors = getattr(runtime, "raise_worker_errors", None)
        if callable(raise_errors):
            raise_errors()
        if _fence_committed(runtime, before_fence, after_fence):
            return True
        time.sleep(_COMMIT_POLL_SECONDS)
    # A busy runtime is not fatal. Stop this bounded chain and let the next
    # scheduled checkpoint continue development.
    return False


def _drain_published_ingress(runtime) -> bool:
    """Align the immutable cut with events published before the actor freeze."""
    quiescent = getattr(runtime, "_is_quiescent", None)
    if not callable(quiescent):
        return True
    deadline = time.monotonic() + _INGRESS_DRAIN_TIMEOUT_SECONDS
    stable = 0
    while time.monotonic() < deadline:
        raise_errors = getattr(runtime, "raise_worker_errors", None)
        if callable(raise_errors):
            raise_errors()
        if quiescent():
            stable += 1
            if stable >= 2:
                return True
        else:
            stable = 0
        time.sleep(_COMMIT_POLL_SECONDS)
    return False


def _formation_checkpoint(supervisor, *, before_cycles: int, before_cut) -> None:
    """Run one immutable formation cut with the v8.62 checkpoint bookkeeping."""
    from v8 import stabilization_noop_retry_v883 as v883

    supervisor._cycles = before_cycles
    if hasattr(supervisor, "_last_developmental_cut"):
        supervisor._last_developmental_cut = before_cut
    prior_stabilizing = bool(getattr(supervisor, "_v82_stabilizing", False))
    supervisor._v82_stabilizing = True
    try:
        # Intermediate active generations only need causal hierarchy formation.
        # Structural correspondence is edge-heavy and is evaluated by the final
        # full checkpoint cut after all transient layers have committed.
        _run_on_active_checkpoint_view(
            supervisor,
            lambda: v883._run_formation_cut_once(
                supervisor, include_correspondence=False
            ),
        )
    finally:
        supervisor._v82_stabilizing = prior_stabilizing


def _coherent_checkpoint_v882(supervisor, *, before_cycles: int, before_cut) -> None:
    """Run bounded immutable generations with checkpoint-scoped commit barriers."""
    runtime = _runtime_for(supervisor)
    freeze = getattr(runtime, "_snapshot_freeze", None) if runtime is not None else None
    was_frozen = bool(freeze is not None and freeze.is_set())
    if freeze is not None and not was_frozen:
        # Stop publishers only long enough to drain the ingress already visible at
        # this checkpoint boundary.  The materialized working views below are
        # immutable tuples, so expensive formation/evidence analysis must not keep
        # every actor parked for the lifetime of that analysis.
        freeze.set()
    try:
        ingress_drained = bool(runtime is None or _drain_published_ingress(runtime))
        if runtime is not None:
            from v8 import information_flow_diagnostics as flow

            flow.emit_bounded(
                "developmental",
                "active_checkpoint_ingress_drain",
                input_count=1,
                output_count=int(ingress_drained),
                rejection_counts=(
                    {} if ingress_drained else {"published_ingress_not_drained": 1}
                ),
            )
    finally:
        if freeze is not None and not was_frozen:
            freeze.clear()

    try:
        for generation in range(_GENERATIONS):
            started = time.monotonic()
            before_fence = _capture_fence(runtime) if runtime is not None else None
            formation_only = bool(runtime is not None and generation + 1 < _GENERATIONS)
            if formation_only:
                _formation_checkpoint(
                    supervisor,
                    before_cycles=before_cycles,
                    before_cut=before_cut,
                )
            else:
                # The last generation deliberately retains the complete public
                # wrapper chain so all evidence producers see the M7-capable cut.
                _run_on_active_checkpoint_view(
                    supervisor,
                    lambda: _BASE_COHERENT_CHECKPOINT(
                        supervisor,
                        before_cycles=before_cycles,
                        before_cut=before_cut,
                    ),
                )
            from v8 import information_flow_diagnostics as flow

            flow.emit_bounded(
                "developmental",
                "active_coherent_checkpoint",
                input_count=generation + 1,
                output_count=1,
                fields={
                    "checkpoint_generation": generation + 1,
                    "checkpoint_phase": "formation" if formation_only else "full_analysis",
                    "elapsed_seconds": max(0.0, time.monotonic() - started),
                },
            )
            if generation + 1 >= _GENERATIONS:
                break
            if runtime is None:
                break
            after_fence = _capture_fence(runtime)
            if not _commit_barrier(supervisor, before_fence, after_fence):
                flow.emit_bounded(
                    "developmental",
                    "active_coherent_checkpoint_barrier",
                    input_count=1,
                    output_count=0,
                    rejection_counts={"proposal_fence_not_committed": 1},
                    fields={"checkpoint_generation": generation + 1},
                )
                break
    finally:
        # Preserve a freeze owned by an enclosing snapshot/finalization operation,
        # but never reacquire the short ingress fence during expensive analysis.
        if freeze is not None and was_frozen:
            freeze.set()


def install_developmental_multigeneration_checkpoint_v882() -> None:
    global _INSTALLED, _BASE_COHERENT_CHECKPOINT
    if _INSTALLED:
        return
    from v8 import incremental_peer_drain_v862 as v862

    _BASE_COHERENT_CHECKPOINT = v862._coherent_checkpoint
    v862._coherent_checkpoint = _coherent_checkpoint_v882
    _INSTALLED = True
