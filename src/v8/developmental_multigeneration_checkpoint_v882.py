from __future__ import annotations

"""v8.82: bounded multi-generation coherent checkpoints during active sampling.

One coherent full cut can only propose the next developmental layer. Higher-order
formation therefore needs commit barriers between successive immutable cuts so that
M3 -> M4 -> M5/M6 -> M7 can become visible while actors are still sampling.

This layer keeps the existing v8.62 checkpoint cadence and all formation/evidence
gates unchanged. It only expands one due coherent checkpoint into a bounded chain of
up to four committed developmental generations.
"""

import time


_INSTALLED = False
_BASE_COHERENT_CHECKPOINT = None
_GENERATIONS = 4
_COMMIT_TIMEOUT_SECONDS = 5.0
_COMMIT_POLL_SECONDS = 0.005


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


def _coherent_checkpoint_v882(supervisor, *, before_cycles: int, before_cut) -> None:
    """Run bounded immutable generations with checkpoint-scoped commit barriers."""
    runtime = _runtime_for(supervisor)
    for generation in range(_GENERATIONS):
        before_fence = _capture_fence(runtime) if runtime is not None else None
        _BASE_COHERENT_CHECKPOINT(
            supervisor,
            before_cycles=before_cycles,
            before_cut=before_cut,
        )
        if generation + 1 >= _GENERATIONS:
            break
        if runtime is None:
            break
        after_fence = _capture_fence(runtime)
        if not _commit_barrier(supervisor, before_fence, after_fence):
            break


def install_developmental_multigeneration_checkpoint_v882() -> None:
    global _INSTALLED, _BASE_COHERENT_CHECKPOINT
    if _INSTALLED:
        return
    from v8 import incremental_peer_drain_v862 as v862

    _BASE_COHERENT_CHECKPOINT = v862._coherent_checkpoint
    v862._coherent_checkpoint = _coherent_checkpoint_v882
    _INSTALLED = True
