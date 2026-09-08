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
_COMMIT_STABLE_CHECKS = 2
_COMMIT_POLL_SECONDS = 0.005


def _runtime_for(supervisor):
    submit = getattr(supervisor, "submit_proposal", None)
    runtime = getattr(submit, "__self__", None)
    if runtime is None:
        return None
    if not hasattr(runtime, "_shard_rings") or not hasattr(runtime, "_shard_inflight"):
        return None
    return runtime


def _peer_proposals_committed(runtime) -> bool:
    """Return true when canonical shard proposal queues are momentarily drained.

    Active actors may continue feeding the experience pipeline.  A developmental
    generation only needs the peer proposals submitted directly to shard rings to be
    committed before the next immutable cut.  Waiting for global runtime quiescence
    is incorrect during active sampling because actors intentionally keep stage
    queues non-empty.
    """
    rings = tuple(getattr(runtime, "_shard_rings", ()))
    inflight = tuple(getattr(runtime, "_shard_inflight", ()))
    if not rings or not inflight:
        return False
    if any(not bool(getattr(ring, "empty", False)) for ring in rings):
        return False
    return all(int(getattr(value, "value", 0)) == 0 for value in inflight)


def _commit_barrier(supervisor) -> bool:
    """Wait only for canonical shard proposal commit; never global quiescence."""
    runtime = _runtime_for(supervisor)
    if runtime is None:
        return False
    deadline = time.monotonic() + _COMMIT_TIMEOUT_SECONDS
    stable = 0
    while time.monotonic() < deadline:
        raise_errors = getattr(runtime, "raise_worker_errors", None)
        if callable(raise_errors):
            raise_errors()
        if _peer_proposals_committed(runtime):
            stable += 1
            if stable >= _COMMIT_STABLE_CHECKS:
                return True
        else:
            stable = 0
        time.sleep(_COMMIT_POLL_SECONDS)
    # A busy active-sampling runtime is not a fatal error. Stop this bounded chain
    # and let the next scheduled coherent checkpoint continue development.
    return False


def _coherent_checkpoint_v882(supervisor, *, before_cycles: int, before_cut) -> None:
    """Run bounded immutable generations with canonical queue barriers between them."""
    for generation in range(_GENERATIONS):
        _BASE_COHERENT_CHECKPOINT(
            supervisor,
            before_cycles=before_cycles,
            before_cut=before_cut,
        )
        if generation + 1 >= _GENERATIONS:
            break
        if not _commit_barrier(supervisor):
            break


def install_developmental_multigeneration_checkpoint_v882() -> None:
    global _INSTALLED, _BASE_COHERENT_CHECKPOINT
    if _INSTALLED:
        return
    from v8 import incremental_peer_drain_v862 as v862

    _BASE_COHERENT_CHECKPOINT = v862._coherent_checkpoint
    v862._coherent_checkpoint = _coherent_checkpoint_v882
    _INSTALLED = True
