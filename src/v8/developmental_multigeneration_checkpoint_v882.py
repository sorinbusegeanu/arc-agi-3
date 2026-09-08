from __future__ import annotations

"""v8.82: bounded multi-generation coherent checkpoints during active sampling.

One coherent full cut can only propose the next developmental layer. Higher-order
formation therefore needs commit barriers between successive immutable cuts so that
M3 -> M4 -> M5/M6 -> M7 can become visible while actors are still sampling.

This layer keeps the existing v8.62 checkpoint cadence and all formation/evidence
gates unchanged. It only expands one due coherent checkpoint into a bounded chain of
up to four committed developmental generations.
"""


_INSTALLED = False
_BASE_COHERENT_CHECKPOINT = None
_GENERATIONS = 4
_COMMIT_TIMEOUT_SECONDS = 5.0


def _runtime_for(supervisor):
    submit = getattr(supervisor, "submit_proposal", None)
    runtime = getattr(submit, "__self__", None)
    if runtime is None:
        return None
    wait = getattr(runtime, "wait_quiescent", None)
    return runtime if callable(wait) else None


def _commit_barrier(supervisor) -> bool:
    runtime = _runtime_for(supervisor)
    if runtime is None:
        return False
    runtime.wait_quiescent(
        timeout=_COMMIT_TIMEOUT_SECONDS,
        stable_checks=2,
        resume_peers=True,
        settle_peers=False,
    )
    return True


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
