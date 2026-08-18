from __future__ import annotations

"""Composition fixes for v8.41 over the existing v8.19/v8.30 authorities."""


_INSTALLED = False


def _feedback_worker_compat(runtime):
    from v8 import lease_dispatch_continuity_v839 as v839
    from v8 import runtime_scaling_v841 as v841

    if hasattr(runtime, "root"):
        return v841._feedback_worker_v841(runtime)
    worker = getattr(runtime, "_v839_actor_feedback", None)
    if worker is None or bool(getattr(worker, "_closed", False)):
        worker = v839._AsyncQueueWorker(
            lambda rows: v839._BASE_RECORD_ACTOR_RESULTS(runtime, tuple(rows)),
            name="v8-actor-feedback",
            error_queue=getattr(runtime, "_error_queue", None),
        )
        runtime._v839_actor_feedback = worker
    return worker


def _peer_input_token_compat(supervisor) -> tuple[int, ...]:
    view = getattr(supervisor, "_v813_live_read_view", None) or getattr(
        supervisor, "read_view", None
    )
    arenas = tuple(getattr(view, "_nodes", ())) + tuple(getattr(view, "_edges", ()))
    if arenas and all(hasattr(arena, "sequence") for arena in arenas):
        return tuple(int(arena.sequence) for arena in arenas)
    return (
        int(supervisor.current_generation()),
        int(supervisor.current_watermark()),
    )


def install_runtime_scaling_v841_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import lease_dispatch_continuity_v839 as v839
    from v8 import optimizer_budget_control_v830 as v830
    from v8 import runtime_scaling_v841 as v841

    # Preserve the complete routing stack:
    # v8.30 budget precheck -> v8.19 runtime/source semantics -> v8.41 nonblocking queue.
    v830._BASE_ROUTE_CANDIDATE = v819._route_candidate_v819
    v819._BASE_ROUTE_CANDIDATE = v841._route_candidate_base_v841

    v839._feedback_worker = _feedback_worker_compat
    v841._peer_input_token = _peer_input_token_compat
    _INSTALLED = True
