from __future__ import annotations


_INSTALLED = False


def _run_actor_jobs_v819(runtime, jobs, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf

    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if (
        coordinator is not None
        and int(coordinator.config.lease_steps) != int(v819._DEFAULT_LEASE_STEPS)
    ):
        # An explicit lease override is authoritative. The original v8.19
        # scheduler already implements configurable short leases and its tests
        # rely on that behavior.
        return perf._BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)
    return perf._adaptive_run_actor_jobs_perf(runtime, jobs, **kwargs)


def install_adaptive_learning_allocation_v819_performance_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module

    actor_module.run_actor_jobs = _run_actor_jobs_v819
    _INSTALLED = True
