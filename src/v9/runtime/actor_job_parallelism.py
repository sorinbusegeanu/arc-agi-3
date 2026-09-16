from __future__ import annotations

from typing import Any


def expand_jobs_for_actor_limit(
    jobs: list[tuple[int, Any, int, int]],
    actor_limit: int,
) -> list[tuple[int, Any, int, int]]:
    """Split game budgets so --actors controls sampler process parallelism."""
    rows = list(jobs)
    if not rows:
        return rows
    target = max(1, int(actor_limit))
    if len(rows) >= target:
        return rows

    replica_counts = [1 for _ in rows]
    maximum_jobs = min(target, sum(max(1, int(row[2])) for row in rows))
    while sum(replica_counts) < maximum_jobs:
        candidates = [
            index
            for index, row in enumerate(rows)
            if replica_counts[index] < max(1, int(row[2]))
        ]
        if not candidates:
            break
        # Split the currently largest per-process budget first. This keeps
        # actor lifetimes balanced while preserving every game's total budget.
        index = max(
            candidates,
            key=lambda value: (
                float(rows[value][2]) / float(replica_counts[value]),
                -value,
            ),
        )
        replica_counts[index] += 1

    expanded: list[tuple[int, Any, int, int]] = []
    actor_id = 1
    for row_index, ((_old_actor_id, spec, steps, seed), replicas) in enumerate(zip(rows, replica_counts)):
        total_steps = max(1, int(steps))
        base, remainder = divmod(total_steps, replicas)
        for replica in range(replicas):
            replica_steps = base + int(replica < remainder)
            if replica_steps <= 0:
                continue
            replica_seed = int(seed) + replica * 1_000_003 + row_index * 10_007
            expanded.append((actor_id, spec, replica_steps, replica_seed))
            actor_id += 1
    return expanded


def install_actor_job_parallelism(coordinator_module: Any) -> None:
    if getattr(coordinator_module, "_actor_job_parallelism_installed", False):
        return
    original = coordinator_module.run_parallel_memory_jobs

    def run_parallel_memory_jobs(runtime: Any, jobs: list[tuple[int, Any, int, int]], *, actor_limit: int, **kwargs: Any):
        expanded = expand_jobs_for_actor_limit(jobs, actor_limit)
        runtime.set_telemetry_gauge("sampling_jobs_before_actor_split", len(jobs))
        runtime.set_telemetry_gauge("sampling_jobs_after_actor_split", len(expanded))
        return original(runtime, expanded, actor_limit=actor_limit, **kwargs)

    coordinator_module.run_parallel_memory_jobs = run_parallel_memory_jobs
    coordinator_module._actor_job_parallelism_installed = True
