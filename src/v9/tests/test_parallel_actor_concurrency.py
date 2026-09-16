from __future__ import annotations

from v9 import ContinuousMemoryRuntime
from v9.cli import resolve_game_specs
from v9.runtime.config import RuntimeConfig
from v9.runtime.parallel_memory_coordinator_v2 import run_parallel_memory_jobs


def test_sampling_prefills_distinct_actor_processes(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "parallel",
            restore=False,
            enable_snapshots=False,
            enable_peers=False,
            shards=2,
            stage_workers=1,
        )
    )
    runtime.start()
    specs = resolve_game_specs("step1")[:4]
    jobs = [
        (index + 1, spec, 8, 1000 + index)
        for index, spec in enumerate(specs)
    ]
    try:
        rows = run_parallel_memory_jobs(
            runtime,
            jobs,
            actor_limit=4,
            stage_workers=1,
            shards=2,
            queue_capacity=256,
            epsilon=0.1,
            env_root=None,
            alfred_backend_factory=None,
            start_method=None,
            progress_interval_seconds=60.0,
            ingest_workers=2,
            derivation_workers=2,
            ingest_queue_capacity=256,
            derivation_queue_capacity=128,
            publication_queue_capacity=256,
            actor_view_refresh_steps=16,
            actor_view_refresh_ms=100.0,
        )
        diagnostics = runtime.unified_telemetry.diagnostic_metrics()
        assert len(rows) == 4
        assert int(diagnostics["actor_prefill_complete"]) == 1
        assert int(diagnostics["actor_prefill_processes"]) == 4
        assert int(diagnostics["actor_slots_target"]) == 4
        assert int(diagnostics["peak_active_actor_processes"]) == 4
        pids = {value for value in str(diagnostics["actor_process_pids"]).split(",") if value}
        assert len(pids) >= 4
    finally:
        runtime.close(normal=False)
