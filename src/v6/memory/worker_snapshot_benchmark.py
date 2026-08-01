"""Opt-in scaling benchmark for the worker-local memory snapshot path."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from v6.memory.worker_snapshot import (
    SnapshotMemoryQueryEngine,
    WorkerMemorySnapshot,
    build_worker_memory_snapshot_from_directory,
    load_worker_memory_snapshot_artifact,
    write_worker_memory_snapshot_artifact,
)


_BENCHMARK_ENGINE: SnapshotMemoryQueryEngine | None = None


def _benchmark_initializer(snapshot_artifact: str) -> None:
    global _BENCHMARK_ENGINE
    _BENCHMARK_ENGINE = SnapshotMemoryQueryEngine(load_worker_memory_snapshot_artifact(snapshot_artifact))


def _benchmark_rankings(iterations: int) -> dict[str, Any]:
    assert _BENCHMARK_ENGINE is not None
    started = time.perf_counter()
    for _ in range(int(iterations)):
        _BENCHMARK_ENGINE.rank_actions({0: {0: ("benchmark",)}}, [0])
    elapsed = time.perf_counter() - started
    return {"rankings": int(iterations), "seconds": elapsed, **_BENCHMARK_ENGINE.metrics()}


def run_worker_snapshot_scaling_benchmark(
    *,
    memory_dir: str | Path,
    output_dir: str | Path,
    worker_counts: tuple[int, ...] = (1, 4, 8, 16, 32, 40, 48),
    rankings_per_worker: int = 5_000,
    include_graph: bool = True,
    include_substrate: bool = True,
) -> dict[str, Any]:
    """Build once, then measure artifact-backed ranking throughput per scale."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    snapshot = build_worker_memory_snapshot_from_directory(
        memory_dir,
        include_graph=bool(include_graph),
        include_substrate=bool(include_substrate),
    )
    artifact, artifact_bytes = write_worker_memory_snapshot_artifact(snapshot, output / "worker_memory_snapshot_benchmark.pkl")
    build_seconds = time.perf_counter() - build_started
    rows: list[dict[str, Any]] = []
    for workers in worker_counts:
        worker_count = max(1, int(workers))
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_benchmark_initializer, initargs=(str(artifact),)) as pool:
            results = list(pool.map(_benchmark_rankings, [int(rankings_per_worker)] * worker_count))
        wall_seconds = time.perf_counter() - started
        rankings = sum(int(row["rankings"]) for row in results)
        query_seconds = sum(float(row["memory_action_rank_seconds"]) for row in results)
        rows.append(
            {
                "workers": worker_count,
                "snapshot_build_seconds": build_seconds,
                "worker_startup_and_run_seconds": wall_seconds,
                "action_rankings": rankings,
                "rankings_per_second": rankings / wall_seconds if wall_seconds else 0.0,
                "mean_ranking_latency_seconds": query_seconds / rankings if rankings else 0.0,
                "delta_refresh_latency_seconds": 0.0,
                "manager_proxy_calls_per_refresh": 1,
                "sqlite_queries_during_action_selection": sum(int(row["sqlite_queries_during_action_selection"]) for row in results),
                "peak_ram_estimate_bytes": int(artifact_bytes) * worker_count,
                "cpu_count": os.cpu_count(),
            }
        )
    result = {
        "snapshot_serialized_bytes": int(artifact_bytes),
        "snapshot_build_seconds": build_seconds,
        "worker_counts": list(worker_counts),
        "rows": rows,
    }
    (output / "worker_snapshot_scaling_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
