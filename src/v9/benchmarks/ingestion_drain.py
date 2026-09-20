from __future__ import annotations

import argparse
import json
import resource
import statistics
import tempfile
import time
from pathlib import Path

from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.canonical_commit import apply_canonical_commit_batch
from v9.runtime.memory_pipeline import IngestionTask, build_commit_plan, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition
from v9.memory.model import MemoryLevel


def _plans(count: int):
    result = []
    watermark = 0
    for index in range(count):
        symbols = (f"token-{index % 7}",) if index % 19 == 0 else ()
        transition = EncodedTransition(
            actor_id=1,
            producer_sequence=index + 1,
            global_step=index,
            environment_identity=("synthetic", "v9716-benchmark", "default", "seed=0"),
            episode_id=7,
            observation_schema_id=11,
            before_signature=index % 17,
            action_id=index % 5,
            after_signature=(index + 1) % 17,
            available_actions_after=5,
            primary_valence=int(index % 101 == 0),
            symbols=symbols,
            curriculum_step="benchmark",
            game_scenario="v9716-benchmark",
        )
        watermark += 1
        result.append(build_commit_plan(prepare_ingestion(IngestionTask(index + 1, watermark, transition))))
        watermark += len(symbols)
    return tuple(result)


def run(*, rows: int = 4096, repeats: int = 3) -> dict[str, object]:
    plans = _plans(rows)
    samples = []
    with tempfile.TemporaryDirectory(prefix="v9716-ingestion-") as directory:
        for repeat in range(repeats):
            runtime = ContinuousMemoryRuntime(
                RuntimeConfig.from_path(Path(directory) / f"repeat-{repeat}", restore=False, enable_snapshots=False)
            )
            commit_started = time.perf_counter()
            result = apply_canonical_commit_batch(runtime, plans)
            commit_seconds = time.perf_counter() - commit_started
            drain_started = time.perf_counter()
            runtime.flush_deferred_memory_updates()
            drain_seconds = time.perf_counter() - drain_started
            memory_counts = {
                f"M{level}": runtime.graph.memory_count(MemoryLevel(level))
                for level in range(8)
            }
            samples.append(
                {
                    "repeat": repeat + 1,
                    "rows": rows,
                    "signature_rows": len(result.signature_rows),
                    "commit_seconds": commit_seconds,
                    "drain_seconds": drain_seconds,
                    "total_seconds": commit_seconds + drain_seconds,
                    "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                    "tracked_shm_bytes": 0,
                    "memory_counts": memory_counts,
                }
            )
            runtime.close(normal=False)
    return {
        "schema": "v9.7.16-fixed-ingestion-drain-v1",
        "rows": rows,
        "repeats": repeats,
        "samples": samples,
        "median_commit_seconds": statistics.median(row["commit_seconds"] for row in samples),
        "median_drain_seconds": statistics.median(row["drain_seconds"] for row in samples),
        "median_total_seconds": statistics.median(row["total_seconds"] for row in samples),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in samples),
        "peak_tracked_shm_bytes": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run(rows=args.rows, repeats=args.repeats)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
