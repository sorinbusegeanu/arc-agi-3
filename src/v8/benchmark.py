from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


def run_benchmark(*, events: int, shards: int, stage_workers: int, root: Path | None = None) -> dict[str, object]:
    owned_root = root is None
    root = Path(tempfile.mkdtemp(prefix="arc-v8-bench-")) if root is None else root
    config = V8RuntimeConfig(
        root=root,
        shards=shards,
        stage_workers=stage_workers,
        stage_ring_capacity=max(8192, min(131072, events)),
        shard_ring_capacity=max(8192, min(131072, events)),
        node_capacity_per_shard=max(50_000, events * 2 // max(1, shards)),
        edge_capacity_per_shard=max(100_000, events * 4 // max(1, shards)),
        action_capacity_per_shard=65_536,
        enable_snapshots=False,
        restore=False,
        snapshot_interval_seconds=9999.0,
    )
    runtime = ContinuousMemoryRuntime(config)
    started = time.perf_counter()
    runtime.start()
    submit_started = time.perf_counter()
    try:
        for index in range(events):
            runtime.submit(
                runtime.make_experience(
                    producer_id=1 + index % 16,
                    producer_sequence=1 + index // 16,
                    source_game_hash=1 + index % 10,
                    global_step=index,
                    context_signature=10 + index % 128,
                    action_id=index % 6,
                    outcome_signature=100 + index % 32,
                    family_signature=200 + index % 16,
                    carrier_signature=300 + index % 64,
                    future_option_delta=float((index % 5) - 2),
                    changed_cells=1 + index % 32,
                    trajectory_signature=400 + index % 256,
                ),
                timeout=30.0,
            )
        submit_seconds = time.perf_counter() - submit_started
        runtime.wait_quiescent(timeout=max(60.0, events / 100.0))
        total_seconds = time.perf_counter() - started
        metrics = runtime.metrics()
        return {
            "events": events,
            "shards": shards,
            "stage_workers": stage_workers,
            "submit_seconds": submit_seconds,
            "total_seconds": total_seconds,
            "submit_events_per_second": events / max(submit_seconds, 1e-9),
            "end_to_end_events_per_second": events / max(total_seconds, 1e-9),
            "metrics": metrics,
            "root": str(root),
            "temporary_root": owned_root,
        }
    finally:
        runtime.close(normal=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc-agi3-v8-benchmark")
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--stage-workers", type=int, default=2)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    payload = run_benchmark(
        events=args.events,
        shards=args.shards,
        stage_workers=args.stage_workers,
        root=None if args.root is None else Path(args.root),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
