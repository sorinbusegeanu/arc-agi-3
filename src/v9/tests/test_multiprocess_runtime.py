from __future__ import annotations

import json

from v9.cli import build_parser, run_continuous


def test_process_topology_is_reported(tmp_path) -> None:
    root = tmp_path / "mp"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "2",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0
    summary = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]["telemetry_diagnostics"]
    assert metrics["actor_processes"] == 2
    assert metrics["stage_worker_processes"] == 2
    assert metrics["shard_worker_processes"] == 2
    assert metrics["multiprocess_transitions_published"] == 12
