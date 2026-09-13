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
    assert metrics["ingest_worker_processes"] == 4
    assert metrics["derivation_worker_processes"] == 4
    assert metrics["multiprocess_transitions_published"] == 12
    assert metrics["sampling_backlog"] == 0


def test_progress_is_written_to_stdout(tmp_path, capsys) -> None:
    root = tmp_path / "progress"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--progress-interval-seconds", "60",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0
    output = capsys.readouterr().out
    assert "v9 progress" in output
    assert "sampled=12/12" in output
    assert "ingested=12" in output
    assert "M0=" in output
    assert "M7=" in output


def test_restored_parallel_run_appends_unique_m0(tmp_path) -> None:
    root = tmp_path / "append"
    argv = [
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--ingest-workers", "2",
        "--derivation-workers", "2",
        "--no-peers",
        "--no-dashboard",
    ]
    assert run_continuous(build_parser().parse_args(argv)) == 0
    first = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    first_m0 = first["metrics"]["memory_levels"]["M0"]

    assert run_continuous(build_parser().parse_args(argv)) == 0
    second = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    second_m0 = second["metrics"]["memory_levels"]["M0"]

    assert second_m0 > first_m0
