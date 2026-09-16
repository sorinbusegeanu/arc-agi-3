from __future__ import annotations

import json
from pathlib import Path

from v9.research.v979_acceptance import evaluate_v979_long_run


def _write_rows(root: Path, rows: list[dict]) -> None:
    path = root / "telemetry" / "dashboard_metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_acceptance_report_passes_stable_telemetry(tmp_path: Path) -> None:
    rows = []
    for index in range(20):
        rows.append({
            "M0_resident": 800,
            "M1_grounded_resident": 700,
            "resident_M0_limit": 1000,
            "resident_M1_grounded_limit": 1000,
            "compaction_backlog": 0,
            "process_rss_bytes": 1_000_000_000 + index * 1_000_000,
            "process_swap_bytes": 0,
            "bounded_view_nodes": 100,
            "bounded_view_edges": 200,
            "bounded_view_node_scan": 400,
            "bounded_view_edge_scan": 600,
            "sampling_backlog": 0,
        })
    _write_rows(tmp_path, rows)
    assert evaluate_v979_long_run(tmp_path).passed


def test_acceptance_report_rejects_limit_violation(tmp_path: Path) -> None:
    rows = [{
        "M0_resident": 1200,
        "M1_grounded_resident": 1200,
        "resident_M0_limit": 1000,
        "resident_M1_grounded_limit": 1000,
        "compaction_backlog": 100,
        "process_rss_bytes": 2_000_000_000,
        "process_swap_bytes": 1_000_000_000,
        "bounded_view_nodes": 100,
        "bounded_view_edges": 100,
        "bounded_view_node_scan": 10_000,
        "bounded_view_edge_scan": 10_000,
        "sampling_backlog": 100,
    } for _ in range(20)]
    _write_rows(tmp_path, rows)
    result = evaluate_v979_long_run(tmp_path)
    assert not result.passed
    assert not result.resident_counts_bounded
