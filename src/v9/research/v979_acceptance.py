from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class V979AcceptanceResult:
    passed: bool
    samples: int
    steady_state_samples: int
    resident_counts_bounded: bool
    compaction_recovers: bool
    rss_stable: bool
    swap_controlled: bool
    bounded_scans_respected: bool
    sampling_backlog_drains: bool
    details: dict[str, Any]


def _value(row: dict[str, Any], key: str, default: Any = 0) -> Any:
    if key in row:
        return row[key]
    for container in ("telemetry_diagnostics", "primary_dashboard"):
        nested = row.get(container)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return default


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _slope_ratio(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 4:
        return 0.0
    half = len(rows) // 2
    early = median(rows[:half])
    late = median(rows[half:])
    return (late - early) / max(1.0, abs(early))


def evaluate_v979_long_run(
    root: str | Path,
    *,
    steady_state_fraction: float = 0.40,
    rss_growth_tolerance: float = 0.15,
) -> V979AcceptanceResult:
    root = Path(root)
    rows = _load(root / "telemetry" / "dashboard_metrics.jsonl")
    if not rows:
        return V979AcceptanceResult(False, 0, 0, False, False, False, False, False, False, {"reason": "no dashboard telemetry"})

    fraction = max(0.10, min(0.80, float(steady_state_fraction)))
    tail_count = max(4, int(len(rows) * fraction))
    steady = rows[-tail_count:]

    m0 = [int(_value(row, "M0_resident", 0)) for row in steady]
    m1 = [int(_value(row, "M1_grounded_resident", 0)) for row in steady]
    m0_limits = [int(_value(row, "resident_M0_limit", 0)) for row in steady]
    m1_limits = [int(_value(row, "resident_M1_grounded_limit", 0)) for row in steady]
    backlogs = [int(_value(row, "compaction_backlog", 0)) for row in steady]
    rss = [int(_value(row, "process_rss_bytes", 0)) for row in steady]
    swap = [int(_value(row, "process_swap_bytes", 0)) for row in steady]
    node_scans = [int(_value(row, "bounded_view_node_scan", 0)) for row in steady]
    edge_scans = [int(_value(row, "bounded_view_edge_scan", 0)) for row in steady]
    view_nodes = [int(_value(row, "bounded_view_nodes", 0)) for row in steady]
    view_edges = [int(_value(row, "bounded_view_edges", 0)) for row in steady]
    sampling_backlog = [int(_value(row, "sampling_backlog", 0)) for row in steady]

    resident_counts_bounded = all(
        (limit <= 0 or value <= limit)
        for value, limit in zip(m0, m0_limits)
    ) and all(
        (limit <= 0 or value <= limit)
        for value, limit in zip(m1, m1_limits)
    )
    compaction_recovers = min(backlogs[-max(1, len(backlogs) // 4):], default=0) == 0 or backlogs[-1] <= max(1, int(median(backlogs) if backlogs else 0))
    rss_growth = _slope_ratio(rss)
    rss_stable = rss_growth <= float(rss_growth_tolerance)
    swap_peak = max(swap, default=0)
    swap_controlled = swap_peak <= max(512 * 1024 * 1024, int(median(swap) * 2 if swap else 0))
    bounded_scans_respected = all(
        scan <= max(64, nodes * 8 + 64)
        for scan, nodes in zip(node_scans, view_nodes)
    ) and all(
        scan <= max(256, edges * 4, nodes * 8)
        for scan, edges, nodes in zip(edge_scans, view_edges, view_nodes)
    )
    sampling_backlog_drains = min(sampling_backlog[-max(1, len(sampling_backlog) // 4):], default=0) == 0

    checks = (
        resident_counts_bounded,
        compaction_recovers,
        rss_stable,
        swap_controlled,
        bounded_scans_respected,
        sampling_backlog_drains,
    )
    details = {
        "steady_state_fraction": fraction,
        "rss_growth_ratio": rss_growth,
        "rss_growth_tolerance": float(rss_growth_tolerance),
        "rss_peak_bytes": max(rss, default=0),
        "swap_peak_bytes": swap_peak,
        "m0_resident_median": median(m0) if m0 else 0,
        "m1_grounded_resident_median": median(m1) if m1 else 0,
        "compaction_backlog_final": backlogs[-1] if backlogs else 0,
        "sampling_backlog_final": sampling_backlog[-1] if sampling_backlog else 0,
        "bounded_view_node_scan_peak": max(node_scans, default=0),
        "bounded_view_edge_scan_peak": max(edge_scans, default=0),
    }
    return V979AcceptanceResult(all(checks), len(rows), len(steady), *checks, details)


def write_v979_acceptance_report(root: str | Path, *, output: str | Path | None = None) -> Path:
    root = Path(root)
    result = evaluate_v979_long_run(root)
    target = Path(output) if output is not None else root / "V979_LONG_RUN_ACCEPTANCE.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate v9.7.9 long-run bounded-residency acceptance")
    parser.add_argument("root")
    parser.add_argument("--output")
    parser.add_argument("--rss-growth-tolerance", type=float, default=0.15)
    args = parser.parse_args(argv)
    result = evaluate_v979_long_run(args.root, rss_growth_tolerance=args.rss_growth_tolerance)
    target = write_v979_acceptance_report(args.root, output=args.output)
    print(json.dumps({"passed": result.passed, "report": str(target), **result.details}, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
