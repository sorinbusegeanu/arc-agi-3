from __future__ import annotations

from pathlib import Path

from v4_5.benchmark.db.store import BenchmarkStore, default_db_path
from v4_5.benchmark.reporting.summaryBuilder import build_summary_payload


def show_run(*, run_id: str, db_path: str | None = None) -> dict:
    store = BenchmarkStore(db_path=Path(db_path) if db_path else default_db_path())
    return build_summary_payload(store, run_id)
