from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from v6.storage.duckdb_queries import load_run_table
from v6.storage.parquet_backend import ParquetStorageBackend
from v6.storage.sqlite_backend import SQLiteStorageBackend


def run_storage_benchmark(*, rows: int, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "game": "bench",
            "seed": 0,
            "step": index,
            "action": index % 5,
            "terminal": 0,
            "reset": 0,
            "changed_cells": index % 7,
        }
        for index in range(int(rows))
    ]

    sqlite_path = output / "sqlite" / "benchmark.sqlite"
    start = time.perf_counter()
    sqlite_backend = SQLiteStorageBackend(sqlite_path, batch_size=1000)
    sqlite_backend.write_interactions(records)
    sqlite_backend.finalize()
    sqlite_write_time = time.perf_counter() - start

    parquet_root = output / "parquet"
    start = time.perf_counter()
    parquet_backend = ParquetStorageBackend(root=parquet_root, game="bench", sampler="benchmark", seed=0, steps=rows, batch_size=1000)
    parquet_backend.write_interactions(records)
    parquet_backend.write_run_summary({"game": "bench", "rows": rows})
    parquet_backend.finalize()
    parquet_write_time = time.perf_counter() - start

    start = time.perf_counter()
    queried = load_run_table(parquet_root, "interactions", game="bench", sampler="benchmark", seed=0, steps=rows)
    duckdb_query_time = time.perf_counter() - start

    result = {
        "rows": int(rows),
        "sqlite_write_time": sqlite_write_time,
        "parquet_write_time": parquet_write_time,
        "sqlite_file_size": _path_size(sqlite_path),
        "parquet_directory_size": _path_size(parquet_root),
        "duckdb_query_time": duckdb_query_time,
        "duckdb_row_count": int(len(queried)),
    }
    (output / "storage_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "storage_benchmark_report.txt").write_text(_format_benchmark(result), encoding="utf-8")
    return result


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _format_benchmark(result: dict) -> str:
    return "\n".join(
        [
            "ARC-AGI3 v6 Storage Benchmark",
            f"rows={result['rows']}",
            f"sqlite_write_time={result['sqlite_write_time']:.4f}",
            f"parquet_write_time={result['parquet_write_time']:.4f}",
            f"sqlite_file_size={result['sqlite_file_size']}",
            f"parquet_directory_size={result['parquet_directory_size']}",
            f"duckdb_query_time={result['duckdb_query_time']:.4f}",
            f"duckdb_row_count={result['duckdb_row_count']}",
        ]
    ) + "\n"
