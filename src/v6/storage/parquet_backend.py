from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from v6.storage.backend import StorageBackend


LARGE_TABLES = {"interactions", "deltas", "contingencies", "future_effects"}
ALL_TABLES = (
    "interactions",
    "deltas",
    "transformation_families",
    "contingencies",
    "future_effects",
    "role_candidates",
    "run_summary",
)


class ParquetStorageBackend(StorageBackend):
    def __init__(
        self,
        *,
        root: str | Path,
        game: str,
        sampler: str,
        seed: int,
        steps: int,
        batch_size: int = 1000,
        compression: str = "zstd",
    ) -> None:
        self.root = Path(root)
        self.game = str(game)
        self.sampler = str(sampler)
        self.seed = int(seed)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.compression = str(compression)
        self.base_path = self.root / f"game={self.game}" / f"sampler={self.sampler}" / f"seed={self.seed}" / f"steps={self.steps}"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.part_counts: dict[str, int] = defaultdict(int)
        _require_pyarrow()

    def write_interactions(self, records: list[dict]) -> None:
        self._write("interactions", records)

    def write_deltas(self, records: list[dict]) -> None:
        self._write("deltas", records)

    def write_transformation_families(self, records: list[dict]) -> None:
        self._write("transformation_families", records)

    def write_contingencies(self, records: list[dict]) -> None:
        self._write("contingencies", records)

    def write_future_effects(self, records: list[dict]) -> None:
        self._write("future_effects", records)

    def write_role_candidates(self, records: list[dict]) -> None:
        self._write("role_candidates", records)

    def write_run_summary(self, record: dict) -> None:
        self._write("run_summary", [record])

    def finalize(self) -> None:
        for table_name in list(self.buffers):
            self._flush(table_name)

    def _write(self, table_name: str, records: list[dict]) -> None:
        if not records:
            return
        self.buffers[table_name].extend(_normalize_record(record) for record in records)
        while len(self.buffers[table_name]) >= self.batch_size:
            self._flush(table_name, max_records=self.batch_size)

    def _flush(self, table_name: str, *, max_records: int | None = None) -> None:
        records = self.buffers.get(table_name, [])
        if not records:
            return
        to_write = records if max_records is None else records[:max_records]
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.part_counts[table_name] += 1
        part = self.part_counts[table_name]
        if table_name in LARGE_TABLES:
            path = self.base_path / f"{table_name}_part_{part:06d}.parquet"
        else:
            path = self.base_path / f"{table_name}.parquet" if part == 1 else self.base_path / f"{table_name}_part_{part:06d}.parquet"
        table = pa.Table.from_pylist(to_write)
        pq.write_table(table, path, compression=self.compression)
        self.buffers[table_name] = [] if max_records is None else records[max_records:]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Parquet storage requires pyarrow and duckdb. Install with: pip install pyarrow duckdb") from exc
