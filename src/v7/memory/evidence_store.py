from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from v7.memory.ids import MemoryId
from v7.memory.schema import ensure_v7_schema


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    memory_id: MemoryId | None
    evidence_type: int
    generation_id: int
    payload: dict[str, Any]
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None


class EvidenceStore:
    """Append-only historical evidence ledger, separate from active cognition."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        ensure_v7_schema(self.connection)

    def close(self) -> None:
        self.connection.close()

    def append_evidence_batch(self, records: Iterable[EvidenceRecord]) -> int:
        rows = [
            (
                None if record.memory_id is None else int(record.memory_id),
                int(record.evidence_type),
                record.source_game,
                record.source_context,
                record.source_global_step,
                json.dumps(record.payload, separators=(",", ":"), sort_keys=True),
                int(record.generation_id),
            )
            for record in records
        ]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO evidence_records(
                    memory_id, evidence_type, source_game, source_context,
                    source_global_step, payload_json, generation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)
