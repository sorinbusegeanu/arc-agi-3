from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
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
        self._parsed_rows: dict[int, list[dict[str, Any]]] = {}
        self._parsed_ids: dict[int, list[int]] = {}
        self._parsed_watermarks: dict[int, int] = {}

    def close(self) -> None:
        self.connection.close()

    def _refresh_parsed_rows(self, evidence_type: int) -> None:
        kind = int(evidence_type)
        watermark = int(self._parsed_watermarks.get(kind, 0))
        rows = self.connection.execute(
            "SELECT evidence_id,memory_id,source_game,source_context,"
            "source_global_step,payload_json,generation_id "
            "FROM evidence_records WHERE evidence_type=? AND evidence_id>? "
            "ORDER BY evidence_id",
            (kind, watermark),
        ).fetchall()
        if not rows:
            return
        parsed = self._parsed_rows.setdefault(kind, [])
        ids = self._parsed_ids.setdefault(kind, [])
        for evidence_id, memory_id, game, context, step, payload_json, generation in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            payload.update(
                {
                    "evidence_id": int(evidence_id),
                    "memory_id": None if memory_id is None else int(memory_id),
                    "source_game": game,
                    "source_context": context,
                    "source_global_step": step,
                    "generation_id": int(generation),
                }
            )
            parsed.append(payload)
            ids.append(int(evidence_id))
        self._parsed_watermarks[kind] = int(rows[-1][0])

    def load_evidence(
        self,
        evidence_type: int,
        *,
        after_evidence_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Return parsed evidence while querying and decoding only new rows."""

        kind = int(evidence_type)
        self._refresh_parsed_rows(kind)
        rows = self._parsed_rows.get(kind, [])
        ids = self._parsed_ids.get(kind, [])
        start = bisect_right(ids, int(after_evidence_id))
        # Callers add derived fields, so keep the cached raw payload immutable.
        return [dict(row) for row in rows[start:]]

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
