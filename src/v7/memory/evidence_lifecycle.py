from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from v7.memory.ids import MemoryId
from v7.memory.schema import ensure_v7_schema


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    memory_id: MemoryId
    generation_id: int
    parent_memory_id: MemoryId | None = None
    relation_type: int = 0
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None


@dataclass(frozen=True, slots=True)
class TransferTrialRecord:
    memory_id: MemoryId
    generation_id: int
    source_game: str
    target_game: str
    success: bool
    score: float = 0.0
    payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ContradictionRecord:
    memory_id: MemoryId
    generation_id: int
    severity: float
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None
    payload: dict[str, object] | None = None


class EvidenceLifecycleStore:
    """Append-only provenance, transfer and contradiction ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        ensure_v7_schema(self.connection)

    def close(self) -> None:
        self.connection.close()

    def append_provenance(self, records: Iterable[ProvenanceRecord]) -> int:
        rows = [(int(r.memory_id), None if r.parent_memory_id is None else int(r.parent_memory_id), int(r.relation_type), r.source_game, r.source_context, r.source_global_step, int(r.generation_id)) for r in records]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany("INSERT INTO provenance_records(memory_id,parent_memory_id,relation_type,source_game,source_context,source_global_step,generation_id) VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def append_transfer_trials(self, records: Iterable[TransferTrialRecord]) -> int:
        rows = [(int(r.memory_id), r.source_game, r.target_game, 1 if r.success else 0, float(r.score), json.dumps(r.payload or {}, separators=(",", ":"), sort_keys=True), int(r.generation_id)) for r in records]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany("INSERT INTO transfer_trials(memory_id,source_game,target_game,success,score,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def append_contradictions(self, records: Iterable[ContradictionRecord]) -> int:
        rows = [(int(r.memory_id), float(r.severity), r.source_game, r.source_context, r.source_global_step, json.dumps(r.payload or {}, separators=(",", ":"), sort_keys=True), int(r.generation_id)) for r in records]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany("INSERT INTO contradiction_records(memory_id,severity,source_game,source_context,source_global_step,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)

    def transfer_summary(self, memory_ids: Iterable[MemoryId]) -> dict[MemoryId, tuple[int, int, float]]:
        ids = tuple(sorted(set(int(memory_id) for memory_id in memory_ids)))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT memory_id, COUNT(*), SUM(success), AVG(score) FROM transfer_trials WHERE memory_id IN ({placeholders}) GROUP BY memory_id", ids).fetchall()
        return {MemoryId(int(memory_id)): (int(total), int(successes or 0), float(avg_score or 0.0)) for memory_id, total, successes, avg_score in rows}

    def contradiction_summary(self, memory_ids: Iterable[MemoryId]) -> dict[MemoryId, tuple[int, float]]:
        ids = tuple(sorted(set(int(memory_id) for memory_id in memory_ids)))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT memory_id, COUNT(*), MAX(severity) FROM contradiction_records WHERE memory_id IN ({placeholders}) GROUP BY memory_id", ids).fetchall()
        return {MemoryId(int(memory_id)): (int(total), float(max_severity or 0.0)) for memory_id, total, max_severity in rows}

    def provenance_parents(self, memory_id: MemoryId) -> tuple[MemoryId, ...]:
        rows = self.connection.execute("SELECT DISTINCT parent_memory_id FROM provenance_records WHERE memory_id=? AND parent_memory_id IS NOT NULL ORDER BY parent_memory_id", (int(memory_id),)).fetchall()
        return tuple(MemoryId(int(row[0])) for row in rows)

    def provenance_source_games(self, memory_id: MemoryId) -> tuple[str, ...]:
        rows = self.connection.execute("""
            WITH RECURSIVE ancestry(memory_id) AS (
                SELECT ?
                UNION
                SELECT p.parent_memory_id FROM provenance_records AS p JOIN ancestry AS a ON p.memory_id=a.memory_id WHERE p.parent_memory_id IS NOT NULL
            )
            SELECT DISTINCT p.source_game FROM provenance_records AS p JOIN ancestry AS a ON p.memory_id=a.memory_id
            WHERE p.source_game IS NOT NULL AND p.source_game<>'' ORDER BY p.source_game
        """, (int(memory_id),)).fetchall()
        return tuple(str(row[0]) for row in rows)

    def transfer_trial_exists(self, memory_id: MemoryId, *, target_game: str, source_global_step: int | None) -> bool:
        rows = self.connection.execute("SELECT payload_json FROM transfer_trials WHERE memory_id=? AND target_game=?", (int(memory_id), str(target_game))).fetchall()
        for (payload_json,) in rows:
            try:
                payload = json.loads(payload_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("source_global_step") == source_global_step:
                return True
        return False
