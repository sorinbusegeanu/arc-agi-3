from __future__ import annotations

"""v8.51 disk-authoritative scientific evidence storage."""

import json
import os
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from v8.model import MemoryUid


_ROOT_ENV = "ARC_AGI3_V8_ROOT"


class DiskBackedEvidenceLedger:
    """Preserve the full evidence ledger without unbounded Python row/id lists."""

    def __init__(self) -> None:
        raw_root = str(os.environ.get(_ROOT_ENV, "")).strip()
        self.path: Path | None = None
        if raw_root:
            self.path = Path(raw_root) / "maintenance" / "evidence.sqlite3"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            database = str(self.path)
        else:
            database = ":memory:"
        self._lock = threading.RLock()
        self._append_listener = None
        self._db = sqlite3.connect(database, timeout=5.0, check_same_thread=False)
        if database != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS evidence ("
            "evidence_id TEXT PRIMARY KEY, available INTEGER NOT NULL, decision INTEGER NOT NULL, "
            "uid TEXT NOT NULL, effect_direction INTEGER NOT NULL, causal TEXT NOT NULL, "
            "payload TEXT NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS evidence_cut ON evidence(available,decision)")
        self._db.execute("CREATE INDEX IF NOT EXISTS evidence_uid ON evidence(uid)")
        self._db.commit()

    @staticmethod
    def _encode(row) -> str:
        return json.dumps(asdict(row), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode(raw: str):
        from v8.evidence import EvidenceRecord

        payload = json.loads(raw)
        payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
        return EvidenceRecord(**payload)

    def append(self, row) -> bool:
        if row.evidence_available_watermark > row.decision_watermark:
            raise ValueError("future evidence cannot influence an earlier decision")
        if not row.quality_valid():
            raise ValueError("invalid scientific evidence quality/normalization")
        with self._lock:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO evidence "
                "(evidence_id,available,decision,uid,effect_direction,causal,payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    str(row.evidence_id),
                    int(row.evidence_available_watermark),
                    int(row.decision_watermark),
                    row.uid.hex(),
                    int(row.effect_direction),
                    str(row.causal_intervention),
                    self._encode(row),
                ),
            )
            inserted = int(cursor.rowcount) > 0
            if inserted:
                self._db.commit()
            listener = self._append_listener if inserted else None
        if listener is not None:
            listener(row)
        return inserted

    def set_append_listener(self, listener, *, replay: bool = False) -> None:
        with self._lock:
            self._append_listener = listener
        if listener is None or not replay:
            return
        last_rowid = 0
        while True:
            with self._lock:
                batch = tuple(
                    self._db.execute(
                        "SELECT rowid,payload FROM evidence WHERE rowid>? ORDER BY rowid LIMIT 256",
                        (last_rowid,),
                    )
                )
            if not batch:
                return
            for rowid, raw in batch:
                listener(self._decode(raw))
                last_rowid = int(rowid)

    def contains(self, evidence_id: str) -> bool:
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM evidence WHERE evidence_id=? LIMIT 1",
                (str(evidence_id),),
            ).fetchone() is not None

    def count(self, watermark: int | None = None) -> int:
        with self._lock:
            if watermark is None:
                row = self._db.execute("SELECT COUNT(*) FROM evidence").fetchone()
            else:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM evidence WHERE available<=? AND decision<=?",
                    (int(watermark), int(watermark)),
                ).fetchone()
        return 0 if row is None else int(row[0])

    def cut(self, watermark: int):
        with self._lock:
            payloads = tuple(
                raw
                for (raw,) in self._db.execute(
                    "SELECT payload FROM evidence WHERE available<=? AND decision<=? ORDER BY rowid",
                    (int(watermark), int(watermark)),
                )
            )
        return tuple(self._decode(raw) for raw in payloads)

    def protected_uids(self) -> set[MemoryUid]:
        zero = MemoryUid.zero().hex()
        with self._lock:
            values = tuple(raw for (raw,) in self._db.execute(
                "SELECT DISTINCT uid FROM evidence WHERE uid<>?", (zero,)
            ))
        result: set[MemoryUid] = set()
        for raw in values:
            text = str(raw)
            if len(text) == 32:
                result.add(MemoryUid(int(text[:16], 16), int(text[16:], 16)))
        return result

    def positive_effect_uids(self) -> set[MemoryUid]:
        zero = MemoryUid.zero().hex()
        with self._lock:
            values = tuple(raw for (raw,) in self._db.execute(
                "SELECT DISTINCT uid FROM evidence "
                "WHERE uid<>? AND effect_direction>0 AND causal<>''", (zero,)
            ))
        result: set[MemoryUid] = set()
        for raw in values:
            text = str(raw)
            if len(text) == 32:
                result.add(MemoryUid(int(text[:16], 16), int(text[16:], 16)))
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "external_sqlite": None if self.path is None else str(self.path),
            "record_count": self.count(),
            "records": [],
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        from v8.evidence import EvidenceRecord

        for raw in state.get("records", []):
            if not isinstance(raw, dict):
                continue
            payload = dict(raw)
            payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
            self.append(EvidenceRecord(**payload))

    def export_jsonl(self, path: str | Path, *, watermark: int | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        query = "SELECT payload FROM evidence ORDER BY rowid"
        params: tuple[object, ...] = ()
        if watermark is not None:
            query = "SELECT payload FROM evidence WHERE available<=? AND decision<=? ORDER BY rowid"
            params = (int(watermark), int(watermark))
        with self._lock, target.open("w", encoding="utf-8") as handle:
            cursor = self._db.execute(query, params)
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for (raw,) in rows:
                    handle.write(str(raw) + "\n")

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()
