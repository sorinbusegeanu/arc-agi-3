from __future__ import annotations

import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock


@dataclass(frozen=True, slots=True)
class SignatureIndexRecord:
    signature: int
    support: int
    contradiction_support: int
    last_derived_support: int
    derivation_dirty: bool
    priority: int

    def __post_init__(self) -> None:
        if min(self.signature, self.support, self.contradiction_support, self.last_derived_support) < 0:
            raise ValueError("signature index values must be non-negative")
        if self.last_derived_support > self.support:
            raise ValueError("last-derived support cannot exceed current support")


class SignatureIndexStore:
    """Persistent signature authority with bounded resident write/cache state."""

    def __init__(
        self,
        path: str | Path,
        *,
        delta_limit: int = 4096,
        page_cache_limit: int = 1024,
        dirty_window_limit: int = 4096,
    ) -> None:
        if min(delta_limit, page_cache_limit, dirty_window_limit) <= 0:
            raise ValueError("signature-index resident bounds must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.delta_limit = int(delta_limit)
        self.page_cache_limit = int(page_cache_limit)
        self.dirty_window_limit = int(dirty_window_limit)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS signatures (
                signature TEXT PRIMARY KEY,
                support INTEGER NOT NULL,
                contradiction_support INTEGER NOT NULL,
                last_derived_support INTEGER NOT NULL,
                derivation_dirty INTEGER NOT NULL,
                priority INTEGER NOT NULL
            )"""
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS signatures_dirty ON signatures(derivation_dirty, priority DESC, signature ASC)"
        )
        self._connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._connection.commit()
        self._delta: OrderedDict[int, SignatureIndexRecord] = OrderedDict()
        self._cache: OrderedDict[int, SignatureIndexRecord] = OrderedDict()

    def close(self) -> None:
        with self._lock:
            self.flush()
            self._connection.close()

    def __enter__(self) -> "SignatureIndexStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def resident_delta_entries(self) -> int:
        return len(self._delta)

    @property
    def resident_cache_entries(self) -> int:
        return len(self._cache)

    def _cache_record(self, record: SignatureIndexRecord) -> SignatureIndexRecord:
        self._cache.pop(record.signature, None)
        self._cache[record.signature] = record
        while len(self._cache) > self.page_cache_limit:
            self._cache.popitem(last=False)
        return record

    @staticmethod
    def _from_row(row: tuple[int, ...] | None, signature: int) -> SignatureIndexRecord:
        if row is None:
            return SignatureIndexRecord(int(signature), 0, 0, 0, False, 0)
        return SignatureIndexRecord(
            int(row[0]), int(row[1]), int(row[2]), int(row[3]), bool(row[4]), int(row[5])
        )

    @staticmethod
    def _db_signature(signature: int) -> str:
        if not 0 <= int(signature) < 1 << 64:
            raise ValueError("signature must be uint64")
        return f"{int(signature):020d}"

    def get(self, signature: int) -> SignatureIndexRecord:
        signature = int(signature)
        if signature < 0:
            raise ValueError("signature must be non-negative")
        with self._lock:
            if signature in self._delta:
                return self._delta[signature]
            cached = self._cache.get(signature)
            if cached is not None:
                self._cache.move_to_end(signature)
                return cached
            row = self._connection.execute(
                "SELECT signature,support,contradiction_support,last_derived_support,derivation_dirty,priority FROM signatures WHERE signature=?",
                (self._db_signature(signature),),
            ).fetchone()
            return self._cache_record(self._from_row(row, signature))

    def observe(
        self,
        signature: int,
        *,
        support_delta: int = 1,
        contradiction_delta: int = 0,
        priority: int | None = None,
    ) -> SignatureIndexRecord:
        if min(support_delta, contradiction_delta) < 0:
            raise ValueError("signature deltas must be non-negative")
        with self._lock:
            current = self.get(signature)
            support = current.support + int(support_delta)
            contradiction = current.contradiction_support + int(contradiction_delta)
            record = SignatureIndexRecord(
                current.signature,
                support,
                contradiction,
                current.last_derived_support,
                support > current.last_derived_support,
                max(current.priority, support - contradiction if priority is None else int(priority)),
            )
            self._delta[current.signature] = record
            self._delta.move_to_end(current.signature)
            self._cache_record(record)
            if len(self._delta) >= self.delta_limit:
                self.flush(max_entries=max(1, self.delta_limit // 2))
            return record

    def mark_derived(self, signature: int, target_support: int) -> SignatureIndexRecord:
        with self._lock:
            current = self.get(signature)
            if target_support > current.support:
                raise ValueError("derived target support exceeds indexed support")
            last = max(current.last_derived_support, int(target_support))
            record = SignatureIndexRecord(
                current.signature,
                current.support,
                current.contradiction_support,
                last,
                current.support > last,
                current.priority,
            )
            self._delta[current.signature] = record
            self._cache_record(record)
            return record

    def flush(self, *, max_entries: int | None = None) -> int:
        with self._lock:
            limit = len(self._delta) if max_entries is None else min(len(self._delta), max(0, int(max_entries)))
            rows = [self._delta.popitem(last=False)[1] for _ in range(limit)]
            if not rows:
                return 0
            with self._connection:
                self._connection.executemany(
                    """INSERT INTO signatures VALUES(?,?,?,?,?,?)
                    ON CONFLICT(signature) DO UPDATE SET
                    support=excluded.support,
                    contradiction_support=excluded.contradiction_support,
                    last_derived_support=excluded.last_derived_support,
                    derivation_dirty=excluded.derivation_dirty,
                    priority=excluded.priority""",
                    tuple(
                        (
                            self._db_signature(row.signature),
                            row.support,
                            row.contradiction_support,
                            row.last_derived_support,
                            int(row.derivation_dirty),
                            row.priority,
                        )
                        for row in rows
                    ),
                )
            return len(rows)

    def dirty_window(self, *, limit: int | None = None) -> tuple[SignatureIndexRecord, ...]:
        with self._lock:
            self.flush()
            selected = min(self.dirty_window_limit, self.dirty_window_limit if limit is None else max(0, int(limit)))
            rows = self._connection.execute(
                """SELECT signature,support,contradiction_support,last_derived_support,derivation_dirty,priority
                FROM signatures WHERE derivation_dirty=1
                ORDER BY priority DESC, signature ASC LIMIT ?""",
                (selected,),
            ).fetchall()
            return tuple(self._cache_record(self._from_row(row, int(row[0]))) for row in rows)

    @property
    def scan_cursor(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT value FROM metadata WHERE key='scan_cursor'").fetchone()
            return 0 if row is None else int(row[0])

    def scan_page(self, *, limit: int = 256) -> tuple[SignatureIndexRecord, ...]:
        if not 0 < limit <= self.dirty_window_limit:
            raise ValueError("scan page limit exceeds the configured dirty window")
        with self._lock:
            self.flush()
            cursor = self.scan_cursor
            rows = self._connection.execute(
                """SELECT signature,support,contradiction_support,last_derived_support,derivation_dirty,priority
                FROM signatures WHERE signature>? ORDER BY signature ASC LIMIT ?""",
                (self._db_signature(cursor), int(limit)),
            ).fetchall()
            if not rows and cursor:
                rows = self._connection.execute(
                    """SELECT signature,support,contradiction_support,last_derived_support,derivation_dirty,priority
                    FROM signatures ORDER BY signature ASC LIMIT ?""",
                    (int(limit),),
                ).fetchall()
            next_cursor = 0 if not rows else int(rows[-1][0])
            with self._connection:
                self._connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('scan_cursor',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(next_cursor),),
                )
            return tuple(self._cache_record(self._from_row(row, int(row[0]))) for row in rows)


__all__ = ["SignatureIndexRecord", "SignatureIndexStore"]
