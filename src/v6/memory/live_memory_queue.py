from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from v6.memory.worker_snapshot import WorkerMemoryOverlay


_LIVE_MEMORY_MANAGERS: list[Any] = []


@dataclass
class LiveMemoryEvent:
    event_type: str
    event_id: str
    global_step: int
    worker_id: str
    priority: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveMemoryWriterConfig:
    memory_dir: str
    queue_maxsize: int = 100_000
    batch_size: int = 1000
    flush_seconds: float = 2.0
    min_priority: float = 0.0
    summary_write_every_batches: int = 50


class LiveMemoryWriter:
    def __init__(self, config: LiveMemoryWriterConfig, queue: multiprocessing.Queue) -> None:
        self.config = config
        self.queue = queue
        self.memory_dir = Path(str(config.memory_dir))
        self.sqlite_path = self.memory_dir / "live_memory.sqlite"
        self.summary_path = self.memory_dir / "live_memory_summary.json"
        self._stop_requested = False
        self.summary: dict[str, Any] = {
            "events_received": 0,
            "events_written": 0,
            "events_dropped_invalid": 0,
            "events_dropped_low_priority": 0,
            "batches_written": 0,
            "last_flush_time": None,
            "event_type_counts": {},
            "queue_stop_received": False,
            "events_dropped": 0,
            "queue_peak_size": 0,
            "queue_block_seconds": 0.0,
        }
        self._batches_since_summary_write = 0

    def run(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        try:
            _configure_live_memory_sqlite(connection)
            _ensure_live_memory_schema(connection)
            self._next_sequence = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM live_memory_events").fetchone()[0])
            batch: list[dict[str, Any]] = []
            last_flush = time.time()
            while not self._stop_requested:
                timeout = max(0.05, float(self.config.flush_seconds))
                try:
                    raw_event = self.queue.get(timeout=timeout)
                except Empty:
                    raw_event = None
                except Exception:
                    raw_event = None
                if raw_event is not None:
                    self.summary["events_received"] = int(self.summary["events_received"]) + 1
                    event = _normalize_event(raw_event)
                    if event is None:
                        self.summary["events_dropped_invalid"] = int(self.summary["events_dropped_invalid"]) + 1
                    elif event["event_type"] == "__stop__":
                        self.summary["queue_stop_received"] = True
                        self._stop_requested = True
                    elif float(event["priority"]) < float(self.config.min_priority):
                        self.summary["events_dropped_low_priority"] = int(self.summary["events_dropped_low_priority"]) + 1
                    else:
                        event["sequence"] = int(self._next_sequence)
                        self._next_sequence += 1
                        batch.append(event)
                        counts = dict(self.summary.get("event_type_counts", {}) or {})
                        counts[event["event_type"]] = int(counts.get(event["event_type"], 0) or 0) + 1
                        self.summary["event_type_counts"] = counts
                now = time.time()
                if batch and (
                    len(batch) >= int(self.config.batch_size)
                    or self._stop_requested
                    or (now - last_flush) >= float(self.config.flush_seconds)
                ):
                    self._flush_batch(connection, batch)
                    batch = []
                    last_flush = now
            if batch:
                self._flush_batch(connection, batch)
        finally:
            connection.close()
            self._write_summary()

    def stop(self) -> None:
        self._stop_requested = True

    def _flush_batch(self, connection: sqlite3.Connection, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        for event in batch:
            try:
                self._write_event(connection, event)
                self.summary["events_written"] = int(self.summary["events_written"]) + 1
            except Exception:
                self.summary["events_dropped_invalid"] = int(self.summary["events_dropped_invalid"]) + 1
                self.summary["events_dropped"] = int(self.summary.get("events_dropped", 0)) + 1
                continue
        connection.commit()
        self.summary["batches_written"] = int(self.summary["batches_written"]) + 1
        self.summary["last_flush_time"] = time.time()
        self._batches_since_summary_write += 1
        if self._batches_since_summary_write >= max(1, int(self.config.summary_write_every_batches)):
            self._write_summary()
            self._batches_since_summary_write = 0

    def _write_event(self, connection: sqlite3.Connection, event: dict[str, Any]) -> None:
        payload = dict(event.get("payload") or {})
        payload_json = json.dumps(payload, sort_keys=True)
        created_at = float(time.time())
        connection.execute(
            """
            INSERT OR REPLACE INTO live_memory_events (
                event_id, sequence, event_type, global_step, worker_id, priority, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event["event_id"]),
                int(event.get("sequence") or 0),
                str(event["event_type"]),
                int(event.get("global_step") or 0),
                str(event.get("worker_id") or ""),
                float(event.get("priority") or 0.0),
                payload_json,
                created_at,
            ),
        )
        _apply_projection(connection, event, payload_json)

    def _write_summary(self) -> None:
        self.summary_path.write_text(json.dumps(self.summary, indent=2, sort_keys=True), encoding="utf-8")


class LiveMemoryReadCache:
    def __init__(self, *, memory_dir: str | Path, refresh_steps: int = 250) -> None:
        self.memory_dir = Path(memory_dir)
        self.sqlite_path = self.memory_dir / "live_memory.sqlite"
        self.refresh_steps = max(1, int(refresh_steps))
        self.last_refresh_step = -1
        self.refresh_count = 0
        self.refresh_failed_count = 0
        self.stable_contingencies: list[dict[str, Any]] = []
        self.replay_candidates: list[dict[str, Any]] = []
        self.contradiction_clusters: list[dict[str, Any]] = []
        self.carrier_candidates: list[dict[str, Any]] = []
        self.future_option_events: list[dict[str, Any]] = []
        self.family_updates: list[dict[str, Any]] = []
        self.overlay = WorkerMemoryOverlay()
        self.last_applied_live_sequence = -1
        self.refresh_rows = 0
        self.refresh_seconds = 0.0
        self.busy_retry_count = 0
        self.busy_wait_seconds = 0.0

    def refresh_if_due(self, step: int) -> bool:
        if self.last_refresh_step < 0 or int(step) - int(self.last_refresh_step) >= self.refresh_steps:
            return self.refresh(force=True, step=step)
        return False

    def refresh(self, force: bool = False, step: int | None = None) -> bool:
        if not force and step is not None and self.last_refresh_step >= 0 and int(step) - int(self.last_refresh_step) < self.refresh_steps:
            return False
        if not self.sqlite_path.exists():
            self.refresh_failed_count += 1
            return False
        started = time.perf_counter()
        try:
            connection = sqlite3.connect(
                f"file:{self.sqlite_path}?mode=ro",
                uri=True,
                timeout=1.0,
            )
        except sqlite3.DatabaseError:
            self.refresh_failed_count += 1
            return False
        try:
            connection.row_factory = sqlite3.Row
            if self.last_applied_live_sequence < 0:
                self.stable_contingencies = _fetch_projection_rows(connection, "SELECT * FROM live_stable_contingencies ORDER BY priority DESC, support_count DESC, key ASC LIMIT 5000")
                self.replay_candidates = _fetch_projection_rows(connection, "SELECT * FROM live_replay_candidates ORDER BY replay_priority DESC, priority DESC, interaction_id ASC LIMIT 5000")
                self.contradiction_clusters = _fetch_projection_rows(connection, "SELECT * FROM live_contradiction_clusters ORDER BY priority DESC, count DESC, contradiction_key ASC LIMIT 5000")
                self.carrier_candidates = _fetch_projection_rows(connection, "SELECT * FROM live_carrier_candidates ORDER BY priority DESC, support_count DESC, carrier_signature ASC LIMIT 5000")
                self.future_option_events = _fetch_projection_rows(connection, "SELECT * FROM live_future_option_events ORDER BY priority DESC, global_step DESC, event_id ASC LIMIT 10000")
                self.family_updates = _fetch_projection_rows(connection, "SELECT * FROM live_family_updates ORDER BY priority DESC, support_count DESC, family_signature ASC LIMIT 5000")
                max_sequence = int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM live_memory_events").fetchone()[0] or 0)
                self.last_applied_live_sequence = max_sequence
                self.overlay.last_sequence = max_sequence
                self.refresh_rows += sum(len(rows) for rows in (self.stable_contingencies, self.replay_candidates, self.contradiction_clusters, self.carrier_candidates, self.future_option_events, self.family_updates))
                initial_events = [
                    {"event_type": "stable_contingency", "event_id": str(row.get("key")), "payload": dict(row)}
                    for row in self.stable_contingencies
                ]
                self.overlay.apply_rows(initial_events, max_sequence)
            else:
                event_rows = connection.execute(
                    "SELECT sequence, event_id, event_type, global_step, worker_id, priority, payload_json FROM live_memory_events WHERE sequence > ? ORDER BY sequence ASC",
                    (int(self.last_applied_live_sequence),),
                ).fetchall()
                decoded: list[dict[str, Any]] = []
                for row in event_rows:
                    try:
                        decoded.append({
                            "sequence": int(row[0]), "event_id": str(row[1]), "event_type": str(row[2]),
                            "global_step": int(row[3] or 0), "payload": json.loads(str(row[6] or "{}")),
                        })
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                if decoded:
                    self.overlay.apply_rows(decoded, int(decoded[-1]["sequence"]))
                    self.last_applied_live_sequence = int(decoded[-1]["sequence"])
                    self.refresh_rows += len(decoded)
                    for event in decoded:
                        payload = dict(event.get("payload") or {})
                        payload.update({"event_id": event["event_id"], "event_type": event["event_type"], "global_step": event["global_step"]})
                        target = {
                            "stable_contingency": self.stable_contingencies,
                            "high_priority_replay": self.replay_candidates,
                            "future_option": self.future_option_events,
                            "future_option_event": self.future_option_events,
                            "family_update": self.family_updates,
                        }.get(str(event["event_type"]))
                        if target is not None:
                            target.append(payload)
        except sqlite3.DatabaseError:
            self.refresh_failed_count += 1
            connection.close()
            return False
        connection.close()
        self.refresh_count += 1
        self.refresh_seconds += time.perf_counter() - started
        if step is not None:
            self.last_refresh_step = int(step)
        elif self.last_refresh_step < 0:
            self.last_refresh_step = 0
        return True


def make_live_memory_queue(maxsize: int) -> multiprocessing.Queue:
    manager = multiprocessing.Manager()
    _LIVE_MEMORY_MANAGERS.append(manager)
    return manager.Queue(maxsize=max(1, int(maxsize)))


def start_live_memory_writer(config: LiveMemoryWriterConfig, queue: multiprocessing.Queue) -> multiprocessing.Process:
    process = multiprocessing.Process(
        target=_writer_entrypoint,
        args=(config, queue),
        name="live-memory-writer",
        daemon=False,
    )
    process.start()
    return process


def stop_live_memory_writer(queue, process, timeout_seconds: float = 30.0) -> dict:
    forced = False
    try:
        queue.put({"event_type": "__stop__"})
    except Exception:
        pass
    process.join(timeout=float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        forced = True
    summary = {
        "writer_exitcode": process.exitcode,
        "writer_forced_terminated": forced,
        "summary_path": None,
        "live_memory_event_counts": None,
    }
    return summary


def _writer_entrypoint(config: LiveMemoryWriterConfig, queue: multiprocessing.Queue) -> None:
    writer = LiveMemoryWriter(config, queue)
    writer.run()


def _configure_live_memory_sqlite(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA busy_timeout=5000;")


def _ensure_live_memory_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS live_memory_events (
            event_id TEXT PRIMARY KEY,
            sequence INTEGER,
            event_type TEXT NOT NULL,
            global_step INTEGER,
            worker_id TEXT,
            priority REAL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_live_memory_events_type_step
            ON live_memory_events(event_type, global_step);
        CREATE INDEX IF NOT EXISTS idx_live_memory_events_priority
            ON live_memory_events(priority);
        CREATE TABLE IF NOT EXISTS live_stable_contingencies (
            key TEXT PRIMARY KEY,
            action INTEGER,
            context_signature TEXT,
            context_level INTEGER,
            transformation_family INTEGER,
            support_count INTEGER,
            confidence REAL,
            priority REAL,
            last_global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS live_replay_candidates (
            interaction_id TEXT PRIMARY KEY,
            replay_priority REAL,
            reason TEXT,
            family_id TEXT,
            context_signature TEXT,
            action_signature TEXT,
            priority REAL,
            last_global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS live_contradiction_clusters (
            contradiction_key TEXT PRIMARY KEY,
            context_signature TEXT,
            action_signature TEXT,
            count INTEGER,
            priority REAL,
            last_global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS live_carrier_candidates (
            carrier_signature TEXT PRIMARY KEY,
            carrier_source TEXT,
            support_count INTEGER,
            linked_family_count INTEGER,
            priority REAL,
            last_global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS live_future_option_events (
            event_id TEXT PRIMARY KEY,
            interaction_id TEXT,
            option_delta REAL,
            motif_type TEXT,
            motif_type_source TEXT,
            priority REAL,
            global_step INTEGER,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS live_family_updates (
            family_signature TEXT PRIMARY KEY,
            family_id INTEGER,
            support_count INTEGER,
            priority REAL,
            last_global_step INTEGER,
            payload_json TEXT
        );
        """
    )
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(live_memory_events)").fetchall()}
    if "sequence" not in columns:
        connection.execute("ALTER TABLE live_memory_events ADD COLUMN sequence INTEGER")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_live_memory_events_sequence ON live_memory_events(sequence)")
    connection.commit()


def _normalize_event(raw_event: Any) -> dict[str, Any] | None:
    if isinstance(raw_event, LiveMemoryEvent):
        payload = asdict(raw_event)
    elif isinstance(raw_event, dict):
        payload = dict(raw_event)
    else:
        return None
    if "event_type" not in payload:
        return None
    payload.setdefault("event_id", f"generated:{time.time_ns()}")
    payload.setdefault("global_step", 0)
    payload.setdefault("worker_id", "")
    payload.setdefault("priority", 0.0)
    payload.setdefault("payload", {})
    try:
        payload["event_type"] = str(payload["event_type"])
        payload["event_id"] = str(payload["event_id"])
        payload["global_step"] = int(payload.get("global_step") or 0)
        payload["worker_id"] = str(payload.get("worker_id") or "")
        payload["priority"] = float(payload.get("priority") or 0.0)
        payload["payload"] = dict(payload.get("payload") or {})
    except Exception:
        return None
    return payload


def _apply_projection(connection: sqlite3.Connection, event: dict[str, Any], payload_json: str) -> None:
    event_type = str(event["event_type"])
    payload = dict(event.get("payload") or {})
    global_step = int(event.get("global_step") or 0)
    priority = float(event.get("priority") or 0.0)
    if event_type == "stable_contingency":
        key = str(payload["key"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_stable_contingencies (
                key, action, context_signature, context_level, transformation_family,
                support_count, confidence, priority, last_global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                None if payload.get("action") is None else int(payload.get("action")),
                str(payload.get("context_signature") or ""),
                None if payload.get("context_level") is None else int(payload.get("context_level")),
                None if payload.get("transformation_family") is None else int(payload.get("transformation_family")),
                int(payload.get("support_count") or 0),
                float(payload.get("confidence") or 0.0),
                priority,
                global_step,
                payload_json,
            ),
        )
    elif event_type == "family_update":
        family_signature = str(payload["family_signature"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_family_updates (
                family_signature, family_id, support_count, priority, last_global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                family_signature,
                None if payload.get("family_id") is None else int(payload.get("family_id")),
                int(payload.get("support_count") or 0),
                priority,
                global_step,
                payload_json,
            ),
        )
    elif event_type == "high_priority_replay":
        interaction_id = str(payload["interaction_id"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_replay_candidates (
                interaction_id, replay_priority, reason, family_id, context_signature,
                action_signature, priority, last_global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                float(payload.get("replay_priority") or 0.0),
                str(payload.get("reason") or ""),
                None if payload.get("family_id") is None else str(payload.get("family_id")),
                None if payload.get("context_signature") is None else str(payload.get("context_signature")),
                None if payload.get("action_signature") is None else str(payload.get("action_signature")),
                priority,
                global_step,
                payload_json,
            ),
        )
    elif event_type == "contradiction_cluster":
        contradiction_key = str(payload["contradiction_key"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_contradiction_clusters (
                contradiction_key, context_signature, action_signature, count, priority, last_global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contradiction_key,
                None if payload.get("context_signature") is None else str(payload.get("context_signature")),
                None if payload.get("action_signature") is None else str(payload.get("action_signature")),
                int(payload.get("count") or 0),
                priority,
                global_step,
                payload_json,
            ),
        )
    elif event_type == "carrier_candidate":
        carrier_signature = str(payload["carrier_signature"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_carrier_candidates (
                carrier_signature, carrier_source, support_count, linked_family_count,
                priority, last_global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                carrier_signature,
                str(payload.get("carrier_source") or "unknown"),
                int(payload.get("support_count") or 0),
                int(payload.get("linked_family_count") or 0),
                priority,
                global_step,
                payload_json,
            ),
        )
    elif event_type == "future_option_event":
        event_id = str(payload["event_id"])
        connection.execute(
            """
            INSERT OR REPLACE INTO live_future_option_events (
                event_id, interaction_id, option_delta, motif_type,
                motif_type_source, priority, global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                None if payload.get("interaction_id") is None else str(payload.get("interaction_id")),
                float(payload.get("option_delta") or 0.0),
                str(payload.get("motif_type") or "unknown"),
                str(payload.get("motif_type_source") or "unknown"),
                priority,
                global_step,
                payload_json,
            ),
        )


def _fetch_projection_rows(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query).fetchall()]
