from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional


class MemoryStore:
    def __init__(self, memory_dir: str, enable_lock: bool = True) -> None:
        self.memory_dir = memory_dir
        self.enable_lock = enable_lock
        self.db_path = os.path.join(memory_dir, "memory.sqlite")
        self.lock_path = os.path.join(memory_dir, "locks", "store.lock")

    def query(self, task_signature: str, game_id: Optional[str] = None) -> Dict[str, Any]:
        self._init_db()
        conn = self._connect(readonly=True)
        try:
            cur = conn.cursor()
            evidence: Dict[str, Any] = {
                "priors": {"action": {}, "coord": {}, "templates": {}},
                "known_failures": {},
                "replay_hints": {},
                "calibration": {},
                "game": {},
            }
            cur.execute(
                "SELECT action_key, attempts_total, effect_total, no_effect_total, avg_changed_cells, avg_bbox_area "
                "FROM action_priors_by_signature WHERE task_signature_v1 = ?",
                (task_signature,),
            )
            for row in cur.fetchall():
                action_key, attempts, effect, no_effect, avg_cells, avg_bbox = row
                evidence["priors"]["action"][action_key] = {
                    "attempts_total": attempts,
                    "effect_total": effect,
                    "no_effect_total": no_effect,
                    "avg_changed_cells": avg_cells,
                    "avg_bbox_area": avg_bbox,
                }
            if not evidence["priors"]["action"]:
                cur.execute(
                    "SELECT action_key, attempts_total, effect_total, no_effect_total, avg_changed_cells, avg_bbox_area "
                    "FROM action_priors_global"
                )
                for row in cur.fetchall():
                    action_key, attempts, effect, no_effect, avg_cells, avg_bbox = row
                    evidence["priors"]["action"][action_key] = {
                        "attempts_total": attempts,
                        "effect_total": effect,
                        "no_effect_total": no_effect,
                        "avg_changed_cells": avg_cells,
                        "avg_bbox_area": avg_bbox,
                    }
            cur.execute(
                "SELECT failure_label, count, params FROM failure_histograms WHERE task_signature_v1 = ?",
                (task_signature,),
            )
            for row in cur.fetchall():
                label, count, params = row
                evidence["known_failures"][label] = {"count": count, "params": params}
            if game_id:
                cur.execute(
                    "SELECT game_id, success_count, attempt_count, last_success_ts, best_known_programs, "
                    "known_noop_signatures, failure_histogram_json FROM game_memory WHERE game_id = ?",
                    (game_id,),
                )
                row = cur.fetchone()
                if row:
                    evidence["game"] = {
                        "game_id": row[0],
                        "success_count": row[1],
                        "attempt_count": row[2],
                        "last_success_ts": row[3],
                        "best_known_programs": _safe_json_load(row[4]),
                        "known_noop_signatures": _safe_json_load(row[5]),
                        "failure_histogram_json": _safe_json_load(row[6]),
                    }
            cur.execute(
                "SELECT candidate_signature_v1, times_considered, times_accepted, times_rejected, avg_score, last_score "
                "FROM candidate_priors WHERE task_signature_v1 = ?",
                (task_signature,),
            )
            for row in cur.fetchall():
                cand_id, considered, accepted, rejected, avg_score, last_score = row
                evidence["priors"]["templates"][cand_id] = {
                    "times_considered": considered,
                    "times_accepted": accepted,
                    "times_rejected": rejected,
                    "avg_score": avg_score,
                    "last_score": last_score,
                }
            return evidence
        finally:
            conn.close()

    def ingest_run_summary(self, run_summary: Dict[str, Any]) -> None:
        self._init_db()
        lock = _FileLock(self.lock_path) if self.enable_lock else _NullLock()
        with lock:
            conn = self._connect(readonly=False)
            try:
                conn.execute("BEGIN")
                self._write_schema(conn)
                self._ingest_action_priors(conn, run_summary, global_table=True)
                self._ingest_action_priors(conn, run_summary, global_table=False)
                self._ingest_hypothesis_priors(conn, run_summary)
                self._ingest_failure_histograms(conn, run_summary)
                self._ingest_game_memory(conn, run_summary)
                self._ingest_events(conn, run_summary)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _ingest_action_priors(
        self, conn: sqlite3.Connection, run_summary: Dict[str, Any], *, global_table: bool
    ) -> None:
        action_efficacy = run_summary.get("action_efficacy", {}) or {}
        per_action = action_efficacy.get("per_action", {})
        task_signature = run_summary.get("task_signature_v1")
        for action_key, stats in per_action.items():
            attempts = int(stats.get("attempts", 0))
            if attempts <= 0:
                continue
            no_effect_rate = float(stats.get("no_effect_rate", 0.0))
            no_effect_total = int(round(no_effect_rate * attempts))
            effect_total = attempts - no_effect_total
            avg_cells = float(stats.get("avg_changed_cells", 0.0))
            avg_bbox = float(stats.get("avg_changed_bbox_area", 0.0))
            if global_table:
                conn.execute(
                    """
                    INSERT INTO action_priors_global(
                        action_key, attempts_total, effect_total, no_effect_total,
                        avg_changed_cells, avg_bbox_area, last_seen_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(action_key) DO UPDATE SET
                        attempts_total = attempts_total + excluded.attempts_total,
                        effect_total = effect_total + excluded.effect_total,
                        no_effect_total = no_effect_total + excluded.no_effect_total,
                        avg_changed_cells = _merge_mean(
                            avg_changed_cells, attempts_total,
                            excluded.avg_changed_cells, excluded.attempts_total
                        ),
                        avg_bbox_area = _merge_mean(
                            avg_bbox_area, attempts_total,
                            excluded.avg_bbox_area, excluded.attempts_total
                        ),
                        last_seen_ts = excluded.last_seen_ts
                    """,
                    (action_key, attempts, effect_total, no_effect_total, avg_cells, avg_bbox, _ts()),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO action_priors_by_signature(
                        task_signature_v1, action_key, attempts_total, effect_total,
                        no_effect_total, avg_changed_cells, avg_bbox_area, last_seen_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_signature_v1, action_key) DO UPDATE SET
                        attempts_total = attempts_total + excluded.attempts_total,
                        effect_total = effect_total + excluded.effect_total,
                        no_effect_total = no_effect_total + excluded.no_effect_total,
                        avg_changed_cells = _merge_mean(
                            avg_changed_cells, attempts_total,
                            excluded.avg_changed_cells, excluded.attempts_total
                        ),
                        avg_bbox_area = _merge_mean(
                            avg_bbox_area, attempts_total,
                            excluded.avg_bbox_area, excluded.attempts_total
                        ),
                        last_seen_ts = excluded.last_seen_ts
                    """,
                    (task_signature, action_key, attempts, effect_total, no_effect_total, avg_cells, avg_bbox, _ts()),
                )

    def _ingest_hypothesis_priors(self, conn: sqlite3.Connection, run_summary: Dict[str, Any]) -> None:
        task_signature = run_summary.get("task_signature_v1")
        outcomes = run_summary.get("hypothesis_outcomes", {}) or {}
        for hyp_id, outcome in outcomes.items():
            considered = 1
            accepted = 1 if outcome.get("supported") else 0
            rejected = 1 if outcome.get("refuted") else 0
            avg_score = float(outcome.get("confidence_update", 0.0))
            conn.execute(
                """
                INSERT INTO candidate_priors(
                    task_signature_v1, candidate_signature_v1, times_considered,
                    times_accepted, times_rejected, avg_score, last_score, reject_histogram_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_signature_v1, candidate_signature_v1) DO UPDATE SET
                    times_considered = times_considered + excluded.times_considered,
                    times_accepted = times_accepted + excluded.times_accepted,
                    times_rejected = times_rejected + excluded.times_rejected,
                    avg_score = _merge_mean(
                        avg_score, times_considered, excluded.avg_score, excluded.times_considered
                    ),
                    last_score = excluded.last_score
                """,
                (task_signature, hyp_id, considered, accepted, rejected, avg_score, avg_score, None),
            )

    def _ingest_failure_histograms(self, conn: sqlite3.Connection, run_summary: Dict[str, Any]) -> None:
        task_signature = run_summary.get("task_signature_v1")
        labels = run_summary.get("failure_labels", []) or []
        for label in labels:
            conn.execute(
                """
                INSERT INTO failure_histograms(task_signature_v1, failure_label, count, params)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_signature_v1, failure_label) DO UPDATE SET
                    count = count + excluded.count
                """,
                (task_signature, label, 1, None),
            )

    def _ingest_game_memory(self, conn: sqlite3.Connection, run_summary: Dict[str, Any]) -> None:
        game_id = run_summary.get("game_id")
        win = bool(run_summary.get("win"))
        failure_hist = json.dumps({label: 1 for label in run_summary.get("failure_labels", []) or []})
        conn.execute(
            """
            INSERT INTO game_memory(
                game_id, success_count, attempt_count, last_success_ts, best_known_programs,
                known_noop_signatures, failure_histogram_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                success_count = success_count + excluded.success_count,
                attempt_count = attempt_count + excluded.attempt_count,
                last_success_ts = CASE WHEN excluded.last_success_ts IS NOT NULL
                    THEN excluded.last_success_ts ELSE last_success_ts END,
                failure_histogram_json = excluded.failure_histogram_json
            """,
            (game_id, 1 if win else 0, 1, _ts() if win else None, None, None, failure_hist),
        )

    def _ingest_events(self, conn: sqlite3.Connection, run_summary: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO events_run_summary_v1(run_id, ingested_ts, run_summary_json)
            VALUES (?, ?, ?)
            """,
            (run_summary.get("run_id"), _ts(), json.dumps(run_summary)),
        )

    def _write_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO schema_meta(schema_version, created_ts, last_compaction_ts, feature_flags)
            VALUES ('1.0', ?, NULL, ?)
            """,
            (_ts(), json.dumps({"candidate_priors": True, "agent_calibration": False})),
        )

    def _init_db(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        conn = self._connect(readonly=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=FULL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta(
                    schema_version TEXT PRIMARY KEY,
                    created_ts REAL,
                    last_compaction_ts REAL,
                    feature_flags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_priors_global(
                    action_key TEXT PRIMARY KEY,
                    attempts_total INTEGER,
                    effect_total INTEGER,
                    no_effect_total INTEGER,
                    avg_changed_cells REAL,
                    avg_bbox_area REAL,
                    last_seen_ts REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_priors_by_signature(
                    task_signature_v1 TEXT,
                    action_key TEXT,
                    attempts_total INTEGER,
                    effect_total INTEGER,
                    no_effect_total INTEGER,
                    avg_changed_cells REAL,
                    avg_bbox_area REAL,
                    last_seen_ts REAL,
                    PRIMARY KEY (task_signature_v1, action_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_priors_signature "
                "ON action_priors_by_signature(task_signature_v1)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS game_memory(
                    game_id TEXT PRIMARY KEY,
                    success_count INTEGER,
                    attempt_count INTEGER,
                    last_success_ts REAL,
                    best_known_programs TEXT,
                    known_noop_signatures TEXT,
                    failure_histogram_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_histograms(
                    task_signature_v1 TEXT,
                    failure_label TEXT,
                    count INTEGER,
                    params TEXT,
                    PRIMARY KEY (task_signature_v1, failure_label)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_priors(
                    task_signature_v1 TEXT,
                    candidate_signature_v1 TEXT,
                    times_considered INTEGER,
                    times_accepted INTEGER,
                    times_rejected INTEGER,
                    avg_score REAL,
                    last_score REAL,
                    reject_histogram_json TEXT,
                    PRIMARY KEY (task_signature_v1, candidate_signature_v1)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_calibration(
                    task_signature_v1 TEXT,
                    agent_id TEXT,
                    role TEXT,
                    suggestions_count INTEGER,
                    accepted_count INTEGER,
                    led_to_progress_count INTEGER,
                    led_to_win_count INTEGER,
                    PRIMARY KEY (task_signature_v1, agent_id, role)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events_run_summary_v1(
                    run_id TEXT,
                    ingested_ts REAL,
                    run_summary_json TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self, readonly: bool) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{self.db_path}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        conn = sqlite3.connect(self.db_path)
        conn.create_function("_merge_mean", 4, _merge_mean)
        return conn


def _merge_mean(old_mean: float, old_n: int, add_mean: float, add_n: int) -> float:
    total = old_n + add_n
    if total <= 0:
        return 0.0
    return (old_mean * old_n + add_mean * add_n) / float(total)


def _safe_json_load(payload: Optional[str]) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def _ts() -> float:
    return time.time()


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FileLock:
    def __init__(self, path: str, timeout_s: float = 10.0) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self.fd: Optional[int] = None

    def __enter__(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                if time.time() - start > self.timeout_s:
                    raise TimeoutError(f"memory store lock timeout: {self.path}")
                time.sleep(0.05)
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.fd is not None:
                os.close(self.fd)
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass
