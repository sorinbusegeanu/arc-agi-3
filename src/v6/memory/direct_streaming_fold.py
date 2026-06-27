from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from v6.evaluation.sampling_job_metrics import (
    compute_sampling_job_metrics,
    compute_sampling_job_temporal_milestones,
)
from v6.memory.compact_memory import CompactMemoryFoldConfig, fold_single_sampling_db_into_main_compact_memory


@dataclass(frozen=True)
class DirectStreamingFoldJob:
    job_id: str
    db_path: str
    game: str
    sampler: str
    seed: int
    steps: int
    horizon: int
    context_depth: int
    global_step_start: int
    global_step_end: int
    memory_dir: str
    delete_raw_after_fold: bool = True


@dataclass(frozen=True)
class DirectStreamingFoldConfig:
    memory_dir: str
    delete_raw_after_fold: bool = True
    cleanup_stale_legacy_temp_on_start: bool = True
    manifest_name: str = "direct_streaming_fold_manifest.sqlite"


@dataclass
class DirectStreamingFoldResult:
    job_id: str
    db_path: str
    status: str
    fold_started_at: float
    fold_finished_at: float | None
    deleted_raw: bool
    error: str | None = None


def ensure_direct_streaming_fold_manifest(memory_dir: str | Path) -> Path:
    root = Path(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "direct_streaming_fold_manifest.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS folded_jobs (
                job_id TEXT PRIMARY KEY,
                db_path TEXT NOT NULL,
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                steps INTEGER,
                horizon INTEGER,
                context_depth INTEGER,
                global_step_start INTEGER,
                global_step_end INTEGER,
                status TEXT NOT NULL,
                fold_started_at REAL,
                fold_finished_at REAL,
                deleted_raw INTEGER DEFAULT 0,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_metrics (
                job_id TEXT PRIMARY KEY,
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                metrics_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS temporal_milestones (
                job_id TEXT PRIMARY KEY,
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                milestones_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fold_summary (
                key TEXT PRIMARY KEY,
                value_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_folded_jobs_status
            ON folded_jobs(status)
            """
        )
        conn.commit()
    return path


def mark_fold_started(job: DirectStreamingFoldJob, *, started_at: float) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO folded_jobs (
                job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                global_step_start, global_step_end, status, fold_started_at, fold_finished_at, deleted_raw, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, NULL, 0, NULL)
            ON CONFLICT(job_id) DO UPDATE SET
                db_path = excluded.db_path,
                game = excluded.game,
                sampler = excluded.sampler,
                seed = excluded.seed,
                steps = excluded.steps,
                horizon = excluded.horizon,
                context_depth = excluded.context_depth,
                global_step_start = excluded.global_step_start,
                global_step_end = excluded.global_step_end,
                status = 'running',
                fold_started_at = excluded.fold_started_at,
                fold_finished_at = NULL,
                deleted_raw = 0,
                error = NULL
            """,
            (
                job.job_id,
                str(job.db_path),
                job.game,
                job.sampler,
                int(job.seed),
                int(job.steps),
                int(job.horizon),
                int(job.context_depth),
                int(job.global_step_start),
                int(job.global_step_end),
                float(started_at),
            ),
        )
        conn.commit()


def mark_fold_finished(job: DirectStreamingFoldJob, *, finished_at: float, deleted_raw: bool) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            UPDATE folded_jobs
            SET status = 'folded',
                fold_finished_at = ?,
                deleted_raw = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (float(finished_at), int(bool(deleted_raw)), job.job_id),
        )
        conn.commit()


def mark_fold_failed(job: DirectStreamingFoldJob, *, finished_at: float, error: str) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            UPDATE folded_jobs
            SET status = 'failed',
                fold_finished_at = ?,
                deleted_raw = 0,
                error = ?
            WHERE job_id = ?
            """,
            (float(finished_at), str(error), job.job_id),
        )
        conn.commit()


def write_job_metrics(job: DirectStreamingFoldJob, metrics: dict[str, Any]) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO job_metrics (job_id, game, sampler, seed, metrics_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(metrics, sort_keys=True)),
        )
        conn.commit()


def write_temporal_milestones(job: DirectStreamingFoldJob, milestones: dict[str, Any]) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO temporal_milestones (job_id, game, sampler, seed, milestones_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(milestones, sort_keys=True)),
        )
        conn.commit()


def load_direct_streamed_job_metrics(memory_dir: str | Path) -> list[dict]:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT metrics_json FROM job_metrics ORDER BY game ASC, sampler ASC, seed ASC").fetchall()
    return [json.loads(str(row[0])) for row in rows if row and row[0]]


def load_direct_streamed_temporal_milestones(memory_dir: str | Path) -> list[dict]:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT milestones_json FROM temporal_milestones ORDER BY game ASC, sampler ASC, seed ASC").fetchall()
    return [json.loads(str(row[0])) for row in rows if row and row[0]]


def direct_streaming_manifest_exists(memory_dir: str | Path) -> bool:
    return (Path(memory_dir) / "direct_streaming_fold_manifest.sqlite").exists()


def direct_streaming_manifest_has_failures(memory_dir: str | Path) -> bool:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'failed'").fetchone()
    return int(row[0] or 0) > 0


def cleanup_stale_legacy_fold_dirs(memory_dir: str | Path) -> dict[str, Any]:
    root = Path(memory_dir)
    deleted: list[str] = []
    for pattern in ("sampling_sidecar_fold_*", "compact_merge_*", "compact_fold_*", "streaming_fold_shards"):
        for path in root.glob(pattern):
            if not path.exists():
                continue
            if path.is_dir():
                import shutil

                shutil.rmtree(path, ignore_errors=True)
                deleted.append(str(path))
    return {"deleted": deleted, "count": len(deleted)}


def _delete_raw_artifacts(db_path: Path) -> bool:
    targets = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
        db_path.with_name("live_graph_compact.json"),
        db_path.with_name("carrier_candidates.json"),
        db_path.with_name("context_contradictions.json"),
        db_path.with_name("memory_lifecycle_summary.json"),
        db_path.with_name("memory_replay_candidates.json"),
        db_path.with_name("efficiency_summary.json"),
    ]
    success = True
    for path in targets:
        if not path.exists():
            continue
        try:
            path.unlink()
        except OSError:
            success = False
    return success


def fold_one_completed_job_directly(
    *,
    job: DirectStreamingFoldJob,
    config: DirectStreamingFoldConfig,
    sampling_config: Any,
) -> DirectStreamingFoldResult:
    started_at = time.time()
    mark_fold_started(job, started_at=started_at)
    try:
        db_path = Path(job.db_path)
        metrics = compute_sampling_job_metrics(
            db_path,
            game=job.game,
            sampler_name=job.sampler,
            seed=int(job.seed),
            config=sampling_config,
        )
        milestones = compute_sampling_job_temporal_milestones(
            db_path,
            game=job.game,
            sampler_name=job.sampler,
            seed=int(job.seed),
        )
        write_job_metrics(job, metrics)
        write_temporal_milestones(job, milestones)
        fold_single_sampling_db_into_main_compact_memory(
            db_path=db_path,
            memory_dir=config.memory_dir,
            fold_config=CompactMemoryFoldConfig(
                global_step_start=int(job.global_step_start),
                global_step_end=int(job.global_step_end),
            ),
        )
        deleted_raw = False
        if bool(job.delete_raw_after_fold and config.delete_raw_after_fold):
            deleted_raw = _delete_raw_artifacts(db_path)
        finished_at = time.time()
        mark_fold_finished(job, finished_at=finished_at, deleted_raw=deleted_raw)
        return DirectStreamingFoldResult(
            job_id=job.job_id,
            db_path=str(job.db_path),
            status="folded",
            fold_started_at=started_at,
            fold_finished_at=finished_at,
            deleted_raw=bool(deleted_raw),
            error=None,
        )
    except Exception as exc:
        finished_at = time.time()
        mark_fold_failed(job, finished_at=finished_at, error=f"{type(exc).__name__}: {exc}")
        return DirectStreamingFoldResult(
            job_id=job.job_id,
            db_path=str(job.db_path),
            status="failed",
            fold_started_at=started_at,
            fold_finished_at=finished_at,
            deleted_raw=False,
            error=f"{type(exc).__name__}: {exc}",
        )


class DirectStreamingFoldWriter:
    def __init__(
        self,
        config: DirectStreamingFoldConfig,
        sampling_config: Any,
    ) -> None:
        self.config = config
        self.sampling_config = sampling_config
        self._queue: queue.Queue[DirectStreamingFoldJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._summary = {
            "direct_streaming_fold_enabled": True,
            "direct_streaming_fold_job_count": 0,
            "direct_streaming_fold_success_count": 0,
            "direct_streaming_fold_failed_count": 0,
            "direct_streaming_fold_deleted_raw_count": 0,
            "direct_streaming_fold_manifest_path": str(ensure_direct_streaming_fold_manifest(config.memory_dir)),
            "direct_streaming_fold_legacy_temp_cleanup_count": 0,
        }

    def start(self) -> None:
        if self.config.cleanup_stale_legacy_temp_on_start:
            cleanup = cleanup_stale_legacy_fold_dirs(self.config.memory_dir)
            self._summary["direct_streaming_fold_legacy_temp_cleanup_count"] = int(cleanup["count"])
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="direct-streaming-fold-writer", daemon=True)
        self._thread.start()

    def submit(self, job: DirectStreamingFoldJob) -> None:
        self._queue.put(job)

    def close(self) -> dict[str, Any]:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join()
        path = ensure_direct_streaming_fold_manifest(self.config.memory_dir)
        with sqlite3.connect(path) as conn:
            for key, value in self._summary.items():
                conn.execute(
                    "INSERT OR REPLACE INTO fold_summary (key, value_json) VALUES (?, ?)",
                    (str(key), json.dumps(value)),
                )
            conn.commit()
        return dict(self._summary)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            self._summary["direct_streaming_fold_job_count"] += 1
            result = fold_one_completed_job_directly(
                job=job,
                config=self.config,
                sampling_config=self.sampling_config,
            )
            if result.status == "folded":
                self._summary["direct_streaming_fold_success_count"] += 1
                if result.deleted_raw:
                    self._summary["direct_streaming_fold_deleted_raw_count"] += 1
            else:
                self._summary["direct_streaming_fold_failed_count"] += 1
