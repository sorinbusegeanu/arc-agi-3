from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tqdm.auto import tqdm

from v6.evaluation.sampling_job_metrics import (
    compute_sampling_job_metrics,
    compute_sampling_job_temporal_milestones,
    compute_sampling_job_validation_payload,
)
from v6.memory.compact_memory import (
    CompactMemoryFoldConfig,
    finalize_main_compact_memory,
    fold_single_sampling_db_into_main_compact_memory,
    merge_compact_memory_shards_into_main,
)
from v6.storage.migration import migrate_sqlite_to_parquet


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
    parquet_export_enabled: bool = False
    parquet_root: str | None = None
    storage_batch_size: int = 1000
    compression: str = "zstd"


@dataclass(frozen=True)
class DirectStreamingFoldConfig:
    memory_dir: str
    delete_raw_after_fold: bool = True
    cleanup_stale_legacy_temp_on_start: bool = True
    manifest_name: str = "direct_streaming_fold_manifest.sqlite"
    fold_workers: int = 8
    shard_root_name: str = "direct_streaming_fold_shards"
    retry_attempts: int = 5
    retry_initial_delay_seconds: float = 5.0
    busy_timeout_ms: int = 60000
    shard_synchronous: str = "off"


@dataclass
class DirectStreamingFoldResult:
    job_id: str
    db_path: str
    status: str
    fold_started_at: float
    fold_finished_at: float | None
    deleted_raw: bool
    error: str | None = None
    raw_db_size_bytes: int = 0
    shard_size_before_bytes: int = 0
    shard_size_after_bytes: int = 0
    shard_bytes_added: int = 0
    fold_seconds: float = 0.0
    fold_write_mb_per_second: float = 0.0
    retry_attempt_count: int = 0
    retryable_error_count: int = 0
    retry_error_history: list[str] | None = None
    metrics: dict[str, Any] | None = None
    milestones: dict[str, Any] | None = None
    validation_payload: dict[str, Any] | None = None
    parquet_exported: bool = False


def ensure_direct_streaming_fold_manifest(memory_dir: str | Path, *, busy_timeout_ms: int = 60000) -> Path:
    root = Path(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "direct_streaming_fold_manifest.sqlite"
    with _connect_manifest(path, busy_timeout_ms=busy_timeout_ms) as conn:
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
                parquet_exported INTEGER DEFAULT 0,
                retry_attempt_count INTEGER DEFAULT 0,
                retryable_error_count INTEGER DEFAULT 0,
                last_retry_error TEXT,
                retry_error_history_json TEXT,
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
            CREATE TABLE IF NOT EXISTS validation_payloads (
                job_id TEXT PRIMARY KEY,
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                payload_json TEXT NOT NULL
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
        _ensure_manifest_column(conn, "folded_jobs", "parquet_exported", "INTEGER DEFAULT 0")
        _ensure_manifest_column(conn, "folded_jobs", "retry_attempt_count", "INTEGER DEFAULT 0")
        _ensure_manifest_column(conn, "folded_jobs", "retryable_error_count", "INTEGER DEFAULT 0")
        _ensure_manifest_column(conn, "folded_jobs", "last_retry_error", "TEXT")
        _ensure_manifest_column(conn, "folded_jobs", "retry_error_history_json", "TEXT")
        conn.commit()
    return path


def mark_fold_started(job: DirectStreamingFoldJob, *, started_at: float) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with _connect_manifest(path) as conn:
        conn.execute(
            """
            INSERT INTO folded_jobs (
                job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                global_step_start, global_step_end, status, fold_started_at, fold_finished_at, deleted_raw, parquet_exported, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, NULL, 0, 0, NULL)
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
                parquet_exported = 0,
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


def mark_fold_finished(job: DirectStreamingFoldJob, *, finished_at: float, deleted_raw: bool, parquet_exported: bool) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with _connect_manifest(path) as conn:
        conn.execute(
            """
            UPDATE folded_jobs
            SET status = 'folded',
                fold_finished_at = ?,
                deleted_raw = ?,
                parquet_exported = ?,
                error = NULL
            WHERE job_id = ?
            """,
            (float(finished_at), int(bool(deleted_raw)), int(bool(parquet_exported)), job.job_id),
        )
        conn.commit()


def mark_fold_failed(job: DirectStreamingFoldJob, *, finished_at: float, error: str) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with _connect_manifest(path) as conn:
        conn.execute(
            """
            UPDATE folded_jobs
            SET status = 'failed',
                fold_finished_at = ?,
                deleted_raw = 0,
                parquet_exported = 0,
                error = ?
            WHERE job_id = ?
            """,
            (float(finished_at), str(error), job.job_id),
        )
        conn.commit()


def write_job_metrics(job: DirectStreamingFoldJob, metrics: dict[str, Any]) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with _connect_manifest(path) as conn:
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
    with _connect_manifest(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO temporal_milestones (job_id, game, sampler, seed, milestones_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(milestones, sort_keys=True)),
        )
        conn.commit()


def write_validation_payload(job: DirectStreamingFoldJob, payload: dict[str, Any]) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir)
    with _connect_manifest(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO validation_payloads (job_id, game, sampler, seed, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(payload, sort_keys=True)),
        )
        conn.commit()


def write_fold_job_manifest_result(
    job: DirectStreamingFoldJob,
    *,
    status: str,
    fold_started_at: float,
    fold_finished_at: float,
    deleted_raw: bool,
    parquet_exported: bool,
    metrics: dict[str, Any] | None,
    milestones: dict[str, Any] | None,
    validation_payload: dict[str, Any] | None,
    retry_attempt_count: int = 0,
    retryable_error_count: int = 0,
    last_retry_error: str | None = None,
    retry_error_history: list[str] | None = None,
    busy_timeout_ms: int = 60000,
    error: str | None = None,
) -> None:
    path = ensure_direct_streaming_fold_manifest(job.memory_dir, busy_timeout_ms=busy_timeout_ms)
    with _connect_manifest(path, busy_timeout_ms=busy_timeout_ms) as conn:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO folded_jobs (
                job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                global_step_start, global_step_end, status, fold_started_at, fold_finished_at, deleted_raw, parquet_exported,
                retry_attempt_count, retryable_error_count, last_retry_error, retry_error_history_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                status = excluded.status,
                fold_started_at = excluded.fold_started_at,
                fold_finished_at = excluded.fold_finished_at,
                deleted_raw = excluded.deleted_raw,
                parquet_exported = excluded.parquet_exported,
                retry_attempt_count = excluded.retry_attempt_count,
                retryable_error_count = excluded.retryable_error_count,
                last_retry_error = excluded.last_retry_error,
                retry_error_history_json = excluded.retry_error_history_json,
                error = excluded.error
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
                str(status),
                float(fold_started_at),
                float(fold_finished_at),
                int(bool(deleted_raw)),
                int(bool(parquet_exported)),
                int(retry_attempt_count),
                int(retryable_error_count),
                None if last_retry_error is None else str(last_retry_error),
                json.dumps(list(retry_error_history or []), sort_keys=False),
                None if error is None else str(error),
            ),
        )
        if metrics is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO job_metrics (job_id, game, sampler, seed, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(metrics, sort_keys=True)),
            )
        if milestones is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO temporal_milestones (job_id, game, sampler, seed, milestones_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(milestones, sort_keys=True)),
            )
        if validation_payload is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO validation_payloads (job_id, game, sampler, seed, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job.job_id, job.game, job.sampler, int(job.seed), json.dumps(validation_payload, sort_keys=True)),
            )
        conn.commit()


def load_direct_streamed_job_metrics(memory_dir: str | Path) -> list[dict]:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with _connect_manifest(path) as conn:
        rows = conn.execute("SELECT metrics_json FROM job_metrics ORDER BY game ASC, sampler ASC, seed ASC").fetchall()
    return [json.loads(str(row[0])) for row in rows if row and row[0]]


def load_direct_streamed_temporal_milestones(memory_dir: str | Path) -> list[dict]:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with _connect_manifest(path) as conn:
        rows = conn.execute("SELECT milestones_json FROM temporal_milestones ORDER BY game ASC, sampler ASC, seed ASC").fetchall()
    return [json.loads(str(row[0])) for row in rows if row and row[0]]


def load_direct_streamed_validation_payloads(memory_dir: str | Path) -> list[dict]:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with _connect_manifest(path) as conn:
        rows = conn.execute("SELECT payload_json FROM validation_payloads ORDER BY game ASC, sampler ASC, seed ASC").fetchall()
    return [json.loads(str(row[0])) for row in rows if row and row[0]]


def direct_streaming_manifest_exists(memory_dir: str | Path) -> bool:
    return (Path(memory_dir) / "direct_streaming_fold_manifest.sqlite").exists()


def direct_streaming_manifest_has_failures(memory_dir: str | Path) -> bool:
    path = ensure_direct_streaming_fold_manifest(memory_dir)
    with _connect_manifest(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'failed'").fetchone()
    return int(row[0] or 0) > 0


def _connect_manifest(path: Path, *, busy_timeout_ms: int = 60000) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=max(1.0, float(busy_timeout_ms) / 1000.0))
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_manifest_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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


def _effective_fold_worker_count(requested_workers: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(requested_workers), int(cpu_count)))


def _shard_root(config: DirectStreamingFoldConfig) -> Path:
    return Path(config.memory_dir) / str(config.shard_root_name)


def _epoch_dir_for_db_path(db_path: str | Path) -> Path | None:
    path = Path(db_path)
    parts = path.parts
    try:
        epochs_index = parts.index("epochs")
    except ValueError:
        return None
    if epochs_index + 1 >= len(parts):
        return None
    return Path(*parts[: epochs_index + 2])


def _rerun_reports_for_retried_epochs(*, memory_dir: str | Path, jobs: list[DirectStreamingFoldJob]) -> list[str]:
    from v6.evaluation.h10b_selective_forgetting import evaluate_h10b_selective_forgetting
    from v6.hypothesis_suite_report import run_hypothesis_suite_report
    from v6.memory.selective_forgetting import run_selective_forgetting_pass

    rebuilt: list[str] = []
    seen: set[Path] = set()
    for job in jobs:
        epoch_dir = _epoch_dir_for_db_path(job.db_path)
        if epoch_dir is None or epoch_dir in seen:
            continue
        seen.add(epoch_dir)
        raw_dir = epoch_dir / "raw"
        reports_dir = epoch_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        if not raw_dir.exists():
            continue
        run_hypothesis_suite_report(
            run_dir=raw_dir,
            memory_dir=Path(memory_dir),
            output_dir=reports_dir,
            scan_all_dbs=False,
            max_db_files=0,
            max_rows=0,
            epoch_id=epoch_dir.name,
        )
        epoch_name = str(epoch_dir.name)
        epoch_number = 0
        if epoch_name.startswith("epoch_"):
            try:
                epoch_number = int(epoch_name.split("_", 1)[1])
            except (IndexError, ValueError):
                epoch_number = 0
        forgetting_summary = run_selective_forgetting_pass(memory_dir=memory_dir, epoch=epoch_number)
        evaluate_h10b_selective_forgetting(
            memory_dir=Path(memory_dir),
            run_dir=raw_dir,
            output_dir=reports_dir / "h10b",
            forgetting_summary=forgetting_summary,
        )
        rebuilt.append(str(epoch_dir))
    return rebuilt


def _make_shard_dirs(config: DirectStreamingFoldConfig, worker_count: int) -> list[Path]:
    shard_root = _shard_root(config)
    if shard_root.exists():
        raise RuntimeError(
            f"direct streaming fold shard root already exists: {shard_root}. "
            "This usually means a previous shard merge failed; inspect it before rerunning."
        )
    shard_root.mkdir(parents=True, exist_ok=False)
    shard_dirs: list[Path] = []
    for index in range(max(1, int(worker_count))):
        shard_dir = shard_root / f"shard_{index:04d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        shard_dirs.append(shard_dir)
    return shard_dirs


def _cleanup_stale_existing_shard_root(config: DirectStreamingFoldConfig) -> bool:
    shard_root = _shard_root(config)
    if not shard_root.exists():
        return False
    manifest_path = ensure_direct_streaming_fold_manifest(config.memory_dir)
    with _connect_manifest(manifest_path) as conn:
        failed_count = int(conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'failed'").fetchone()[0] or 0)
        running_count = int(conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'running'").fetchone()[0] or 0)
    if failed_count > 0 or running_count > 0:
        raise RuntimeError(
            f"direct streaming fold shard root already exists: {shard_root}. "
            "Manifest still reports failed or running fold jobs; resolve them before continuing."
        )
    shutil.rmtree(shard_root, ignore_errors=True)
    return not shard_root.exists()


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


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            total += int(child.stat().st_size)
        except OSError:
            continue
    return total


def is_retryable_fold_error(exc: Exception) -> bool:
    if not isinstance(exc, (sqlite3.OperationalError, sqlite3.DatabaseError)):
        return False
    message = str(exc).lower()
    retryable_fragments = (
        "database is locked",
        "database is busy",
        "database table is locked",
        "database schema is locked",
        "locked",
        "busy",
        "disk i/o error",
        "unable to open database file",
    )
    malformed_fragments = (
        "malformed",
        "corrupt",
        "disk image is malformed",
        "not a database",
    )
    if any(fragment in message for fragment in malformed_fragments):
        return False
    return any(fragment in message for fragment in retryable_fragments)


def _retry_delay_seconds(config: DirectStreamingFoldConfig, attempt_index: int) -> float:
    base = max(0.0, float(config.retry_initial_delay_seconds))
    schedule = [base, base * 2.0, base * 4.0, base * 8.0, 60.0]
    if attempt_index < len(schedule):
        return float(min(schedule[attempt_index], 60.0))
    return 60.0


def _fold_one_completed_job_to_shard_once(
    *,
    job: DirectStreamingFoldJob,
    config: DirectStreamingFoldConfig,
    sampling_config: Any,
    shard_dir: str | Path,
) -> DirectStreamingFoldResult:
    started_at = time.time()
    db_path = Path(job.db_path)
    raw_db_size_bytes = int(db_path.stat().st_size) if db_path.exists() else 0
    shard_path = Path(shard_dir)
    shard_size_before_bytes = _tree_size(shard_path)
    metrics = compute_sampling_job_metrics(
        db_path,
        game=job.game,
        sampler_name=job.sampler,
        seed=int(job.seed),
        config=sampling_config,
        busy_timeout_ms=int(config.busy_timeout_ms),
    )
    milestones = compute_sampling_job_temporal_milestones(
        db_path,
        game=job.game,
        sampler_name=job.sampler,
        seed=int(job.seed),
        busy_timeout_ms=int(config.busy_timeout_ms),
    )
    validation_payload = compute_sampling_job_validation_payload(
        db_path,
        game=job.game,
        sampler_name=job.sampler,
        seed=int(job.seed),
        config=sampling_config,
        busy_timeout_ms=int(config.busy_timeout_ms),
    )
    parquet_exported = False
    if bool(job.parquet_export_enabled):
        if not job.parquet_root:
            raise RuntimeError("parquet export enabled but parquet_root is missing")
        migrate_sqlite_to_parquet(
            sqlite_path=db_path,
            parquet_root=Path(job.parquet_root),
            game=job.game,
            sampler=job.sampler,
            seed=int(job.seed),
            steps=int(job.steps),
            batch_size=int(job.storage_batch_size),
            compression=str(job.compression),
            run_summary={
                "horizon": int(job.horizon),
                "context_depth": int(job.context_depth),
                "global_step_start": int(job.global_step_start),
                "global_step_end": int(job.global_step_end),
            },
        )
        parquet_exported = True
    fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=shard_dir,
        fold_config=CompactMemoryFoldConfig(
            global_step_start=int(job.global_step_start),
            global_step_end=int(job.global_step_end),
        ),
        finalize_after_fold=False,
        sqlite_synchronous=str(config.shard_synchronous),
        temporary_shard=True,
        busy_timeout_ms=int(config.busy_timeout_ms),
    )
    finished_at = time.time()
    shard_size_after_bytes = _tree_size(shard_path)
    shard_bytes_added = max(0, int(shard_size_after_bytes - shard_size_before_bytes))
    fold_seconds = max(0.0, float(finished_at - started_at))
    fold_write_mb_per_second = float(shard_bytes_added) / (1024.0 * 1024.0 * fold_seconds) if fold_seconds > 0.0 else 0.0
    return DirectStreamingFoldResult(
        job_id=job.job_id,
        db_path=str(job.db_path),
        status="folded",
        fold_started_at=started_at,
        fold_finished_at=finished_at,
        deleted_raw=False,
        error=None,
        raw_db_size_bytes=int(raw_db_size_bytes),
        shard_size_before_bytes=int(shard_size_before_bytes),
        shard_size_after_bytes=int(shard_size_after_bytes),
        shard_bytes_added=int(shard_bytes_added),
        fold_seconds=float(fold_seconds),
        fold_write_mb_per_second=float(fold_write_mb_per_second),
        retry_error_history=[],
        metrics=metrics,
        milestones=milestones,
        validation_payload=validation_payload,
        parquet_exported=bool(parquet_exported),
    )


def fold_one_completed_job_to_shard(
    *,
    job: DirectStreamingFoldJob,
    config: DirectStreamingFoldConfig,
    sampling_config: Any,
    shard_dir: str | Path,
) -> DirectStreamingFoldResult:
    last_error: Exception | None = None
    started_at = time.time()
    attempts = max(1, int(config.retry_attempts))
    retry_error_history: list[str] = []
    retryable_error_count = 0
    for attempt_index in range(attempts):
        if not Path(job.db_path).exists():
            finished_at = time.time()
            error = f"FileNotFoundError: raw sqlite db missing: {job.db_path}"
            write_fold_job_manifest_result(
                job,
                status="failed",
                fold_started_at=started_at,
                fold_finished_at=finished_at,
                deleted_raw=False,
                parquet_exported=False,
                metrics=None,
                milestones=None,
                validation_payload=None,
                retry_attempt_count=attempt_index + 1,
                retryable_error_count=retryable_error_count,
                last_retry_error=retry_error_history[-1] if retry_error_history else None,
                retry_error_history=retry_error_history,
                busy_timeout_ms=int(config.busy_timeout_ms),
                error=error,
            )
            return DirectStreamingFoldResult(
                job_id=job.job_id,
                db_path=str(job.db_path),
                status="failed",
                fold_started_at=started_at,
                fold_finished_at=finished_at,
                deleted_raw=False,
                error=error,
                retry_attempt_count=attempt_index + 1,
                retryable_error_count=retryable_error_count,
                retry_error_history=list(retry_error_history),
            )
        try:
            result = _fold_one_completed_job_to_shard_once(
                job=job,
                config=config,
                sampling_config=sampling_config,
                shard_dir=shard_dir,
            )
            result.retry_attempt_count = attempt_index + 1
            result.retryable_error_count = retryable_error_count
            result.retry_error_history = list(retry_error_history)
            write_fold_job_manifest_result(
                job,
                status="folded",
                fold_started_at=result.fold_started_at,
                fold_finished_at=float(result.fold_finished_at or time.time()),
                deleted_raw=False,
                parquet_exported=bool(result.parquet_exported),
                metrics=result.metrics,
                milestones=result.milestones,
                validation_payload=result.validation_payload,
                retry_attempt_count=attempt_index + 1,
                retryable_error_count=retryable_error_count,
                last_retry_error=retry_error_history[-1] if retry_error_history else None,
                retry_error_history=retry_error_history,
                busy_timeout_ms=int(config.busy_timeout_ms),
                error=None,
            )
            return result
        except Exception as exc:
            last_error = exc
            if is_retryable_fold_error(exc):
                retryable_error_count += 1
                retry_error_history.append(f"{type(exc).__name__}: {exc}")
            if not is_retryable_fold_error(exc) or attempt_index >= (attempts - 1):
                break
            time.sleep(_retry_delay_seconds(config, attempt_index))
    finished_at = time.time()
    error = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "RuntimeError: unknown fold failure"
    write_fold_job_manifest_result(
        job,
        status="failed",
        fold_started_at=started_at,
        fold_finished_at=finished_at,
        deleted_raw=False,
        parquet_exported=False,
        metrics=None,
        milestones=None,
        validation_payload=None,
        retry_attempt_count=attempts,
        retryable_error_count=retryable_error_count,
        last_retry_error=retry_error_history[-1] if retry_error_history else None,
        retry_error_history=retry_error_history,
        busy_timeout_ms=int(config.busy_timeout_ms),
        error=error,
    )
    return DirectStreamingFoldResult(
        job_id=job.job_id,
        db_path=str(job.db_path),
        status="failed",
        fold_started_at=started_at,
        fold_finished_at=finished_at,
        deleted_raw=False,
        error=error,
        retry_attempt_count=attempts,
        retryable_error_count=retryable_error_count,
        retry_error_history=list(retry_error_history),
    )


def merge_direct_fold_shards(
    *,
    memory_dir: str | Path,
    shard_dirs: list[Path],
    fold_config: CompactMemoryFoldConfig,
    workers: int,
) -> dict[str, Any]:
    return merge_compact_memory_shards_into_main(
        memory_dir=memory_dir,
        shard_dirs=[str(path) for path in shard_dirs],
        fold_config=fold_config,
        parallel_workers=max(1, int(workers)),
        progress=True,
        progress_desc="merge fold shards",
    )


class DirectStreamingFoldWriter:
    def __init__(
        self,
        config: DirectStreamingFoldConfig,
        sampling_config: Any,
    ) -> None:
        self.config = config
        self.sampling_config = sampling_config
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._futures: list[tuple[Any, DirectStreamingFoldJob]] = []
        self._executor: ProcessPoolExecutor | None = None
        self._progress = None
        self._shard_dirs: list[Path] = []
        self._successful_jobs: list[tuple[DirectStreamingFoldJob, DirectStreamingFoldResult]] = []
        self._next_shard_index = 0
        self._effective_workers = 1
        self._summary = {
            "direct_streaming_fold_enabled": True,
            "direct_streaming_fold_job_count": 0,
            "direct_streaming_fold_success_count": 0,
            "direct_streaming_fold_failed_count": 0,
            "direct_streaming_fold_deleted_raw_count": 0,
            "direct_streaming_fold_manifest_path": str(ensure_direct_streaming_fold_manifest(config.memory_dir)),
            "direct_streaming_fold_legacy_temp_cleanup_count": 0,
            "direct_streaming_fold_worker_count": 0,
            "direct_streaming_fold_shard_count": 0,
            "direct_streaming_fold_shard_root": str(_shard_root(config)),
            "direct_streaming_fold_shards_deleted": False,
            "direct_streaming_fold_merge_started_at": None,
            "direct_streaming_fold_merge_finished_at": None,
            "direct_streaming_fold_merge_seconds": None,
            "direct_streaming_fold_jobs_submitted": 0,
            "direct_streaming_fold_jobs_completed": 0,
            "direct_streaming_fold_jobs_failed": 0,
            "direct_streaming_fold_raw_deleted_after_shard_fold_count": 0,
            "direct_streaming_fold_global_step_start": None,
            "direct_streaming_fold_global_step_end": None,
            "direct_streaming_fold_finalized_main_memory": False,
            "direct_streaming_fold_failed_job_ids": [],
            "direct_streaming_fold_failed_errors": [],
            "direct_streaming_fold_total_raw_bytes": 0,
            "direct_streaming_fold_total_shard_bytes_added": 0,
            "direct_streaming_fold_mean_job_seconds": 0.0,
            "direct_streaming_fold_mean_write_mb_per_second": 0.0,
            "direct_streaming_shard_synchronous": str(config.shard_synchronous),
            "direct_streaming_fold_stale_shard_root_removed": False,
            "direct_streaming_fold_retryable_error_count": 0,
            "direct_streaming_fold_max_retry_attempt_count": 0,
        }

    def start(self) -> None:
        if self.config.cleanup_stale_legacy_temp_on_start:
            cleanup = cleanup_stale_legacy_fold_dirs(self.config.memory_dir)
            self._summary["direct_streaming_fold_legacy_temp_cleanup_count"] = int(cleanup["count"])
        if self._executor is not None:
            return
        self._summary["direct_streaming_fold_stale_shard_root_removed"] = bool(_cleanup_stale_existing_shard_root(self.config))
        self._effective_workers = _effective_fold_worker_count(int(self.config.fold_workers))
        self._shard_dirs = _make_shard_dirs(self.config, self._effective_workers)
        self._executor = ProcessPoolExecutor(max_workers=self._effective_workers, max_tasks_per_child=1)
        self._summary["direct_streaming_fold_worker_count"] = int(self._effective_workers)
        self._summary["direct_streaming_fold_shard_count"] = int(len(self._shard_dirs))

    def submit(self, job: DirectStreamingFoldJob) -> None:
        if self._executor is None or not self._shard_dirs:
            raise RuntimeError("direct streaming fold writer was not started")
        shard_dir = self._shard_dirs[self._next_shard_index % len(self._shard_dirs)]
        self._next_shard_index += 1
        future = self._executor.submit(
            fold_one_completed_job_to_shard,
            job=job,
            config=self.config,
            sampling_config=self.sampling_config,
            shard_dir=str(shard_dir),
        )
        self._futures.append((future, job))
        self._submitted_jobs += 1
        self._summary["direct_streaming_fold_job_count"] = int(self._submitted_jobs)
        self._summary["direct_streaming_fold_jobs_submitted"] = int(self._submitted_jobs)

    def close(self) -> dict[str, Any]:
        merge_error: Exception | None = None
        self._progress = tqdm(
            total=int(self._submitted_jobs),
            desc="direct fold",
            unit="job",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            future_jobs = {future: job for future, job in self._futures}
            for future in as_completed(list(future_jobs)):
                job = future_jobs[future]
                result = future.result()
                self._completed_jobs += 1
                self._summary["direct_streaming_fold_jobs_completed"] = int(self._completed_jobs)
                if self._progress is not None:
                    self._progress.update(1)
                self._record_result(job, result)
            if self._summary["direct_streaming_fold_failed_count"] == 0 and self._summary["direct_streaming_fold_success_count"] > 0:
                merge_started_at = time.time()
                self._summary["direct_streaming_fold_merge_started_at"] = float(merge_started_at)
                merge_direct_fold_shards(
                    memory_dir=self.config.memory_dir,
                    shard_dirs=self._shard_dirs,
                    fold_config=CompactMemoryFoldConfig(
                        global_step_start=int(self._summary.get("direct_streaming_fold_global_step_start", 0) or 0),
                        global_step_end=int(self._summary.get("direct_streaming_fold_global_step_end", 0) or 0),
                    ),
                    workers=self._effective_workers,
                )
                finalize_main_compact_memory(
                    memory_dir=self.config.memory_dir,
                    fold_config=CompactMemoryFoldConfig(
                        global_step_start=int(self._summary.get("direct_streaming_fold_global_step_start", 0) or 0),
                        global_step_end=int(self._summary.get("direct_streaming_fold_global_step_end", 0) or 0),
                    ),
                )
                merge_finished_at = time.time()
                self._summary["direct_streaming_fold_merge_finished_at"] = float(merge_finished_at)
                self._summary["direct_streaming_fold_merge_seconds"] = float(merge_finished_at - merge_started_at)
                self._summary["direct_streaming_fold_finalized_main_memory"] = True
                if bool(self.config.delete_raw_after_fold):
                    for job, result in self._successful_jobs:
                        deleted_raw = _delete_raw_artifacts(Path(job.db_path))
                        result.deleted_raw = bool(deleted_raw)
                        if deleted_raw:
                            self._summary["direct_streaming_fold_deleted_raw_count"] += 1
                            self._summary["direct_streaming_fold_raw_deleted_after_shard_fold_count"] += 1
                        write_fold_job_manifest_result(
                            job,
                            status="folded",
                            fold_started_at=float(result.fold_started_at),
                            fold_finished_at=float(result.fold_finished_at or merge_finished_at),
                            deleted_raw=bool(deleted_raw),
                            parquet_exported=bool(result.parquet_exported),
                            metrics=result.metrics,
                            milestones=result.milestones,
                            validation_payload=result.validation_payload,
                            retry_attempt_count=int(result.retry_attempt_count or 0),
                            retryable_error_count=int(result.retryable_error_count or 0),
                            last_retry_error=(result.retry_error_history or [None])[-1],
                            retry_error_history=list(result.retry_error_history or []),
                            error=None,
                        )
                shutil.rmtree(_shard_root(self.config), ignore_errors=True)
                self._summary["direct_streaming_fold_shards_deleted"] = True
        except Exception as exc:
            merge_error = exc
        finally:
            if self._progress is not None:
                self._progress.close()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
            self._write_summary()
        if merge_error is not None:
            raise RuntimeError(f"direct streaming fold merge failed: {merge_error}") from merge_error
        return dict(self._summary)

    def _record_result(self, job: DirectStreamingFoldJob, result: DirectStreamingFoldResult) -> None:
        if result.status == "folded":
            self._summary["direct_streaming_fold_success_count"] += 1
            self._successful_jobs.append((job, result))
            success_count = int(self._summary["direct_streaming_fold_success_count"] or 0)
            start = int(job.global_step_start)
            end = int(job.global_step_end)
            current_start = self._summary.get("direct_streaming_fold_global_step_start")
            current_end = self._summary.get("direct_streaming_fold_global_step_end")
            self._summary["direct_streaming_fold_global_step_start"] = start if current_start is None else min(int(current_start), start)
            self._summary["direct_streaming_fold_global_step_end"] = end if current_end is None else max(int(current_end), end)
            self._summary["direct_streaming_fold_total_raw_bytes"] += int(result.raw_db_size_bytes or 0)
            self._summary["direct_streaming_fold_total_shard_bytes_added"] += int(result.shard_bytes_added or 0)
            total_seconds = float(self._summary.get("direct_streaming_fold_mean_job_seconds", 0.0) or 0.0) * max(0, success_count - 1) + float(result.fold_seconds or 0.0)
            total_rate = float(self._summary.get("direct_streaming_fold_mean_write_mb_per_second", 0.0) or 0.0) * max(0, success_count - 1) + float(result.fold_write_mb_per_second or 0.0)
            self._summary["direct_streaming_fold_mean_job_seconds"] = float(total_seconds / success_count)
            self._summary["direct_streaming_fold_mean_write_mb_per_second"] = float(total_rate / success_count)
        else:
            self._summary["direct_streaming_fold_failed_count"] += 1
            self._summary["direct_streaming_fold_failed_job_ids"].append(str(job.job_id))
            self._summary.setdefault("direct_streaming_fold_failed_retry_attempt_counts", []).append(int(result.retry_attempt_count or 0))
            if result.error:
                self._summary["direct_streaming_fold_failed_errors"].append(str(result.error))
        self._summary["direct_streaming_fold_retryable_error_count"] = int(self._summary.get("direct_streaming_fold_retryable_error_count", 0) or 0) + int(result.retryable_error_count or 0)
        self._summary["direct_streaming_fold_max_retry_attempt_count"] = max(
            int(self._summary.get("direct_streaming_fold_max_retry_attempt_count", 0) or 0),
            int(result.retry_attempt_count or 0),
        )
        self._summary["direct_streaming_fold_jobs_failed"] = int(self._summary["direct_streaming_fold_failed_count"])

    def _write_summary(self) -> None:
        path = ensure_direct_streaming_fold_manifest(self.config.memory_dir)
        with _connect_manifest(path) as conn:
            for key, value in self._summary.items():
                conn.execute(
                    "INSERT OR REPLACE INTO fold_summary (key, value_json) VALUES (?, ?)",
                    (str(key), json.dumps(value)),
                )
            conn.commit()


def retry_direct_streaming_fold_failures(
    *,
    manifest_path: str | Path,
    memory_dir: str | Path,
    workers: int = 2,
    delete_raw_after_fold: bool = True,
    finalize_after_success: bool = False,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    with _connect_manifest(manifest) as conn:
        rows = conn.execute(
            """
            SELECT job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                   global_step_start, global_step_end
            FROM folded_jobs
            WHERE status = 'failed'
            ORDER BY game ASC, sampler ASC, seed ASC, global_step_start ASC
            """
        ).fetchall()
    jobs: list[DirectStreamingFoldJob] = []
    for row in rows:
        db_path = Path(str(row[1]))
        if not db_path.exists():
            continue
        jobs.append(
            DirectStreamingFoldJob(
                job_id=str(row[0]),
                db_path=str(db_path),
                game=str(row[2]),
                sampler=str(row[3]),
                seed=int(row[4]),
                steps=int(row[5]),
                horizon=int(row[6]),
                context_depth=int(row[7]),
                global_step_start=int(row[8]),
                global_step_end=int(row[9]),
                memory_dir=str(memory_dir),
                delete_raw_after_fold=bool(delete_raw_after_fold),
            )
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    retry_config = DirectStreamingFoldConfig(
        memory_dir=str(memory_dir),
        delete_raw_after_fold=bool(delete_raw_after_fold),
        cleanup_stale_legacy_temp_on_start=False,
        fold_workers=max(1, int(workers)),
        shard_root_name=f"direct_streaming_fold_retry_shards_{timestamp}",
    )
    summary = {
        "direct_streaming_fold_retry_count": int(len(jobs)),
        "direct_streaming_fold_remaining_failed_count": 0,
        "direct_streaming_fold_finalized_main_memory": False,
        "direct_streaming_fold_retry_manifest_path": str(manifest),
        "direct_streaming_fold_reports_rebuilt": [],
    }
    if not jobs:
        with _connect_manifest(manifest) as conn:
            remaining = int(conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'failed'").fetchone()[0] or 0)
            summary["direct_streaming_fold_remaining_failed_count"] = remaining
            for key, value in summary.items():
                conn.execute("INSERT OR REPLACE INTO fold_summary (key, value_json) VALUES (?, ?)", (str(key), json.dumps(value)))
            conn.commit()
        return summary
    effective_workers = min(_effective_fold_worker_count(int(workers)), len(jobs))
    shard_dirs = _make_shard_dirs(retry_config, effective_workers)
    progress = tqdm(total=len(jobs), desc="retry direct fold", unit="job", dynamic_ncols=True, leave=True)
    try:
        with ProcessPoolExecutor(max_workers=effective_workers, max_tasks_per_child=1) as executor:
            future_map = {}
            for index, job in enumerate(jobs):
                shard_dir = shard_dirs[index % len(shard_dirs)]
                sampling_config = SimpleNamespace(steps=int(job.steps), horizon=int(job.horizon))
                future = executor.submit(
                    fold_one_completed_job_to_shard,
                    job=job,
                    config=retry_config,
                    sampling_config=sampling_config,
                    shard_dir=str(shard_dir),
                )
                future_map[future] = job
            success_count = 0
            for future in as_completed(list(future_map)):
                result = future.result()
                progress.update(1)
                if result.status == "folded":
                    success_count += 1
        if success_count > 0:
            min_step = min(int(job.global_step_start) for job in jobs)
            max_step = max(int(job.global_step_end) for job in jobs)
            merge_direct_fold_shards(
                memory_dir=memory_dir,
                shard_dirs=shard_dirs,
                fold_config=CompactMemoryFoldConfig(global_step_start=min_step, global_step_end=max_step),
                workers=effective_workers,
            )
            shutil.rmtree(_shard_root(retry_config), ignore_errors=True)
        with _connect_manifest(manifest) as conn:
            remaining = int(conn.execute("SELECT COUNT(*) FROM folded_jobs WHERE status = 'failed'").fetchone()[0] or 0)
        summary["direct_streaming_fold_remaining_failed_count"] = remaining
        if remaining == 0 and bool(finalize_after_success) and jobs:
            min_step = min(int(job.global_step_start) for job in jobs)
            max_step = max(int(job.global_step_end) for job in jobs)
            finalize_main_compact_memory(
                memory_dir=memory_dir,
                fold_config=CompactMemoryFoldConfig(global_step_start=min_step, global_step_end=max_step),
            )
            summary["direct_streaming_fold_finalized_main_memory"] = True
            summary["direct_streaming_fold_reports_rebuilt"] = _rerun_reports_for_retried_epochs(
                memory_dir=memory_dir,
                jobs=jobs,
            )
    finally:
        progress.close()
    with _connect_manifest(manifest) as conn:
        for key, value in summary.items():
            conn.execute("INSERT OR REPLACE INTO fold_summary (key, value_json) VALUES (?, ?)", (str(key), json.dumps(value)))
        conn.commit()
    return summary
