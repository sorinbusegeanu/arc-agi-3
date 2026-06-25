from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any


def cleanup_epoch_artifacts(*, epoch_dir: str | Path, memory_dir: str | Path) -> dict[str, Any]:
    epoch_path = Path(epoch_dir)
    raw_dir = epoch_path / "raw"
    reports_dir = epoch_path / "reports"
    cleanup_dir = epoch_path / "cleanup"
    status_dir = epoch_path / "status"
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    if not reports_dir.exists():
        raise RuntimeError("cleanup refused because reports directory is missing")

    disk_before = _tree_size(epoch_path)
    raw_files = [path for path in raw_dir.rglob("*") if path.is_file()] if raw_dir.exists() else []
    deleted_files_sample: list[str] = []
    raw_bytes_deleted = 0
    deletion_errors: list[str] = []
    for path in raw_files:
        try:
            raw_bytes_deleted += path.stat().st_size
            if len(deleted_files_sample) < 20:
                deleted_files_sample.append(str(path))
            path.unlink()
        except OSError as exc:
            deletion_errors.append(f"{path}: {exc}")
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    memory_paths = [
        Path(memory_dir) / "current_state.sqlite",
        Path(memory_dir) / "graph.sqlite",
        Path(memory_dir) / "replay_queue.sqlite",
    ]
    for sqlite_path in memory_paths:
        if sqlite_path.exists():
            _sqlite_cleanup(sqlite_path)

    disk_after = _tree_size(epoch_path)
    summary = {
        "epoch_id": epoch_path.name,
        "disk_before_cleanup_bytes": disk_before,
        "disk_after_cleanup_bytes": disk_after,
        "disk_freed_bytes": max(0, disk_before - disk_after),
        "raw_files_deleted_count": len(raw_files) - len(deletion_errors),
        "raw_bytes_deleted": raw_bytes_deleted,
        "temp_files_deleted_count": 0,
        "temp_bytes_deleted": 0,
        "memory_db_size_bytes": _file_size(memory_paths[0]),
        "graph_db_size_bytes": _file_size(memory_paths[1]),
        "replay_queue_db_size_bytes": _file_size(memory_paths[2]),
        "reports_size_bytes": _tree_size(reports_dir),
        "kept_files": sorted(
            str(path.relative_to(epoch_path))
            for path in epoch_path.rglob("*")
            if path.is_file()
        ),
        "deleted_files_sample": deleted_files_sample,
        "deletion_errors": deletion_errors,
    }
    (cleanup_dir / "cleanup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def disk_usage_snapshot(path: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    used_percent = (float(usage.used) / float(usage.total) * 100.0) if usage.total else 0.0
    return {
        "disk_total_bytes": int(usage.total),
        "disk_used_bytes": int(usage.used),
        "disk_free_bytes": int(usage.free),
        "disk_used_percent": used_percent,
    }


def stop_due_to_disk(path: str | Path, *, threshold_percent: float) -> tuple[bool, dict[str, Any]]:
    snapshot = disk_usage_snapshot(path)
    triggered = float(snapshot["disk_used_percent"]) >= float(threshold_percent)
    snapshot["stop_if_disk_above_percent"] = float(threshold_percent)
    snapshot["disk_stop_triggered"] = bool(triggered)
    return triggered, snapshot


def _sqlite_cleanup(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
        connection.commit()


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return _file_size(path)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += _file_size(child)
    return total


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0
