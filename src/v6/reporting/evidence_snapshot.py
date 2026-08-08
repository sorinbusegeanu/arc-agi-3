from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SQLITE_NAMES = {
    "current_state.sqlite",
    "graph.sqlite",
    "replay_queue.sqlite",
    "direct_streaming_fold_manifest.sqlite",
}
_COPY_NAMES = {
    "memory_summary.json",
    "direct_streaming_fold_manifest.json",
    "v61_schema_migration.json",
    "v621_schema_migration.json",
    "v63_schema_migration.json",
}


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=60.0) as source_conn:
        with sqlite3.connect(target, timeout=60.0) as target_conn:
            source_conn.backup(target_conn)


def _copy_evidence_tree(source: Path, target: Path) -> None:
    """Copy only canonical compact-memory evidence required by evaluators.

    Raw sampling DBs and parquet artifacts live under run_dir and are not part
    of the compact-memory evidence contract. Recursively copying them made the
    report phase scale with unrelated retained artifacts.
    """
    for name in sorted(_SQLITE_NAMES):
        path = source / name
        if path.is_file():
            _sqlite_backup(path, target / name)
    for name in sorted(_COPY_NAMES):
        path = source / name
        if path.is_file():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        try:
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        except OSError:
            pass


@contextmanager
def read_only_evidence_snapshot(memory_dir: Path | None) -> Iterator[Path | None]:
    if memory_dir is None:
        yield None
        return
    source = Path(memory_dir)
    with tempfile.TemporaryDirectory(prefix="arc_agi3_v61_evidence_") as temporary:
        target = Path(temporary) / "memory"
        target.mkdir(parents=True, exist_ok=True)
        _copy_evidence_tree(source, target)
        _make_read_only(target)
        yield target


def memory_fingerprint(memory_dir: Path | None) -> str | None:
    """Cheap mutation guard for canonical evidence files.

    REPORT uses a physical read-only snapshot, so hashing every byte of large
    SQLite files twice is unnecessary. Size and nanosecond mtime detect source
    mutation while keeping the guard independent of database size.
    """
    if memory_dir is None:
        return None
    root = Path(memory_dir)
    digest = hashlib.sha256()
    for name in sorted(_SQLITE_NAMES | _COPY_NAMES):
        path = root / name
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()
