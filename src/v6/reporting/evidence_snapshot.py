from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
_COPY_SUFFIXES = {".json", ".jsonl", ".txt", ".parquet"}


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=60.0) as source_conn:
        with sqlite3.connect(target, timeout=60.0) as target_conn:
            source_conn.backup(target_conn)


def _copy_evidence_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(("-wal", "-shm", "-journal")):
            continue
        relative = path.relative_to(source)
        destination = target / relative
        suffix = path.suffix.lower()
        if suffix in _SQLITE_SUFFIXES:
            _sqlite_backup(path, destination)
        elif suffix in _COPY_SUFFIXES or path.name in {
            "memory_summary.json",
            "direct_streaming_fold_manifest.json",
        }:
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
    if memory_dir is None:
        return None
    root = Path(memory_dir)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(("-wal", "-shm", "-journal")):
            continue
        if path.suffix.lower() not in _SQLITE_SUFFIXES | {".json"}:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()
