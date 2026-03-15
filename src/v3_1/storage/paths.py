from __future__ import annotations

from pathlib import Path


def filesystem_safe(value: str) -> str:
    return value.replace(":", "_")


def session_root(root_dir: str, session_id: str) -> Path:
    return Path(root_dir) / filesystem_safe(session_id)


def visualization_root(root_dir: str, session_id: str) -> Path:
    return session_root(root_dir, session_id) / "visualization"


def round_root(root_dir: str, session_id: str, round_id: int) -> Path:
    return session_root(root_dir, session_id) / f"round_{round_id:03d}"


def get_persistent_memory_db_path(storage_root: str) -> Path:
    return Path(storage_root) / "persistent_memory.sqlite"


def get_session_memory_snapshot_path(session_dir: str, round_id: int, pass_id: int) -> Path:
    return Path(session_dir) / f"round_{round_id:03d}" / f"memory_pass{pass_id}_round{round_id:03d}.json"
