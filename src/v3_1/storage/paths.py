from __future__ import annotations

from pathlib import Path


def filesystem_safe(value: str) -> str:
    return value.replace(":", "_")


def session_root(root_dir: str, session_id: str) -> Path:
    return Path(root_dir) / filesystem_safe(session_id)


def round_root(root_dir: str, session_id: str, round_id: int) -> Path:
    return session_root(root_dir, session_id) / f"round_{round_id:03d}"
