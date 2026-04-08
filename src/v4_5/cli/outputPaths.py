from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def default_run_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path("runs") / "v4_5" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_output_dir(configured_output_dir: str | None) -> Path:
    if configured_output_dir:
        path = Path(configured_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return default_run_output_dir()


def debug_log_path() -> Path:
    path = Path("runs") / "debug.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reset_debug_log() -> Path:
    path = debug_log_path()
    path.write_text("", encoding="utf-8")
    return path
