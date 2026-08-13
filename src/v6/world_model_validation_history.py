from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

LINK_HISTORY = "higher_order_world_model_validation_link_history"
_INSTALLED = False
_ORIGINAL_ARCHIVE: Any = None


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _epoch_key(conn: sqlite3.Connection, diagnostic_epoch_id: str | int | None) -> str:
    if diagnostic_epoch_id is not None:
        return str(diagnostic_epoch_id)
    max_step = -1
    if _exists(conn, "world_model_components"):
        row = conn.execute("SELECT MAX(COALESCE(last_seen_global_step, first_seen_global_step, -1)) FROM world_model_components").fetchone()
        max_step = int((row or [-1])[0] or -1)
    return f"step:{max_step}"


def _archive_with_links(memory_dir: Path, diagnostic_epoch_id: str | int | None) -> int:
    changed = int(_ORIGINAL_ARCHIVE(memory_dir, diagnostic_epoch_id) or 0)
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return changed
    with sqlite3.connect(db) as conn:
        if not _exists(conn, "world_model_links"):
            return changed
        if not _exists(conn, LINK_HISTORY):
            conn.execute(f'CREATE TABLE "{LINK_HISTORY}" AS SELECT * FROM world_model_links WHERE 0')
            conn.execute(f'ALTER TABLE "{LINK_HISTORY}" ADD COLUMN diagnostic_epoch_id TEXT')
        else:
            source_info = conn.execute("PRAGMA table_info(world_model_links)").fetchall()
            target_columns = set(_columns(conn, LINK_HISTORY))
            for row in source_info:
                name = str(row[1])
                if name in target_columns:
                    continue
                declared = str(row[2] or "").strip()
                suffix = f" {declared}" if declared else ""
                conn.execute(f'ALTER TABLE "{LINK_HISTORY}" ADD COLUMN "{name}"{suffix}')
        columns = _columns(conn, "world_model_links")
        target_columns = set(_columns(conn, LINK_HISTORY))
        common = [name for name in columns if name in target_columns]
        required = {"component_signature", "linked_type", "linked_key"}
        if not required <= set(common):
            conn.commit()
            return changed
        conn.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS idx_world_validation_link_history_identity '
            f'ON "{LINK_HISTORY}" (component_signature, linked_type, linked_key, diagnostic_epoch_id)'
        )
        cols = ", ".join(f'"{name}"' for name in common)
        before = conn.total_changes
        conn.execute(
            f'INSERT OR REPLACE INTO "{LINK_HISTORY}" ({cols}, diagnostic_epoch_id) '
            f'SELECT {cols}, ? FROM world_model_links',
            (_epoch_key(conn, diagnostic_epoch_id),),
        )
        conn.commit()
        return changed + (conn.total_changes - before)


def install_world_model_validation_history() -> None:
    global _INSTALLED, _ORIGINAL_ARCHIVE
    if _INSTALLED:
        return
    from v6 import higher_order_evidence_history as evidence
    _ORIGINAL_ARCHIVE = evidence.archive_world_model_validation_evidence
    evidence.archive_world_model_validation_evidence = _archive_with_links
    _INSTALLED = True
