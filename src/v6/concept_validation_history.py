from __future__ import annotations

import sqlite3
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_VALIDATE: Any = None
_ORIGINAL_DIAGNOSTICS: Any = None
_ORIGINAL_MOTIF_ROWS: Any = None
_ACTIVE_HISTORY: ContextVar[dict[str, Any] | None] = ContextVar(
    "v6_concept_validation_history", default=None
)

TRANSFER_HISTORY = "concept_validation_role_transfer_history"
FUTURE_HISTORY = "concept_validation_future_option_event_history"
MOTIF_HISTORY = "concept_validation_future_option_motif_history"


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _archive(conn: sqlite3.Connection, source: str, target: str, key_sql: str) -> int:
    if not _exists(conn, source):
        return 0
    if not _exists(conn, target):
        conn.execute(f'CREATE TABLE "{target}" AS SELECT * FROM "{source}" WHERE 0')
    source_info = conn.execute(f'PRAGMA table_info("{source}")').fetchall()
    target_columns = set(_columns(conn, target))
    for row in source_info:
        name = str(row[1])
        if name in target_columns:
            continue
        declared = str(row[2] or "").strip()
        suffix = f" {declared}" if declared else ""
        conn.execute(f'ALTER TABLE "{target}" ADD COLUMN "{name}"{suffix}')
    index_name = f"idx_{target[:40]}_identity"
    try:
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{target}" ({key_sql})')
    except sqlite3.IntegrityError:
        conn.execute(f'DELETE FROM "{target}" WHERE rowid NOT IN (SELECT MAX(rowid) FROM "{target}" GROUP BY {key_sql})')
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{target}" ({key_sql})')
    source_cols = _columns(conn, source)
    target_cols = set(_columns(conn, target))
    common = [name for name in source_cols if name in target_cols]
    if not common:
        return 0
    cols = ", ".join(f'"{name}"' for name in common)
    before = conn.total_changes
    conn.execute(f'INSERT OR IGNORE INTO "{target}" ({cols}) SELECT {cols} FROM "{source}"')
    return conn.total_changes - before


def archive_validation_evidence(memory_dir: Path) -> dict[str, int]:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return {"transfer": 0, "future": 0, "motif": 0}
    with sqlite3.connect(db) as conn:
        result = {
            "transfer": _archive(conn, "role_transfer_attempts", TRANSFER_HISTORY, '"attempt_id"'),
            "future": _archive(conn, "future_option_events", FUTURE_HISTORY, '"event_id"'),
            "motif": _archive(conn, "future_option_motifs", MOTIF_HISTORY, '"motif_signature", "last_seen_global_step"'),
        }
        conn.commit()
        return result


def _dedupe(rows: list[sqlite3.Row], key: str) -> list[sqlite3.Row]:
    by_id: dict[str, sqlite3.Row] = {}
    no_id: list[sqlite3.Row] = []
    for row in rows:
        value = row[key] if key in row.keys() else None
        if value in (None, ""):
            no_id.append(row)
        else:
            by_id[str(value)] = row
    return no_id + list(by_id.values())


def _empty_history_context() -> dict[str, Any]:
    return {"transfer_rows": [], "future_rows": [], "motif_rows": [], "merged_cache": {}, "motif_cache": None}


def _history_from_connection(conn: sqlite3.Connection) -> dict[str, Any]:
    payload = _empty_history_context()
    if _exists(conn, TRANSFER_HISTORY):
        payload["transfer_rows"] = list(conn.execute(f'SELECT * FROM "{TRANSFER_HISTORY}"').fetchall())
    if _exists(conn, FUTURE_HISTORY):
        payload["future_rows"] = list(conn.execute(f'SELECT * FROM "{FUTURE_HISTORY}"').fetchall())
    if _exists(conn, MOTIF_HISTORY):
        payload["motif_rows"] = [dict(row) for row in conn.execute(
            f'SELECT motif_signature, source_role_ids_json, motif_stability_score, is_emergent, last_seen_global_step '
            f'FROM "{MOTIF_HISTORY}" WHERE last_seen_global_step IS NOT NULL'
        ).fetchall()]
    return payload


def _load_history(memory_dir: Path) -> dict[str, Any]:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return _empty_history_context()
    uri = f"file:{db.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60.0) as conn:
        conn.row_factory = sqlite3.Row
        return _history_from_connection(conn)


def _validate_with_history(*args: Any, **kwargs: Any):
    config = kwargs.get("config")
    if config is None and len(args) > 1:
        config = args[1]
    if config is None or not bool(getattr(config, "enabled", False)):
        return _ORIGINAL_VALIDATE(*args, **kwargs)
    memory_dir_value = kwargs.get("memory_dir")
    if memory_dir_value is None and args:
        memory_dir_value = args[0]
    if memory_dir_value is None:
        return _ORIGINAL_VALIDATE(*args, **kwargs)
    ctx = _load_history(Path(memory_dir_value))
    token = _ACTIVE_HISTORY.set(ctx)
    try:
        result = _ORIGINAL_VALIDATE(*args, **kwargs)
    finally:
        _ACTIVE_HISTORY.reset(token)
    if bool(kwargs.get("validate_world_models", False)):
        from v6 import higher_order_evidence_history as evidence_history
        evidence_history.archive_world_model_validation_evidence(Path(memory_dir_value), kwargs.get("diagnostic_epoch_id"))
    if isinstance(result, dict):
        result = dict(result)
        result["cumulative_validation_history"] = {
            "transfer_rows": len(ctx["transfer_rows"]),
            "future_option_event_rows": len(ctx["future_rows"]),
            "future_option_motif_rows": len(ctx["motif_rows"]),
            "history_loaded_once": True,
        }
    return result


def _diagnostics(*args: Any, **kwargs: Any):
    state_conn = kwargs.get("state_conn")
    if state_conn is None:
        return _ORIGINAL_DIAGNOSTICS(*args, **kwargs)
    ctx = _ACTIVE_HISTORY.get()
    direct_context = ctx is None
    if ctx is None:
        ctx = _history_from_connection(state_conn)
    current_transfer = list(kwargs.get("transfer_rows") or [])
    current_future = list(kwargs.get("future_rows") or [])
    cache_key = (id(current_transfer), len(current_transfer), id(current_future), len(current_future))
    cache = ctx["merged_cache"]
    merged = cache.get(cache_key)
    if merged is None:
        transfer_rows = _dedupe(list(ctx["transfer_rows"]) + current_transfer, "attempt_id")
        future_rows = _dedupe(list(ctx["future_rows"]) + current_future, "event_id")
        from v6 import higher_order_substrate as substrate
        merged = (transfer_rows, future_rows, substrate._build_transfer_history_index(transfer_rows))
        cache[cache_key] = merged
    transfer_rows, future_rows, transfer_history = merged
    updated = dict(kwargs)
    updated["transfer_rows"] = transfer_rows
    updated["future_rows"] = future_rows
    updated["transfer_history"] = transfer_history
    result = _ORIGINAL_DIAGNOSTICS(*args, **updated)
    if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], dict):
        events, diagnostics, state = result
        diagnostics = dict(diagnostics)
        diagnostics["validation_history_applied"] = True
        diagnostics["validation_transfer_history_row_count"] = len(transfer_rows)
        diagnostics["validation_future_option_history_row_count"] = len(future_rows)
        diagnostics["validation_history_loaded_once"] = not direct_context
        return events, diagnostics, state
    return result


def _motif_rows(state_conn: sqlite3.Connection):
    ctx = _ACTIVE_HISTORY.get()
    direct_context = ctx is None
    if ctx is None:
        ctx = _history_from_connection(state_conn)
    steps, current = _ORIGINAL_MOTIF_ROWS(state_conn)
    if not ctx["motif_rows"]:
        return steps, current
    if not direct_context and ctx["motif_cache"] is not None:
        return ctx["motif_cache"]
    keyed = {(str(row.get("motif_signature") or ""), int(row.get("last_seen_global_step") or 0)): row for row in list(ctx["motif_rows"]) + list(current)}
    rows = sorted(keyed.values(), key=lambda row: (int(row.get("last_seen_global_step") or 0), str(row.get("motif_signature") or "")))
    result = ([int(row["last_seen_global_step"]) for row in rows], rows)
    if not direct_context:
        ctx["motif_cache"] = result
    return result


def install_concept_validation_history() -> None:
    global _INSTALLED, _ORIGINAL_VALIDATE, _ORIGINAL_DIAGNOSTICS, _ORIGINAL_MOTIF_ROWS
    if _INSTALLED:
        return
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite
    _ORIGINAL_VALIDATE = substrate.validate_incremental_promotions_only
    _ORIGINAL_DIAGNOSTICS = substrate._build_functional_explanation_diagnostics
    _ORIGINAL_MOTIF_ROWS = fast._motif_rows
    substrate.validate_incremental_promotions_only = _validate_with_history
    suite.validate_incremental_promotions_only = _validate_with_history
    substrate._build_functional_explanation_diagnostics = _diagnostics
    fast._motif_rows = _motif_rows
    _INSTALLED = True
