from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_ROLE_DERIVE: Any = None
_ORIGINAL_FUTURE_DERIVE: Any = None
_ORIGINAL_DIAGNOSTICS: Any = None
_ORIGINAL_MOTIF_ROWS: Any = None

TRANSFER_HISTORY = "concept_validation_role_transfer_history"
FUTURE_HISTORY = "concept_validation_future_option_event_history"
MOTIF_HISTORY = "concept_validation_future_option_motif_history"


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _archive(conn: sqlite3.Connection, source: str, target: str, key_sql: str) -> int:
    if not _exists(conn, source):
        return 0
    if not _exists(conn, target):
        conn.execute(f"CREATE TABLE {target} AS SELECT * FROM {source} WHERE 0")
        conn.execute(f"CREATE UNIQUE INDEX idx_{target}_identity ON {target} ({key_sql})")
    source_cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({source})").fetchall()]
    target_cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({target})").fetchall()}
    common = [name for name in source_cols if name in target_cols]
    cols = ", ".join(common)
    before = conn.total_changes
    conn.execute(f"INSERT OR IGNORE INTO {target} ({cols}) SELECT {cols} FROM {source}")
    return conn.total_changes - before


def archive_validation_evidence(memory_dir: Path) -> dict[str, int]:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return {"transfer": 0, "future": 0, "motif": 0}
    with sqlite3.connect(db) as conn:
        result = {
            "transfer": _archive(conn, "role_transfer_attempts", TRANSFER_HISTORY, "attempt_id"),
            "future": _archive(conn, "future_option_events", FUTURE_HISTORY, "event_id"),
            "motif": _archive(
                conn,
                "future_option_motifs",
                MOTIF_HISTORY,
                "motif_signature, last_seen_global_step",
            ),
        }
        conn.commit()
        return result


def _derive_role(*args: Any, **kwargs: Any):
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    archive_validation_evidence(memory_dir)
    result = _ORIGINAL_ROLE_DERIVE(*args, **kwargs)
    archive_validation_evidence(memory_dir)
    return result


def _derive_future(*args: Any, **kwargs: Any):
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    archive_validation_evidence(memory_dir)
    result = _ORIGINAL_FUTURE_DERIVE(*args, **kwargs)
    archive_validation_evidence(memory_dir)
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


def _diagnostics(*args: Any, **kwargs: Any):
    state_conn = kwargs.get("state_conn")
    if state_conn is None:
        return _ORIGINAL_DIAGNOSTICS(*args, **kwargs)
    transfer_rows = list(kwargs.get("transfer_rows") or [])
    future_rows = list(kwargs.get("future_rows") or [])
    if _exists(state_conn, TRANSFER_HISTORY):
        transfer_rows = _dedupe(
            list(state_conn.execute(f"SELECT * FROM {TRANSFER_HISTORY}").fetchall()) + transfer_rows,
            "attempt_id",
        )
    if _exists(state_conn, FUTURE_HISTORY):
        future_rows = _dedupe(
            list(state_conn.execute(f"SELECT * FROM {FUTURE_HISTORY}").fetchall()) + future_rows,
            "event_id",
        )
    from v6 import higher_order_substrate as substrate
    updated = dict(kwargs)
    updated["transfer_rows"] = transfer_rows
    updated["future_rows"] = future_rows
    updated["transfer_history"] = substrate._build_transfer_history_index(transfer_rows)
    result = _ORIGINAL_DIAGNOSTICS(*args, **updated)
    if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], dict):
        events, diagnostics, state = result
        diagnostics = dict(diagnostics)
        diagnostics["validation_history_applied"] = True
        diagnostics["validation_transfer_history_row_count"] = len(transfer_rows)
        diagnostics["validation_future_option_history_row_count"] = len(future_rows)
        return events, diagnostics, state
    return result


def _motif_rows(state_conn: sqlite3.Connection):
    steps, current = _ORIGINAL_MOTIF_ROWS(state_conn)
    if not _exists(state_conn, MOTIF_HISTORY):
        return steps, current
    historical = [dict(row) for row in state_conn.execute(
        f"SELECT motif_signature, source_role_ids_json, motif_stability_score, is_emergent, last_seen_global_step "
        f"FROM {MOTIF_HISTORY} WHERE last_seen_global_step IS NOT NULL"
    ).fetchall()]
    keyed = {
        (str(row.get("motif_signature") or ""), int(row.get("last_seen_global_step") or 0)): row
        for row in historical + list(current)
    }
    rows = sorted(
        keyed.values(),
        key=lambda row: (int(row.get("last_seen_global_step") or 0), str(row.get("motif_signature") or "")),
    )
    return [int(row["last_seen_global_step"]) for row in rows], rows


def install_concept_validation_history() -> None:
    global _INSTALLED, _ORIGINAL_ROLE_DERIVE, _ORIGINAL_FUTURE_DERIVE
    global _ORIGINAL_DIAGNOSTICS, _ORIGINAL_MOTIF_ROWS
    if _INSTALLED:
        return
    from v6 import concept_validation_fastpath as fast
    from v6 import future_options as future
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite
    _ORIGINAL_ROLE_DERIVE = substrate.derive_role_transfer_attempts_only
    _ORIGINAL_FUTURE_DERIVE = future.derive_future_option_memory
    _ORIGINAL_DIAGNOSTICS = substrate._build_functional_explanation_diagnostics
    _ORIGINAL_MOTIF_ROWS = fast._motif_rows
    substrate.derive_role_transfer_attempts_only = _derive_role
    suite.derive_role_transfer_attempts_only = _derive_role
    future.derive_future_option_memory = _derive_future
    suite.derive_future_option_memory = _derive_future
    substrate._build_functional_explanation_diagnostics = _diagnostics
    fast._motif_rows = _motif_rows
    _INSTALLED = True
