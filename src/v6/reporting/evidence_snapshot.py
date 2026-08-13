from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_SQLITE_NAMES = {"current_state.sqlite", "graph.sqlite", "replay_queue.sqlite", "direct_streaming_fold_manifest.sqlite"}
_COPY_NAMES = {"memory_summary.json", "direct_streaming_fold_manifest.json", "v61_schema_migration.json", "v621_schema_migration.json", "v63_schema_migration.json"}


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=60.0) as source_conn:
        with sqlite3.connect(target, timeout=60.0) as target_conn:
            source_conn.backup(target_conn)


def _copy_evidence_tree(source: Path, target: Path) -> None:
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


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _preserve_current_table(conn: sqlite3.Connection, table: str) -> None:
    if not _exists(conn, table):
        return
    target = f"report_current_{table}"
    if _exists(conn, target):
        conn.execute(f'DROP TABLE "{target}"')
    conn.execute(f'CREATE TABLE "{target}" AS SELECT * FROM "{table}"')


def _replace_from_history(conn: sqlite3.Connection, live: str, history: str, *, latest_identity: str | None = None, latest_order: str | None = None) -> int:
    if not _exists(conn, live) or not _exists(conn, history):
        return 0
    if int(conn.execute(f'SELECT COUNT(*) FROM "{history}"').fetchone()[0]) <= 0:
        return 0
    _preserve_current_table(conn, live)
    live_columns = _columns(conn, live)
    history_columns = set(_columns(conn, history))
    common = [column for column in live_columns if column in history_columns]
    if not common:
        return 0
    cols = ", ".join(f'"{column}"' for column in common)
    conn.execute(f'DELETE FROM "{live}"')
    if latest_identity and latest_identity in common:
        order_expr = latest_order or "rowid"
        select_sql = (
            f'SELECT {cols} FROM ('
            f'SELECT {cols}, ROW_NUMBER() OVER ('
            f'PARTITION BY "{latest_identity}" ORDER BY {order_expr} DESC, rowid DESC'
            f') AS report_rank FROM "{history}") WHERE report_rank=1'
        )
    else:
        select_sql = f'SELECT {cols} FROM "{history}"'
    conn.execute(f'INSERT OR REPLACE INTO "{live}" ({cols}) {select_sql}')
    return int(conn.execute(f'SELECT COUNT(*) FROM "{live}"').fetchone()[0])


def _verified_concepts(conn: sqlite3.Connection) -> set[str]:
    verified: set[str] = set()
    if _exists(conn, "concept_promotion_validation_diagnostics"):
        for signature, payload_json in conn.execute("SELECT concept_signature, payload_json FROM concept_promotion_validation_diagnostics").fetchall():
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and bool(payload.get("promoted")):
                verified.add(str(signature))
    if _exists(conn, "concept_promotion_state"):
        columns = set(_columns(conn, "concept_promotion_state"))
        if {"concept_signature", "currently_promoted", "validation_status"} <= columns:
            for signature, promoted, status in conn.execute("SELECT concept_signature, currently_promoted, validation_status FROM concept_promotion_state").fetchall():
                if int(promoted or 0) == 1 and str(status or "").strip().lower() in {"passed", "validated"}:
                    verified.add(str(signature))
    return verified


def _row_value(row: sqlite3.Row, key: str) -> Any:
    return row[key] if key in row.keys() else None


def _refresh_h11_chain_status(conn: sqlite3.Connection) -> None:
    if not _exists(conn, "future_option_transfer_links"):
        return
    link_columns = set(_columns(conn, "future_option_transfer_links"))
    required = {"motif_signature", "role_signature", "concept_signature", "motif_provenance_status", "transfer_provenance_status", "concept_validation_status"}
    if not required <= link_columns:
        return
    motif_status: dict[str, str] = {}
    if _exists(conn, "future_option_motifs") and {"motif_signature", "provenance_status"} <= set(_columns(conn, "future_option_motifs")):
        motif_status = {str(signature): str(status or "missing") for signature, status in conn.execute("SELECT motif_signature, provenance_status FROM future_option_motifs").fetchall()}
    verified_concepts = _verified_concepts(conn)
    transfer_keys: set[tuple[str, str, str, str, str]] = set()
    if _exists(conn, "role_transfer_attempts"):
        transfer_columns = set(_columns(conn, "role_transfer_attempts"))
        role_column = "source_role_signature" if "source_role_signature" in transfer_columns else "role_signature" if "role_signature" in transfer_columns else None
        scope_columns = {"source_game_key", "target_game_key", "source_context_key", "target_context_key"}
        if role_column and scope_columns <= transfer_columns:
            provenance_filter = ""
            if {"provenance_mode", "provenance_status"} <= transfer_columns:
                provenance_filter = " WHERE provenance_mode='single_source' AND provenance_status='verified'"
            for row in conn.execute(f'SELECT "{role_column}", source_game_key, target_game_key, source_context_key, target_context_key FROM role_transfer_attempts{provenance_filter}').fetchall():
                transfer_keys.add(tuple(str(value or "") for value in row))
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT rowid, * FROM future_option_transfer_links").fetchall():
        updates: dict[str, Any] = {}
        motif = str(_row_value(row, "motif_signature") or "")
        concept = str(_row_value(row, "concept_signature") or "")
        source_role = str(_row_value(row, "source_role_signature") or _row_value(row, "role_signature") or "")
        scope_key = (source_role, str(_row_value(row, "source_game_key") or ""), str(_row_value(row, "target_game_key") or ""), str(_row_value(row, "source_context_key") or ""), str(_row_value(row, "target_context_key") or ""))
        if motif_status.get(motif) == "verified":
            updates["motif_provenance_status"] = "verified"
        if concept in verified_concepts:
            updates["concept_validation_status"] = "verified"
            if "promoted_concept_count" in link_columns:
                updates["promoted_concept_count"] = max(1, int(_row_value(row, "promoted_concept_count") or 0))
        if scope_key in transfer_keys:
            updates["transfer_provenance_status"] = "verified"
        if updates:
            assignments = ", ".join(f'"{key}"=?' for key in updates)
            conn.execute(f'UPDATE future_option_transfer_links SET {assignments} WHERE rowid=?', (*updates.values(), int(row["rowid"])))


def _materialize_cumulative_reporting_projection(database: Path) -> dict[str, int]:
    if not database.exists():
        return {}
    projected: dict[str, int] = {}
    with sqlite3.connect(database, timeout=60.0) as conn:
        mappings = (
            ("role_transfer_attempts", "concept_validation_role_transfer_history", None, None),
            ("future_option_events", "concept_validation_future_option_event_history", None, None),
            ("future_option_motifs", "concept_validation_future_option_motif_history", "motif_signature", "COALESCE(last_seen_global_step,-1)"),
            ("role_neighborhood_signatures", "higher_order_role_neighborhood_history", None, None),
            ("role_candidates", "higher_order_role_candidate_history", None, None),
            ("role_links", "higher_order_role_link_history", None, None),
            ("concept_candidates", "higher_order_concept_candidate_history", None, None),
            ("concept_links", "higher_order_concept_link_history", None, None),
            ("world_model_components", "higher_order_world_model_component_history", None, None),
            ("world_model_links", "higher_order_world_model_link_history", None, None),
            ("world_model_family_links", "higher_order_world_model_family_link_history", None, None),
            ("future_option_links", "higher_order_future_option_link_history", None, None),
            ("future_option_attention_links", "higher_order_future_option_attention_history", None, None),
            ("future_option_transfer_links", "higher_order_future_option_transfer_history", None, None),
            ("future_option_motif_observations", "higher_order_future_option_observation_history", None, None),
        )
        for live, history, identity, order in mappings:
            count = _replace_from_history(conn, live, history, latest_identity=identity, latest_order=order)
            if count:
                projected[live] = count
        _refresh_h11_chain_status(conn)
        if _exists(conn, "memory_summary"):
            conn.execute("INSERT INTO memory_summary(key,value_json) VALUES('reporting_cumulative_projection',?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (json.dumps(projected, sort_keys=True),))
        conn.commit()
    return projected


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
        _materialize_cumulative_reporting_projection(target / "current_state.sqlite")
        _make_read_only(target)
        yield target


def memory_fingerprint(memory_dir: Path | None) -> str | None:
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
