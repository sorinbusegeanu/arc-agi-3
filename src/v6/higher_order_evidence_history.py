from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6 import concept_validation_history as validation_history

_INSTALLED = False
_ORIGINAL_ROLE_CANDIDATES: Any = None
_ORIGINAL_ROLE_TRANSFERS: Any = None
_ORIGINAL_CONCEPTS: Any = None
_ORIGINAL_WORLD_MODELS: Any = None
_ORIGINAL_FUTURE_OPTIONS: Any = None

ROLE_TABLES = {
    "role_neighborhood_signatures": ("higher_order_role_neighborhood_history", ("carrier_signature",)),
    "role_candidates": ("higher_order_role_candidate_history", ("role_signature",)),
    "role_links": ("higher_order_role_link_history", ("role_signature", "linked_type", "linked_key")),
}
CONCEPT_TABLES = {
    "concept_candidates": ("higher_order_concept_candidate_history", ("concept_signature",)),
    "concept_links": ("higher_order_concept_link_history", ("concept_signature", "linked_type", "linked_key")),
}
WORLD_TABLES = {
    "world_model_components": ("higher_order_world_model_component_history", ("component_signature",)),
    "world_model_links": ("higher_order_world_model_link_history", ("component_signature", "linked_type", "linked_key")),
    "world_model_family_links": ("higher_order_world_model_family_link_history", ("component_signature", "family_signature")),
}
FUTURE_LINK_TABLES = {
    "future_option_links": ("higher_order_future_option_link_history", ("motif_signature", "linked_type", "linked_key")),
    "future_option_attention_links": ("higher_order_future_option_attention_history", ("event_id",)),
    "future_option_transfer_links": (
        "higher_order_future_option_transfer_history",
        (
            "motif_signature",
            "role_signature",
            "concept_signature",
            "source_game_key",
            "target_game_key",
            "source_context_key",
            "target_context_key",
        ),
    ),
    "future_option_motif_observations": (
        "higher_order_future_option_observation_history",
        ("motif_signature", "event_id"),
    ),
}
WORLD_VALIDATION_HISTORY = "higher_order_world_model_validation_history"


def _db(memory_dir: Path) -> Path:
    return Path(memory_dir) / "current_state.sqlite"


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _ensure_history(conn: sqlite3.Connection, source: str, target: str, keys: tuple[str, ...]) -> None:
    if not _exists(conn, source):
        return
    if not _exists(conn, target):
        conn.execute(f'CREATE TABLE "{target}" AS SELECT * FROM "{source}" WHERE 0')
    source_columns = conn.execute(f'PRAGMA table_info("{source}")').fetchall()
    target_columns = set(_columns(conn, target))
    for row in source_columns:
        name = str(row[1])
        if name in target_columns:
            continue
        declared = str(row[2] or "").strip()
        suffix = f" {declared}" if declared else ""
        conn.execute(f'ALTER TABLE "{target}" ADD COLUMN "{name}"{suffix}')
    key_sql = ", ".join(f'"{key}"' for key in keys)
    index_name = f"idx_{target[:40]}_identity"
    try:
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{target}" ({key_sql})')
    except sqlite3.IntegrityError:
        conn.execute(
            f'DELETE FROM "{target}" WHERE rowid NOT IN '
            f'(SELECT MAX(rowid) FROM "{target}" GROUP BY {key_sql})'
        )
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{target}" ({key_sql})')


def _archive_table(
    conn: sqlite3.Connection,
    source: str,
    target: str,
    keys: tuple[str, ...],
    *,
    immutable: bool = False,
) -> int:
    if not _exists(conn, source):
        return 0
    _ensure_history(conn, source, target, keys)
    source_columns = _columns(conn, source)
    target_columns = set(_columns(conn, target))
    common = [name for name in source_columns if name in target_columns]
    if not common:
        return 0
    cols = ", ".join(f'"{name}"' for name in common)
    before = conn.total_changes
    verb = "INSERT OR IGNORE" if immutable else "INSERT OR REPLACE"
    conn.execute(f'{verb} INTO "{target}" ({cols}) SELECT {cols} FROM "{source}"')
    return conn.total_changes - before


def _archive_group(memory_dir: Path, tables: dict[str, tuple[str, tuple[str, ...]]]) -> None:
    path = _db(memory_dir)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        for source, (target, keys) in tables.items():
            _archive_table(conn, source, target, keys)
        conn.commit()


def _history_count(memory_dir: Path, table: str) -> int:
    path = _db(memory_dir)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        if not _exists(conn, table):
            return 0
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _source_count(memory_dir: Path, table: str) -> int:
    path = _db(memory_dir)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        if not _exists(conn, table):
            return 0
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _memory_dir(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path:
    return Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])


def _update_summary(memory_dir: Path, key: str, updates: dict[str, Any]) -> None:
    path = _db(memory_dir)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        if not _exists(conn, "memory_summary"):
            return
        row = conn.execute("SELECT value_json FROM memory_summary WHERE key=?", (key,)).fetchone()
        payload: dict[str, Any] = {}
        if row and row[0]:
            try:
                value = json.loads(str(row[0]))
                if isinstance(value, dict):
                    payload.update(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        payload.update(updates)
        conn.execute(
            "INSERT INTO memory_summary(key,value_json) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, json.dumps(payload, sort_keys=True)),
        )
        conn.commit()


def _archive_validation_tables(memory_dir: Path) -> None:
    validation_history.archive_validation_evidence(memory_dir)


def archive_world_model_validation_evidence(
    memory_dir: Path,
    diagnostic_epoch_id: str | int | None,
) -> int:
    """Persist post-validation component observations without changing current state."""
    path = _db(memory_dir)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        if not _exists(conn, "world_model_components"):
            return 0
        if not _exists(conn, WORLD_VALIDATION_HISTORY):
            conn.execute(
                f'CREATE TABLE "{WORLD_VALIDATION_HISTORY}" AS '
                'SELECT * FROM world_model_components WHERE 0'
            )
            conn.execute(f'ALTER TABLE "{WORLD_VALIDATION_HISTORY}" ADD COLUMN diagnostic_epoch_id TEXT')
            conn.execute(f'ALTER TABLE "{WORLD_VALIDATION_HISTORY}" ADD COLUMN state_historically_coherent INTEGER')
            conn.execute(f'ALTER TABLE "{WORLD_VALIDATION_HISTORY}" ADD COLUMN state_currently_coherent INTEGER')
            conn.execute(f'ALTER TABLE "{WORLD_VALIDATION_HISTORY}" ADD COLUMN state_validation_status TEXT')
        else:
            current_cols = conn.execute("PRAGMA table_info(world_model_components)").fetchall()
            history_cols = set(_columns(conn, WORLD_VALIDATION_HISTORY))
            for row in current_cols:
                name = str(row[1])
                if name in history_cols:
                    continue
                declared = str(row[2] or "").strip()
                suffix = f" {declared}" if declared else ""
                conn.execute(f'ALTER TABLE "{WORLD_VALIDATION_HISTORY}" ADD COLUMN "{name}"{suffix}')
        conn.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS idx_world_model_validation_history_identity '
            f'ON "{WORLD_VALIDATION_HISTORY}" (component_signature, diagnostic_epoch_id)'
        )
        component_cols = _columns(conn, "world_model_components")
        history_cols = set(_columns(conn, WORLD_VALIDATION_HISTORY))
        common = [name for name in component_cols if name in history_cols]
        col_sql = ", ".join(f'"{name}"' for name in common)
        max_step = conn.execute(
            "SELECT MAX(COALESCE(last_seen_global_step, first_seen_global_step, -1)) FROM world_model_components"
        ).fetchone()[0]
        epoch_key = str(diagnostic_epoch_id) if diagnostic_epoch_id is not None else f"step:{int(max_step or -1)}"
        state_exists = _exists(conn, "world_model_component_state")
        if state_exists:
            select_state = (
                "COALESCE(s.historically_coherent,0), COALESCE(s.currently_coherent,0), "
                "COALESCE(s.validation_status,'')"
            )
            join_state = (
                " LEFT JOIN world_model_component_state AS s "
                "ON s.component_signature=c.component_signature"
            )
        else:
            select_state = "NULL, NULL, NULL"
            join_state = ""
        before = conn.total_changes
        component_select = ", ".join(f'c."{name}"' for name in common)
        conn.execute(
            f'INSERT OR REPLACE INTO "{WORLD_VALIDATION_HISTORY}" '
            f'({col_sql}, diagnostic_epoch_id, state_historically_coherent, '
            f'state_currently_coherent, state_validation_status) '
            f'SELECT {component_select}, ?, {select_state} '
            f'FROM world_model_components AS c{join_state}',
            (epoch_key,),
        )
        conn.commit()
        return conn.total_changes - before


def _derive_role_candidates(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, ROLE_TABLES)
    _archive_group(memory_dir, CONCEPT_TABLES)
    _archive_group(memory_dir, WORLD_TABLES)
    _archive_group(memory_dir, {"future_option_transfer_links": FUTURE_LINK_TABLES["future_option_transfer_links"]})
    result = _ORIGINAL_ROLE_CANDIDATES(*args, **kwargs)
    _archive_group(memory_dir, ROLE_TABLES)
    if isinstance(result, dict):
        result = dict(result)
        result["current_role_candidate_count"] = _source_count(memory_dir, "role_candidates")
        result["cumulative_role_candidate_count"] = _history_count(memory_dir, ROLE_TABLES["role_candidates"][0])
        result["history_archive_boundary"] = "before_role_candidate_clear"
        result["historical_rows_restored_into_current_state"] = False
    return result


def _derive_role_transfers(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    historical_before = _history_count(memory_dir, validation_history.TRANSFER_HISTORY)
    budget = int(kwargs.get("max_transfer_attempts", 0) or 0)
    result = _ORIGINAL_ROLE_TRANSFERS(*args, **kwargs)
    _archive_validation_tables(memory_dir)
    current = _source_count(memory_dir, "role_transfer_attempts")
    cumulative = _history_count(memory_dir, validation_history.TRANSFER_HISTORY)
    extra = {
        "configured_transfer_attempt_budget": budget,
        "historical_transfer_attempt_count_before": historical_before,
        "current_transfer_attempt_count": current,
        "new_transfer_attempt_count_this_derivation": max(0, cumulative - historical_before),
        "cumulative_transfer_attempt_count": cumulative,
        "transfer_attempt_generation_window": budget,
        "historical_rows_restored_into_current_state": False,
    }
    if isinstance(result, dict):
        result = {**result, **extra}
    _update_summary(memory_dir, "higher_order_transfer_summary", extra)
    return result


def _derive_concepts(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_group(memory_dir, CONCEPT_TABLES)
    _archive_group(memory_dir, WORLD_TABLES)
    result = _ORIGINAL_CONCEPTS(*args, **kwargs)
    _archive_group(memory_dir, CONCEPT_TABLES)
    if isinstance(result, dict):
        result = dict(result)
        result["current_concept_candidate_count"] = _source_count(memory_dir, "concept_candidates")
        result["cumulative_concept_candidate_count"] = _history_count(memory_dir, CONCEPT_TABLES["concept_candidates"][0])
        result["historical_rows_restored_into_current_state"] = False
    return result


def _derive_world_models(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_group(memory_dir, WORLD_TABLES)
    result = _ORIGINAL_WORLD_MODELS(*args, **kwargs)
    _archive_group(memory_dir, WORLD_TABLES)
    if isinstance(result, dict):
        result = dict(result)
        result["current_world_model_component_count"] = _source_count(memory_dir, "world_model_components")
        result["cumulative_world_model_component_count"] = _history_count(memory_dir, WORLD_TABLES["world_model_components"][0])
        result["historical_rows_restored_into_current_state"] = False
    return result


def _derive_future_options(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, FUTURE_LINK_TABLES)
    historical_events = _history_count(memory_dir, validation_history.FUTURE_HISTORY)
    historical_motifs = _history_count(memory_dir, validation_history.MOTIF_HISTORY)
    event_budget = int(kwargs.get("max_events", 0) or 0)
    motif_budget = int(kwargs.get("max_motifs", 0) or 0)
    result = _ORIGINAL_FUTURE_OPTIONS(*args, **kwargs)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, FUTURE_LINK_TABLES)
    current_events = _source_count(memory_dir, "future_option_events")
    current_motifs = _source_count(memory_dir, "future_option_motifs")
    cumulative_events = _history_count(memory_dir, validation_history.FUTURE_HISTORY)
    cumulative_motifs = _history_count(memory_dir, validation_history.MOTIF_HISTORY)
    extra = {
        "configured_future_option_event_budget": event_budget,
        "configured_future_option_motif_budget": motif_budget,
        "historical_future_option_event_count_before": historical_events,
        "historical_future_option_motif_observation_count_before": historical_motifs,
        "current_future_option_event_count": current_events,
        "current_future_option_motif_count": current_motifs,
        "new_future_option_event_count_this_derivation": max(0, cumulative_events - historical_events),
        "new_future_option_motif_observation_count_this_derivation": max(0, cumulative_motifs - historical_motifs),
        "cumulative_future_option_event_count": cumulative_events,
        "cumulative_future_option_motif_observation_count": cumulative_motifs,
        "future_option_event_generation_window": event_budget,
        "future_option_motif_generation_window": motif_budget,
        "historical_rows_restored_into_current_state": False,
    }
    if isinstance(result, dict):
        result = {**result, **extra}
    _update_summary(memory_dir, "future_option_derivation_summary", extra)
    return result


def install_higher_order_evidence_history() -> None:
    global _INSTALLED, _ORIGINAL_ROLE_CANDIDATES, _ORIGINAL_ROLE_TRANSFERS
    global _ORIGINAL_CONCEPTS, _ORIGINAL_WORLD_MODELS, _ORIGINAL_FUTURE_OPTIONS
    if _INSTALLED:
        return
    from v6 import future_options
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    _ORIGINAL_ROLE_CANDIDATES = substrate.derive_role_candidates_only
    _ORIGINAL_ROLE_TRANSFERS = substrate.derive_role_transfer_attempts_only
    _ORIGINAL_CONCEPTS = substrate.derive_concept_candidates_only
    _ORIGINAL_WORLD_MODELS = substrate.derive_world_model_components_only
    _ORIGINAL_FUTURE_OPTIONS = future_options.derive_future_option_memory

    substrate.derive_role_candidates_only = _derive_role_candidates
    suite.derive_role_candidates_only = _derive_role_candidates
    substrate.derive_role_transfer_attempts_only = _derive_role_transfers
    suite.derive_role_transfer_attempts_only = _derive_role_transfers
    substrate.derive_concept_candidates_only = _derive_concepts
    suite.derive_concept_candidates_only = _derive_concepts
    substrate.derive_world_model_components_only = _derive_world_models
    suite.derive_world_model_components_only = _derive_world_models
    future_options.derive_future_option_memory = _derive_future_options
    suite.derive_future_option_memory = _derive_future_options
    _INSTALLED = True
