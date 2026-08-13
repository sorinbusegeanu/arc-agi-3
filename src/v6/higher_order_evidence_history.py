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
    cols = ", ".join(f'"{name}"' for name in common)
    before = conn.total_changes
    verb = "INSERT OR IGNORE" if immutable else "INSERT OR REPLACE"
    conn.execute(f'{verb} INTO "{target}" ({cols}) SELECT {cols} FROM "{source}"')
    return conn.total_changes - before


def _restore_table(
    conn: sqlite3.Connection,
    source: str,
    target: str,
    keys: tuple[str, ...],
    *,
    replace: bool = False,
) -> int:
    if not _exists(conn, source) or not _exists(conn, target):
        return 0
    source_columns = _columns(conn, source)
    target_columns = set(_columns(conn, target))
    common = [name for name in source_columns if name in target_columns]
    cols = ", ".join(f'"{name}"' for name in common)
    before = conn.total_changes
    if replace:
        conn.execute(f'INSERT OR REPLACE INTO "{source}" ({cols}) SELECT {cols} FROM "{target}"')
    else:
        predicate = " AND ".join(f'current."{key}" IS history."{key}"' for key in keys)
        conn.execute(
            f'INSERT INTO "{source}" ({cols}) SELECT {cols} FROM "{target}" AS history '
            f'WHERE NOT EXISTS (SELECT 1 FROM "{source}" AS current WHERE {predicate})'
        )
    return conn.total_changes - before


def _archive_group(memory_dir: Path, tables: dict[str, tuple[str, tuple[str, ...]]]) -> None:
    path = _db(memory_dir)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        for source, (target, keys) in tables.items():
            _archive_table(conn, source, target, keys)
        conn.commit()


def _restore_group(
    memory_dir: Path,
    tables: dict[str, tuple[str, tuple[str, ...]]],
    *,
    replace_entities: set[str] | None = None,
) -> None:
    path = _db(memory_dir)
    if not path.exists():
        return
    replace_entities = replace_entities or set()
    with sqlite3.connect(path) as conn:
        for source, (target, keys) in tables.items():
            _restore_table(conn, source, target, keys, replace=source in replace_entities)
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


def _restore_validation_tables(memory_dir: Path) -> None:
    path = _db(memory_dir)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        for source, target, keys in (
            ("role_transfer_attempts", validation_history.TRANSFER_HISTORY, ("attempt_id",)),
            ("future_option_events", validation_history.FUTURE_HISTORY, ("event_id",)),
        ):
            if _exists(conn, source) and _exists(conn, target):
                _restore_table(conn, source, target, keys)
        if _exists(conn, "future_option_motifs") and _exists(conn, validation_history.MOTIF_HISTORY):
            cols = _columns(conn, "future_option_motifs")
            history_cols = set(_columns(conn, validation_history.MOTIF_HISTORY))
            common = [name for name in cols if name in history_cols]
            col_sql = ", ".join(f'"{name}"' for name in common)
            conn.execute(
                f'INSERT OR REPLACE INTO future_option_motifs ({col_sql}) '
                f'SELECT {col_sql} FROM ('
                f'SELECT {col_sql}, ROW_NUMBER() OVER ('
                f'PARTITION BY motif_signature ORDER BY COALESCE(last_seen_global_step,-1) DESC, rowid DESC'
                f') AS rank FROM "{validation_history.MOTIF_HISTORY}") WHERE rank=1'
            )
        conn.commit()


def _derive_role_candidates(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, ROLE_TABLES)
    _archive_group(memory_dir, CONCEPT_TABLES)
    _archive_group(memory_dir, WORLD_TABLES)
    _archive_group(memory_dir, {"future_option_transfer_links": FUTURE_LINK_TABLES["future_option_transfer_links"]})
    result = _ORIGINAL_ROLE_CANDIDATES(*args, **kwargs)
    _archive_group(memory_dir, ROLE_TABLES)
    _restore_group(memory_dir, ROLE_TABLES, replace_entities={"role_neighborhood_signatures", "role_candidates"})
    if isinstance(result, dict):
        result = dict(result)
        result["cumulative_role_candidate_count"] = _source_count(memory_dir, "role_candidates")
        result["history_archive_boundary"] = "before_role_candidate_clear"
    return result


def _derive_role_transfers(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    historical = _history_count(memory_dir, validation_history.TRANSFER_HISTORY)
    budget = int(kwargs.get("max_transfer_attempts", 0) or 0)
    updated = dict(kwargs)
    if budget > 0:
        updated["max_transfer_attempts"] = historical + budget
    result = _ORIGINAL_ROLE_TRANSFERS(*args, **updated)
    _archive_validation_tables(memory_dir)
    _restore_validation_tables(memory_dir)
    cumulative = _source_count(memory_dir, "role_transfer_attempts")
    extra = {
        "configured_transfer_attempt_budget": budget,
        "historical_transfer_attempt_count_before": historical,
        "new_transfer_attempt_count_this_derivation": max(0, cumulative - historical),
        "cumulative_transfer_attempt_count": cumulative,
        "transfer_attempt_generation_window": int(updated.get("max_transfer_attempts", 0) or 0),
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
    _restore_group(memory_dir, CONCEPT_TABLES, replace_entities={"concept_candidates"})
    if isinstance(result, dict):
        result = dict(result)
        result["cumulative_concept_candidate_count"] = _source_count(memory_dir, "concept_candidates")
    return result


def _derive_world_models(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_group(memory_dir, WORLD_TABLES)
    result = _ORIGINAL_WORLD_MODELS(*args, **kwargs)
    _archive_group(memory_dir, WORLD_TABLES)
    _restore_group(memory_dir, WORLD_TABLES, replace_entities={"world_model_components"})
    if isinstance(result, dict):
        result = dict(result)
        result["cumulative_world_model_component_count"] = _source_count(memory_dir, "world_model_components")
    return result


def _derive_future_options(*args: Any, **kwargs: Any):
    memory_dir = _memory_dir(args, kwargs)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, FUTURE_LINK_TABLES)
    historical_events = _history_count(memory_dir, validation_history.FUTURE_HISTORY)
    historical_motifs = 0
    path = _db(memory_dir)
    if path.exists():
        with sqlite3.connect(path) as conn:
            if _exists(conn, validation_history.MOTIF_HISTORY):
                historical_motifs = int(
                    conn.execute(
                        f'SELECT COUNT(DISTINCT motif_signature) FROM "{validation_history.MOTIF_HISTORY}"'
                    ).fetchone()[0]
                )
    event_budget = int(kwargs.get("max_events", 0) or 0)
    motif_budget = int(kwargs.get("max_motifs", 0) or 0)
    updated = dict(kwargs)
    if event_budget > 0:
        updated["max_events"] = historical_events + event_budget
    if motif_budget > 0:
        updated["max_motifs"] = historical_motifs + motif_budget
    result = _ORIGINAL_FUTURE_OPTIONS(*args, **updated)
    _archive_validation_tables(memory_dir)
    _archive_group(memory_dir, FUTURE_LINK_TABLES)
    _restore_validation_tables(memory_dir)
    _restore_group(memory_dir, FUTURE_LINK_TABLES)
    cumulative_events = _source_count(memory_dir, "future_option_events")
    cumulative_motifs = _source_count(memory_dir, "future_option_motifs")
    extra = {
        "configured_future_option_event_budget": event_budget,
        "configured_future_option_motif_budget": motif_budget,
        "historical_future_option_event_count_before": historical_events,
        "historical_future_option_motif_count_before": historical_motifs,
        "new_future_option_event_count_this_derivation": max(0, cumulative_events - historical_events),
        "new_future_option_motif_count_this_derivation": max(0, cumulative_motifs - historical_motifs),
        "cumulative_future_option_event_count": cumulative_events,
        "cumulative_future_option_motif_count": cumulative_motifs,
        "future_option_event_generation_window": int(updated.get("max_events", 0) or 0),
        "future_option_motif_generation_window": int(updated.get("max_motifs", 0) or 0),
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
