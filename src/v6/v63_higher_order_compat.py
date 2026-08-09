from __future__ import annotations

import sqlite3
from typing import Any


_INSTALLED = False
_ORIGINAL_BUILD_RELATIONAL: Any = None


def install_v63_higher_order_compat() -> None:
    """Preserve legacy single-concept rows as non-validating diagnostics.

    v6.3 canonical M5 remains relational: H08 eligibility still requires at
    least two concept links.  Legacy rows are retained only so historical
    prediction-event accounting and old report readers remain reproducible.
    """
    global _INSTALLED, _ORIGINAL_BUILD_RELATIONAL
    if _INSTALLED:
        return
    from v6 import v63_higher_order_semantics as semantics

    _ORIGINAL_BUILD_RELATIONAL = semantics._build_relational_world_models
    semantics._build_relational_world_models = _build_relational_with_legacy_diagnostics
    _INSTALLED = True


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    ]


def _snapshot_rows(
    conn: sqlite3.Connection,
    table: str,
    *,
    component_ids: set[str] | None = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    columns = _table_columns(conn, table)
    if not columns:
        return [], []
    quoted = ", ".join(f'"{column}"' for column in columns)
    if component_ids is None:
        rows = conn.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
    elif not component_ids:
        rows = []
    else:
        key = "component_signature"
        placeholders = ",".join("?" for _ in component_ids)
        rows = conn.execute(
            f'SELECT {quoted} FROM "{table}" WHERE "{key}" IN ({placeholders})',
            tuple(sorted(component_ids)),
        ).fetchall()
    return columns, [tuple(row) for row in rows]


def _restore_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> None:
    if not columns or not rows:
        return
    quoted = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})',
        rows,
    )


def _build_relational_with_legacy_diagnostics(
    state_conn: sqlite3.Connection,
    *,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    legacy_ids = {
        str(row[0])
        for row in state_conn.execute(
            "SELECT component_signature FROM world_model_components "
            "WHERE COALESCE(linked_concept_count, 0) < 2"
        ).fetchall()
    }
    component_snapshot = _snapshot_rows(
        state_conn, "world_model_components", component_ids=legacy_ids
    )
    link_snapshot = _snapshot_rows(
        state_conn, "world_model_links", component_ids=legacy_ids
    )
    family_snapshot = _snapshot_rows(
        state_conn, "world_model_family_links", component_ids=legacy_ids
    )
    state_snapshot = _snapshot_rows(
        state_conn, "world_model_component_state", component_ids=legacy_ids
    )

    summary = _ORIGINAL_BUILD_RELATIONAL(
        state_conn,
        max_world_model_family_links=max_world_model_family_links,
    )

    _restore_rows(state_conn, "world_model_components", *component_snapshot)
    _restore_rows(state_conn, "world_model_links", *link_snapshot)
    _restore_rows(state_conn, "world_model_family_links", *family_snapshot)
    _restore_rows(state_conn, "world_model_component_state", *state_snapshot)

    if legacy_ids:
        placeholders = ",".join("?" for _ in legacy_ids)
        params = tuple(sorted(legacy_ids))
        state_conn.execute(
            f"""
            UPDATE world_model_components
            SET component_type='legacy_single_concept_diagnostic',
                candidate_only=1,
                is_coherent=0
            WHERE component_signature IN ({placeholders})
            """,
            params,
        )
        state_conn.execute(
            f"""
            UPDATE world_model_component_state
            SET currently_coherent=0,
                validation_status='legacy_single_concept_noncanonical'
            WHERE component_signature IN ({placeholders})
            """,
            params,
        )

    relational_count = int(
        state_conn.execute(
            "SELECT COUNT(*) FROM world_model_components "
            "WHERE COALESCE(linked_concept_count,0) >= 2"
        ).fetchone()[0]
    )
    legacy_count = int(
        state_conn.execute(
            "SELECT COUNT(*) FROM world_model_components "
            "WHERE COALESCE(linked_concept_count,0) < 2"
        ).fetchone()[0]
    )
    total_count = relational_count + legacy_count
    coherent_relational = int(
        state_conn.execute(
            "SELECT COUNT(*) FROM world_model_components "
            "WHERE COALESCE(linked_concept_count,0) >= 2 "
            "AND COALESCE(is_coherent,0)=1"
        ).fetchone()[0]
    )
    state_conn.commit()

    result = dict(summary)
    result.update(
        {
            "world_model_component_count": total_count,
            "relational_world_model_component_count": relational_count,
            "legacy_single_concept_diagnostic_count": legacy_count,
            "coherent_world_model_component_count": coherent_relational,
            "candidate_only_world_model_component_count": total_count - coherent_relational,
            "world_model_semantics_version": "v63_relational_multiconcept_v1",
        }
    )
    return result
