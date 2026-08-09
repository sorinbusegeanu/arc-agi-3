from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_INSTALLED = False
_ORIGINAL_ENSURE_CURRENT_STATE_SCHEMA: Any = None
_ORIGINAL_FOLD_SINGLE_DB: Any = None
_ORIGINAL_MERGE_STATE_SET_BASED: Any = None
_ORIGINAL_MERGE_STATE_ROWS: Any = None
_ORIGINAL_FOLD_LIVE_SYSTEM: Any = None
_ORIGINAL_RESTORE_COMPACT: Any = None
_ORIGINAL_REPAIR_ROLE_EMERGENCE: Any = None
_ORIGINAL_H04_BASE: Any = None


def install_v63_emergence_timing_completion() -> None:
    """Install explicit threshold-crossing timestamps across the compact-memory path."""
    global _INSTALLED
    global _ORIGINAL_ENSURE_CURRENT_STATE_SCHEMA
    global _ORIGINAL_FOLD_SINGLE_DB
    global _ORIGINAL_MERGE_STATE_SET_BASED
    global _ORIGINAL_MERGE_STATE_ROWS
    global _ORIGINAL_FOLD_LIVE_SYSTEM
    global _ORIGINAL_RESTORE_COMPACT
    global _ORIGINAL_REPAIR_ROLE_EMERGENCE
    global _ORIGINAL_H04_BASE

    from v6.memory import compact_memory as compact
    from v6.memory import compact_memory_restore as restore
    from v6 import v63_higher_order_semantics as higher_semantics
    from v6 import hypothesis_h04_report as h04

    if not _INSTALLED:
        _ORIGINAL_ENSURE_CURRENT_STATE_SCHEMA = compact._ensure_current_state_schema
        _ORIGINAL_FOLD_SINGLE_DB = compact._fold_single_db
        _ORIGINAL_MERGE_STATE_SET_BASED = compact._merge_state_tables_set_based
        _ORIGINAL_MERGE_STATE_ROWS = compact._merge_state_tables
        _ORIGINAL_FOLD_LIVE_SYSTEM = compact.fold_live_system_into_compact_memory
        _ORIGINAL_RESTORE_COMPACT = restore.load_compact_memory_into_system
        _ORIGINAL_REPAIR_ROLE_EMERGENCE = higher_semantics._repair_role_emergence_steps
        _ORIGINAL_H04_BASE = h04._evaluate_h04_carrier_emergence_base
        _INSTALLED = True

    # Reapply on every migration entry point because earlier compatibility
    # installers can refresh bindings.
    compact._ensure_current_state_schema = _ensure_current_state_schema
    compact._fold_single_db = _fold_single_db
    compact._merge_state_tables_set_based = _merge_state_tables_set_based
    compact._merge_state_tables = _merge_state_tables
    compact.fold_live_system_into_compact_memory = _fold_live_system_into_compact_memory
    restore.load_compact_memory_into_system = _restore_compact_memory_into_system
    main_module = sys.modules.get("v6.main")
    if main_module is not None:
        setattr(main_module, "load_compact_memory_into_system", _restore_compact_memory_into_system)
    higher_semantics._repair_role_emergence_steps = _repair_role_emergence_steps
    h04._evaluate_h04_carrier_emergence_base = _evaluate_h04_base


def _ensure_emergence_column(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='carrier_candidates'"
    ).fetchone()
    if table is None:
        return
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(carrier_candidates)").fetchall()
    }
    if "first_emergent_global_step" not in columns:
        connection.execute(
            "ALTER TABLE carrier_candidates ADD COLUMN first_emergent_global_step INTEGER"
        )


def _ensure_current_state_schema(path: Path) -> bool:
    result = bool(_ORIGINAL_ENSURE_CURRENT_STATE_SCHEMA(path))
    with sqlite3.connect(path) as connection:
        _ensure_emergence_column(connection)
        connection.commit()
    return result


def _load_sidecar_timings(db_path: Path) -> dict[str, dict[str, int | None]]:
    sidecar = Path(db_path).with_name("carrier_candidates.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}
    timings: dict[str, dict[str, int | None]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        signature = str(item.get("carrier_signature") or item.get("carrier_id") or "")
        if not signature:
            continue
        first_seen = _optional_int(item.get("first_seen_global_step"))
        first_emergent = _optional_int(item.get("first_emergent_global_step"))
        if (
            str(item.get("status") or "") != "emergent_carrier"
            or str(item.get("carrier_source") or "") == "context_action_fallback"
        ):
            first_emergent = None
        current = timings.setdefault(
            signature,
            {"first_seen_global_step": None, "first_emergent_global_step": None},
        )
        current["first_seen_global_step"] = _minimum_optional(
            current["first_seen_global_step"], first_seen
        )
        current["first_emergent_global_step"] = _minimum_optional(
            current["first_emergent_global_step"], first_emergent
        )
    return timings


def _restore_sidecar_timings(
    state_conn: sqlite3.Connection,
    db_path: Path,
) -> None:
    _ensure_emergence_column(state_conn)
    for signature, timing in _load_sidecar_timings(db_path).items():
        first_seen = timing["first_seen_global_step"]
        first_emergent = timing["first_emergent_global_step"]
        state_conn.execute(
            """
            UPDATE carrier_candidates
            SET first_seen_global_step = CASE
                    WHEN ? IS NULL THEN first_seen_global_step
                    WHEN first_seen_global_step IS NULL THEN ?
                    ELSE MIN(first_seen_global_step, ?)
                END,
                first_emergent_global_step = CASE
                    WHEN ? IS NULL THEN first_emergent_global_step
                    WHEN first_emergent_global_step IS NULL THEN ?
                    ELSE MIN(first_emergent_global_step, ?)
                END
            WHERE carrier_signature = ?
            """,
            (
                first_seen, first_seen, first_seen,
                first_emergent, first_emergent, first_emergent,
                signature,
            ),
        )


def _fold_single_db(*args: Any, **kwargs: Any) -> None:
    state_conn = kwargs.get("state_conn")
    db_path = kwargs.get("db_path")
    if isinstance(state_conn, sqlite3.Connection):
        _ensure_emergence_column(state_conn)
    _ORIGINAL_FOLD_SINGLE_DB(*args, **kwargs)
    if isinstance(state_conn, sqlite3.Connection) and db_path is not None:
        # v63_temporal_semantics_completion historically overloaded first_seen
        # with the emergence timestamp. Restore observation time and persist the
        # threshold crossing in its own column instead.
        _restore_sidecar_timings(state_conn, Path(db_path))


def _merge_explicit_emergence_from_attached(
    state_conn: sqlite3.Connection,
    shard_path: Path,
) -> None:
    _ensure_emergence_column(state_conn)
    alias = "v63_emergence_shard"
    state_conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_path),))
    try:
        shard_columns = {
            str(row[1])
            for row in state_conn.execute(
                f"PRAGMA {alias}.table_info(carrier_candidates)"
            ).fetchall()
        }
        if "first_emergent_global_step" not in shard_columns:
            return
        state_conn.execute(
            f"""
            UPDATE carrier_candidates
            SET first_emergent_global_step = CASE
                WHEN first_emergent_global_step IS NULL THEN (
                    SELECT shard.first_emergent_global_step
                    FROM {alias}.carrier_candidates AS shard
                    WHERE shard.carrier_signature = carrier_candidates.carrier_signature
                )
                WHEN (
                    SELECT shard.first_emergent_global_step
                    FROM {alias}.carrier_candidates AS shard
                    WHERE shard.carrier_signature = carrier_candidates.carrier_signature
                ) IS NULL THEN first_emergent_global_step
                ELSE MIN(
                    first_emergent_global_step,
                    (
                        SELECT shard.first_emergent_global_step
                        FROM {alias}.carrier_candidates AS shard
                        WHERE shard.carrier_signature = carrier_candidates.carrier_signature
                    )
                )
            END
            WHERE EXISTS (
                SELECT 1
                FROM {alias}.carrier_candidates AS shard
                WHERE shard.carrier_signature = carrier_candidates.carrier_signature
                  AND shard.first_emergent_global_step IS NOT NULL
            )
            """
        )
    finally:
        state_conn.commit()
        state_conn.execute(f"DETACH DATABASE {alias}")


def _merge_state_tables_set_based(
    temp_state_path: Path,
    state_conn: sqlite3.Connection,
    fold_config: Any,
) -> None:
    _ensure_emergence_column(state_conn)
    _ORIGINAL_MERGE_STATE_SET_BASED(temp_state_path, state_conn, fold_config)
    _merge_explicit_emergence_from_attached(state_conn, Path(temp_state_path))


def _merge_state_tables(
    temp_state: sqlite3.Connection,
    state_conn: sqlite3.Connection,
    fold_config: Any,
) -> None:
    _ensure_emergence_column(state_conn)
    _ORIGINAL_MERGE_STATE_ROWS(temp_state, state_conn, fold_config)
    temp_columns = {
        str(row[1])
        for row in temp_state.execute("PRAGMA table_info(carrier_candidates)").fetchall()
    }
    if "first_emergent_global_step" not in temp_columns:
        return
    for signature, first_emergent in temp_state.execute(
        "SELECT carrier_signature, first_emergent_global_step FROM carrier_candidates "
        "WHERE first_emergent_global_step IS NOT NULL"
    ).fetchall():
        state_conn.execute(
            """
            UPDATE carrier_candidates
            SET first_emergent_global_step = CASE
                WHEN first_emergent_global_step IS NULL THEN ?
                ELSE MIN(first_emergent_global_step, ?)
            END
            WHERE carrier_signature = ?
            """,
            (int(first_emergent), int(first_emergent), str(signature)),
        )


def _fold_live_system_into_compact_memory(system: Any, memory_dir: str | Path) -> dict[str, Any]:
    result = _ORIGINAL_FOLD_LIVE_SYSTEM(system, memory_dir)
    from v6.memory import compact_memory as compact

    paths = compact.ensure_memory_layout(memory_dir)
    candidates = list(getattr(system.carrier_tracker, "build_candidates", lambda: [])())
    with sqlite3.connect(paths.current_state) as state_conn:
        _ensure_emergence_column(state_conn)
        for candidate in candidates:
            first_seen = _optional_int(getattr(candidate, "first_seen_global_step", None))
            first_emergent = _optional_int(getattr(candidate, "first_emergent_global_step", None))
            if (
                str(getattr(candidate, "status", "")) != "emergent_carrier"
                or str(getattr(candidate, "carrier_source", "")) == "context_action_fallback"
            ):
                first_emergent = None
            state_conn.execute(
                """
                UPDATE carrier_candidates
                SET first_seen_global_step = CASE
                        WHEN ? IS NULL THEN first_seen_global_step
                        WHEN first_seen_global_step IS NULL THEN ?
                        ELSE MIN(first_seen_global_step, ?)
                    END,
                    first_emergent_global_step = CASE
                        WHEN ? IS NULL THEN first_emergent_global_step
                        WHEN first_emergent_global_step IS NULL THEN ?
                        ELSE MIN(first_emergent_global_step, ?)
                    END
                WHERE carrier_signature = ?
                """,
                (
                    first_seen, first_seen, first_seen,
                    first_emergent, first_emergent, first_emergent,
                    str(getattr(candidate, "carrier_signature", "")),
                ),
            )
        state_conn.commit()
    return result


def _restore_compact_memory_into_system(
    system: Any,
    memory_dir: str | Path,
    *,
    restore_graph: bool = True,
    restore_substrate: bool = True,
) -> dict[str, Any]:
    summary = _ORIGINAL_RESTORE_COMPACT(
        system,
        memory_dir,
        restore_graph=restore_graph,
        restore_substrate=restore_substrate,
    )
    tracker = getattr(system, "carrier_tracker", None)
    if tracker is None:
        return summary
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return summary
    try:
        with sqlite3.connect(f"file:{current_state.resolve()}?mode=ro", uri=True) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(carrier_candidates)").fetchall()
            }
            if "first_emergent_global_step" not in columns:
                return summary
            rows = connection.execute(
                """
                SELECT carrier_signature, first_emergent_global_step
                FROM carrier_candidates
                WHERE COALESCE(is_emergent, 0)=1
                  AND carrier_source != 'context_action_fallback'
                  AND first_emergent_global_step IS NOT NULL
                """
            ).fetchall()
    except sqlite3.Error:
        return summary
    first_steps = getattr(tracker, "_v63_first_emergent_steps", None)
    if first_steps is None:
        first_steps = {}
        tracker._v63_first_emergent_steps = first_steps
    for signature, first_emergent in rows:
        first_steps[str(signature)] = int(first_emergent)
    summary = dict(summary)
    summary["carrier_emergence_thresholds_restored"] = len(rows)
    return summary


def _repair_role_emergence_steps(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
) -> None:
    """Reconstruct role emergence only from carriers that already emerged."""
    columns = {
        str(row[1])
        for row in state_conn.execute("PRAGMA table_info(role_candidates)").fetchall()
    }
    if "first_emergent_global_step" not in columns:
        state_conn.execute(
            "ALTER TABLE role_candidates ADD COLUMN first_emergent_global_step INTEGER"
        )
    _ensure_emergence_column(state_conn)

    role_carriers: dict[str, list[str]] = defaultdict(list)
    for row in state_conn.execute(
        "SELECT role_signature, linked_key FROM role_links "
        "WHERE linked_type='carrier' ORDER BY role_signature, linked_key"
    ).fetchall():
        role_carriers[str(row[0])].append(str(row[1]))

    carrier_meta = {
        str(row[0]): {
            "emergent": None if row[1] is None else int(row[1]),
            "last": None if row[2] is None else int(row[2]),
        }
        for row in state_conn.execute(
            "SELECT carrier_signature, first_emergent_global_step, last_seen_global_step "
            "FROM carrier_candidates"
        ).fetchall()
    }

    from v6 import higher_order_substrate as substrate

    carrier_links = substrate._carrier_links_by_carrier(state_conn)
    contexts = {
        f"context:{context}"
        for links in carrier_links.values()
        for context in links.get("context", set())
    }
    context_games = (
        substrate._context_games_for_context_nodes(graph_conn, contexts)
        if contexts
        else {}
    )
    emergent_roles = {
        str(row[0])
        for row in state_conn.execute(
            "SELECT role_signature FROM role_candidates WHERE COALESCE(is_emergent,0)=1"
        ).fetchall()
    }

    for role, carriers in role_carriers.items():
        if role not in emergent_roles:
            state_conn.execute(
                "UPDATE role_candidates SET first_emergent_global_step=NULL "
                "WHERE role_signature=?",
                (role,),
            )
            continue
        ordered = sorted(
            (int(carrier_meta[carrier]["emergent"]), carrier)
            for carrier in carriers
            if carrier in carrier_meta and carrier_meta[carrier]["emergent"] is not None
        )
        seen_carriers: set[str] = set()
        families: set[str] = set()
        role_contexts: set[str] = set()
        games: set[str] = set()
        emergence_step: int | None = None
        index = 0
        while index < len(ordered):
            step = int(ordered[index][0])
            batch: list[str] = []
            while index < len(ordered) and int(ordered[index][0]) == step:
                batch.append(str(ordered[index][1]))
                index += 1
            for carrier in batch:
                seen_carriers.add(carrier)
                links = carrier_links.get(carrier, {})
                families.update(str(value) for value in links.get("family", set()))
                new_contexts = {str(value) for value in links.get("context", set())}
                role_contexts.update(new_contexts)
                for context in new_contexts:
                    games.update(
                        str(value)
                        for value in context_games.get(f"context:{context}", set())
                    )
            stability = (
                0.25 * min(1.0, len(seen_carriers) / 3.0)
                + 0.25 * min(1.0, len(families) / 3.0)
                + 0.25 * min(1.0, len(role_contexts) / 3.0)
                + 0.25 * min(1.0, len(games) / 2.0)
            )
            if (
                len(seen_carriers) >= 2
                and len(families) >= 1
                and (len(role_contexts) >= 2 or len(games) >= 2)
                and stability >= 0.50
            ):
                emergence_step = step
                break
        state_conn.execute(
            "UPDATE role_candidates SET first_emergent_global_step=? WHERE role_signature=?",
            (emergence_step, role),
        )


def _evaluate_h04_base(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Feed H04 the explicit emergence threshold before v6.3 normalization."""
    result = dict(_ORIGINAL_H04_BASE(*args, **kwargs))
    memory_dir = kwargs.get("memory_dir")
    if memory_dir is None and len(args) >= 2:
        memory_dir = args[1]
    if memory_dir is None:
        return result
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return result

    from v6 import hypothesis_h04_report as h04

    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_emergence_column(connection)
        rows = connection.execute(
            """
            SELECT carrier_signature, support_count, is_emergent,
                   first_emergent_global_step, carrier_source
            FROM carrier_candidates
            """
        ).fetchall()
        family_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT carrier_signature, COUNT(DISTINCT linked_key)
                FROM carrier_links
                WHERE linked_type='family' AND linked_key IS NOT NULL AND linked_key != ''
                GROUP BY carrier_signature
                """
            ).fetchall()
        }

    emergent = [row for row in rows if int(row["is_emergent"] or 0) == 1]
    usable_emergent: list[sqlite3.Row] = []
    for row in emergent:
        linked_family_count = family_counts.get(str(row["carrier_signature"]), 0)
        specificity = float(row["support_count"] or 0.0) / max(1, linked_family_count)
        if (
            linked_family_count <= int(h04.MAX_LINKED_FAMILIES_PER_CARRIER)
            and specificity >= float(h04.MIN_CARRIER_SPECIFICITY)
        ):
            usable_emergent.append(row)

    first_emergent = min(
        (
            int(row["first_emergent_global_step"])
            for row in emergent
            if row["first_emergent_global_step"] is not None
        ),
        default=None,
    )
    first_usable = min(
        (
            int(row["first_emergent_global_step"])
            for row in usable_emergent
            if row["first_emergent_global_step"] is not None
        ),
        default=None,
    )
    if first_emergent is None and first_usable is None:
        return result

    metrics = dict(result.get("core_metrics") or {})
    if first_emergent is not None:
        result["first_emergent_carrier_step"] = first_emergent
        metrics["first_emergent_carrier_step"] = first_emergent
    if first_usable is not None:
        result["first_usable_emergent_carrier_step"] = first_usable
        metrics["first_usable_emergent_carrier_step"] = first_usable
    first_stable = result.get("first_stable_transformation_family_step")
    if first_stable is None:
        first_stable = metrics.get("first_stable_transformation_family_step")
    strict_usable = (
        None
        if first_stable is None or first_usable is None
        else int(first_stable) < int(first_usable)
    )
    metrics["h03_before_h04"] = strict_usable
    metrics["h03_before_h04_usable"] = strict_usable
    metrics["temporal_order_comparison"] = "strict_before"
    result["h03_before_h04"] = strict_usable
    result["h03_before_h04_usable"] = strict_usable
    result["core_metrics"] = metrics
    _recompute_h04_decision(result)
    if strict_usable is True:
        result["missing_evidence"] = [
            str(message)
            for message in list(result.get("missing_evidence") or [])
            if "H03-before-H04 temporal order" not in str(message)
            and "H04 temporal order failed" not in str(message)
        ]
    return result


def _recompute_h04_decision(result: dict[str, Any]) -> None:
    metrics = dict(result.get("core_metrics") or {})
    carrier_count = int(metrics.get("carrier_candidate_count") or 0)
    usable_emergent = int(metrics.get("usable_emergent_carrier_count") or 0)
    emergent_count = int(metrics.get("emergent_carrier_count") or 0)
    emergent_fallback = int(metrics.get("emergent_context_action_fallback_count") or 0)
    usable_count = int(metrics.get("usable_carrier_count") or 0)
    cross_ok = (
        int(metrics.get("carrier_cross_family_count") or 0) >= 2
        or int(metrics.get("carrier_cross_context_count") or 0) >= 2
    )
    explains = int(metrics.get("usable_carrier_explains_edge_count") or 0)
    anchors = int(metrics.get("usable_carrier_anchors_edge_count") or 0)
    strict = metrics.get("h03_before_h04_usable")
    timing_source = str(metrics.get("carrier_timing_source") or result.get("carrier_timing_source") or "unknown")
    first_stable = metrics.get("first_stable_transformation_family_step")
    ready = (
        usable_emergent > 0
        and emergent_fallback == 0
        and usable_count > 0
        and cross_ok
        and explains > 0
        and anchors > 0
    )
    if carrier_count <= 0:
        decision = "INVALID" if first_stable is not None else "INCONCLUSIVE"
    elif ready and strict is False and timing_source == "real_evidence":
        decision = "INVALID"
    elif ready and strict is False:
        decision = "PARTIALLY_VALID"
    elif ready and strict is True and timing_source == "real_evidence":
        decision = "VALID"
    elif ready and strict is None:
        decision = "PARTIALLY_VALID"
    elif ready and strict is True:
        decision = "PARTIALLY_VALID"
    elif emergent_count > 0 and emergent_fallback == emergent_count:
        decision = "INVALID"
    elif carrier_count > 0:
        decision = "PARTIALLY_VALID"
    else:
        decision = "INCONCLUSIVE"
    if metrics.get("h04_graph_quality_pass") is not True and decision == "VALID":
        decision = "PARTIALLY_VALID"
    result["decision"] = decision


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _minimum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(int(left), int(right))
