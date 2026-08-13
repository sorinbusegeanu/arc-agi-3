from __future__ import annotations

import json
import math
import sqlite3
import time
from bisect import bisect_right
from collections import defaultdict
from contextvars import ContextVar
from hashlib import sha1
from pathlib import Path
from typing import Any

_INSTALLED = False
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("v6_concept_validation_fastpath", default=None)
_ORIGINALS: dict[str, Any] = {}


def _timed(name: str, callback, *args, **kwargs):
    ctx = _ACTIVE.get()
    started = time.perf_counter()
    try:
        return callback(*args, **kwargs)
    finally:
        if ctx is not None:
            timings = ctx.setdefault("timings", {})
            timings[name] = float(timings.get(name, 0.0) or 0.0) + (time.perf_counter() - started)
            counts = ctx.setdefault("call_counts", {})
            counts[name] = int(counts.get(name, 0) or 0) + 1


def _connection_key(conn: sqlite3.Connection) -> int:
    return id(conn)


def _prediction_rows(state_conn: sqlite3.Connection) -> tuple[list[int], list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    ctx = _ACTIVE.get()
    key = _connection_key(state_conn)
    cache_key = ("prediction_rows", key)
    if ctx is not None and cache_key in ctx["cache"]:
        return ctx["cache"][cache_key]
    started = time.perf_counter()
    columns = _ORIGINALS["_prediction_result_columns"](state_conn)
    required = {"id", "global_step", "context_signature"}
    rows: list[dict[str, Any]] = []
    contradiction_rows: list[dict[str, Any]] = []
    if required <= columns:
        selected = ["id", "global_step", "context_signature"]
        for optional in ("predicted_family", "actual_family", "context_contradiction"):
            if optional in columns:
                selected.append(optional)
        sql = "SELECT " + ", ".join(selected) + " FROM prediction_results ORDER BY global_step ASC, id ASC"
        rows = [dict(row) for row in state_conn.execute(sql).fetchall()]
        if "context_contradiction" in columns:
            contradiction_rows = [row for row in rows if int(row.get("context_contradiction") or 0) == 1]
    result = (
        [int(row["global_step"]) for row in rows], rows,
        [int(row["global_step"]) for row in contradiction_rows], contradiction_rows,
    )
    if ctx is not None:
        ctx["cache"][cache_key] = result
        ctx["index_stats"]["prediction_rows"] = len(rows)
        ctx["index_stats"]["contradiction_rows"] = len(contradiction_rows)
        ctx["timings"]["index.prediction_rows"] = ctx["timings"].get("index.prediction_rows", 0.0) + (time.perf_counter() - started)
    return result


def _motif_rows(state_conn: sqlite3.Connection) -> tuple[list[int], list[dict[str, Any]]]:
    ctx = _ACTIVE.get()
    key = ("motif_rows", _connection_key(state_conn))
    if ctx is not None and key in ctx["cache"]:
        return ctx["cache"][key]
    started = time.perf_counter()
    table = state_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='future_option_motifs'").fetchone()
    rows: list[dict[str, Any]] = []
    if table is not None:
        rows = [dict(row) for row in state_conn.execute(
            "SELECT motif_signature, source_role_ids_json, motif_stability_score, is_emergent, last_seen_global_step "
            "FROM future_option_motifs WHERE last_seen_global_step IS NOT NULL "
            "ORDER BY last_seen_global_step ASC, motif_signature ASC"
        ).fetchall()]
    result = ([int(row["last_seen_global_step"]) for row in rows], rows)
    if ctx is not None:
        ctx["cache"][key] = result
        ctx["index_stats"]["future_option_motif_rows"] = len(rows)
        ctx["timings"]["index.future_option_motifs"] = ctx["timings"].get("index.future_option_motifs", 0.0) + (time.perf_counter() - started)
    return result


def _transfer_step_rows(transfer_rows: list[sqlite3.Row]) -> tuple[list[int], list[sqlite3.Row]]:
    ctx = _ACTIVE.get()
    fingerprint = (id(transfer_rows), len(transfer_rows))
    key = ("transfer_steps", fingerprint)
    if ctx is not None and key in ctx["cache"]:
        return ctx["cache"][key]
    started = time.perf_counter()
    ordered = sorted(
        (row for row in transfer_rows if row["last_seen_global_step"] is not None),
        key=lambda row: (int(row["last_seen_global_step"]), str(row["attempt_id"])),
    )
    result = ([int(row["last_seen_global_step"]) for row in ordered], ordered)
    if ctx is not None:
        ctx["cache"][key] = result
        ctx["index_stats"]["transfer_rows"] = len(ordered)
        ctx["timings"]["index.transfer_rows"] = ctx["timings"].get("index.transfer_rows", 0.0) + (time.perf_counter() - started)
    return result


def _role_score_bundle(substrate, *, source_roles: list[str], step: int, transfer_rows, transfer_history, rate_cache, scope: tuple[str, str, str, str] | None = None):
    ctx = _ACTIVE.get()
    roles_key = tuple(source_roles)
    scope_key = scope or ("", "", "", "")
    key = (roles_key, int(step), *scope_key)
    cache = None if ctx is None else ctx["role_score_cache"]
    if cache is not None and key in cache:
        return cache[key]
    generic_rates = [
        substrate._prior_role_success_rate(
            transfer_rows,
            role=role,
            before_step=int(step),
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )[0]
        for role in source_roles
    ]
    scoped_rates: list[float] = []
    if scope is not None:
        source_game_key, source_context_key, target_game_key, target_context_key = scope
        for role in source_roles:
            rate, count = substrate._prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=int(step),
                source_game_key=source_game_key,
                source_context_key=source_context_key,
                target_game_key=target_game_key,
                target_context_key=target_context_key,
                transfer_history=transfer_history,
                rate_cache=rate_cache,
            )
            if count > 0:
                scoped_rates.append(rate)
    result = (generic_rates, scoped_rates)
    if cache is not None:
        cache[key] = result
    return result


def _transfer_explanation_events(*, candidate_signature: str, source_roles: list[str], first_seen_global_step: int | None, transfer_rows: list[sqlite3.Row], transfer_history=None, rate_cache=None) -> list[dict[str, Any]]:
    from v6 import higher_order_substrate as substrate
    if first_seen_global_step is None or not source_roles:
        return []
    started = time.perf_counter()
    steps, ordered = _transfer_step_rows(transfer_rows)
    start = bisect_right(steps, int(first_seen_global_step))
    source_role_set = set(source_roles)
    feature_step_cache: dict[int, int | None] = {}
    events: list[dict[str, Any]] = []
    for row in ordered[start:]:
        source_role = str(row["role_signature"] or "")
        step = int(row["last_seen_global_step"])
        target_role = str(row["observed_role_signature"] or row["predicted_role_signature"] or "unknown")
        source_game_key = str(row["source_game_key"] or "")
        source_context_key = str(row["source_context_key"] or "")
        target_game_key = str(row["target_game_key"] or "")
        target_context_key = str(row["target_context_key"] or "")
        generic_rates, scoped_rates = _role_score_bundle(
            substrate,
            source_roles=source_roles,
            step=step,
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
            scope=(source_game_key, source_context_key, target_game_key, target_context_key),
        )
        best_single = max(generic_rates, default=0.0)
        combination_rates = scoped_rates if len(scoped_rates) >= 2 else generic_rates
        concept_score = substrate._combined_role_score(combination_rates) if len(combination_rates) >= 2 else best_single
        if step not in feature_step_cache:
            if transfer_history is not None:
                feature_step_cache[step] = max(
                    (prior for role in source_roles for prior in [transfer_history.max_step_before(role=role, step=step)] if prior is not None),
                    default=None,
                )
            else:
                idx = bisect_right(steps, step - 1)
                feature_step_cache[step] = max(
                    (int(item["last_seen_global_step"]) for item in ordered[:idx] if str(item["role_signature"] or "") in source_role_set),
                    default=None,
                )
        outcome = float(int(row["reuse_success"] or 0))
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"transfer:{source_role}:{target_role}:{source_game_key}:{source_context_key}:{target_game_key}:{target_context_key}:{row['attempt_id']}",
            "event_type": "transfer",
            "evaluation_scope": "later_global_step",
            "best_single_role_score": best_single,
            "lower_level_baseline_score": best_single,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - best_single,
            "behavioral_gain": -abs(concept_score - outcome) + abs(best_single - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": step,
            "_feature_global_step_max": feature_step_cache[step],
            "_label_used_as_feature": False,
            "_source_role_signature": source_role,
            "_predicted_target_role_signature": str(row["predicted_role_signature"] or ""),
            "_observed_target_role_signature": str(row["observed_role_signature"] or ""),
            "_carrier_ids": [str(value) for value in (row["source_carrier_signature"], row["target_carrier_signature"]) if value not in (None, "")],
            "_context_keys": [value for value in (source_context_key, target_context_key) if value],
            "_game_keys": [value for value in (source_game_key, target_game_key) if value],
        })
    ctx = _ACTIVE.get()
    if ctx is not None:
        ctx["timings"]["events.transfer"] = ctx["timings"].get("events.transfer", 0.0) + (time.perf_counter() - started)
        ctx["event_counts"]["transfer"] += len(events)
    return events


def _future_option_motif_explanation_events(state_conn: sqlite3.Connection, *, candidate_signature: str, source_roles: list[str], first_seen_global_step: int | None, future_rows: list[sqlite3.Row], role_rate_cache=None) -> list[dict[str, Any]]:
    from v6 import higher_order_substrate as substrate
    if first_seen_global_step is None or not source_roles:
        return []
    started = time.perf_counter()
    role_set = set(source_roles)
    role_rates: dict[str, float] = {}
    for role in source_roles:
        cache_key = (role, int(first_seen_global_step))
        if role_rate_cache is not None and cache_key in role_rate_cache:
            role_rates[role] = role_rate_cache[cache_key]
        else:
            values = [
                1.0 if float(row["option_delta"] or 0.0) > 0.0 else 0.0
                for row in future_rows
                if str(row["source_role_id"] or row["owner_key"] or "") == role
                and row["last_seen_global_step"] is not None
                and int(row["last_seen_global_step"]) < int(first_seen_global_step)
            ]
            role_rates[role] = sum(values) / len(values) if values else 0.0
            if role_rate_cache is not None:
                role_rate_cache[cache_key] = role_rates[role]
    steps, rows = _motif_rows(state_conn)
    start = bisect_right(steps, int(first_seen_global_step))
    events: list[dict[str, Any]] = []
    for row in rows[start:]:
        try:
            motif_roles = {str(value) for value in json.loads(str(row.get("source_role_ids_json") or "[]"))}
        except (TypeError, ValueError, json.JSONDecodeError):
            motif_roles = set()
        rates = [role_rates[role] for role in sorted(role_set & motif_roles)]
        baseline = max(rates, default=0.0)
        concept_score = substrate._combined_role_score(rates) if len(rates) >= 2 else baseline
        outcome = float(int(row.get("is_emergent") or 0))
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"future_option_motif:{row['motif_signature']}",
            "event_type": "future_option_motif",
            "evaluation_scope": "later_global_step",
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": int(row["last_seen_global_step"]),
            "_feature_global_step_max": int(first_seen_global_step) - 1,
            "_label_used_as_feature": False,
            "_source_role_ids": sorted(motif_roles),
        })
    ctx = _ACTIVE.get()
    if ctx is not None:
        ctx["timings"]["events.future_option_motif"] = ctx["timings"].get("events.future_option_motif", 0.0) + (time.perf_counter() - started)
        ctx["event_counts"]["future_option_motif"] += len(events)
    return events


def _prediction_explanation_events(state_conn: sqlite3.Connection, *, candidate_signature: str, source_roles: list[str], first_seen_global_step: int | None, transfer_rows: list[sqlite3.Row], role_links, transfer_history=None, rate_cache=None) -> list[dict[str, Any]]:
    from v6 import higher_order_substrate as substrate
    columns = _ORIGINALS["_prediction_result_columns"](state_conn)
    required = {"id", "global_step", "context_signature", "predicted_family", "actual_family"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return []
    started = time.perf_counter()
    steps, rows, _contr_steps, _contr_rows = _prediction_rows(state_conn)
    start = bisect_right(steps, int(first_seen_global_step))
    source_role_family_ids = sorted({str(family) for role in source_roles for family in role_links.get(role, {}).get("family", set()) if family not in (None, "")})
    events: list[dict[str, Any]] = []
    for row in rows[start:]:
        step = int(row["global_step"])
        generic_rates, _ = _role_score_bundle(substrate, source_roles=source_roles, step=step, transfer_rows=transfer_rows, transfer_history=transfer_history, rate_cache=rate_cache)
        baseline = max(generic_rates, default=0.0)
        concept_score = substrate._combined_role_score(generic_rates) if len(generic_rates) >= 2 else baseline
        predicted_family = str(row.get("predicted_family") or "")
        actual_family = str(row.get("actual_family") or "")
        outcome = float(predicted_family == actual_family)
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"prediction:prediction_result:{row['id']}:later_global_step",
            "event_type": "prediction",
            "evaluation_scope": "later_global_step",
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "candidate_role_family_ids": source_role_family_ids,
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": step,
            "_feature_global_step_max": transfer_history.max_any_step_before(step) if transfer_history is not None else None,
            "_label_used_as_feature": False,
            "_context_keys": [str(row.get("context_signature") or "")] if row.get("context_signature") else [],
            "_family_ids": [family for family in (predicted_family, actual_family) if family],
        })
    ctx = _ACTIVE.get()
    if ctx is not None:
        ctx["timings"]["events.prediction"] = ctx["timings"].get("events.prediction", 0.0) + (time.perf_counter() - started)
        ctx["event_counts"]["prediction"] += len(events)
    return events


def _contradiction_resolution_explanation_events(state_conn: sqlite3.Connection, *, candidate_signature: str, source_roles: list[str], first_seen_global_step: int | None, transfer_rows: list[sqlite3.Row], role_links, transfer_history=None, rate_cache=None) -> list[dict[str, Any]]:
    from v6 import higher_order_substrate as substrate
    columns = _ORIGINALS["_prediction_result_columns"](state_conn)
    required = {"id", "global_step", "context_signature", "context_contradiction"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return []
    started = time.perf_counter()
    _steps, _rows, contradiction_steps, contradiction_rows = _prediction_rows(state_conn)
    start = bisect_right(contradiction_steps, int(first_seen_global_step))
    source_role_family_ids = sorted({str(family) for role in source_roles for family in role_links.get(role, {}).get("family", set()) if family not in (None, "")})
    available_family_columns = {"predicted_family", "actual_family"} <= columns
    events: list[dict[str, Any]] = []
    for row in contradiction_rows[start:]:
        step = int(row["global_step"])
        generic_rates, _ = _role_score_bundle(substrate, source_roles=source_roles, step=step, transfer_rows=transfer_rows, transfer_history=transfer_history, rate_cache=rate_cache)
        failure_rates = [1.0 - rate for rate in generic_rates]
        baseline = max(failure_rates, default=0.0)
        concept_score = substrate._combined_role_score(failure_rates) if len(failure_rates) >= 2 else baseline
        predicted_family = str(row.get("predicted_family") or "") if available_family_columns else ""
        actual_family = str(row.get("actual_family") or "") if available_family_columns else ""
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"contradiction_resolution:{row['id']}:later_global_step",
            "event_type": "contradiction_resolution",
            "evaluation_scope": "later_global_step",
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "candidate_role_family_ids": source_role_family_ids,
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": concept_score - baseline,
            "_outcome": 1.0,
            "_evaluation_global_step": step,
            "_feature_global_step_max": transfer_history.max_any_step_before(step) if transfer_history is not None else None,
            "_label_used_as_feature": False,
            "_context_keys": [str(row.get("context_signature") or "")] if row.get("context_signature") else [],
            "_family_ids": [family for family in (predicted_family, actual_family) if family],
        })
    ctx = _ACTIVE.get()
    if ctx is not None:
        ctx["timings"]["events.contradiction"] = ctx["timings"].get("events.contradiction", 0.0) + (time.perf_counter() - started)
        ctx["event_counts"]["contradiction_resolution"] += len(events)
    return events


def _evidence_frontier(memory_dir: Path) -> dict[str, int]:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return {}
    result: dict[str, int] = {}
    with sqlite3.connect(db) as conn:
        for table, column in (
            ("role_transfer_attempts", "last_seen_global_step"),
            ("future_option_events", "last_seen_global_step"),
            ("future_option_motifs", "last_seen_global_step"),
            ("prediction_results", "global_step"),
        ):
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if exists is None:
                continue
            value = conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            result[f"{table}.max_step"] = int(value or 0)
            result[f"{table}.count"] = int(count or 0)
    return result


def _load_frontier(memory_dir: Path) -> dict[str, Any] | None:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT value_json FROM memory_summary WHERE key='concept_validation_fastpath_frontier'").fetchone()
    if row is None:
        return None
    try:
        value = json.loads(str(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _store_frontier(memory_dir: Path, payload: dict[str, Any]) -> None:
    db = Path(memory_dir) / "current_state.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO memory_summary(key, value_json) VALUES('concept_validation_fastpath_frontier', ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (json.dumps(payload, sort_keys=True),),
        )
        conn.commit()


def _validate_incremental_promotions_only(*args: Any, **kwargs: Any) -> dict[str, Any]:
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    validate_roles_and_concepts = bool(kwargs.get("validate_roles_and_concepts"))
    frontier_before = _evidence_frontier(memory_dir) if validate_roles_and_concepts else {}
    previous_frontier = _load_frontier(memory_dir) if validate_roles_and_concepts else None
    ctx: dict[str, Any] = {
        "cache": {},
        "role_score_cache": {},
        "timings": {},
        "call_counts": {},
        "event_counts": defaultdict(int),
        "index_stats": {},
    }
    token = _ACTIVE.set(ctx)
    started = time.perf_counter()
    try:
        result = _ORIGINALS["validate_incremental_promotions_only"](*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        _ACTIVE.reset(token)
    if not isinstance(result, dict):
        return result
    profile = {
        "total_seconds": elapsed,
        "timings": {key: float(value) for key, value in sorted(ctx["timings"].items())},
        "call_counts": dict(ctx["call_counts"]),
        "event_counts": {key: int(value) for key, value in sorted(ctx["event_counts"].items())},
        "index_stats": dict(ctx["index_stats"]),
        "role_score_cache_entries": len(ctx["role_score_cache"]),
        "evidence_frontier": frontier_before,
        "previous_evidence_frontier": previous_frontier,
        "frontier_changed": previous_frontier != frontier_before if validate_roles_and_concepts else None,
    }
    result["concept_validation_fastpath_profile"] = profile
    if validate_roles_and_concepts:
        _store_frontier(memory_dir, frontier_before)
    return result


def install_concept_validation_fastpath() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    for name in (
        "validate_incremental_promotions_only",
        "_prediction_result_columns",
        "_transfer_explanation_events",
        "_future_option_motif_explanation_events",
        "_prediction_explanation_events",
        "_contradiction_resolution_explanation_events",
    ):
        _ORIGINALS[name] = getattr(substrate, name)

    substrate._transfer_explanation_events = _transfer_explanation_events
    substrate._future_option_motif_explanation_events = _future_option_motif_explanation_events
    substrate._prediction_explanation_events = _prediction_explanation_events
    substrate._contradiction_resolution_explanation_events = _contradiction_resolution_explanation_events
    substrate.validate_incremental_promotions_only = _validate_incremental_promotions_only
    suite.validate_incremental_promotions_only = _validate_incremental_promotions_only
    _INSTALLED = True
