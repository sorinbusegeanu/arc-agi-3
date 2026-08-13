from __future__ import annotations

import json
import sqlite3
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

_INSTALLED = False


def _enabled(config: Any) -> bool:
    return bool(getattr(config, "enabled", False))


def _safe_transfer(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast

    rows = kwargs.get("transfer_rows") or []
    if rows:
        keys = set(rows[0].keys())
        if not {"source_carrier_signature", "target_carrier_signature"} <= keys:
            return fast._ORIGINALS["_transfer_explanation_events"](*args, **kwargs)
    return fast._transfer_explanation_events(*args, **kwargs)


def _timed_canonical(name: str, callback):
    def wrapped(*args: Any, **kwargs: Any):
        from v6 import concept_validation_fastpath as fast

        ctx = fast._ACTIVE.get()
        started = time.perf_counter()
        try:
            result = callback(*args, **kwargs)
        finally:
            if ctx is not None:
                elapsed = time.perf_counter() - started
                timings = ctx.setdefault("timings", {})
                timings[name] = float(timings.get(name, 0.0) or 0.0) + elapsed
                counts = ctx.setdefault("call_counts", {})
                counts[name] = int(counts.get(name, 0) or 0) + 1
        if ctx is not None and isinstance(result, list):
            event_counts = ctx.setdefault("event_counts", defaultdict(int))
            event_counts[name.removeprefix("events.")] += len(result)
        return result

    return wrapped


def _future_role_prefix(future_rows: list[sqlite3.Row]):
    from v6 import concept_validation_fastpath as fast

    ctx = fast._ACTIVE.get()
    key = ("future_role_prefix", id(future_rows), len(future_rows))
    if ctx is not None and key in ctx["cache"]:
        return ctx["cache"][key]
    started = time.perf_counter()
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in future_rows:
        step = row["last_seen_global_step"]
        if step is None:
            continue
        role = str(row["source_role_id"] or row["owner_key"] or "")
        if not role:
            continue
        grouped[role].append((int(step), 1 if float(row["option_delta"] or 0.0) > 0.0 else 0))
    result: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for role, values in grouped.items():
        values.sort()
        prefix = [0]
        for _step, positive in values:
            prefix.append(prefix[-1] + positive)
        result[role] = (
            tuple(step for step, _positive in values),
            tuple(prefix),
        )
    if ctx is not None:
        ctx["cache"][key] = result
        ctx["index_stats"]["future_roles"] = len(result)
        ctx["index_stats"]["future_option_rows_indexed"] = sum(len(value[0]) for value in result.values())
        ctx["timings"]["index.future_option_roles"] = ctx["timings"].get("index.future_option_roles", 0.0) + (time.perf_counter() - started)
    return result


def _future_option_motif_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    future_rows: list[sqlite3.Row],
    role_rate_cache: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate

    if first_seen_global_step is None or not source_roles:
        return []
    started = time.perf_counter()
    index = _future_role_prefix(future_rows)
    role_rates: dict[str, float] = {}
    for role in source_roles:
        cache_key = (role, int(first_seen_global_step))
        if role_rate_cache is not None and cache_key in role_rate_cache:
            role_rates[role] = role_rate_cache[cache_key]
            continue
        steps, prefix = index.get(role, ((), (0,)))
        count = bisect_left(steps, int(first_seen_global_step))
        rate = (float(prefix[count]) / float(count)) if count else 0.0
        role_rates[role] = rate
        if role_rate_cache is not None:
            role_rate_cache[cache_key] = rate

    motif_steps, rows = fast._motif_rows(state_conn)
    start = bisect_right(motif_steps, int(first_seen_global_step))
    role_set = set(source_roles)
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
        events.append(
            {
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
            }
        )
    ctx = fast._ACTIVE.get()
    if ctx is not None:
        ctx["timings"]["events.future_option_motif"] = ctx["timings"].get("events.future_option_motif", 0.0) + (time.perf_counter() - started)
        ctx["event_counts"]["future_option_motif"] += len(events)
    return events


def _state_fingerprint(memory_dir: Path) -> str:
    db = Path(memory_dir) / "current_state.sqlite"
    digest = sha256()
    with sqlite3.connect(db) as conn:
        for table, columns in (
            ("role_candidates", "role_signature, carrier_signature, first_seen_global_step, last_seen_global_step, is_promoted"),
            ("role_links", "role_signature, link_type, target_signature, first_seen_global_step, last_seen_global_step"),
            ("concept_candidates", "concept_signature, first_seen_global_step, last_seen_global_step, promotion_score, is_promoted"),
            ("concept_links", "concept_signature, link_type, target_signature, first_seen_global_step, last_seen_global_step"),
        ):
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if exists is None:
                continue
            digest.update(table.encode("utf-8"))
            for row in conn.execute(f"SELECT {columns} FROM {table} ORDER BY 1, 2").fetchall():
                digest.update(repr(tuple(row)).encode("utf-8"))
    return digest.hexdigest()


def _config_fingerprint(config: Any) -> str:
    values = {
        name: getattr(config, name)
        for name in dir(config)
        if not name.startswith("_") and not callable(getattr(config, name))
    }
    return sha256(json.dumps(values, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _load_cache(memory_dir: Path) -> dict[str, Any] | None:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_summary'").fetchone()
        if exists is None:
            return None
        row = conn.execute("SELECT value_json FROM memory_summary WHERE key='concept_validation_fastpath_cache'").fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(str(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _store_cache(memory_dir: Path, payload: dict[str, Any]) -> None:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return
    with sqlite3.connect(db) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_summary'").fetchone()
        if exists is None:
            return
        conn.execute(
            "INSERT INTO memory_summary(key, value_json) VALUES('concept_validation_fastpath_cache', ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (json.dumps(payload, sort_keys=True),),
        )
        conn.commit()


def _cache_safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    reused = dict(summary)
    for key in tuple(reused):
        if key.endswith("_demoted") or key.endswith("_promoted_this_run"):
            if isinstance(reused[key], (int, float)):
                reused[key] = 0
    reused["concept_validation_cache_hit"] = True
    reused["concept_validation_reused_unchanged_evidence"] = True
    return reused


def _validate(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast

    config = kwargs.get("config")
    if config is None and len(args) > 1:
        config = args[1]
    if not _enabled(config):
        return fast._ORIGINALS["validate_incremental_promotions_only"](*args, **kwargs)

    memory_dir_raw = kwargs.get("memory_dir") if "memory_dir" in kwargs else (args[0] if args else None)
    memory_dir = Path(memory_dir_raw)
    validate_roles_and_concepts = bool(kwargs.get("validate_roles_and_concepts", False))
    reset = bool(kwargs.get("validation_state_reset_applied_this_run", False))
    frontier = fast._evidence_frontier(memory_dir) if validate_roles_and_concepts else {}
    state_fingerprint = _state_fingerprint(memory_dir) if validate_roles_and_concepts else ""
    config_fingerprint = _config_fingerprint(config)

    if validate_roles_and_concepts and not reset:
        previous = _load_cache(memory_dir)
        if (
            previous
            and previous.get("evidence_frontier") == frontier
            and previous.get("state_fingerprint") == state_fingerprint
            and previous.get("config_fingerprint") == config_fingerprint
            and isinstance(previous.get("summary"), dict)
        ):
            result = _cache_safe_summary(dict(previous["summary"]))
            result["concept_validation_fastpath_profile"] = {
                "cache_hit": True,
                "evidence_frontier": frontier,
                "state_fingerprint": state_fingerprint,
                "total_seconds": 0.0,
            }
            return result

    result = fast._validate_incremental_promotions_only(*args, **kwargs)
    if validate_roles_and_concepts and isinstance(result, dict):
        _store_cache(
            memory_dir,
            {
                "evidence_frontier": frontier,
                "state_fingerprint": state_fingerprint,
                "config_fingerprint": config_fingerprint,
                "summary": result,
            },
        )
    return result


def install_concept_validation_fastpath_compat() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    # Restore final semantic helpers captured before the fast-path replaced them,
    # then wrap them only for sub-timing. This preserves v6.3/v0.4.2 behavior.
    canonical_prediction = fast._ORIGINALS["_prediction_explanation_events"]
    canonical_contradiction = fast._ORIGINALS["_contradiction_resolution_explanation_events"]

    substrate._transfer_explanation_events = _safe_transfer
    substrate._future_option_motif_explanation_events = _future_option_motif_explanation_events
    substrate._prediction_explanation_events = _timed_canonical("events.prediction", canonical_prediction)
    substrate._contradiction_resolution_explanation_events = _timed_canonical("events.contradiction_resolution", canonical_contradiction)
    substrate.validate_incremental_promotions_only = _validate
    suite.validate_incremental_promotions_only = _validate
    _INSTALLED = True
