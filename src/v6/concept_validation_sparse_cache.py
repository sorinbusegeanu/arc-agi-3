from __future__ import annotations

import os
import time
from typing import Any

_INSTALLED = False
_ORIGINAL: Any = None
_ORIGINAL_PRIOR_RATE: Any = None
_ORIGINAL_FUNCTIONAL_DIAGNOSTICS: Any = None
_LAST_PROFILE: dict[str, Any] = {}


def _inc(ctx: dict[str, Any] | None, key: str, amount: int = 1) -> None:
    if ctx is None:
        return
    stats = ctx.setdefault("cache_stats", {})
    stats[key] = int(stats.get(key, 0)) + int(amount)
    index_stats = ctx.setdefault("index_stats", {})
    index_stats[f"sparse_cache.{key}"] = int(stats[key])


def _add_timing(ctx: dict[str, Any] | None, key: str, seconds: float) -> None:
    if ctx is None:
        return
    timings = ctx.setdefault("timings", {})
    timings[key] = float(timings.get(key, 0.0) or 0.0) + float(seconds)


def _sparse_role_score_bundle(
    substrate: Any,
    *,
    source_roles: list[str],
    step: int,
    transfer_rows: Any,
    transfer_history: Any,
    rate_cache: Any,
    scope: tuple[str, str, str, str] | None = None,
):
    from v6 import concept_validation_fastpath as fast

    started = time.perf_counter()
    ctx = fast._ACTIVE.get()
    roles_key = tuple(source_roles)
    step_i = int(step)

    generic_cache = None
    scoped_cache = None
    if ctx is not None:
        generic_cache = ctx.setdefault("generic_role_score_cache", {})
        scoped_cache = ctx.setdefault("scoped_role_score_cache", {})

    generic_key = (roles_key, step_i)
    if generic_cache is not None and generic_key in generic_cache:
        generic_rates = generic_cache[generic_key]
        _inc(ctx, "generic_hits")
    else:
        generic_rates = [
            substrate._prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=step_i,
                transfer_history=transfer_history,
                rate_cache=rate_cache,
            )[0]
            for role in source_roles
        ]
        if generic_cache is not None:
            generic_cache[generic_key] = generic_rates
            _inc(ctx, "generic_misses")

    scoped_rates: list[float] = []
    if scope is not None:
        scope_key = (roles_key, step_i, *scope)
        if scoped_cache is not None and scope_key in scoped_cache:
            scoped_rates = scoped_cache[scope_key]
            _inc(ctx, "scoped_hits")
        else:
            source_game_key, source_context_key, target_game_key, target_context_key = scope
            sparse_index = getattr(transfer_history, "by_source_target_scope", None)
            if isinstance(sparse_index, dict):
                for role in source_roles:
                    series = sparse_index.get(
                        (
                            role,
                            source_game_key or "",
                            source_context_key or "",
                            target_game_key or "",
                            target_context_key or "",
                        )
                    )
                    if series is None:
                        continue
                    rate, count = series.rate_before(step_i)
                    if count > 0:
                        scoped_rates.append(rate)
            else:
                for role in source_roles:
                    rate, count = substrate._prior_role_success_rate(
                        transfer_rows,
                        role=role,
                        before_step=step_i,
                        source_game_key=source_game_key,
                        source_context_key=source_context_key,
                        target_game_key=target_game_key,
                        target_context_key=target_context_key,
                        transfer_history=transfer_history,
                        rate_cache=rate_cache,
                    )
                    if count > 0:
                        scoped_rates.append(rate)
            if scoped_cache is not None:
                scoped_cache[scope_key] = scoped_rates
                _inc(ctx, "scoped_misses")
                if not scoped_rates:
                    _inc(ctx, "scoped_empty")

    if ctx is not None:
        index_stats = ctx.setdefault("index_stats", {})
        index_stats["sparse_cache.generic_entries"] = len(generic_cache or {})
        index_stats["sparse_cache.scoped_entries"] = len(scoped_cache or {})
        _add_timing(ctx, "role_score_bundle.sparse_total", time.perf_counter() - started)
    return generic_rates, scoped_rates


def _profiled_prior_role_success_rate(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast

    ctx = fast._ACTIVE.get()
    started = time.perf_counter()
    try:
        return _ORIGINAL_PRIOR_RATE(*args, **kwargs)
    finally:
        if ctx is not None:
            _inc(ctx, "prior_role_success_rate_calls")
            _add_timing(ctx, "prior_role_success_rate.total", time.perf_counter() - started)


def _profiled_functional_diagnostics(*args: Any, **kwargs: Any):
    from v6 import concept_validation_fastpath as fast

    ctx = fast._ACTIVE.get()
    started = time.perf_counter()
    result = _ORIGINAL_FUNCTIONAL_DIAGNOSTICS(*args, **kwargs)
    elapsed = time.perf_counter() - started
    if ctx is None:
        return result

    candidate = str(kwargs.get("candidate_signature") or "unknown")
    events = result[0] if isinstance(result, tuple) and result else []
    per_concept = ctx.setdefault("per_concept", {})
    record = per_concept.setdefault(candidate, {"calls": 0, "seconds": 0.0})
    record["calls"] = int(record.get("calls", 0)) + 1
    record["seconds"] = float(record.get("seconds", 0.0) or 0.0) + elapsed

    type_counts: dict[str, dict[str, int]] = {}
    for event in events if isinstance(events, list) else []:
        event_type = str(event.get("event_type") or "unknown")
        item = type_counts.setdefault(event_type, {"examined": 0, "relevant": 0, "accepted": 0, "invalid": 0})
        item["examined"] += 1
        if bool(event.get("is_relevant")):
            item["relevant"] += 1
        if bool(event.get("invalid")):
            item["invalid"] += 1
        elif bool(event.get("is_relevant")):
            item["accepted"] += 1
    record["event_types"] = type_counts
    record["events_examined"] = sum(item["examined"] for item in type_counts.values())
    record["events_relevant"] = sum(item["relevant"] for item in type_counts.values())
    record["events_accepted"] = sum(item["accepted"] for item in type_counts.values())
    transfer = type_counts.get("transfer", {})
    record["transfer_rows_examined"] = int(transfer.get("examined", 0))
    record["transfer_rows_accepted"] = int(transfer.get("accepted", 0))
    _add_timing(ctx, "functional_diagnostics.total", elapsed)
    _inc(ctx, "functional_diagnostics_calls")
    return result


def _merge_profile(profile: dict[str, Any], ctx: dict[str, Any], elapsed: float) -> dict[str, Any]:
    merged = dict(profile)
    merged.setdefault("outer_total_seconds", elapsed)
    timings = dict(merged.get("timings") or {})
    for key, value in dict(ctx.get("timings") or {}).items():
        timings[key] = max(float(timings.get(key, 0.0) or 0.0), float(value))
    merged["timings"] = timings

    call_counts = dict(merged.get("call_counts") or {})
    for key, value in dict(ctx.get("call_counts") or {}).items():
        call_counts[key] = max(int(call_counts.get(key, 0) or 0), int(value))
    merged["call_counts"] = call_counts

    index_stats = dict(merged.get("index_stats") or {})
    index_stats.update(dict(ctx.get("index_stats") or {}))
    merged["index_stats"] = index_stats
    merged["event_counts"] = dict(ctx.get("event_counts") or merged.get("event_counts") or {})
    merged["sparse_cache"] = {
        "generic_entries": len(ctx.get("generic_role_score_cache") or {}),
        "scoped_entries": len(ctx.get("scoped_role_score_cache") or {}),
        **{key: int(value) for key, value in sorted(dict(ctx.get("cache_stats") or {}).items())},
    }
    merged["per_concept"] = dict(ctx.get("per_concept") or {})
    merged["worker"] = {
        "pid": os.getpid(),
        "seconds": float(elapsed),
        "concept_count": len(merged["per_concept"]),
    }
    merged["role_score_cache_entries"] = len(ctx.get("role_score_cache") or {})
    return merged


def _profiled_validate(*args: Any, **kwargs: Any):
    global _LAST_PROFILE
    from v6 import concept_validation_fastpath as fast

    if fast._ACTIVE.get() is not None:
        return _ORIGINAL(*args, **kwargs)

    config = kwargs.get("config")
    if config is None and len(args) > 1:
        config = args[1]
    if config is not None and not bool(getattr(config, "enabled", False)):
        return _ORIGINAL(*args, **kwargs)

    validate_roles_and_concepts = bool(kwargs.get("validate_roles_and_concepts", False))
    if not validate_roles_and_concepts:
        return _ORIGINAL(*args, **kwargs)

    ctx: dict[str, Any] = {
        "cache": {},
        "role_score_cache": {},
        "generic_role_score_cache": {},
        "scoped_role_score_cache": {},
        "timings": {},
        "call_counts": {},
        "event_counts": {},
        "index_stats": {},
        "cache_stats": {},
        "per_concept": {},
    }
    token = fast._ACTIVE.set(ctx)
    started = time.perf_counter()
    try:
        result = _ORIGINAL(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        fast._ACTIVE.reset(token)

    if isinstance(result, dict):
        existing = result.get("concept_validation_fastpath_profile")
        profile = _merge_profile(dict(existing) if isinstance(existing, dict) else {}, ctx, elapsed)
        result["concept_validation_fastpath_profile"] = profile
        _LAST_PROFILE = profile
    return result


def get_last_concept_validation_profile() -> dict[str, Any]:
    return dict(_LAST_PROFILE)


def install_concept_validation_sparse_cache() -> None:
    global _INSTALLED, _ORIGINAL, _ORIGINAL_PRIOR_RATE, _ORIGINAL_FUNCTIONAL_DIAGNOSTICS
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    fast._role_score_bundle = _sparse_role_score_bundle

    _ORIGINAL_PRIOR_RATE = substrate._prior_role_success_rate
    substrate._prior_role_success_rate = _profiled_prior_role_success_rate

    _ORIGINAL_FUNCTIONAL_DIAGNOSTICS = substrate._build_functional_explanation_diagnostics
    substrate._build_functional_explanation_diagnostics = _profiled_functional_diagnostics

    _ORIGINAL = substrate.validate_incremental_promotions_only
    substrate.validate_incremental_promotions_only = _profiled_validate
    suite.validate_incremental_promotions_only = _profiled_validate
    _INSTALLED = True
