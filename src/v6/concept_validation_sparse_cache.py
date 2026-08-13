from __future__ import annotations

import time
from typing import Any

_INSTALLED = False
_ORIGINAL: Any = None


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
    """Equivalent role-score lookup with generic/scoped caches split.

    Generic role history depends only on (roles, step), not on the event scope.
    Scoped history is sparse in the transfer-history index, so missing exact
    role/scope series are skipped without invoking the generic lookup helper.
    """
    from v6 import concept_validation_fastpath as fast

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
        stats = ctx.setdefault("cache_stats", {})
        stats["generic_hits"] = int(stats.get("generic_hits", 0)) + 1
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
            stats = ctx.setdefault("cache_stats", {})
            stats["generic_misses"] = int(stats.get("generic_misses", 0)) + 1

    scoped_rates: list[float] = []
    if scope is not None:
        scope_key = (roles_key, step_i, *scope)
        if scoped_cache is not None and scope_key in scoped_cache:
            scoped_rates = scoped_cache[scope_key]
            stats = ctx.setdefault("cache_stats", {})
            stats["scoped_hits"] = int(stats.get("scoped_hits", 0)) + 1
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
                stats = ctx.setdefault("cache_stats", {})
                stats["scoped_misses"] = int(stats.get("scoped_misses", 0)) + 1
                if not scoped_rates:
                    stats["scoped_empty"] = int(stats.get("scoped_empty", 0)) + 1

    return generic_rates, scoped_rates


def _profiled_validate(*args: Any, **kwargs: Any):
    """Ensure profiling exists even when later runtime wrappers are installed."""
    from v6 import concept_validation_fastpath as fast

    if fast._ACTIVE.get() is not None:
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
        profile = dict(existing) if isinstance(existing, dict) else {}
        profile.setdefault("outer_total_seconds", elapsed)
        profile["sparse_cache"] = {
            "generic_entries": len(ctx["generic_role_score_cache"]),
            "scoped_entries": len(ctx["scoped_role_score_cache"]),
            **{key: int(value) for key, value in sorted(ctx["cache_stats"].items())},
        }
        # If the inner canonical fastpath did not create a profile, preserve
        # enough diagnostics to prove which helpers executed.
        if not existing:
            profile["timings"] = {key: float(value) for key, value in sorted(ctx["timings"].items())}
            profile["call_counts"] = {key: int(value) for key, value in sorted(ctx["call_counts"].items())}
            profile["index_stats"] = dict(ctx["index_stats"])
            profile["role_score_cache_entries"] = len(ctx["role_score_cache"])
        result["concept_validation_fastpath_profile"] = profile
    return result


def install_concept_validation_sparse_cache() -> None:
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    fast._role_score_bundle = _sparse_role_score_bundle

    _ORIGINAL = substrate.validate_incremental_promotions_only
    substrate.validate_incremental_promotions_only = _profiled_validate
    suite.validate_incremental_promotions_only = _profiled_validate
    _INSTALLED = True
