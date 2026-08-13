from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL: Any = None


def _validate_with_active_profile(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from v6 import concept_validation_fastpath as fast

    ctx = fast._ACTIVE.get()
    if ctx is None:
        return _ORIGINAL(*args, **kwargs)

    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    validate_roles_and_concepts = bool(kwargs.get("validate_roles_and_concepts"))
    frontier_before = fast._evidence_frontier(memory_dir) if validate_roles_and_concepts else {}
    previous_frontier = fast._load_frontier(memory_dir) if validate_roles_and_concepts else None

    ctx.setdefault("cache", {})
    ctx.setdefault("role_score_cache", {})
    ctx.setdefault("timings", {})
    ctx.setdefault("call_counts", {})
    events = ctx.get("event_counts")
    if not isinstance(events, defaultdict):
        ctx["event_counts"] = defaultdict(int, dict(events or {}))
    ctx.setdefault("index_stats", {})

    started = time.perf_counter()
    result = fast._ORIGINALS["validate_incremental_promotions_only"](*args, **kwargs)
    elapsed = time.perf_counter() - started
    if not isinstance(result, dict):
        return result

    result["concept_validation_fastpath_profile"] = {
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
    if validate_roles_and_concepts:
        fast._store_frontier(memory_dir, frontier_before)
    return result


def install_concept_validation_profiler_context_fix() -> None:
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath as fast

    _ORIGINAL = fast._validate_incremental_promotions_only
    fast._validate_incremental_promotions_only = _validate_with_active_profile
    _INSTALLED = True
