from __future__ import annotations

"""Broaden explicit transfer evaluation without broadening autonomous M7 control.

A strategy must still pass the existing causal/probe, provenance and target structural
or normalized-grounding gates. This layer only removes the circular requirement that
it already appear in the cross-context fallback list before its first explicit
cross-game transfer probe can be evaluated.
"""


_INSTALLED = False
_BASE_GROUNDED_TRANSFER = None
_BASE_ORDERED_SEQUENCES = None


def _all_exact_strategy_rows(view):
    refresh = getattr(view, "_refresh_strategy_cache", None)
    if callable(refresh):
        refresh()
    unique = {}
    for rows in getattr(view, "_strategy_by_context", {}).values():
        for row in rows:
            unique[row.strategy_uid] = row
    for row in getattr(view, "_strategy_fallback", ()):
        unique[row.strategy_uid] = row
    return tuple(
        sorted(
            unique.values(),
            key=lambda row: (
                int(row.strategy_uid.hi),
                int(row.strategy_uid.lo),
            ),
        )
    )


def _with_expanded_fallback(view, callback, *, cache_attr: str):
    original = tuple(getattr(view, "_strategy_fallback", ()))
    expanded = _all_exact_strategy_rows(view)
    if expanded == original:
        return callback()
    prior_cache = getattr(view, cache_attr, None)
    view._strategy_fallback = expanded
    try:
        setattr(view, cache_attr, None)
        return callback()
    finally:
        view._strategy_fallback = list(original) if isinstance(getattr(view, "_strategy_fallback", None), list) else original
        # Do not restore an old cache built from the narrower candidate pool. The
        # result produced during this explicit transfer call remains the valid cache.
        if getattr(view, cache_attr, None) is None and prior_cache is not None:
            setattr(view, cache_attr, prior_cache)


def _grounded_transfer_candidates_v856(view, game_id: str):
    return _with_expanded_fallback(
        view,
        lambda: _BASE_GROUNDED_TRANSFER(view, game_id),
        cache_attr="_v837_transfer_index_key",
    )


def _ordered_transfer_candidates_v856(view, game_id: str):
    return _with_expanded_fallback(
        view,
        lambda: _BASE_ORDERED_SEQUENCES(view, game_id),
        cache_attr="_v854_ordered_key",
    )


def install_adaptive_memory_transfer_candidate_pool_v856() -> None:
    global _INSTALLED, _BASE_GROUNDED_TRANSFER, _BASE_ORDERED_SEQUENCES
    if _INSTALLED:
        return

    from v8 import environment_neutrality_v837 as v837
    from v8 import learning_transfer_correctness_v854 as v854

    _BASE_GROUNDED_TRANSFER = v837._grounded_transfer_index
    _BASE_ORDERED_SEQUENCES = v854._ordered_sequences

    v837._grounded_transfer_index = _grounded_transfer_candidates_v856
    v854._ordered_sequences = _ordered_transfer_candidates_v856

    _INSTALLED = True
