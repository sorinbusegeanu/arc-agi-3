from __future__ import annotations

from collections import defaultdict

from v6 import concept_validation_fastpath as fast
from v6 import concept_validation_sparse_cache as sparse


class _History:
    by_source_target_scope: dict[tuple[str, str, str, str, str], object] = {}


def test_generic_role_rates_are_reused_across_distinct_scopes(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    class _Substrate:
        @staticmethod
        def _prior_role_success_rate(
            transfer_rows,
            *,
            role,
            before_step,
            transfer_history=None,
            rate_cache=None,
            **kwargs,
        ):
            calls.append((role, int(before_step)))
            return (0.75 if role == "role-a" else 0.25, 4)

    ctx = {
        "cache": {},
        "role_score_cache": {},
        "generic_role_score_cache": {},
        "scoped_role_score_cache": {},
        "timings": {},
        "call_counts": {},
        "event_counts": defaultdict(int),
        "index_stats": {},
        "cache_stats": {},
    }
    token = fast._ACTIVE.set(ctx)
    try:
        for index in range(100):
            generic, scoped = sparse._sparse_role_score_bundle(
                _Substrate,
                source_roles=["role-a", "role-b"],
                step=100,
                transfer_rows=[],
                transfer_history=_History(),
                rate_cache={},
                scope=("source", f"ctx-{index}", "target", f"target-{index}"),
            )
            assert generic == [0.75, 0.25]
            assert scoped == []
    finally:
        fast._ACTIVE.reset(token)

    assert calls == [("role-a", 100), ("role-b", 100)]
    assert ctx["cache_stats"]["generic_misses"] == 1
    assert ctx["cache_stats"]["generic_hits"] == 99
    assert ctx["cache_stats"]["scoped_misses"] == 100
    assert ctx["cache_stats"]["scoped_empty"] == 100
    assert len(ctx["generic_role_score_cache"]) == 1
    assert len(ctx["scoped_role_score_cache"]) == 100


def test_sparse_exact_scope_matches_transfer_history_series() -> None:
    class _Series:
        def rate_before(self, step: int):
            assert step == 50
            return 0.6, 5

    class _HistoryWithScope:
        by_source_target_scope = {
            ("role-a", "sg", "sc", "tg", "tc"): _Series(),
        }

    class _Substrate:
        @staticmethod
        def _prior_role_success_rate(*args, role, **kwargs):
            return (0.4 if role == "role-a" else 0.2, 10)

    ctx = {
        "cache": {},
        "role_score_cache": {},
        "generic_role_score_cache": {},
        "scoped_role_score_cache": {},
        "timings": {},
        "call_counts": {},
        "event_counts": defaultdict(int),
        "index_stats": {},
        "cache_stats": {},
    }
    token = fast._ACTIVE.set(ctx)
    try:
        generic, scoped = sparse._sparse_role_score_bundle(
            _Substrate,
            source_roles=["role-a", "role-b"],
            step=50,
            transfer_rows=[],
            transfer_history=_HistoryWithScope(),
            rate_cache={},
            scope=("sg", "sc", "tg", "tc"),
        )
    finally:
        fast._ACTIVE.reset(token)

    assert generic == [0.4, 0.2]
    assert scoped == [0.6]
