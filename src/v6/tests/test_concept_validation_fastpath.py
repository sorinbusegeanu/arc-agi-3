from __future__ import annotations

import sqlite3
from collections import defaultdict

from v6 import concept_validation_fastpath as fast
from v6 import concept_validation_fastpath_compat as compat
from v6 import higher_order_substrate as substrate
from v6 import hypothesis_suite_report as suite


def _prediction_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE prediction_results ("
        "id INTEGER PRIMARY KEY, global_step INTEGER, context_signature TEXT, "
        "predicted_family TEXT, actual_family TEXT, context_contradiction INTEGER)"
    )
    conn.executemany(
        "INSERT INTO prediction_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 10, "c1", "f1", "f1", 0),
            (2, 20, "c2", "f1", "f2", 1),
            (3, 30, "c3", "f2", "f2", 0),
        ],
    )
    return conn


def test_fastpath_is_installed_without_overriding_canonical_prediction_semantics() -> None:
    assert fast._INSTALLED is True
    assert compat._INSTALLED is True
    assert substrate._transfer_explanation_events is compat._safe_transfer
    assert substrate._future_option_motif_explanation_events is compat._future_option_motif_explanation_events
    assert substrate.validate_incremental_promotions_only is compat._validate
    assert suite.validate_incremental_promotions_only is compat._validate
    assert substrate._prediction_explanation_events is not fast._prediction_explanation_events


def test_prediction_row_index_is_built_once_per_validation_context() -> None:
    conn = _prediction_db()
    ctx = {
        "cache": {}, "role_score_cache": {}, "timings": {}, "call_counts": {},
        "event_counts": defaultdict(int), "index_stats": {},
    }
    token = fast._ACTIVE.set(ctx)
    try:
        first = fast._prediction_rows(conn)
        second = fast._prediction_rows(conn)
    finally:
        fast._ACTIVE.reset(token)
        conn.close()
    assert first is second
    assert ctx["index_stats"]["prediction_rows"] == 3
    assert ctx["index_stats"]["contradiction_rows"] == 1
    assert len([key for key in ctx["cache"] if key[0] == "prediction_rows"]) == 1


def test_transfer_step_index_uses_strictly_later_rows() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE t (attempt_id TEXT, role_signature TEXT, reuse_success INTEGER, "
        "last_seen_global_step INTEGER, observed_role_signature TEXT, predicted_role_signature TEXT, "
        "source_carrier_signature TEXT, target_carrier_signature TEXT, source_game_key TEXT, "
        "target_game_key TEXT, source_context_key TEXT, target_context_key TEXT)"
    )
    conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("a", "r", 1, 10, "x", "x", "c1", "c2", "g1", "g2", "k1", "k2"),
            ("b", "r", 0, 20, "x", "x", "c1", "c2", "g1", "g2", "k1", "k2"),
            ("c", "r", 1, 30, "x", "x", "c1", "c2", "g1", "g2", "k1", "k2"),
        ],
    )
    rows = conn.execute("SELECT * FROM t").fetchall()
    steps, ordered = fast._transfer_step_rows(rows)
    from bisect import bisect_right
    start = bisect_right(steps, 20)
    assert [row["attempt_id"] for row in ordered[start:]] == ["c"]
    conn.close()


def test_disabled_validation_delegates_without_fastpath_side_effects(tmp_path, monkeypatch) -> None:
    class Config:
        enabled = False

    called = []
    monkeypatch.setitem(
        fast._ORIGINALS,
        "validate_incremental_promotions_only",
        lambda *args, **kwargs: called.append((args, kwargs)) or {"enabled": False},
    )
    result = compat._validate(memory_dir=tmp_path, config=Config(), validate_roles_and_concepts=True)
    assert result == {"enabled": False}
    assert len(called) == 1
