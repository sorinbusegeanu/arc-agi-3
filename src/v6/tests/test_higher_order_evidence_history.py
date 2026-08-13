from __future__ import annotations

import sqlite3
from pathlib import Path

from v6 import higher_order_evidence_history as evidence
from v6 import concept_validation_history as validation_history


def _seed_minimal(memory_dir: Path) -> Path:
    memory_dir.mkdir()
    db = memory_dir / "current_state.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE memory_summary(key TEXT PRIMARY KEY, value_json TEXT)")
        conn.execute(
            "CREATE TABLE role_transfer_attempts (attempt_id TEXT PRIMARY KEY, role_signature TEXT, "
            "reuse_success INTEGER, last_seen_global_step INTEGER, source_game_key TEXT, "
            "source_context_key TEXT, target_game_key TEXT, target_context_key TEXT)"
        )
        conn.execute(
            "CREATE TABLE future_option_events (event_id TEXT PRIMARY KEY, source_role_id TEXT, "
            "owner_key TEXT, option_delta REAL, last_seen_global_step INTEGER)"
        )
        conn.execute(
            "CREATE TABLE future_option_motifs (motif_signature TEXT PRIMARY KEY, source_role_ids_json TEXT, "
            "motif_stability_score REAL, is_emergent INTEGER, last_seen_global_step INTEGER)"
        )
    return db


def test_role_candidate_stage_archives_transfer_before_destructive_clear(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db = _seed_minimal(memory_dir)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('old','r1',1,10,'g1','c1','g2','c2')")
        conn.commit()

    def destructive_role_stage(*args, **kwargs):
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM role_transfer_attempts")
            conn.commit()
        return {"role_candidate_count": 0}

    monkeypatch.setattr(evidence, "_ORIGINAL_ROLE_CANDIDATES", destructive_role_stage)
    result = evidence._derive_role_candidates(memory_dir=memory_dir)

    assert result["history_archive_boundary"] == "before_role_candidate_clear"
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            f"SELECT COUNT(*) FROM {validation_history.TRANSFER_HISTORY}"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0] == 0


def test_transfer_cap_is_per_derivation_budget_and_history_is_restored(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db = _seed_minimal(memory_dir)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('old','r1',1,10,'g1','c1','g2','c2')")
        conn.commit()
    validation_history.archive_validation_evidence(memory_dir)
    seen = {}

    def fake_transfer(*args, **kwargs):
        seen["max_transfer_attempts"] = kwargs["max_transfer_attempts"]
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM role_transfer_attempts")
            conn.execute("INSERT INTO role_transfer_attempts VALUES ('old','r1',1,10,'g1','c1','g2','c2')")
            conn.execute("INSERT INTO role_transfer_attempts VALUES ('new','r1',1,20,'g1','c1','g3','c3')")
            conn.commit()
        return {"transfer_attempt_count": 2}

    monkeypatch.setattr(evidence, "_ORIGINAL_ROLE_TRANSFERS", fake_transfer)
    result = evidence._derive_role_transfers(memory_dir=memory_dir, max_transfer_attempts=1)

    assert seen["max_transfer_attempts"] == 2
    assert result["configured_transfer_attempt_budget"] == 1
    assert result["historical_transfer_attempt_count_before"] == 1
    assert result["new_transfer_attempt_count_this_derivation"] == 1
    assert result["cumulative_transfer_attempt_count"] == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0] == 2
        assert conn.execute(
            f"SELECT COUNT(*) FROM {validation_history.TRANSFER_HISTORY}"
        ).fetchone()[0] == 2


def test_future_caps_advance_beyond_history_and_restore_latest_motif(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db = _seed_minimal(memory_dir)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO future_option_events VALUES ('e1','r1','r1',1.0,10)")
        conn.execute("INSERT INTO future_option_motifs VALUES ('m1','[\"r1\"]',0.5,0,10)")
        conn.commit()
    validation_history.archive_validation_evidence(memory_dir)
    seen = {}

    def fake_future(*args, **kwargs):
        seen["max_events"] = kwargs["max_events"]
        seen["max_motifs"] = kwargs["max_motifs"]
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM future_option_events")
            conn.execute("DELETE FROM future_option_motifs")
            conn.execute("INSERT INTO future_option_events VALUES ('e1','r1','r1',1.0,10)")
            conn.execute("INSERT INTO future_option_events VALUES ('e2','r1','r1',-1.0,20)")
            conn.execute("INSERT INTO future_option_motifs VALUES ('m1','[\"r1\"]',0.9,1,20)")
            conn.execute("INSERT INTO future_option_motifs VALUES ('m2','[\"r1\"]',0.7,1,20)")
            conn.commit()
        return {"future_option_event_count": 2, "future_option_motif_count": 2}

    monkeypatch.setattr(evidence, "_ORIGINAL_FUTURE_OPTIONS", fake_future)
    result = evidence._derive_future_options(memory_dir=memory_dir, max_events=1, max_motifs=1)

    assert seen == {"max_events": 2, "max_motifs": 2}
    assert result["historical_future_option_event_count_before"] == 1
    assert result["new_future_option_event_count_this_derivation"] == 1
    assert result["cumulative_future_option_event_count"] == 2
    assert result["cumulative_future_option_motif_count"] == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM future_option_motifs").fetchone()[0] == 2
        assert conn.execute(
            "SELECT motif_stability_score,last_seen_global_step FROM future_option_motifs WHERE motif_signature='m1'"
        ).fetchone() == (0.9, 20)
