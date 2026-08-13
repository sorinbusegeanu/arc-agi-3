from __future__ import annotations

import sqlite3
from pathlib import Path

from v6 import concept_validation_history as history


def _seed_tables(db: Path) -> None:
    with sqlite3.connect(db) as conn:
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
            "CREATE TABLE future_option_motifs (motif_signature TEXT, source_role_ids_json TEXT, "
            "motif_stability_score REAL, is_emergent INTEGER, last_seen_global_step INTEGER)"
        )


def test_validation_history_survives_operational_table_replacement(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db = memory_dir / "current_state.sqlite"
    _seed_tables(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('a1','r1',1,10,'g1','c1','g2','c2')")
        conn.execute("INSERT INTO future_option_events VALUES ('e1','r1','r1',1.0,10)")
        conn.execute("INSERT INTO future_option_motifs VALUES ('m1','[\"r1\"]',0.8,1,10)")
        conn.commit()
    history.archive_validation_evidence(memory_dir)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM role_transfer_attempts")
        conn.execute("DELETE FROM future_option_events")
        conn.execute("DELETE FROM future_option_motifs")
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('a2','r1',0,20,'g1','c1','g3','c3')")
        conn.execute("INSERT INTO future_option_events VALUES ('e2','r1','r1',-1.0,20)")
        conn.execute("INSERT INTO future_option_motifs VALUES ('m1','[\"r1\"]',0.9,1,20)")
        conn.commit()
    history.archive_validation_evidence(memory_dir)
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {history.TRANSFER_HISTORY}").fetchone()[0] == 2
        assert conn.execute(f"SELECT COUNT(*) FROM {history.FUTURE_HISTORY}").fetchone()[0] == 2
        assert conn.execute(f"SELECT COUNT(*) FROM {history.MOTIF_HISTORY}").fetchone()[0] == 2


def test_diagnostics_receive_historical_and_current_transfer_rows(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "state.sqlite"
    _seed_tables(db)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('old','r1',1,10,'g1','c1','g2','c2')")
        conn.execute("INSERT INTO future_option_events VALUES ('old-e','r1','r1',1.0,10)")
        conn.commit()
        history._archive(conn, "role_transfer_attempts", history.TRANSFER_HISTORY, "attempt_id")
        history._archive(conn, "future_option_events", history.FUTURE_HISTORY, "event_id")
        conn.execute("DELETE FROM role_transfer_attempts")
        conn.execute("DELETE FROM future_option_events")
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('new','r1',0,20,'g1','c1','g3','c3')")
        conn.execute("INSERT INTO future_option_events VALUES ('new-e','r1','r1',-1.0,20)")
        conn.commit()
        current_transfer = conn.execute("SELECT * FROM role_transfer_attempts").fetchall()
        current_future = conn.execute("SELECT * FROM future_option_events").fetchall()
        seen = {}

        def fake(*args, **kwargs):
            seen["transfer"] = len(kwargs["transfer_rows"])
            seen["future"] = len(kwargs["future_rows"])
            return [], {}, {}

        monkeypatch.setattr(history, "_ORIGINAL_DIAGNOSTICS", fake)
        _events, diagnostics, _state = history._diagnostics(
            state_conn=conn,
            transfer_rows=current_transfer,
            future_rows=current_future,
            transfer_history=None,
        )
        assert seen == {"transfer": 2, "future": 2}
        assert diagnostics["validation_history_applied"] is True
        assert diagnostics["validation_transfer_history_row_count"] == 2
        assert diagnostics["validation_future_option_history_row_count"] == 2
