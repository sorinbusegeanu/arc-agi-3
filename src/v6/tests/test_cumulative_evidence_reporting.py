from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.reporting import evidence_snapshot


def test_report_projection_separates_current_and_history(tmp_path: Path) -> None:
    db = tmp_path / "current_state.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE role_transfer_attempts (attempt_id TEXT PRIMARY KEY, reuse_success INTEGER)")
        conn.execute("CREATE TABLE concept_validation_role_transfer_history (attempt_id TEXT PRIMARY KEY, reuse_success INTEGER)")
        conn.execute("INSERT INTO role_transfer_attempts VALUES ('new',1)")
        conn.executemany("INSERT INTO concept_validation_role_transfer_history VALUES (?,1)", [("old",), ("new",)])
        conn.commit()
    projection = evidence_snapshot._materialize_cumulative_reporting_projection(db)
    assert projection["role_transfer_attempts"] == 2
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0] == 2
        assert conn.execute("SELECT attempt_id FROM report_current_role_transfer_attempts").fetchall() == [("new",)]


def test_report_projection_revalidates_h11_chain(tmp_path: Path) -> None:
    db = tmp_path / "current_state.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE role_transfer_attempts (attempt_id TEXT PRIMARY KEY, source_role_signature TEXT, source_game_key TEXT, target_game_key TEXT, source_context_key TEXT, target_context_key TEXT, provenance_mode TEXT, provenance_status TEXT);
            CREATE TABLE concept_validation_role_transfer_history AS SELECT * FROM role_transfer_attempts WHERE 0;
            CREATE TABLE future_option_motifs (motif_signature TEXT PRIMARY KEY, last_seen_global_step INTEGER, provenance_status TEXT);
            CREATE TABLE concept_validation_future_option_motif_history AS SELECT * FROM future_option_motifs WHERE 0;
            CREATE TABLE future_option_transfer_links (motif_signature TEXT, role_signature TEXT, source_role_signature TEXT, concept_signature TEXT, source_game_key TEXT, target_game_key TEXT, source_context_key TEXT, target_context_key TEXT, motif_provenance_status TEXT, transfer_provenance_status TEXT, concept_validation_status TEXT, promoted_concept_count INTEGER);
            CREATE TABLE higher_order_future_option_transfer_history AS SELECT * FROM future_option_transfer_links WHERE 0;
            CREATE TABLE concept_promotion_validation_diagnostics (concept_signature TEXT, payload_json TEXT);
        """)
        conn.execute("INSERT INTO concept_validation_role_transfer_history VALUES (?,?,?,?,?,?,?,?)", ("a1","r1","g1","g2","c1","c2","single_source","verified"))
        conn.execute("INSERT INTO concept_validation_future_option_motif_history VALUES (?,?,?)", ("m1",20,"verified"))
        conn.execute("INSERT INTO higher_order_future_option_transfer_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ("m1","r1","r1","concept-1","g1","g2","c1","c2","proxy","proxy","proxy",0))
        conn.execute("INSERT INTO concept_promotion_validation_diagnostics VALUES (?,?)", ("concept-1",json.dumps({"promoted":True})))
        conn.commit()
    evidence_snapshot._materialize_cumulative_reporting_projection(db)
    with sqlite3.connect(db) as conn:
        row=conn.execute("SELECT motif_provenance_status,transfer_provenance_status,concept_validation_status,promoted_concept_count FROM future_option_transfer_links").fetchone()
    assert row == ("verified","verified","verified",1)
