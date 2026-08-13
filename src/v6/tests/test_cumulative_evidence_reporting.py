from __future__ import annotations

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
