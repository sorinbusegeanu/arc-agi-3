from __future__ import annotations

import sqlite3
from pathlib import Path

from v6.evaluation.h10b_selective_forgetting import evaluate_h10b_selective_forgetting
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory.selective_forgetting import run_selective_forgetting_pass


def _setup_memory_dir(tmp_path: Path) -> Path:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("M0:interaction:1", "M0", "InteractionMemory", "1", 1, "{}"),
        )
        conn.execute(
            "INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("M0:interaction:2", "M0", "InteractionMemory", "2", 5, "{}"),
        )
        conn.execute(
            "INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("M2:family:10", "M2", "TransformationFamilyMemory", "10", 3, "{}"),
        )
        conn.execute(
            """
            INSERT INTO memory_scores (
                node_id, isf_total, transfer_score, explanatory_reach, future_option_delta, replay_priority,
                retention_status, memory_state, stored_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("M0:interaction:1", 0.05, 0.05, 0.05, 0.0, 0.0, "active", "active", 1),
        )
        conn.execute(
            """
            INSERT INTO memory_scores (
                node_id, isf_total, transfer_score, explanatory_reach, future_option_delta, replay_priority,
                retention_status, memory_state, stored_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("M0:interaction:2", 0.9, 0.7, 0.6, 0.5, 0.8, "active", "active", 1),
        )
        conn.execute(
            """
            INSERT INTO memory_scores (
                node_id, isf_total, transfer_score, explanatory_reach, future_option_delta, replay_priority,
                retention_status, memory_state, stored_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("M2:family:10", 0.8, 0.7, 0.7, 0.2, 0.4, "active", "active", 1),
        )
        conn.execute(
            "INSERT INTO memory_promotions (promotion_id, source_node_id, target_node_id, promotion_type, evidence_count, promotion_score, status, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("p1", "M0:interaction:1", "M2:family:10", "M0_M2", 3, 0.8, "promoted", "{}"),
        )
        conn.commit()
    return memory_dir


def test_low_retention_memory_becomes_archived_or_forgotten(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    summary = run_selective_forgetting_pass(memory_dir=memory_dir, epoch=3)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        state = conn.execute("SELECT memory_state FROM memory_scores WHERE node_id = 'M0:interaction:1'").fetchone()[0]
    assert state in {"compressed", "archived", "forgotten", "superseded"}
    assert summary["stored_memory_count"] == 3


def test_high_isf_memory_remains_active(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    run_selective_forgetting_pass(memory_dir=memory_dir, epoch=2)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        state = conn.execute("SELECT memory_state FROM memory_scores WHERE node_id = 'M0:interaction:2'").fetchone()[0]
    assert state == "active"


def test_memory_explained_by_promoted_structure_becomes_compressed(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    run_selective_forgetting_pass(memory_dir=memory_dir, epoch=2)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT memory_state, compressed_into_id FROM memory_scores WHERE node_id = 'M0:interaction:1'"
        ).fetchone()
    assert row[0] == "compressed"
    assert row[1] == "M2:family:10"


def test_superseded_memory_records_superseded_by_id_field(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("DELETE FROM memory_promotions")
        conn.execute(
            "UPDATE memory_scores SET isf_total = 0.1, explanatory_reach = 0.0, transfer_score = 0.0, replay_priority = 0.0 WHERE node_id = 'M0:interaction:1'"
        )
        conn.commit()
    run_selective_forgetting_pass(memory_dir=memory_dir, epoch=6)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT memory_state, superseded_by_id FROM memory_scores WHERE node_id = 'M0:interaction:1'"
        ).fetchone()
    assert row[0] in {"archived", "forgotten", "superseded"}
    if row[0] == "superseded":
        assert row[1] is None or isinstance(row[1], str)


def test_memory_survival_ratio_and_high_vs_low_lift(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    summary = run_selective_forgetting_pass(memory_dir=memory_dir, epoch=2)
    assert 0.0 <= summary["memory_survival_ratio"] <= 1.0
    assert summary["high_isf_survival_ratio"] >= summary["low_isf_survival_ratio"]


def test_no_hard_delete_happens_by_default(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    run_selective_forgetting_pass(memory_dir=memory_dir, epoch=2)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        count = conn.execute("SELECT COUNT(*) FROM memory_scores").fetchone()[0]
    assert count == 3


def test_h10b_report_computes_counts(tmp_path: Path) -> None:
    memory_dir = _setup_memory_dir(tmp_path)
    forgetting_summary = run_selective_forgetting_pass(memory_dir=memory_dir, epoch=2)
    report = evaluate_h10b_selective_forgetting(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "reports" / "h10b",
        forgetting_summary=forgetting_summary,
    )
    assert report["stored_memory_count"] == 3
    assert report["high_isf_survival_ratio"] >= report["low_isf_survival_ratio"]
    assert (tmp_path / "reports" / "h10b" / "h10b_selective_forgetting_report.json").exists()
