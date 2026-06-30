from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence
from v6.memory.compact_memory import ensure_memory_layout


def _build_memory(memory_dir: Path, *, fallback_only: bool = False) -> None:
    memory = ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory.current_state) as connection:
        connection.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, stability_score, is_emergent, carrier_timing_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "carrier-a",
                "carrier-a",
                "context_action_fallback" if fallback_only else "object",
                4,
                2,
                8,
                10,
                0.5,
                1,
                "real_evidence",
            ),
        )
        connection.execute(
            """
            INSERT INTO temporal_milestones (
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step, first_stable_contingency_step,
                first_prediction_violation_step, first_high_replay_priority_step, first_transformation_family_step,
                first_stable_transformation_family_step, first_carrier_candidate_step, first_emergent_carrier_step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("tt01", "mixed", 0, 1, 2, 3, 4, 5, 3, 6, 7, 8),
        )
        connection.execute("INSERT INTO carrier_links VALUES (?, ?, ?, ?, ?, ?)", ("carrier-a", "family", "family-a", 1, 5, 10))
        connection.execute("INSERT INTO carrier_links VALUES (?, ?, ?, ?, ?, ?)", ("carrier-a", "family", "family-b", 1, 5, 10))
        connection.execute("INSERT INTO carrier_links VALUES (?, ?, ?, ?, ?, ?)", ("carrier-a", "context", "ctx-a", 1, 5, 10))
        connection.execute("INSERT INTO carrier_links VALUES (?, ?, ?, ?, ?, ?)", ("carrier-a", "context", "ctx-b", 1, 5, 10))
        connection.commit()
    with sqlite3.connect(memory.graph) as connection:
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("carrier:carrier-a", "carrier", "carrier-a", 5, 10, 4))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("family:family-a", "family", "family-a", 3, 10, 4))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("context:ctx-a", "context", "ctx-a", 1, 10, 4))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("carrier:carrier-a->explains->family:family-a", "carrier:carrier-a", "family:family-a", "explains", 5, 10, 4, 1.0))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("carrier:carrier-a->anchors->contingency:c1", "carrier:carrier-a", "contingency:c1", "anchors", 5, 10, 4, 1.0))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("carrier:carrier-a->appears_in->context:ctx-a", "carrier:carrier-a", "context:ctx-a", "appears_in", 5, 10, 4, 1.0))
        connection.commit()
    memory.summary_json.write_text(json.dumps({"fold_summary": {"carrier_candidates_added": 1}}, indent=2), encoding="utf-8")


def test_h04_returns_partially_valid_or_valid_from_compact_memory(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _build_memory(memory_dir)

    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "out")

    assert result["decision"] == "VALID"
    assert result["core_metrics"]["carrier_candidate_count"] == 1
    assert result["core_metrics"]["carrier_cross_family_count"] >= 2
    assert result["core_metrics"]["carrier_cross_context_count"] >= 2
    assert result["core_metrics"]["carrier_timing_source"] == "real_evidence"
    assert (tmp_path / "out" / "h04_carrier_emergence_report.json").exists()


def test_h04_rejects_context_action_fallback_as_emergent(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _build_memory(memory_dir, fallback_only=True)

    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "out")

    assert result["decision"] == "INVALID"
    assert result["core_metrics"]["emergent_context_action_fallback_count"] == 1


def test_h04_without_links_is_only_partially_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _build_memory(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as connection:
        connection.execute("DELETE FROM carrier_links")
        connection.commit()

    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "out")

    assert result["decision"] == "PARTIALLY_VALID"
