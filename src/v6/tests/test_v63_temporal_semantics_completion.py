from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.carrier_emergence import CarrierEmergenceTracker
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory.v63_temporal_semantics_completion import (
    _repair_fold_threshold_timing,
    _stable_transformation_family_threshold_step,
    install_v63_temporal_semantics_completion,
)
from v6.v63_higher_order_semantics import install_v63_higher_order_semantics
from v6.v63_semantics import normalize_h04_result


def _sampling_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "ez01" / "mixed" / "steps_1000" / "seed_0.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sampling_metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE prediction_results (
                interaction_id INTEGER PRIMARY KEY,
                global_step INTEGER,
                actual_family INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO sampling_metadata(key, value) VALUES (?, ?)",
            [
                ("transformation_family_stable_support", json.dumps(5)),
                ("global_step_offset", json.dumps(0)),
            ],
        )
        connection.executemany(
            "INSERT INTO prediction_results(interaction_id, global_step, actual_family) VALUES (?, ?, ?)",
            [
                (1, 1, 7),
                (2, 2, 7),
                (3, 3, 7),
                (4, 4, 7),
                (5, 5, 7),
                (6, 6, 8),
            ],
        )
        connection.commit()
    db_path.with_name("carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "carrier:object_id:1",
                    "carrier_signature": "object_id:1",
                    "carrier_source": "object",
                    "support_count": 4,
                    "distinct_family_count": 2,
                    "first_seen_global_step": 1,
                    "first_emergent_global_step": 8,
                    "last_seen_global_step": 10,
                    "status": "emergent_carrier",
                },
                {
                    "carrier_id": "carrier:fallback",
                    "carrier_signature": "fallback",
                    "carrier_source": "context_action_fallback",
                    "support_count": 10,
                    "first_seen_global_step": 1,
                    "first_emergent_global_step": 2,
                    "last_seen_global_step": 10,
                    "status": "emergent_carrier",
                },
            ]
        ),
        encoding="utf-8",
    )
    return db_path


def test_fold_repairs_h03_h04_threshold_crossing_milestones(tmp_path: Path) -> None:
    db_path = _sampling_db(tmp_path)
    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as state_conn:
        state_conn.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count,
                linked_family_count, first_seen_global_step, last_seen_global_step,
                carrier_timing_source, stability_score, is_emergent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "carrier:object_id:1",
                "object_id:1",
                "object",
                4,
                2,
                8,
                10,
                "real_evidence",
                0.8,
                1,
            ),
        )
        repaired = _repair_fold_threshold_timing(
            db_path=db_path,
            state_conn=state_conn,
        )
        milestone = state_conn.execute(
            """
            SELECT first_stable_transformation_family_step,
                   first_emergent_carrier_step
            FROM temporal_milestones
            WHERE game='ez01' AND sampler='mixed' AND seed=0
            """
        ).fetchone()
        carrier_step = state_conn.execute(
            "SELECT first_seen_global_step FROM carrier_candidates WHERE carrier_signature='object_id:1'"
        ).fetchone()[0]

    assert _stable_transformation_family_threshold_step(db_path) == 5
    assert repaired == {
        "first_stable_transformation_family_step": 5,
        "first_emergent_carrier_step": 8,
    }
    assert milestone == (5, 8)
    assert carrier_step == 8


def test_restored_emergent_carrier_keeps_persisted_threshold_step() -> None:
    install_v63_higher_order_semantics()
    install_v63_temporal_semantics_completion()
    tracker = CarrierEmergenceTracker(
        min_support=3,
        min_distinct_contexts=2,
        min_prediction_lift=0.05,
        min_compression_gain=0.01,
    )
    tracker.import_candidate(
        carrier_signature="object_id:restored",
        carrier_source="object",
        support_count=4,
        linked_family_count=2,
        first_seen_global_step=8,
        last_seen_global_step=20,
        stability_score=0.8,
        is_emergent=True,
    )
    candidate = tracker._build_candidate("object_id:restored")
    assert candidate.status == "emergent_carrier"
    assert candidate.first_emergent_global_step == 8


def test_h04_strict_order_accepts_actual_threshold_precedence() -> None:
    result = {
        "decision": "VALID",
        "carrier_timing_source": "real_evidence",
        "first_stable_transformation_family_step": 5,
        "first_emergent_carrier_step": 8,
        "first_usable_emergent_carrier_step": 8,
        "core_metrics": {},
        "missing_evidence": [],
    }
    normalized = normalize_h04_result(result)
    assert normalized["decision"] == "VALID"
    assert normalized["h03_before_h04"] is True
    assert normalized["h03_before_h04_usable"] is True
    assert normalized["temporal_order_comparison"] == "strict_before"
