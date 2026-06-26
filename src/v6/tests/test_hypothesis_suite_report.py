from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.cli import main
from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
from v6.hypothesis_suite_report import SUITE_JSON_NAME, build_hypothesis_suite_summary
from v6.memory.compact_memory import ensure_memory_layout


def _write_suite_input_report(run_dir: Path) -> None:
    payload = {
        "games": ["tt01", "pb02"],
        "samplers": ["random_baseline", "mixed"],
        "seeds": [0],
        "runs": [
            {
                "game": "tt01",
                "sampler_name": "random_baseline",
                "run_status": "ok",
                "total_interactions": 100,
                "stable_contingency_count": 3,
                "prediction_accuracy": 0.7,
                "unique_transformation_families": 2,
                "mean_isf_prediction_error": 0.5,
                "high_priority_replay_count": 2,
            },
            {
                "game": "pb02",
                "sampler_name": "mixed",
                "run_status": "ok",
                "total_interactions": 120,
                "stable_contingency_count": 4,
                "prediction_accuracy": 0.8,
                "unique_transformation_families": 2,
                "mean_isf_prediction_error": 0.6,
                "high_priority_replay_count": 3,
            },
        ],
        "validation": {
            "mean_isf_total": 1.8,
            "max_isf_total": 3.2,
            "mean_isf_prediction_error": 0.6,
            "mean_isf_learning_value": 0.4,
            "mean_isf_transfer_potential": 0.3,
            "mean_isf_explanatory_potential": 0.2,
            "high_isf_interaction_count": 8,
            "context_contradiction_count": 6,
            "contradicted_context_count": 4,
            "contradicted_context_action_count": 5,
            "repeated_contradiction_count": 2,
            "context_expansion_suggested_count": 3,
            "memory_record_count": 12,
            "memory_replay_candidate_count": 10,
            "memory_mean_replay_priority": 0.53,
            "memory_max_replay_priority": 0.95,
            "high_priority_replay_count": 4,
            "carrier_candidate_count": 0,
            "emergent_carrier_count": 0,
            "carrier_spatial_candidate_count": 0,
            "carrier_object_candidate_count": 0,
            "emergent_object_carrier_count": 0,
            "carrier_context_action_fallback_candidate_count": 0,
            "emergent_context_action_fallback_count": 0,
        },
        "temporal_milestones": {
            "by_game_sampler_seed": [
                {
                    "game": "tt01",
                    "sampler": "random_baseline",
                    "seed": 0,
                    "first_interaction_step": 1,
                    "first_contingency_candidate_step": 2,
                    "first_stable_contingency_step": None,
                    "first_prediction_violation_step": 4,
                    "first_high_replay_priority_step": 5,
                    "first_transformation_family_step": 3,
                    "first_stable_transformation_family_step": None,
                    "first_carrier_candidate_step": None,
                    "first_emergent_carrier_step": None,
                }
            ]
        },
    }
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "interaction_sampling_v05c_report.txt").write_text("stub\n", encoding="utf-8")


def _write_suite_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY,
                isf_prediction_error REAL,
                memory_replay_priority REAL,
                memory_replay_candidate INTEGER
            );
            CREATE TABLE prediction_results (
                interaction_id INTEGER NOT NULL,
                context_signature TEXT,
                action INTEGER,
                context_contradiction INTEGER,
                context_expansion_suggested INTEGER,
                actual_family INTEGER
            );
            CREATE TABLE contingencies (
                contingency_id INTEGER PRIMARY KEY,
                context_action TEXT,
                effect_signature TEXT,
                support INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO interactions (id, isf_prediction_error, memory_replay_priority, memory_replay_candidate) VALUES (?, ?, ?, ?)",
            [
                (1, 1.0, 0.95, 1),
                (2, 1.0, 0.90, 1),
                (3, 0.0, 0.30, 1),
                (4, 0.0, 0.25, 1),
            ],
        )
        connection.executemany(
            """
            INSERT INTO prediction_results (
                interaction_id,
                context_signature,
                action,
                context_contradiction,
                context_expansion_suggested,
                actual_family
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "ctx-a", 0, 1, 1, 10),
                (2, "ctx-b", 1, 1, 1, 10),
                (3, "ctx-c", 0, 0, 0, 11),
                (4, "ctx-d", 1, 0, 0, 11),
            ],
        )
        connection.executemany(
            "INSERT INTO contingencies VALUES (?, ?, ?, ?)",
            [
                (1, "ctxA|0", "eff-1", 3),
                (2, "ctxB|1", "eff-1", 4),
                (3, "ctxC|0", "eff-2", 3),
                (4, "ctxD|1", "eff-2", 2),
            ],
        )
        connection.commit()


def test_hypothesis_suite_report_creates_subdirectories_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_suite_input_report(run_dir)
    _write_suite_db(run_dir / "seed_0.sqlite")

    assert (
        main(
            [
                "hypothesis-suite-report",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(out_dir),
                "--scan-all-dbs",
                "--max-db-files",
                "1000",
            ]
        )
        == 0
    )

    assert (out_dir / "h01").is_dir()
    assert (out_dir / "h02").is_dir()
    assert (out_dir / "h03").is_dir()
    assert (out_dir / SUITE_JSON_NAME).exists()
    summary = json.loads((out_dir / SUITE_JSON_NAME).read_text(encoding="utf-8"))
    assert "H01 decision" in summary
    assert "H02 decision" in summary
    assert "H03 decision" in summary
    assert "h02a_replay_attention_decision" in summary["H02 core metrics"]
    assert "h02b_pre_carrier_timing_decision" in summary["H02 core metrics"]
    assert summary["H04 decision"] == "NOT_IMPLEMENTED"
    assert summary["per_game_status_table"]


def test_suite_summary_temporal_order_ratios_ignore_null_cases(tmp_path: Path) -> None:
    summary = build_hypothesis_suite_summary(
        run_dir=tmp_path,
        h01={"decision": "VALID", "missing_evidence": [], "per_game_contingency_counts": {}, "per_sampler_contingency_counts": {}, "stable_contingency_count": 1, "total_interaction_count": 10, "mean_prediction_accuracy": 0.5},
        h02={"decision": "VALID", "missing_evidence": []},
        h03={"decision": "VALID", "missing_evidence": []},
    )

    assert summary["temporal_order_diagnostics"]["h01_before_h03_ratio"] is None
    assert summary["temporal_order_diagnostics"]["temporal_order_cases_available"] == 0


def test_suite_summary_allows_temporal_milestone_nulls(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_suite_input_report(run_dir)
    report = json.loads((run_dir / "interaction_sampling_v05c_report.json").read_text(encoding="utf-8"))
    report["temporal_milestones"]["by_game_sampler_seed"][0]["first_prediction_violation_step"] = None
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = build_hypothesis_suite_summary(
        run_dir=run_dir,
        h01={"decision": "VALID", "missing_evidence": [], "per_game_contingency_counts": {}, "per_sampler_contingency_counts": {}, "stable_contingency_count": 1, "total_interaction_count": 10, "mean_prediction_accuracy": 0.5},
        h02={"decision": "PARTIALLY_VALID", "missing_evidence": []},
        h03={"decision": "VALID", "missing_evidence": []},
    )

    row = summary["temporal_order_diagnostics"]["per_case"][0]
    assert row["h02_before_h03"] is None


def test_hypothesis_reports_can_read_from_compact_memory_after_raw_cleanup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    memory = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(memory.current_state) as connection:
        connection.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, '["ctx"]|a0|efamily:a', "tt01", "mixed", 0, "family:a", 25, 1, 10, 1.0, 0.0, 0.9, 1),
        )
        connection.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "family:a", "family:a", "unknown", "unknown", "unknown", 10, 3, 1, 10, 1.0),
        )
        connection.execute(
            """
            INSERT INTO contradiction_clusters (
                cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                max_prediction_error, mean_replay_priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctx-a", "ctx-a", 2, 1, 10, 1.0, 0.9),
        )
        connection.commit()
    with sqlite3.connect(memory.replay_queue) as connection:
        connection.execute(
            """
            INSERT INTO replay_queue (
                replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("1", "interaction", "1", 0.9, "contradiction_linked", 1, 10, "{}"),
        )
        connection.commit()
    memory.summary_json.write_text(json.dumps({"fold_summary": {"stable_contingencies_added": 1}, "total_interactions_seen": 10}, indent=2), encoding="utf-8")

    h01 = evaluate_h01_contingency_emergence(run_dir=run_dir, output_dir=out_dir / "h01", memory_dir=memory.root)
    h02 = evaluate_h02_prediction_violation_attention(run_dir=run_dir, output_dir=out_dir / "h02", memory_dir=memory.root)
    h03 = evaluate_h03_transformation_family_formation(run_dir=run_dir, output_dir=out_dir / "h03", memory_dir=memory.root)

    assert h01["evidence_source"] == "compact_memory"
    assert h02["evidence_source"] == "compact_memory"
    assert h03["evidence_source"] == "compact_memory"
