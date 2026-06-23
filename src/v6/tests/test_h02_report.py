from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from v6.cli import build_parser, main
from v6.hypothesis_h02_report import (
    DIRECT_LINKAGE_UNAVAILABLE_MESSAGE,
    H02_JSON_NAME,
    H02_MD_NAME,
    H02_TXT_NAME,
    evaluate_h02_prediction_violation_attention,
)


def _write_v05c_report(run_dir: Path, **validation_overrides) -> None:
    validation = {
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
        "high_priority_replay_count": 2,
        "carrier_candidate_count": 3,
        "emergent_carrier_count": 0,
        "emergent_object_carrier_count": 0,
        "emergent_context_action_fallback_count": 0,
    }
    validation.update(validation_overrides)
    payload = {
        "runs": [],
        "sampler_comparison": [],
        "best_by_game": [],
        "summary_by_family": [],
        "validation": validation,
    }
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_direct_linkage_db(path: Path) -> None:
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
                context_expansion_suggested INTEGER
            );
            """
        )
        interaction_rows = [
            (1, 1.0, 0.95, 1),
            (2, 1.0, 0.90, 1),
            (3, 1.0, 0.85, 1),
            (4, 0.0, 0.40, 1),
            (5, 0.0, 0.35, 1),
            (6, 0.0, 0.30, 1),
            (7, 0.0, 0.25, 1),
            (8, 0.0, 0.20, 1),
            (9, 0.0, 0.20, 1),
            (10, 0.0, 0.15, 1),
        ]
        connection.executemany(
            "INSERT INTO interactions (id, isf_prediction_error, memory_replay_priority, memory_replay_candidate) VALUES (?, ?, ?, ?)",
            interaction_rows,
        )
        prediction_rows = [
            (1, "ctx-a", 0, 1, 1),
            (2, "ctx-a", 0, 1, 1),
            (3, "ctx-b", 1, 1, 1),
            (4, "ctx-c", 1, 0, 0),
            (5, "ctx-d", 2, 0, 0),
        ]
        connection.executemany(
            """
            INSERT INTO prediction_results (
                interaction_id,
                context_signature,
                action,
                context_contradiction,
                context_expansion_suggested
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            prediction_rows,
        )
        connection.commit()


def test_h02_json_only_report_can_be_partially_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "PARTIALLY_VALID"
    assert result["db_found"] is False
    assert result["prediction_violation_replay_lift"] is None
    assert DIRECT_LINKAGE_UNAVAILABLE_MESSAGE in result["missing_evidence"]


def test_h02_missing_input_report_is_inconclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "INCONCLUSIVE"
    assert result["input_report_found"] is False


def test_h02_synthetic_db_with_replay_lift_can_be_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_direct_linkage_db(run_dir / "seed_0.sqlite")

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "VALID"
    assert result["db_found"] is True
    assert result["prediction_violation_replay_lift"] is not None
    assert result["prediction_violation_replay_lift"] > 1.25
    assert result["high_priority_replay_prediction_violation_ratio"] > result["prediction_violation_base_ratio"]


def test_h02_zero_prediction_error_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir, mean_isf_prediction_error=0.0)

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "INVALID"


def test_h02_cli_creates_json_txt_and_md_outputs(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v6.cli",
            "hypothesis-h02-report",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--max-rows",
            "1000",
        ],
    )

    assert main() == 0
    assert (output_dir / H02_JSON_NAME).exists()
    assert (output_dir / H02_TXT_NAME).exists()
    assert (output_dir / H02_MD_NAME).exists()


def test_cli_accepts_h02_report_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "hypothesis-h02-report",
            "--run-dir",
            "runs/v6/hypothesis_tests/h02_prediction_violation_attention_cd1",
            "--output-dir",
            "runs/v6/hypothesis_tests/results/h02_prediction_violation_attention_cd1",
        ]
    )

    assert args.command == "hypothesis-h02-report"
    assert args.run_dir.endswith("h02_prediction_violation_attention_cd1")
    assert args.max_rows == 1000000
