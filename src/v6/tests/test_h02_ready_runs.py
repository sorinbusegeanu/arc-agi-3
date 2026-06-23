from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h02_report import (
    H02_JSON_NAME,
    H02_MD_NAME,
    H02_READY_JSON_NAME,
    H02_READY_TXT_NAME,
    H02_TXT_NAME,
    find_h02_ready_runs,
    run_h02_on_best_ready_run,
)


def _write_v05c_report(run_dir: Path, **validation_overrides) -> None:
    validation = {
        "mean_isf_prediction_error": 0.6,
        "context_contradiction_count": 6,
        "repeated_contradiction_count": 2,
        "context_expansion_suggested_count": 3,
        "memory_record_count": 12,
        "memory_replay_candidate_count": 10,
        "high_priority_replay_count": 2,
        "carrier_object_candidate_count": 4,
        "emergent_object_carrier_count": 0,
        "carrier_context_action_fallback_candidate_count": 1,
        "emergent_context_action_fallback_count": 0,
    }
    validation.update(validation_overrides)
    payload = {"validation": validation}
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_direct_linkage_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE interactions (
                interaction_id INTEGER PRIMARY KEY,
                prediction_error REAL,
                replay_priority REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO interactions VALUES (?, ?, ?)",
            [
                (1, 1.0, 0.95),
                (2, 1.0, 0.90),
                (3, 1.0, 0.85),
                (4, 0.0, 0.20),
                (5, 0.0, 0.25),
                (6, 0.0, 0.30),
            ],
        )
        connection.commit()


def test_find_h02_ready_runs_no_reports(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    runs_root.mkdir()

    result = find_h02_ready_runs(runs_root, output_dir)

    assert result["candidate_count"] == 0
    assert result["ready_count"] == 0
    assert result["recommended_run"] is None
    assert (output_dir / H02_READY_JSON_NAME).exists()
    assert (output_dir / H02_READY_TXT_NAME).exists()


def test_old_report_missing_carrier_fields_is_not_ready(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    run_dir = runs_root / "old_run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(
        run_dir,
        carrier_object_candidate_count=None,
        emergent_object_carrier_count=None,
        carrier_context_action_fallback_candidate_count=None,
        emergent_context_action_fallback_count=None,
    )
    _write_direct_linkage_db(run_dir / "seed_0.sqlite")

    result = find_h02_ready_runs(runs_root, output_dir)

    assert result["candidate_count"] == 1
    assert result["ready_count"] == 0
    entry = result["runs"][0]
    assert entry["h02_ready"] is False
    assert entry["missing_required_carrier_fields"] == [
        "emergent_object_carrier_count",
        "emergent_context_action_fallback_count",
        "carrier_object_candidate_count",
        "carrier_context_action_fallback_candidate_count",
    ]


def test_ready_report_is_recommended(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    run_dir = runs_root / "ready_run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_direct_linkage_db(run_dir / "seed_0.sqlite")

    result = find_h02_ready_runs(runs_root, output_dir)

    assert result["candidate_count"] == 1
    assert result["ready_count"] == 1
    assert result["recommended_run"]["run_dir"] == str(run_dir)
    assert result["runs"][0]["h02_ready"] is True
    text = (output_dir / H02_READY_TXT_NAME).read_text(encoding="utf-8")
    assert "--max-db-files 20" in text
    assert "--prefer-db seed_0.sqlite" in text


def test_ready_run_sorting_prefers_larger_memory_record_count(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    low_run = runs_root / "low_memory"
    high_run = runs_root / "high_memory"
    low_run.mkdir(parents=True)
    high_run.mkdir(parents=True)
    _write_v05c_report(low_run, memory_record_count=10)
    _write_v05c_report(high_run, memory_record_count=50)
    _write_direct_linkage_db(low_run / "seed_0.sqlite")
    _write_direct_linkage_db(high_run / "seed_0.sqlite")

    result = find_h02_ready_runs(runs_root, output_dir)

    assert result["ready_count"] == 2
    assert result["recommended_run"]["run_dir"] == str(high_run)
    assert result["runs"][0]["run_dir"] == str(high_run)


def test_run_best_without_ready_run_writes_inventory_only(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    run_dir = runs_root / "not_ready"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir, mean_isf_prediction_error=0.0)
    _write_direct_linkage_db(run_dir / "seed_0.sqlite")

    result = run_h02_on_best_ready_run(runs_root, output_dir)

    assert result["recommended_run"] is None
    assert (output_dir / H02_READY_JSON_NAME).exists()
    assert (output_dir / H02_READY_TXT_NAME).exists()
    assert not (output_dir / H02_JSON_NAME).exists()
    assert not (output_dir / H02_TXT_NAME).exists()
    assert not (output_dir / H02_MD_NAME).exists()
    text = (output_dir / H02_READY_TXT_NAME).read_text(encoding="utf-8")
    assert (
        "No H02-ready existing v05c run found. Generate a new v05c run with current code, then rerun find-h02-ready-runs."
        in text
    )


def test_run_best_with_ready_run_creates_h02_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "out"
    run_dir = runs_root / "ready_run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_direct_linkage_db(run_dir / "seed_0.sqlite")

    result = run_h02_on_best_ready_run(runs_root, output_dir, max_db_files=1)

    assert result["recommended_run"]["run_dir"] == str(run_dir)
    assert result["run_best_parameters"]["max_db_files"] == 1
    assert (output_dir / H02_READY_JSON_NAME).exists()
    assert (output_dir / H02_READY_TXT_NAME).exists()
    assert (output_dir / H02_JSON_NAME).exists()
    assert (output_dir / H02_TXT_NAME).exists()
    assert (output_dir / H02_MD_NAME).exists()
