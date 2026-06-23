from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h02_report import (
    DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE,
    DIRECT_LINKAGE_UNAVAILABLE_MESSAGE,
    compute_prediction_violation_replay_lift_from_existing_db,
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
        rows = [
            (1, 1.0, 0.95),
            (2, 1.0, 0.90),
            (3, 1.0, 0.85),
            (4, 0.0, 0.20),
            (5, 0.0, 0.25),
            (6, 0.0, 0.30),
        ]
        connection.executemany("INSERT INTO interactions VALUES (?, ?, ?)", rows)
        connection.commit()


def _write_direct_linkage_db_with_rows(path: Path, rows: list[tuple[int, float, float]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE interactions (
                interaction_id INTEGER PRIMARY KEY,
                isf_prediction_error REAL,
                memory_replay_priority REAL
            )
            """
        )
        connection.executemany("INSERT INTO interactions VALUES (?, ?, ?)", rows)
        connection.commit()


def _write_empty_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata (k TEXT, v TEXT)")
        connection.commit()


def test_compute_replay_lift_single_table_positive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    db_path = run_dir / "single.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE interactions (
                interaction_id INTEGER PRIMARY KEY,
                prediction_error REAL,
                replay_priority REAL
            )
            """
        )
        rows = [
            (1, 1.0, 0.95),
            (2, 1.0, 0.90),
            (3, 1.0, 0.85),
            (4, 0.0, 0.20),
            (5, 0.0, 0.25),
            (6, 0.0, 0.30),
        ]
        connection.executemany("INSERT INTO interactions VALUES (?, ?, ?)", rows)
        connection.commit()

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir)

    assert result["direct_replay_lift_available"] is True
    assert result["prediction_violation_replay_lift"] > 1.25
    assert result["candidate_tables_used"] == ["interactions"]
    assert result["high_priority_threshold_method"] in {"sql_percentile", "fallback_max_0_9"}


def test_sharded_run_stops_early_on_first_ranked_usable_db(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    usable_dir = run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_30000"
    usable_dir.mkdir(parents=True)
    _write_direct_linkage_db_with_rows(
        usable_dir / "seed_0.sqlite",
        [(1, 1.0, 0.95), (2, 1.0, 0.90), (3, 0.0, 0.20), (4, 0.0, 0.25)],
    )
    for index in range(29):
        shard_dir = run_dir / "sampling_v05c" / "tt01" / "random_baseline" / "steps_30000"
        shard_dir.mkdir(parents=True, exist_ok=True)
        _write_empty_db(shard_dir / f"seed_{index + 1}.sqlite")

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir, max_db_files=5)

    assert result["sqlite_db_count_total"] == 30
    assert result["sqlite_db_count_inspected"] == 1
    assert result["sqlite_db_inspection_truncated"] is True
    assert result["direct_replay_lift_available"] is True
    assert result["selected_db_path"].endswith("seed_0.sqlite")


def test_max_db_files_respected_when_no_usable_shards_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for index in range(30):
        _write_empty_db(run_dir / f"seed_{index}.sqlite")

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir, max_db_files=7)

    assert result["sqlite_db_count_total"] == 30
    assert result["sqlite_db_count_inspected"] == 7
    assert result["sqlite_db_skipped_count"] == 23
    assert result["sqlite_db_inspection_truncated"] is True
    assert result["direct_replay_lift_available"] is False
    assert DIRECT_LINKAGE_SHARD_LIMIT_MESSAGE in result["missing_evidence"]


def test_prefer_db_is_selected_first(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_empty_db(run_dir / "alpha.sqlite")
    preferred = run_dir / "preferred.sqlite"
    _write_direct_linkage_db_with_rows(
        preferred,
        [(1, 1.0, 0.95), (2, 1.0, 0.90), (3, 0.0, 0.20), (4, 0.0, 0.25)],
    )

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir, prefer_db="preferred.sqlite", max_db_files=2)

    assert result["direct_replay_lift_available"] is True
    assert result["selected_db_path"] == str(preferred)
    assert result["inspected_db_paths"][0] == str(preferred)


def test_scan_all_dbs_prefers_larger_row_count_while_default_stops_early(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_30000"
    second = run_dir / "sampling_v05c" / "tt01" / "low_confidence" / "steps_30000"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_direct_linkage_db_with_rows(
        first / "seed_0.sqlite",
        [(1, 1.0, 0.95), (2, 1.0, 0.85), (3, 0.0, 0.20), (4, 0.0, 0.30)],
    )
    _write_direct_linkage_db_with_rows(
        second / "seed_0.sqlite",
        [(i, 1.0 if i <= 6 else 0.0, 0.95 if i <= 6 else 0.20) for i in range(1, 13)],
    )

    early = compute_prediction_violation_replay_lift_from_existing_db(run_dir, max_db_files=5)
    scan_all = compute_prediction_violation_replay_lift_from_existing_db(run_dir, max_db_files=5, scan_all_dbs=True)

    assert early["direct_replay_lift_available"] is True
    assert early["sqlite_db_count_inspected"] == 1
    assert early["selected_db_path"].endswith("mixed/steps_30000/seed_0.sqlite")
    assert scan_all["direct_replay_lift_available"] is True
    assert scan_all["sqlite_db_count_inspected"] == 2
    assert scan_all["selected_db_path"].endswith("low_confidence/steps_30000/seed_0.sqlite")


def test_strong_replay_lift_with_missing_object_fields_is_partially_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        emergent_object_carrier_count=None,
        emergent_context_action_fallback_count=None,
    )
    _write_direct_linkage_db(run_dir / "strong.sqlite")

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "PARTIALLY_VALID"
    assert "partially supported" in result["scientific_conclusion"].lower()
    assert "object-carrier absence evidence is unavailable" in result["scientific_conclusion"]
    assert "Aggregate object-carrier absence evidence is unavailable." in result["missing_evidence"]


def test_strong_replay_lift_with_object_fields_absent_is_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_direct_linkage_db(run_dir / "strong.sqlite")

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "VALID"


def test_equal_priority_table_is_invalid_when_direct_lift_is_one(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    with sqlite3.connect(run_dir / "equal.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE interactions (
                interaction_id INTEGER PRIMARY KEY,
                prediction_error REAL,
                replay_priority REAL
            )
            """
        )
        rows = [
            (1, 1.0, 0.50),
            (2, 1.0, 0.50),
            (3, 0.0, 0.50),
            (4, 0.0, 0.50),
        ]
        connection.executemany("INSERT INTO interactions VALUES (?, ?, ?)", rows)
        connection.commit()

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["direct_replay_lift_available"] is True
    assert result["prediction_violation_replay_lift"] == 1.0
    assert result["decision"] == "INVALID"


def test_missing_schema_columns_reports_exact_missing_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    with sqlite3.connect(run_dir / "aggregate_only.sqlite") as connection:
        connection.execute("CREATE TABLE metadata (k TEXT, v TEXT)")
        connection.commit()

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["direct_replay_lift_available"] is False
    assert DIRECT_LINKAGE_UNAVAILABLE_MESSAGE in result["missing_evidence"]


def test_joined_schema_computes_lift(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with sqlite3.connect(run_dir / "joined.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE memory_table (
                interaction_id INTEGER,
                replay_priority REAL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE prediction_table (
                interaction_id INTEGER,
                prediction_error REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO memory_table VALUES (?, ?)",
            [(1, 0.9), (2, 0.85), (3, 0.25), (4, 0.20)],
        )
        connection.executemany(
            "INSERT INTO prediction_table VALUES (?, ?)",
            [(1, 1.0), (2, 1.0), (3, 0.0), (4, 0.0)],
        )
        connection.commit()

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir)

    assert result["direct_replay_lift_available"] is True
    assert result["prediction_violation_replay_lift"] > 1.25
    assert result["candidate_tables_used"] == ["memory_table", "prediction_table"]


def test_common_interactions_fast_path_uses_ranked_db_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    db_path = run_dir / "fast.sqlite"
    _write_direct_linkage_db_with_rows(
        db_path,
        [(1, 1.0, 0.95), (2, 1.0, 0.90), (3, 1.0, 0.85), (4, 0.0, 0.20), (5, 0.0, 0.25), (6, 0.0, 0.30)],
    )

    result = compute_prediction_violation_replay_lift_from_existing_db(run_dir, max_db_files=1)

    assert result["direct_replay_lift_available"] is True
    assert result["candidate_tables_used"] == ["interactions"]
    assert result["replay_priority_metric_source"] == "interactions.memory_replay_priority"
    assert result["prediction_violation_metric_source"] == "interactions.isf_prediction_error"


def test_missing_db_keeps_valid_aggregate_json_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["db_found"] is False
    assert result["prediction_violation_replay_lift"] is None
    assert result["decision"] == "PARTIALLY_VALID"
    assert DIRECT_LINKAGE_UNAVAILABLE_MESSAGE in result["missing_evidence"]


def test_missing_aggregate_report_is_inconclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    run_dir.mkdir()
    with sqlite3.connect(run_dir / "single.sqlite") as connection:
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
            [(1, 1.0, 0.9), (2, 0.0, 0.2)],
        )
        connection.commit()

    result = evaluate_h02_prediction_violation_attention(run_dir, output_dir)

    assert result["decision"] == "INCONCLUSIVE"
