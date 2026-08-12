from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import v6.hypothesis_suite_performance as perf
import v6.hypothesis_suite_report as suite


def _valid(**kwargs):
    return {"decision": "VALID", "core_metrics": {}, "missing_evidence": []}


def test_performance_policy_is_installed():
    assert suite.evaluate_hypotheses_read_only is perf._evaluate_hypotheses_threaded


def test_shared_evidence_cache_counts_common_tables(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    with sqlite3.connect(memory_dir / "current_state.sqlite") as connection:
        connection.execute("CREATE TABLE role_transfer_attempts(id INTEGER)")
        connection.executemany("INSERT INTO role_transfer_attempts VALUES (?)", [(1,), (2,), (3,)])
        connection.execute("CREATE TABLE future_option_transfer_links(id INTEGER)")
        connection.executemany("INSERT INTO future_option_transfer_links VALUES (?)", [(1,), (2,)])
        connection.execute("CREATE TABLE memory_summary(key TEXT, value_json TEXT)")
        connection.execute(
            "INSERT INTO memory_summary VALUES (?, ?)",
            ("future_option_derivation_summary", json.dumps({"roles_with_transfer_attempts": 3})),
        )
    cache = perf.build_suite_evidence_cache(memory_dir)
    assert cache.table_counts["role_transfer_attempts"] == 3
    assert cache.table_counts["future_option_transfer_links"] == 2
    assert cache.summary_values["future_option_derivation_summary"]["roles_with_transfer_attempts"] == 3
    assert cache.rows_indexed >= 5


def test_report_evaluators_run_concurrently(tmp_path, monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return _valid()

    for name in (
        "evaluate_h01_contingency_emergence",
        "evaluate_h02_prediction_violation_attention",
        "evaluate_h03_transformation_family_formation",
        "evaluate_h04_carrier_emergence",
        "evaluate_h05_role_emergence",
        "evaluate_h12_efficiency_emergence",
    ):
        monkeypatch.setattr(suite, name, slow)
    monkeypatch.setattr(suite, "_V64_EVALUATOR_WORKERS", 4, raising=False)

    results = suite.evaluate_hypotheses_read_only(
        run_dir=tmp_path / "run",
        evidence_memory_dir=tmp_path / "memory",
        output_dir=tmp_path / "reports",
        suite_mode="fast",
        max_db_files=0,
        max_rows=100,
        scan_all_dbs=False,
        incremental_promotion_validation=True,
        promotion_min_incremental_coverage=0.05,
        promotion_min_cross_context_or_game_evidence=2,
        promotion_min_behavioral_or_predictive_lift=0.01,
        promotion_min_relevant_heldout_event_count=20,
        promotion_population_comparability_threshold=0.8,
        promotion_demotion_failure_limit=2,
        h11_provenance_sample_limit=10,
        h11_write_full_provenance_jsonl=False,
        max_h11_main_report_bytes=10000,
    )
    assert max_active > 1
    assert results["H01"]["performance_profile"]["evaluation_seconds"] > 0
    profile = json.loads((tmp_path / "reports" / "hypothesis_evaluator_profile.json").read_text())
    assert profile["evaluator_workers"] == 4
    assert "H01" in profile["evaluators"]


def _build_h11_fixture(memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as connection:
        connection.execute(
            """CREATE TABLE future_option_transfer_links(
                motif_signature TEXT, role_signature TEXT, concept_signature TEXT,
                source_role_signature TEXT, source_game_key TEXT, target_game_key TEXT,
                source_context_key TEXT, target_context_key TEXT,
                source_game_is_surrogate INTEGER, target_game_is_surrogate INTEGER,
                source_context_is_surrogate INTEGER, target_context_is_surrogate INTEGER,
                provenance_mode TEXT, motif_provenance_status TEXT,
                transfer_provenance_status TEXT, concept_validation_status TEXT,
                transfer_attempt_count INTEGER, successful_transfer_count INTEGER,
                strong_transfer_success_count INTEGER, promoted_concept_count INTEGER
            )"""
        )
        rows = [
            ("m1", "r1", "c1", "r1", "g1", "g2", "ctx1", "ctx2", 0, 0, 0, 0, "single_source", "verified", "verified", "verified", 1, 1, 1, 1),
            ("m1", "r1", "c1", "r1", "g1", "g3", "ctx1", "ctx3", 0, 0, 0, 0, "single_source", "verified", "verified", "verified", 1, 1, 1, 1),
        ]
        connection.executemany(
            "INSERT INTO future_option_transfer_links VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        connection.execute("CREATE TABLE future_option_motifs(motif_signature TEXT, is_emergent INTEGER)")
        connection.execute("INSERT INTO future_option_motifs VALUES ('m1', 1)")
        connection.execute(
            """CREATE TABLE role_transfer_attempts(
                role_signature TEXT, source_role_signature TEXT, reuse_success INTEGER,
                provenance_mode TEXT, source_game_key TEXT, target_game_key TEXT,
                source_context_key TEXT, target_context_key TEXT,
                source_game_is_surrogate INTEGER, target_game_is_surrogate INTEGER,
                source_context_is_surrogate INTEGER, target_context_is_surrogate INTEGER,
                provenance_status TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO role_transfer_attempts VALUES ('r1','r1',1,'single_source','g1','g2','ctx1','ctx2',0,0,0,0,'verified')"
        )
        connection.execute("CREATE TABLE concept_candidates(concept_signature TEXT, is_promoted INTEGER)")
        connection.execute("INSERT INTO concept_candidates VALUES ('c1', 1)")
        connection.execute("CREATE TABLE memory_summary(key TEXT, value_json TEXT)")
        connection.execute(
            "INSERT INTO memory_summary VALUES (?, ?)",
            ("future_option_derivation_summary", json.dumps({"roles_with_transfer_attempts": 1})),
        )


def test_h11_large_population_uses_streaming_path(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "reports"
    _build_h11_fixture(memory_dir)
    monkeypatch.setattr(perf, "LARGE_H11_LINK_THRESHOLD", 1)
    cache = perf.build_suite_evidence_cache(memory_dir)
    result = perf._h11_streaming_large(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=output_dir,
        already_derived=True,
        provenance_sample_limit=1,
        write_full_provenance_jsonl=False,
        max_main_report_bytes=100000,
        suite_evidence_cache=cache,
    )
    assert result["streaming_evaluation"] is True
    assert result["future_option_transfer_link_count"] == 2
    assert result["fully_verified_link_count"] == 2
    assert result["verified_transfer_pair_count"] == 2
    assert result["motif_transfer_chain_provenance_sample_count"] == 1
