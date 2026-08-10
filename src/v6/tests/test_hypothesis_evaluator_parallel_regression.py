import v6.hypothesis_suite_report as suite
from v6.reporting.framework import apply_decision_envelope


def _valid_result(**kwargs):
    return {
        "decision": "VALID",
        "core_metrics": {"fixture": 1},
        "missing_evidence": [],
    }


def _call_suite(tmp_path, *, suite_mode="full"):
    return suite.evaluate_hypotheses_read_only(
        run_dir=tmp_path / "run",
        evidence_memory_dir=tmp_path / "memory",
        output_dir=tmp_path / "reports",
        suite_mode=suite_mode,
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


def test_report_evaluators_accept_runtime_installed_bindings(tmp_path, monkeypatch):
    evaluator_names = [
        "evaluate_h01_contingency_emergence",
        "evaluate_h02_prediction_violation_attention",
        "evaluate_h03_transformation_family_formation",
        "evaluate_h04_carrier_emergence",
        "evaluate_h05_role_emergence",
        "evaluate_h06_role_transfer",
        "evaluate_h07_concept_emergence",
        "evaluate_h08_world_model_coherence",
        "evaluate_h09_future_option_motifs",
        "evaluate_h10_future_option_attention",
        "evaluate_h11_future_option_transfer_concepts",
        "evaluate_h12_efficiency_emergence",
    ]

    for name in evaluator_names:
        def local_evaluator(**kwargs):
            return _valid_result(**kwargs)
        monkeypatch.setattr(suite, name, local_evaluator)

    results = _call_suite(tmp_path)

    assert set(results) == {f"H{i:02d}" for i in range(1, 13)}
    assert all(result["decision"] == "VALID" for result in results.values())
    assert not any("evaluator_error" in result for result in results.values())


def test_evaluator_exception_is_not_mislabeled_as_insufficient_evidence(tmp_path, monkeypatch):
    def failing_evaluator(**kwargs):
        raise RuntimeError("evaluator regression sentinel")

    monkeypatch.setattr(suite, "evaluate_h01_contingency_emergence", failing_evaluator)
    monkeypatch.setattr(suite, "evaluate_h02_prediction_violation_attention", _valid_result)
    monkeypatch.setattr(suite, "evaluate_h03_transformation_family_formation", _valid_result)
    monkeypatch.setattr(suite, "evaluate_h04_carrier_emergence", _valid_result)
    monkeypatch.setattr(suite, "evaluate_h05_role_emergence", _valid_result)
    monkeypatch.setattr(suite, "evaluate_h12_efficiency_emergence", _valid_result)

    results = _call_suite(tmp_path, suite_mode="fast")

    assert results["H01"]["decision"] == "EVALUATOR_ERROR"
    assert results["H01"]["evaluator_error"]["type"] == "RuntimeError"
    assert "sentinel" in results["H01"]["evaluator_error"]["message"]


def test_decision_envelope_preserves_evaluator_error(tmp_path, monkeypatch):
    class Contract:
        hypothesis_id = "H01"
        required_tables = ()
        required_report_fields = ()
        proxy_markers = ()
        minimum_coverage = None
        dependencies = ()

    monkeypatch.setattr('v6.reporting.framework.get_contract', lambda _hypothesis_id: Contract())
    result = apply_decision_envelope(
        "H01",
        {
            "decision": "EVALUATOR_ERROR",
            "evaluator_error": {"type": "RuntimeError", "message": "sentinel"},
            "core_metrics": {},
            "missing_evidence": [],
        },
        memory_dir=tmp_path,
        provenance={},
        dependency_results={},
        memory_unchanged=True,
    )
    assert result["raw_decision"] == "EVALUATOR_ERROR"
    assert result["final_decision"] == "EVALUATOR_ERROR"


def test_report_phase_has_no_evaluator_worker_fanout():
    import inspect

    signature = inspect.signature(suite.evaluate_hypotheses_read_only)
    assert "evaluator_workers" not in signature.parameters
    source = inspect.getsource(suite.run_hypothesis_suite_report)
    assert "evaluator_workers" not in source
    assert "ProcessPoolExecutor" not in inspect.getsource(suite)
    assert "ThreadPoolExecutor" not in inspect.getsource(suite)
