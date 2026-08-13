from __future__ import annotations

import json

from v7.cli import run_events
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import EpisodeEvidence, ScientificDerivationKernels
from v7.memory.development import ScientificArtifactComparator
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.ids import MemoryLevel
from v7.memory.models import EdgeMutation
from v7.memory.reporting import StrictHypothesisReporter
from v7.memory.restart import RuntimeSnapshotStore
from v7.memory.scoring import PerformanceProbe
from v7.memory.writer import CanonicalMemoryWriter
from v7.performance import PerformanceSample, PerformanceValidationSuite, REQUIRED_PERFORMANCE_METRICS
from v7.runtime import V7Runtime, V7RuntimeConfig


def test_restart_restores_canonical_identity_cognition_edges_and_dirty_plan(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    pipeline = MemoryLearningPipeline(writer)
    evidence = EpisodeEvidence(10, 2, 100, True, prediction_error=0.5, future_option_delta=1.0)
    m1 = pipeline.observe_episode(evidence)
    other = pipeline.observe_episode(EpisodeEvidence(11, 2, 101, True))
    m2 = pipeline.derive_m2(action_id=2, member_ids=(m1, other), outcome_class=7)
    writer.apply_edge_batch((EdgeMutation(m1, 9, m2, support_delta=3),))
    writer.commit_generation()
    durable = DurableGenerationStore(tmp_path / "state.sqlite")
    try:
        snapshots = RuntimeSnapshotStore(durable)
        snapshots.persist(writer)
        restored = snapshots.restore()
        assert restored.published_view.generation_id == writer.published_view.generation_id
        assert restored.published_view.nodes == writer.published_view.nodes
        assert restored.published_view.neighbors((m1,), 9) == ((m2,),)
        assert getattr(restored, "_edge_support")[(m1, 9, m2)] == 3
        assert restored.canonical_memory_id(ScientificDerivationKernels.m1_from_episode(evidence).key) == m1
        assert m1 in restored.published_view.score_inputs(context_signature=10, action_ids=(2,))[0].contingency_ids
        assert restored.dirty_derivation_plan().total_count == writer.dirty_derivation_plan().total_count
    finally:
        durable.close()


def test_runtime_resumes_and_reuses_existing_memory_id(tmp_path) -> None:
    config = V7RuntimeConfig.from_path(tmp_path)
    evidence = EpisodeEvidence(20, 3, 200, True, source_global_step=1)
    runtime = V7Runtime(config)
    try:
        first = runtime.observe(evidence)
        assert int(runtime.commit(run_lifecycle=False).state.generation_id) == 1
    finally:
        runtime.close()
    runtime = V7Runtime(config)
    try:
        assert runtime.observe(evidence) == first
        assert int(runtime.commit(run_lifecycle=False).state.generation_id) == 2
    finally:
        runtime.close()


def test_cli_jsonl_runs_one_generation(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"context_signature": 1, "action_id": 2, "outcome_signature": 3, "success": True}) + "\n", encoding="utf-8")
    assert run_events(tmp_path / "run", events) == {"events": 1, "generation": 1, "memories": 1}


def test_hypothesis_contracts_require_real_evidence_and_separate_gates() -> None:
    reporter = StrictHypothesisReporter()
    assert reporter.evaluate("H01", raw_decision="VALID", evidence={"evidence_rows": 2}).final_decision == "INSUFFICIENT_EVIDENCE"
    assert reporter.evaluate("H01", raw_decision="VALID", evidence={"evidence_rows": 2, "measurement": 0.8, "proxy_only": True}).final_decision == "INSUFFICIENT_EVIDENCE"
    valid = reporter.evaluate("H01", raw_decision="VALID", evidence={"evidence_rows": 2, "measurement": 0.8})
    assert valid.raw_decision == valid.final_decision == "VALID"
    assert reporter.evaluate("H01", raw_decision="VALID", evidence={"evidence_rows": 2, "measurement": 0.8}, dependency_gate="FAIL").final_decision == "INSUFFICIENT_EVIDENCE"


def test_all_h01_h12_contracts_exist() -> None:
    assert tuple(sorted(StrictHypothesisReporter().contracts)) == tuple(f"H{index:02d}" for index in range(1, 13))


def test_scientific_artifact_comparator_finds_precise_difference() -> None:
    comparator = ScientificArtifactComparator()
    assert comparator.compare({"M1": [1, 2]}, {"M1": [1, 2]}).matched
    result = comparator.compare({"M1": {"support": 2}}, {"M1": {"support": 3}})
    assert not result.matched
    assert result.mismatches[0].path == "$.M1.support"


def test_performance_probe_measures_and_checks_budget() -> None:
    measurement = PerformanceProbe.measure("noop", lambda: None, iterations=10)
    assert measurement.iterations == 10
    assert measurement.total_seconds >= 0
    assert PerformanceProbe.within_budget(measurement, max_mean_seconds=1.0)


def test_performance_validation_requires_complete_metric_set() -> None:
    suite = PerformanceValidationSuite()
    baseline = {}
    candidate = {}
    for metric in REQUIRED_PERFORMANCE_METRICS:
        throughput = metric.endswith("items_per_second")
        unit = "items/s" if throughput else "seconds"
        baseline[metric] = PerformanceSample(metric, 10.0, unit, lower_is_better=not throughput)
        candidate[metric] = PerformanceSample(metric, 12.0 if throughput else 8.0, unit, lower_is_better=not throughput)
    rows = suite.compare(baseline, candidate)
    assert len(rows) == len(REQUIRED_PERFORMANCE_METRICS)
    assert all(row.improved for row in rows)


def test_restored_hierarchy_retains_levels(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    pipeline = MemoryLearningPipeline(writer)
    a = pipeline.observe_episode(EpisodeEvidence(1, 1, 1, True))
    b = pipeline.observe_episode(EpisodeEvidence(2, 1, 2, True))
    family = pipeline.derive_m2(action_id=1, member_ids=(a, b), outcome_class=1)
    writer.commit_generation()
    durable = DurableGenerationStore(tmp_path / "state.sqlite")
    try:
        store = RuntimeSnapshotStore(durable)
        store.persist(writer)
        restored = store.restore()
        assert restored.published_view.nodes[a].level == MemoryLevel.M1
        assert restored.published_view.nodes[family].level == MemoryLevel.M2
    finally:
        durable.close()
