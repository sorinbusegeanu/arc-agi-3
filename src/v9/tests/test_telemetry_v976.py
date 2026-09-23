from __future__ import annotations

from threading import Event, Thread

from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.telemetry import (
    ConsolidationSample,
    HGTInferenceSample,
    HGTTrainingSample,
    ModelEvolutionSample,
    OptimizationSample,
)
from v9.telemetry.dashboard import PRIMARY_KEYS



def test_unified_telemetry_covers_all_v976_categories_and_persists(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    provenance = runtime.telemetry_provenance(
        decision_uid="D1",
        curriculum_step="step3",
        environment_family="gym",
        game_scenario="FrozenLake-v1",
    )

    runtime.record_deliberation_metrics(
        reasoning_cycles=3,
        initial_score=1.0,
        final_score=1.5,
        best_score=2.0,
        changed=True,
        behavior_improved=True,
        reasoning_cost=4.0,
        stop_reason="AMBIGUITY_RESOLVED",
        provenance=provenance,
    )
    runtime.record_hgt_inference(
        HGTInferenceSample(
            consequence_error=0.2,
            strategy_ranking_correct=True,
            candidate_refinement_success=True,
            subgraph_nodes=320,
            subgraph_edges=900,
            inference_latency_ms=12.0,
            relevance_precision=0.8,
            correspondence_accuracy=0.7,
            behavior_delta=0.1,
        ),
        provenance=provenance,
    )
    runtime.record_hgt_training(
        HGTTrainingSample(
            training_loss=0.5,
            validation_loss=0.6,
            training_step_latency_ms=18.0,
            training_examples_seen=5000,
            effective_batch_size=32,
            gradient_norm=1.2,
            learning_rate=1e-4,
            training_steps=100,
            examples_per_second=220.0,
            gpu_memory_bytes=4_000_000_000,
            gpu_utilization=75.0,
            historical_retention=0.93,
            current_curriculum_gain=0.08,
            cross_family_validation_gain=0.04,
            loss_by_head={"relevance": 0.2, "optimization": 0.3},
        ),
        provenance=provenance,
    )
    runtime.record_model_evolution(
        ModelEvolutionSample(
            model_version="hgt-2",
            parent_model_version="hgt-1",
            training_examples_since_parent=5000,
            current_stage_delta=0.08,
            historical_retention_delta=-0.01,
            cross_family_transfer_delta=0.03,
            reasoning_improvement_delta=0.05,
            inference_latency_delta_ms=1.0,
            promotion_result="PUBLISHED",
        ),
        provenance=provenance,
    )
    runtime.record_hgt_consolidation(
        ConsolidationSample(
            hydra_bytes_retired=1024,
            hydra_nodes_retired=10,
            hydra_nodes_replaced_by_abstractions=3,
            replay_examples_before=100,
            replay_examples_after=25,
            representative_retention_ratio=0.9,
            hgt_retention_before=0.95,
            hgt_retention_after=0.94,
            reactivation_examples=5,
        ),
        provenance=provenance,
    )
    runtime.record_optimization(
        OptimizationSample(
            initial_solution_cost=20.0,
            optimized_solution_cost=12.0,
            initial_solution_reliability=0.8,
            optimized_solution_reliability=0.85,
            optimization_cycles=4,
            candidates_generated=8,
            candidates_refined=4,
            candidates_executed=2,
            predicted_cost=11.0,
            realized_cost=12.0,
            predicted_reliability=0.84,
            realized_success=True,
            outcome_preserved=True,
            reasoning_cost=5.0,
            source_environment_family="sokoban",
            target_environment_family="minigrid",
        ),
        provenance=provenance,
    )

    metrics = runtime.metrics()
    dashboard = metrics["primary_dashboard"]
    diagnostics = metrics["telemetry_diagnostics"]

    assert tuple(dashboard) == PRIMARY_KEYS
    assert len(dashboard) == 24
    assert dashboard["ModelVersion"] == "hgt-2"
    assert dashboard["GPU_memory_GB"] == round(4_000_000_000 / (1024.0 ** 3), 2)

    assert diagnostics["reasoning_cycles"] == 3.0
    assert diagnostics["initial_candidate_score"] == 1.0
    assert diagnostics["best_candidate_score"] == 2.0
    assert diagnostics["hgt_consequence_error"] == 0.2
    assert diagnostics["hgt_strategy_ranking_accuracy"] == 1.0
    assert diagnostics["hgt_candidate_refinement_success"] == 1.0
    assert diagnostics["hgt_training_loss"] == 0.5
    assert diagnostics["hgt_validation_loss"] == 0.6
    assert diagnostics["historical_retention"] == 0.93
    assert diagnostics["current_curriculum_gain"] == 0.08
    assert diagnostics["cross_family_validation_gain"] == 0.04
    assert diagnostics["reasoning_stop_reasons"]["AMBIGUITY_RESOLVED"] == 1
    assert diagnostics["replay_compression_ratio"] == 0.75
    assert diagnostics["mean_relative_efficiency_gain"] == 0.4
    assert diagnostics["optimization_outcome_preservation_rate"] == 1.0
    assert diagnostics["last_provenance"]["decision_uid"] == "D1"

    runtime.close()
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    restored_metrics = restored.metrics()
    assert restored_metrics["primary_dashboard"]["ModelVersion"] == "hgt-2"
    assert restored_metrics["telemetry_diagnostics"]["model_promotions"] == 1


def test_prediction_error_is_exposed_as_primary_runtime_metric(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.submit(runtime.make_experience(
        producer_id=1,
        producer_sequence=1,
        environment_instance_id=7,
        global_step=0,
        context_signature=3,
        action_id=2,
        outcome_signature=4,
        prediction_error=0.75,
    ))
    assert runtime.metrics()["prediction_error"] == 0.75


def test_dashboard_exposes_all_memory_levels(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    dashboard = runtime.metrics()["primary_dashboard"]
    for level in range(8):
        assert f"M{level}_count" in dashboard
    assert "M4_validated" in dashboard


def test_metrics_waits_for_an_in_progress_runtime_mutation(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    mutation_started = Event()
    release_mutation = Event()
    metrics_finished = Event()
    failures: list[BaseException] = []

    def hold_runtime_mutation() -> None:
        with runtime._lock:
            mutation_started.set()
            release_mutation.wait(timeout=2.0)

    def read_metrics() -> None:
        try:
            runtime.metrics()
        except BaseException as exc:
            failures.append(exc)
        finally:
            metrics_finished.set()

    mutation = Thread(target=hold_runtime_mutation)
    reader = Thread(target=read_metrics)
    mutation.start()
    assert mutation_started.wait(timeout=1.0)
    reader.start()
    assert not metrics_finished.wait(timeout=0.05)
    release_mutation.set()
    mutation.join(timeout=1.0)
    reader.join(timeout=1.0)

    assert metrics_finished.is_set()
    assert not failures


def test_hgt_probability_metrics_are_bounded(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.record_hgt_inference(
        HGTInferenceSample(
            consequence_error=0.0,
            strategy_ranking_correct=True,
            candidate_refinement_success=True,
            subgraph_nodes=1,
            subgraph_edges=1,
            inference_latency_ms=1.0,
            relevance_precision=1.5,
            correspondence_accuracy=2.0,
        )
    )
    metrics = runtime.metrics()["telemetry_diagnostics"]
    assert metrics["relevance_precision"] == 1.0
    assert metrics["correspondence_accuracy"] == 1.0
