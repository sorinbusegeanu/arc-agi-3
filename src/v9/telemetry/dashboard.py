from __future__ import annotations

from typing import Any


PRIMARY_KEYS = (
    "success_rate",
    "behavioral_success_rate",
    "behavioral_success_gain",
    "trajectory_efficiency",
    "M0_count",
    "M1_count",
    "M2_count",
    "M3_count",
    "M4_count",
    "M4_validated",
    "M5_count",
    "M6_count",
    "M7_count",
    "memory_growth_rate",
    "compression_ratio",
    "prediction_error",
    "cross_family_transfer",
    "false_transfer_rate",
    "reasoning_cycles",
    "final_vs_initial_candidate_improvement",
    "deliberation_behavior_improvement",
    "HGT_consequence_error",
    "HGT_strategy_ranking_accuracy",
    "HGT_candidate_refinement_success",
    "HGT_training_loss",
    "HGT_validation_loss",
    "historical_retention",
    "current_curriculum_gain",
    "cross_family_validation_gain",
    "ModelVersion",
    "GPU_memory",
    "sampling_rate",
    "ingestion_rate",
    "derivation_rate",
    "coordinator_action_requests",
)


def build_primary_dashboard(runtime_metrics: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    levels = dict(runtime_metrics.get("memory_levels", {}))
    transfer_trials = int(runtime_metrics.get("transfer_trials", 0))
    transfer_scopes = int(runtime_metrics.get("transfer_trust_scopes", 0))
    false_transfers = int(runtime_metrics.get("failed_transfer_scopes", 0))
    deliberation_decisions = int(diagnostic.get("deliberation_decisions", 0))
    reasoning_cycles = int(diagnostic.get("reasoning_cycles", 0))
    initial_sum = float(diagnostic.get("initial_candidate_score", 0.0))
    best_sum = float(diagnostic.get("best_candidate_score", 0.0))
    dashboard = {
        "success_rate": float(runtime_metrics.get("success_rate", 0.0)),
        "behavioral_success_rate": float(diagnostic.get("behavioral_success_rate", 0.0)),
        "behavioral_success_gain": float(diagnostic.get("behavioral_success_gain", 0.0)),
        "trajectory_efficiency": float(runtime_metrics.get("trajectory_efficiency", 0.0)),
        "M0_count": int(levels.get("M0", 0)),
        "M1_count": int(levels.get("M1", 0)),
        "M2_count": int(levels.get("M2", 0)),
        "M3_count": int(levels.get("M3", 0)),
        "M4_count": int(levels.get("M4", 0)),
        "M4_validated": int(runtime_metrics.get("m4_validated", 0)),
        "M5_count": int(levels.get("M5", 0)),
        "M6_count": int(levels.get("M6", 0)),
        "M7_count": int(levels.get("M7", 0)),
        "memory_growth_rate": float(runtime_metrics.get("persistent_memory_growth_ratio", 0.0)),
        "compression_ratio": float(runtime_metrics.get("compression_ratio", 0.0)),
        "prediction_error": float(runtime_metrics.get("prediction_error", 0.0)),
        "cross_family_transfer": float(runtime_metrics.get("cross_family_transfer", 0.0)),
        "false_transfer_rate": (false_transfers / transfer_scopes) if transfer_scopes else 0.0,
        "reasoning_cycles": (reasoning_cycles / deliberation_decisions) if deliberation_decisions else 0.0,
        "final_vs_initial_candidate_improvement": ((best_sum - initial_sum) / deliberation_decisions) if deliberation_decisions else 0.0,
        "deliberation_behavior_improvement": float(diagnostic.get("changed_decisions_with_better_outcome_rate", 0.0)),
        "HGT_consequence_error": float(diagnostic.get("hgt_consequence_error", 0.0)),
        "HGT_strategy_ranking_accuracy": float(diagnostic.get("hgt_strategy_ranking_accuracy", 0.0)),
        "HGT_candidate_refinement_success": float(diagnostic.get("hgt_candidate_refinement_success", 0.0)),
        "HGT_training_loss": float(diagnostic.get("hgt_training_loss", 0.0)),
        "HGT_validation_loss": float(diagnostic.get("hgt_validation_loss", 0.0)),
        "historical_retention": float(diagnostic.get("historical_retention", 0.0)),
        "current_curriculum_gain": float(diagnostic.get("current_curriculum_gain", 0.0)),
        "cross_family_validation_gain": float(diagnostic.get("cross_family_validation_gain", 0.0)),
        "ModelVersion": str(diagnostic.get("model_version", "untrained")),
        "GPU_memory": int(diagnostic.get("gpu_memory_bytes", 0)),
        "inference_latency": float(diagnostic.get("inference_latency_ms", 0.0)),
        "training_step_latency": float(diagnostic.get("training_step_latency_ms", 0.0)),
        "active_actor_processes": int(diagnostic.get("active_actor_processes", 0)),
        "active_ingest_workers": int(diagnostic.get("active_ingest_workers", 0)),
        "active_derivation_workers": int(diagnostic.get("active_derivation_workers", 0)),
        "sampled_steps": int(diagnostic.get("sampled_steps", 0)),
        "ingested_steps": int(diagnostic.get("ingested_steps", 0)),
        "sampling_backlog": int(diagnostic.get("sampling_backlog", 0)),
        "ingest_queue_depth": int(diagnostic.get("ingest_queue_depth", 0)),
        "derivation_queue_depth": int(diagnostic.get("derivation_queue_depth", 0)),
        "sampling_rate": float(diagnostic.get("sampling_rate", 0.0)),
        "ingestion_rate": float(diagnostic.get("ingestion_rate", 0.0)),
        "derivation_rate": float(diagnostic.get("derivation_rate", 0.0)),
        "memory_result_queue_depth": int(diagnostic.get("memory_result_queue_depth", 0)),
        "canonical_apply_rate": float(diagnostic.get("canonical_apply_rate", 0.0)),
        "canonical_apply_latency_ms": float(diagnostic.get("canonical_apply_latency_ms", 0.0)),
        "canonical_batch_size": int(diagnostic.get("canonical_batch_size", 0)),
        "coordinator_action_requests": int(diagnostic.get("coordinator_action_requests", 0)),
        "policy_snapshot_generation": int(diagnostic.get("policy_snapshot_generation", 0)),
        "policy_snapshot_refreshes": int(diagnostic.get("policy_snapshot_refreshes", 0)),
    }
    return {key: dashboard[key] for key in PRIMARY_KEYS}
