from __future__ import annotations

from typing import Any


PRIMARY_KEYS = (
    "success_rate",
    "trajectory_efficiency",
    "M0_count",
    "M3_count",
    "M4_validated",
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
    "inference_latency",
    "training_step_latency",
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
        "trajectory_efficiency": float(runtime_metrics.get("trajectory_efficiency", 0.0)),
        "M0_count": int(levels.get("M0", 0)),
        "M3_count": int(levels.get("M3", 0)),
        "M4_validated": int(runtime_metrics.get("m4_validated", 0)),
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
    }
    return {key: dashboard[key] for key in PRIMARY_KEYS}
