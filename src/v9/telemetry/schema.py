from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TelemetryProvenance:
    run_uid: str | None = None
    decision_uid: str | None = None
    graph_generation: int | None = None
    model_version: str | None = None
    scientific_config_id: str | None = None
    curriculum_step: str | None = None
    environment_family: str | None = None
    game_scenario: str | None = None
    context_uid: str | None = None
    lineage_uid: str | None = None
    memory_level: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class HGTInferenceSample:
    consequence_error: float
    strategy_ranking_correct: bool
    candidate_refinement_success: bool
    subgraph_nodes: int
    subgraph_edges: int
    inference_latency_ms: float
    relevance_precision: float | None = None
    correspondence_accuracy: float | None = None
    behavior_delta: float | None = None


@dataclass(frozen=True, slots=True)
class HGTTrainingSample:
    training_loss: float
    validation_loss: float
    training_step_latency_ms: float
    training_examples_seen: int
    effective_batch_size: int
    gradient_norm: float
    learning_rate: float
    training_steps: int
    examples_per_second: float
    gpu_memory_bytes: int
    gpu_utilization: float
    historical_retention: float
    current_curriculum_gain: float
    cross_family_validation_gain: float
    loss_by_head: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelEvolutionSample:
    model_version: str
    parent_model_version: str | None
    training_examples_since_parent: int
    current_stage_delta: float
    historical_retention_delta: float
    cross_family_transfer_delta: float
    reasoning_improvement_delta: float
    inference_latency_delta_ms: float
    promotion_result: str


@dataclass(frozen=True, slots=True)
class ConsolidationSample:
    hydra_bytes_retired: int
    hydra_nodes_retired: int
    hydra_nodes_replaced_by_abstractions: int
    replay_examples_before: int
    replay_examples_after: int
    representative_retention_ratio: float
    hgt_retention_before: float
    hgt_retention_after: float
    reactivation_examples: int = 0


@dataclass(frozen=True, slots=True)
class OptimizationSample:
    initial_solution_cost: float
    optimized_solution_cost: float
    initial_solution_reliability: float
    optimized_solution_reliability: float
    optimization_cycles: int
    candidates_generated: int
    candidates_refined: int
    candidates_executed: int
    predicted_cost: float
    realized_cost: float
    predicted_reliability: float
    realized_success: bool
    outcome_preserved: bool
    reasoning_cost: float
    source_environment_family: str | None = None
    target_environment_family: str | None = None

    @property
    def relative_efficiency_gain(self) -> float:
        if self.initial_solution_cost <= 0:
            return 0.0
        return (self.initial_solution_cost - self.optimized_solution_cost) / self.initial_solution_cost
