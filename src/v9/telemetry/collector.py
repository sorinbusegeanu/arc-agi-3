from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from .schema import (
    ConsolidationSample,
    HGTInferenceSample,
    HGTTrainingSample,
    ModelEvolutionSample,
    OptimizationSample,
    TelemetryProvenance,
)


class UnifiedTelemetry:
    SCHEMA_VERSION = 1

    def __init__(self, *, model_version: str = "untrained") -> None:
        self.model_version = str(model_version)
        self.counters: Counter[str] = Counter()
        self.sums: dict[str, float] = {}
        self.gauges: dict[str, float | int | str] = {}
        self.stop_reasons: Counter[str] = Counter()
        self.loss_by_head: dict[str, float] = {}
        self.last_provenance: dict[str, Any] = {}

    def _sum(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def set_gauge(self, key: str, value: float | int | str) -> None:
        self.gauges[str(key)] = value

    def record_deliberation(
        self,
        *,
        reasoning_cycles: int,
        initial_score: float,
        final_score: float,
        best_score: float,
        changed: bool,
        behavior_improved: bool | None,
        reasoning_cost: float,
        stop_reason: str,
        provenance: TelemetryProvenance | None = None,
    ) -> None:
        self.counters["deliberation_decisions"] += 1
        self.counters["reasoning_cycles"] += max(0, int(reasoning_cycles))
        self.counters["deliberation_changed_decisions"] += int(bool(changed))
        if behavior_improved is not None:
            self.counters["deliberation_outcome_observations"] += 1
            self.counters["deliberation_behavior_improvements"] += int(bool(behavior_improved))
        self._sum("initial_candidate_score", initial_score)
        self._sum("final_candidate_score", final_score)
        self._sum("best_candidate_score", best_score)
        self._sum("candidate_improvement", best_score - initial_score)
        self._sum("reasoning_cost", reasoning_cost)
        self.stop_reasons[str(stop_reason)] += 1
        if provenance:
            self.last_provenance = provenance.as_dict()

    def record_hgt_inference(self, sample: HGTInferenceSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.counters["hgt_inference_samples"] += 1
        self.counters["hgt_strategy_ranking_correct"] += int(sample.strategy_ranking_correct)
        self.counters["hgt_candidate_refinement_success"] += int(sample.candidate_refinement_success)
        self._sum("hgt_consequence_error", sample.consequence_error)
        self._sum("hgt_subgraph_nodes", sample.subgraph_nodes)
        self._sum("hgt_subgraph_edges", sample.subgraph_edges)
        self._sum("hgt_inference_latency_ms", sample.inference_latency_ms)
        if sample.relevance_precision is not None:
            self.counters["hgt_relevance_samples"] += 1
            self._sum("hgt_relevance_precision", sample.relevance_precision)
        if sample.correspondence_accuracy is not None:
            self.counters["hgt_correspondence_samples"] += 1
            self._sum("hgt_correspondence_accuracy", sample.correspondence_accuracy)
        if sample.behavior_delta is not None:
            self.counters["hgt_behavior_samples"] += 1
            self._sum("hgt_behavior_delta", sample.behavior_delta)
            self.counters["hgt_behavior_improvements"] += int(sample.behavior_delta > 0)
            self.counters["hgt_behavior_regressions"] += int(sample.behavior_delta < 0)
        if provenance:
            self.last_provenance = provenance.as_dict()

    def record_hgt_training(self, sample: HGTTrainingSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.counters["hgt_training_reports"] += 1
        self.gauges.update({
            "hgt_training_loss": float(sample.training_loss),
            "hgt_validation_loss": float(sample.validation_loss),
            "training_step_latency_ms": float(sample.training_step_latency_ms),
            "training_examples_seen": int(sample.training_examples_seen),
            "effective_batch_size": int(sample.effective_batch_size),
            "gradient_norm": float(sample.gradient_norm),
            "learning_rate": float(sample.learning_rate),
            "training_steps": int(sample.training_steps),
            "examples_per_second": float(sample.examples_per_second),
            "gpu_memory_bytes": int(sample.gpu_memory_bytes),
            "gpu_utilization": float(sample.gpu_utilization),
            "historical_retention": float(sample.historical_retention),
            "current_curriculum_gain": float(sample.current_curriculum_gain),
            "cross_family_validation_gain": float(sample.cross_family_validation_gain),
            "train_validation_gap": float(sample.validation_loss - sample.training_loss),
        })
        self.loss_by_head = {str(k): float(v) for k, v in sample.loss_by_head.items()}
        if provenance:
            self.last_provenance = provenance.as_dict()

    def record_model_evolution(self, sample: ModelEvolutionSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.model_version = str(sample.model_version)
        self.counters["model_evaluations"] += 1
        if str(sample.promotion_result).upper() in {"PROMOTED", "ACCEPTED", "PUBLISHED"}:
            self.counters["model_promotions"] += 1
        self.gauges.update({
            "model_version": self.model_version,
            "parent_model_version": sample.parent_model_version or "",
            "training_examples_since_parent": int(sample.training_examples_since_parent),
            "current_stage_delta": float(sample.current_stage_delta),
            "historical_retention_delta": float(sample.historical_retention_delta),
            "cross_family_transfer_delta": float(sample.cross_family_transfer_delta),
            "reasoning_improvement_delta": float(sample.reasoning_improvement_delta),
            "inference_latency_delta_ms": float(sample.inference_latency_delta_ms),
            "promotion_result": str(sample.promotion_result),
        })
        if provenance:
            self.last_provenance = provenance.as_dict()

    def record_consolidation(self, sample: ConsolidationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.counters["consolidation_events"] += 1
        self.counters["hydra_bytes_retired"] += max(0, int(sample.hydra_bytes_retired))
        self.counters["hydra_nodes_retired"] += max(0, int(sample.hydra_nodes_retired))
        self.counters["hydra_nodes_replaced_by_abstractions"] += max(0, int(sample.hydra_nodes_replaced_by_abstractions))
        self.counters["replay_examples_before"] += max(0, int(sample.replay_examples_before))
        self.counters["replay_examples_after"] += max(0, int(sample.replay_examples_after))
        self.counters["reactivation_examples"] += max(0, int(sample.reactivation_examples))
        self.gauges.update({
            "representative_retention_ratio": float(sample.representative_retention_ratio),
            "hgt_retention_before_consolidation": float(sample.hgt_retention_before),
            "hgt_retention_after_consolidation": float(sample.hgt_retention_after),
        })
        if provenance:
            self.last_provenance = provenance.as_dict()

    def record_optimization(self, sample: OptimizationSample, *, provenance: TelemetryProvenance | None = None) -> None:
        self.counters["optimization_events"] += 1
        self.counters["optimization_cycles"] += max(0, int(sample.optimization_cycles))
        self.counters["optimization_candidates_generated"] += max(0, int(sample.candidates_generated))
        self.counters["optimization_candidates_refined"] += max(0, int(sample.candidates_refined))
        self.counters["optimization_candidates_executed"] += max(0, int(sample.candidates_executed))
        self.counters["optimization_outcome_preserved"] += int(sample.outcome_preserved)
        self.counters["optimization_realized_success"] += int(sample.realized_success)
        self._sum("initial_solution_cost", sample.initial_solution_cost)
        self._sum("optimized_solution_cost", sample.optimized_solution_cost)
        self._sum("initial_solution_reliability", sample.initial_solution_reliability)
        self._sum("optimized_solution_reliability", sample.optimized_solution_reliability)
        self._sum("predicted_cost", sample.predicted_cost)
        self._sum("realized_cost", sample.realized_cost)
        self._sum("predicted_reliability", sample.predicted_reliability)
        self._sum("optimization_reasoning_cost", sample.reasoning_cost)
        self._sum("relative_efficiency_gain", sample.relative_efficiency_gain)
        if provenance:
            self.last_provenance = provenance.as_dict()

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    def diagnostic_metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_version": self.model_version,
            **dict(self.counters),
            **dict(self.sums),
            **dict(self.gauges),
            "loss_by_head": dict(self.loss_by_head),
            "reasoning_stop_reasons": dict(self.stop_reasons),
            "last_provenance": dict(self.last_provenance),
        }
        d = self.counters["deliberation_decisions"]
        result["decisions_changed_by_deliberation_rate"] = self._ratio(self.counters["deliberation_changed_decisions"], d)
        result["changed_decisions_with_better_outcome_rate"] = self._ratio(
            self.counters["deliberation_behavior_improvements"],
            self.counters["deliberation_outcome_observations"],
        )
        i = self.counters["hgt_inference_samples"]
        result["hgt_consequence_error"] = self._ratio(self.sums.get("hgt_consequence_error", 0.0), i)
        result["hgt_strategy_ranking_accuracy"] = self._ratio(self.counters["hgt_strategy_ranking_correct"], i)
        result["hgt_candidate_refinement_success"] = self._ratio(self.counters["hgt_candidate_refinement_success"], i)
        result["inference_latency_ms"] = self._ratio(self.sums.get("hgt_inference_latency_ms", 0.0), i)
        result["mean_subgraph_nodes"] = self._ratio(self.sums.get("hgt_subgraph_nodes", 0.0), i)
        result["mean_subgraph_edges"] = self._ratio(self.sums.get("hgt_subgraph_edges", 0.0), i)
        result["relevance_precision"] = self._ratio(self.sums.get("hgt_relevance_precision", 0.0), self.counters["hgt_relevance_samples"])
        result["correspondence_accuracy"] = self._ratio(self.sums.get("hgt_correspondence_accuracy", 0.0), self.counters["hgt_correspondence_samples"])
        result["hgt_behavior_improvement_rate"] = self._ratio(self.counters["hgt_behavior_improvements"], self.counters["hgt_behavior_samples"])
        result["hgt_behavior_regression_rate"] = self._ratio(self.counters["hgt_behavior_regressions"], self.counters["hgt_behavior_samples"])
        o = self.counters["optimization_events"]
        result["mean_relative_efficiency_gain"] = self._ratio(self.sums.get("relative_efficiency_gain", 0.0), o)
        result["optimization_outcome_preservation_rate"] = self._ratio(self.counters["optimization_outcome_preserved"], o)
        result["replay_compression_ratio"] = 1.0 - self._ratio(self.counters["replay_examples_after"], self.counters["replay_examples_before"]) if self.counters["replay_examples_before"] else 0.0
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_version": self.model_version,
            "counters": dict(self.counters),
            "sums": dict(self.sums),
            "gauges": dict(self.gauges),
            "stop_reasons": dict(self.stop_reasons),
            "loss_by_head": dict(self.loss_by_head),
            "last_provenance": dict(self.last_provenance),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "UnifiedTelemetry":
        if int(state.get("schema_version", 0)) != cls.SCHEMA_VERSION:
            raise ValueError("incompatible telemetry state")
        result = cls(model_version=str(state.get("model_version", "untrained")))
        result.counters.update({str(k): int(v) for k, v in dict(state.get("counters", {})).items()})
        result.sums = {str(k): float(v) for k, v in dict(state.get("sums", {})).items()}
        result.gauges = dict(state.get("gauges", {}))
        result.stop_reasons.update({str(k): int(v) for k, v in dict(state.get("stop_reasons", {})).items()})
        result.loss_by_head = {str(k): float(v) for k, v in dict(state.get("loss_by_head", {})).items()}
        result.last_provenance = dict(state.get("last_provenance", {}))
        return result
