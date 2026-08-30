from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    ChainStatus,
    Confidence,
    ExperimentPurpose,
    FailureCategory,
    FailureLevel,
    PredictionDirection,
)


@dataclass(frozen=True)
class ChainEdgeEvidence:
    edge: str
    status: ChainStatus
    evidence_count: int = 0
    evidence_ids: tuple[str, ...] = ()
    blocker: str | None = None


@dataclass(frozen=True)
class ChainAuditResult:
    edges: tuple[ChainEdgeEvidence, ...]
    first_broken_link: str | None
    complete: bool


@dataclass(frozen=True)
class RuntimeHypothesis:
    hypothesis_id: str
    claim: str
    target_chain_edge: str
    status: str = "ACTIVE"
    confidence: Confidence = Confidence.MEDIUM
    failure_level: FailureLevel = FailureLevel.LOCAL
    category: FailureCategory = FailureCategory.UNKNOWN
    parent_hypothesis_id: str | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentPrediction:
    hypothesis_id: str
    metric: str
    direction: PredictionDirection
    expected_min_effect: float = 0.0
    flat_tolerance: float = 0.0
    falsifier: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    purpose: ExperimentPurpose
    code_revision: str
    snapshot_id: str | None
    games: tuple[str, ...]
    seeds: tuple[int, ...]
    interaction_budget: int
    conditions: Mapping[str, Mapping[str, Any]]
    predictions: tuple[ExperimentPrediction, ...]

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.code_revision:
            raise ValueError("code_revision is required")
        if not self.games:
            raise ValueError("at least one game is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.interaction_budget <= 0:
            raise ValueError("interaction_budget must be positive")
        if "control" not in self.conditions:
            raise ValueError("a control condition is required")
        if not self.predictions:
            raise ValueError("at least one preregistered prediction is required")


@dataclass(frozen=True)
class MetricResult:
    metric: str
    control_mean: float
    treatment_mean: float
    control_variance: float = 0.0
    treatment_variance: float = 0.0
    sample_count: int = 1

    @property
    def effect(self) -> float:
        return self.treatment_mean - self.control_mean
