from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HypothesisStatus(str, Enum):
    UNTESTED = "UNTESTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUPPORTED = "SUPPORTED"
    FALSIFIED = "FALSIFIED"


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    hypothesis: str
    status: HypothesisStatus
    effect: float | None
    trials: int
    reason: str


@dataclass(frozen=True, slots=True)
class HypothesisAssessment:
    hypothesis: str
    raw_decision: str
    quality_gate: str
    dependency_gate: str
    final_decision: HypothesisStatus
    blocker: str | None
    evidence_counts: dict[str, int]
    scientific_config_id: str


def untested_assessment(hypothesis: str, *, scientific_config_id: str, blocker: str) -> HypothesisAssessment:
    return HypothesisAssessment(hypothesis, HypothesisStatus.UNTESTED.value, "NOT_EVALUATED", "NOT_EVALUATED", HypothesisStatus.UNTESTED, blocker, {}, scientific_config_id)
