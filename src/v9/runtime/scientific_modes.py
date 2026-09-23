from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScientificVisibilityMode(str, Enum):
    """Actor-visible publication regimes defined by the v9.7.16 design."""

    ASYNC_DEVELOPMENT = "ASYNC_DEVELOPMENT"
    MATCHED_REASONING = "MATCHED_REASONING"


class LearnedDevelopmentalFeedbackProfile(str, Enum):
    """Whether learned HGT proposals may influence Hydra development."""

    DISABLED = "DISABLED"
    ENABLED = "ENABLED"


@dataclass(frozen=True, slots=True)
class H17StabilityMetrics:
    reversal_rate: float
    churn_rate: float
    structural_persistence: float
    created_memories: int
    retired_memories: int
    active_memories: int
    useful_novel_structures: int
    prediction_quality: float
    transfer_quality: float
    memory_growth: int

    def __post_init__(self) -> None:
        if min(self.created_memories, self.retired_memories, self.active_memories, self.useful_novel_structures) < 0:
            raise ValueError("H17 memory counts must be non-negative")


def coerce_visibility_mode(value: ScientificVisibilityMode | str) -> ScientificVisibilityMode:
    if isinstance(value, ScientificVisibilityMode):
        return value
    try:
        return ScientificVisibilityMode(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unsupported scientific visibility mode: {value!r}") from exc


def coerce_feedback_profile(
    value: LearnedDevelopmentalFeedbackProfile | str,
) -> LearnedDevelopmentalFeedbackProfile:
    if isinstance(value, LearnedDevelopmentalFeedbackProfile):
        return value
    try:
        return LearnedDevelopmentalFeedbackProfile(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unsupported learned developmental feedback profile: {value!r}") from exc


__all__ = [
    "H17StabilityMetrics",
    "LearnedDevelopmentalFeedbackProfile",
    "ScientificVisibilityMode",
    "coerce_feedback_profile",
    "coerce_visibility_mode",
]
