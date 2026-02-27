from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SignalEntry:
    signal_id: str
    value_start: float
    value_end: float
    delta: float
    weight: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProgressEstimate:
    progress_scalar: float
    progress_delta: float
    confidence: float
    direction: str


@dataclass
class GoalHints:
    likely_goal_type: str
    stop_condition_predicates: List[str]
    stall_risk: float = 0.0


@dataclass
class GoalDetectorReport:
    progress_estimate: ProgressEstimate
    signals: Dict[str, Any]
    goal_hints: GoalHints
    run_summary: Dict[str, Any]
