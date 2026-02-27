from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CandidateAction:
    type: str  # simple|coord
    action_id: str
    x: Optional[int] = None
    y: Optional[int] = None

    def key(self) -> Tuple[Any, ...]:
        if self.type == "coord":
            return (self.action_id, self.y, self.x)
        return (self.action_id,)


@dataclass
class CandidateMeta:
    source: str
    expected_signatures: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    refutes: List[str] = field(default_factory=list)
    novelty: float = 0.0
    disambiguation: float = 0.0
    expected_change: float = 0.0
    loop_risk: float = 0.0
    expected_progress: float = 0.0
    hypothesis_align: float = 0.0
    action_cost: float = 0.0
    score: float = 0.0


@dataclass
class PlannerState:
    recent_states: List[str] = field(default_factory=list)
    recent_actions: List[CandidateAction] = field(default_factory=list)
    recent_noop_actions: List[Tuple[Any, ...]] = field(default_factory=list)
    recent_state_actions: List[Tuple[str, Tuple[Any, ...]]] = field(default_factory=list)
    recent_state_action_noops: List[Tuple[str, Tuple[Any, ...]]] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    pending_tests: List[Dict[str, Any]] = field(default_factory=list)
    mode_history: List[str] = field(default_factory=list)


@dataclass
class DecisionTrace:
    mode: str
    candidates: List[Dict[str, Any]]
    chosen: Dict[str, Any]
    warnings: List[str]


@dataclass
class PlannerInputs:
    mechanic_prior: Optional[Dict[str, Any]] = None
    hypotheses_report: Optional[Dict[str, Any]] = None
    simple_report: Optional[Dict[str, Any]] = None
    full_report: Optional[Dict[str, Any]] = None
    goal_report: Optional[Dict[str, Any]] = None
    transition_graph: Optional[Dict[str, Any]] = None
