from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .planner_types import PlannerState
from .simple_explorer_types import SimpleFrontierState
from .full_explorer_types import FullFrontierState
from .memory_types import MemoryState


@dataclass
class Disagreement:
    type: str
    participants: List[str]
    opened_step: int
    resolved_step: Optional[int]
    resolution_tests: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "open"


@dataclass
class Blackboard:
    run_id: str
    game_id: str
    seed: int
    step_idx: int
    state_hash: str
    primary_grid: Dict[str, Any]
    fp_current: Dict[str, Any]
    history: List[Dict[str, Any]]
    budgets: Dict[str, int]
    phase: str
    action_schema: Dict[str, Any]
    fp_history: List[Dict[str, Any]] = field(default_factory=list)

    simple_explorer: Optional[Dict[str, Any]] = None
    full_explorer: Optional[Dict[str, Any]] = None
    rule_proposer: Optional[Dict[str, Any]] = None
    mechanic_classifier: Optional[Dict[str, Any]] = None
    goal_detector: Optional[Dict[str, Any]] = None
    planner: Optional[Dict[str, Any]] = None
    simple_explorer_meta: Optional[Dict[str, Any]] = None
    full_explorer_meta: Optional[Dict[str, Any]] = None
    rule_proposer_meta: Optional[Dict[str, Any]] = None
    mechanic_classifier_meta: Optional[Dict[str, Any]] = None
    goal_detector_meta: Optional[Dict[str, Any]] = None
    action_selection_report: Optional[Dict[str, Any]] = None
    planner_decision: Optional[Dict[str, Any]] = None
    planner_inputs_audit: Optional[Dict[str, Any]] = None
    memory: Optional[MemoryState] = None
    memory_meta: Optional[Dict[str, Any]] = None
    memory_evidence: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    fp_step_buffer: List[Dict[str, Any]] = field(default_factory=list)
    transition_events: List[Any] = field(default_factory=list)
    hypotheses_engine: Optional[List[Any]] = None
    hypotheses_engine_meta: Optional[Dict[str, Any]] = None
    test_selector_report: Optional[Dict[str, Any]] = None
    hypothesis_conf_deltas: List[float] = field(default_factory=list)
    conflict_open: bool = False

    disagreements: List[Disagreement] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    planner_state: PlannerState = field(default_factory=PlannerState)
    resolution_tests: List[Dict[str, Any]] = field(default_factory=list)
    simple_frontier_state: SimpleFrontierState = field(default_factory=SimpleFrontierState)
    full_frontier_state: FullFrontierState = field(default_factory=FullFrontierState)
