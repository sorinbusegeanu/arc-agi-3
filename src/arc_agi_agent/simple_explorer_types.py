from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


StateKey = str
ActionKey = str


@dataclass
class RunSummary:
    game_id: str
    seed: int
    steps_executed: int
    unique_states: int
    unique_transitions: int
    loops_detected: Dict[str, int]
    timeouts_or_errors: List[str]


@dataclass
class ActionEffectStats:
    attempts: int
    no_effect_rate: float
    avg_changed_cells: float
    avg_changed_bbox_area: float
    dominant_event_signatures: List[Tuple[str, float]]
    typical_motion_vectors: List[Tuple[float, float, int]]
    common_block_conditions: List[str]


@dataclass
class TransitionEdge:
    from_state: StateKey
    action: ActionKey
    to_state: StateKey
    count: int
    avg_changed_cells: float
    avg_changed_bbox_area: float
    event_signature_histogram: Dict[str, int]
    example_steps: List[int] = field(default_factory=list)


@dataclass
class TransitionNode:
    state: StateKey
    height: int
    width: int
    palette_size: int
    object_count: int


@dataclass
class TransitionGraph:
    nodes: Dict[StateKey, TransitionNode]
    edges: Dict[Tuple[StateKey, ActionKey, StateKey], TransitionEdge]


@dataclass
class FrontierEntry:
    state: StateKey
    untried_actions: List[ActionKey]
    action_attempt_counts: Dict[ActionKey, int]


@dataclass
class SimpleExplorerReport:
    run_summary: RunSummary
    action_effect_model: Dict[ActionKey, ActionEffectStats]
    transition_graph: TransitionGraph
    frontier: Dict[StateKey, FrontierEntry]
    artifacts: Dict[str, Any]


@dataclass
class SimpleFrontierState:
    state_untried: Dict[StateKey, List[ActionKey]] = field(default_factory=dict)
    state_action_attempts: Dict[StateKey, Dict[ActionKey, int]] = field(default_factory=dict)
    state_action_outcomes: Dict[StateKey, Dict[ActionKey, Dict[StateKey, int]]] = field(default_factory=dict)
    last_processed_step: int = -1
