from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .frontier_priority import CoordActionCandidate


StateKey = str
ActionFamilyKey = str
CoordActionKey = Tuple[str, int, int]


@dataclass
class RunSummary:
    game_id: str
    seed: int
    steps_executed: int
    unique_states: int
    unique_transitions: int
    unique_coord_actions_tried: int
    loops_detected: Dict[str, int]
    termination_reason: str


@dataclass
class CoordActionEffectStats:
    attempts_total: int
    attempts_by_coord_selector: Dict[str, int]
    no_effect_rate: float
    avg_changed_cells: float
    avg_changed_bbox_area: float
    dominant_event_signatures: List[Tuple[str, float]]
    hotspots: List[Tuple[int, int, float]]
    negative_zones: List[Tuple[int, int, float]]


@dataclass
class TransitionEdge:
    from_state: StateKey
    action_id: str
    x: int
    y: int
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
    edges: Dict[Tuple[StateKey, str, int, int, StateKey], TransitionEdge]


@dataclass
class FrontierEntry:
    state: StateKey
    pending_candidates: int
    attempt_counts_by_action_family: Dict[str, int]
    cooldowns: Dict[str, int]
    banlist: List[Tuple[str, int, int]]


@dataclass
class FullExplorerReport:
    run_summary: RunSummary
    coord_action_effect_model: Dict[ActionFamilyKey, CoordActionEffectStats]
    transition_graph: TransitionGraph
    frontier: Dict[StateKey, FrontierEntry]
    artifacts: Dict[str, Any]


@dataclass
class FullFrontierState:
    state_frontier: Dict[StateKey, List[CoordActionCandidate]] = field(default_factory=dict)
    state_noop_counts: Dict[StateKey, Dict[CoordActionKey, int]] = field(default_factory=dict)
    state_banlist: Dict[StateKey, Set[CoordActionKey]] = field(default_factory=dict)
    global_noop_counts: Dict[CoordActionKey, int] = field(default_factory=dict)
    coord_tried: Dict[CoordActionKey, int] = field(default_factory=dict)
    action_coord_tried: Dict[str, Set[Tuple[int, int]]] = field(default_factory=dict)
    action_attempts: Dict[str, int] = field(default_factory=dict)
    last_processed_step: int = -1
