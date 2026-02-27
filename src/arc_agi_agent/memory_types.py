from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionEfficacy:
    attempts: int = 0
    no_effect_count: int = 0
    effect_count: int = 0
    avg_changed_cells: float = 0.0
    avg_changed_bbox_area: float = 0.0
    last_step_seen: int = 0
    event_signature_counts: Dict[str, int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class StateActionEfficacy:
    attempts: int = 0
    no_effect_count: int = 0
    last_effect_step: Optional[int] = None
    last_step_seen: int = 0


@dataclass
class CoordStat:
    attempts: int = 0
    no_effect_count: int = 0
    avg_changed_cells: float = 0.0
    last_step_seen: int = 0


@dataclass
class TemplateStats:
    times_considered: int = 0
    times_triggered: int = 0
    times_scored_positive: int = 0
    last_step_triggered: Optional[int] = None
    supporting_events: Dict[str, int] = field(default_factory=dict)


@dataclass
class MemoryState:
    version: str = "1.0"
    per_action: Dict[str, ActionEfficacy] = field(default_factory=dict)
    per_state_action: Dict[str, StateActionEfficacy] = field(default_factory=dict)
    coord_heatmaps: Dict[str, Dict[str, CoordStat]] = field(default_factory=dict)
    event_sig_window: List[Dict[str, int]] = field(default_factory=list)
    object_delta_window: List[Dict[str, int]] = field(default_factory=list)
    template_stats: Dict[str, TemplateStats] = field(default_factory=dict)
    recent_actions_by_state: Dict[str, List[str]] = field(default_factory=dict)
    mechanic_by_fingerprint: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    last_update_debug: Optional[Dict[str, Any]] = None


@dataclass
class MemoryUpdateInputs:
    ctx: Dict[str, Any]
    state_hash_before: str
    state_hash_after: str
    action: Dict[str, Any]
    action_schema: Dict[str, Any]
    fp_report_before: Optional[Dict[str, Any]] = None
    fp_report_after: Optional[Dict[str, Any]] = None
    diff_summary: Optional[Dict[str, Any]] = None
    fp_diff: Optional[Dict[str, Any]] = None
    planner_decision: Optional[Dict[str, Any]] = None
    goal_report: Optional[Dict[str, Any]] = None
    mechanic_classifier: Optional[Dict[str, Any]] = None
    rule_proposer: Optional[Dict[str, Any]] = None
    simple_report: Optional[Dict[str, Any]] = None
    full_report: Optional[Dict[str, Any]] = None
