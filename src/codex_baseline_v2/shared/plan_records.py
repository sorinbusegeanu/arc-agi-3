from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _to_dict(obj: object) -> Dict[str, Any]:
    return dict(obj.__dict__)  # dataclasses only in this module


@dataclass(frozen=True)
class SkillSpecV1:
    schema_version: str
    skill_id: str
    skill_type: str
    parameter_names: List[str]
    precondition_ids: List[str]
    expected_effect_node_ids: List[str]
    average_duration_steps: float
    success_rate: float
    failure_mode_labels: List[str]
    source_trace_ids: List[str]
    total_attempt_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    active: bool = True
    executions_this_round: int = 0
    latest_termination_reason_this_round: Optional[str] = None
    repeated_contact_no_effect_count_this_round: int = 0
    target_x_this_round: Optional[int] = None
    target_y_this_round: Optional[int] = None
    historical_contact_no_effect_count: int = 0
    latest_prior_round_termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SkillSpecV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.2")),
            skill_id=str(payload.get("skill_id", "")),
            skill_type=str(payload.get("skill_type", "unknown")),
            parameter_names=list(payload.get("parameter_names", [])),
            precondition_ids=list(payload.get("precondition_ids", [])),
            expected_effect_node_ids=list(payload.get("expected_effect_node_ids", [])),
            average_duration_steps=float(payload.get("average_duration_steps", 0.0)),
            success_rate=float(payload.get("success_rate", 0.0)),
            failure_mode_labels=list(payload.get("failure_mode_labels", [])),
            source_trace_ids=list(payload.get("source_trace_ids", [])),
            total_attempt_count=int(payload.get("total_attempt_count", 0)),
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            active=bool(payload.get("active", True)),
            executions_this_round=int(payload.get("executions_this_round", 0)),
            latest_termination_reason_this_round=payload.get("latest_termination_reason_this_round"),
            repeated_contact_no_effect_count_this_round=int(payload.get("repeated_contact_no_effect_count_this_round", 0)),
            target_x_this_round=payload.get("target_x_this_round"),
            target_y_this_round=payload.get("target_y_this_round"),
            historical_contact_no_effect_count=int(payload.get("historical_contact_no_effect_count", 0)),
            latest_prior_round_termination_reason=payload.get("latest_prior_round_termination_reason"),
        )


@dataclass(frozen=True)
class SkillExecutionRecordV1:
    schema_version: str
    execution_id: str
    skill_id: str
    parameter_values: List[str]
    start_step: int
    end_step: int
    success: bool
    termination_reason: str
    observed_event_ids: List[str]
    observed_topology_delta_ids: List[str]
    updated_confidence_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SkillExecutionRecordV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.2")),
            execution_id=str(payload.get("execution_id", "")),
            skill_id=str(payload.get("skill_id", "")),
            parameter_values=list(payload.get("parameter_values", [])),
            start_step=int(payload.get("start_step", 0)),
            end_step=int(payload.get("end_step", 0)),
            success=bool(payload.get("success", False)),
            termination_reason=str(payload.get("termination_reason", "")),
            observed_event_ids=list(payload.get("observed_event_ids", [])),
            observed_topology_delta_ids=list(payload.get("observed_topology_delta_ids", [])),
            updated_confidence_delta=float(payload.get("updated_confidence_delta", 0.0)),
        )


@dataclass(frozen=True)
class PlannerBeliefStateV1:
    schema_version: str
    current_area_id: Optional[str]
    current_avatar_track_id: Optional[str]
    candidate_subgoal_ids: List[str]
    active_latent_state_ids: List[str]
    uncertain_latent_state_ids: List[str]
    reachable_frontier_ids: List[str]
    candidate_skill_ids: List[str]
    plan_memory_refs: List[str]
    recovery_reconstruction_debug: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerBeliefStateV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.2")),
            current_area_id=payload.get("current_area_id"),
            current_avatar_track_id=payload.get("current_avatar_track_id"),
            candidate_subgoal_ids=list(payload.get("candidate_subgoal_ids", [])),
            active_latent_state_ids=list(payload.get("active_latent_state_ids", [])),
            uncertain_latent_state_ids=list(payload.get("uncertain_latent_state_ids", [])),
            reachable_frontier_ids=list(payload.get("reachable_frontier_ids", [])),
            candidate_skill_ids=list(payload.get("candidate_skill_ids", [])),
            plan_memory_refs=list(payload.get("plan_memory_refs", [])),
            recovery_reconstruction_debug=dict(payload.get("recovery_reconstruction_debug", {})),
        )


@dataclass(frozen=True)
class PlanNodeV1:
    schema_version: str
    plan_node_id: str
    parent_id: Optional[str]
    depth: int
    subgoal_id: Optional[str]
    skill_id: Optional[str]
    estimated_cost: float
    estimated_success: float
    estimated_information_gain: float
    estimated_goal_progress: float
    blocked: bool
    notes: Optional[str]
    candidate_cluster_key: Optional[str] = None
    movement_cluster_key: Optional[str] = None
    cluster_failure_count: int = 0
    cluster_exhausted_flag: bool = False
    same_row_penalty: float = 0.0
    post_jump_exclusion_flag: bool = False
    distance_from_last_failed_cluster: Optional[float] = None
    excluded_by_cooldown: bool = False
    neighbor_of_failed_cluster: bool = False
    row_band_id: Optional[str] = None
    current_round_failed_neighbor_count: int = 0
    current_round_failed_row_count: int = 0
    recent_failed_movement_cluster_match: bool = False
    blocked_by_exact_cluster_exhaustion: bool = False
    blocked_by_neighbor_cooldown: bool = False
    blocked_by_post_jump_exclusion: bool = False
    blocked_by_unreachable: bool = False
    candidate_source: Optional[str] = None
    # Export-only debug semantics:
    # blocking_reason_codes = why this node was excluded.
    blocking_reason_codes: List[str] = field(default_factory=list)
    # surviving_unblocked_candidate_count = number of candidates left after all hard filters in this pass.
    surviving_unblocked_candidate_count: int = 0
    pre_filter_survived: bool = False
    hard_filter_applied: bool = False
    post_filter_survived: bool = False
    rank_removed_reason: Optional[str] = None
    # pre_filter_rank_position = score/model order before hard blocking.
    pre_filter_rank_position: Optional[int] = None
    # post_filter_rank_position = final rank among surviving unblocked candidates only.
    post_filter_rank_position: Optional[int] = None
    selected_after_rerank: bool = False
    candidate_final_status: Optional[str] = None
    blocking_reason_details: Dict[str, Any] = field(default_factory=dict)
    final_score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanNodeV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.2")),
            plan_node_id=str(payload.get("plan_node_id", "")),
            parent_id=payload.get("parent_id"),
            depth=int(payload.get("depth", 0)),
            subgoal_id=payload.get("subgoal_id"),
            skill_id=payload.get("skill_id"),
            estimated_cost=float(payload.get("estimated_cost", 0.0)),
            estimated_success=float(payload.get("estimated_success", 0.0)),
            estimated_information_gain=float(payload.get("estimated_information_gain", 0.0)),
            estimated_goal_progress=float(payload.get("estimated_goal_progress", 0.0)),
            blocked=bool(payload.get("blocked", False)),
            notes=payload.get("notes"),
            candidate_cluster_key=payload.get("candidate_cluster_key"),
            movement_cluster_key=payload.get("movement_cluster_key"),
            cluster_failure_count=int(payload.get("cluster_failure_count", 0)),
            cluster_exhausted_flag=bool(payload.get("cluster_exhausted_flag", False)),
            same_row_penalty=float(payload.get("same_row_penalty", 0.0)),
            post_jump_exclusion_flag=bool(payload.get("post_jump_exclusion_flag", False)),
            distance_from_last_failed_cluster=payload.get("distance_from_last_failed_cluster"),
            excluded_by_cooldown=bool(payload.get("excluded_by_cooldown", False)),
            neighbor_of_failed_cluster=bool(payload.get("neighbor_of_failed_cluster", False)),
            row_band_id=payload.get("row_band_id"),
            current_round_failed_neighbor_count=int(payload.get("current_round_failed_neighbor_count", 0)),
            current_round_failed_row_count=int(payload.get("current_round_failed_row_count", 0)),
            recent_failed_movement_cluster_match=bool(payload.get("recent_failed_movement_cluster_match", False)),
            blocked_by_exact_cluster_exhaustion=bool(payload.get("blocked_by_exact_cluster_exhaustion", False)),
            blocked_by_neighbor_cooldown=bool(payload.get("blocked_by_neighbor_cooldown", False)),
            blocked_by_post_jump_exclusion=bool(payload.get("blocked_by_post_jump_exclusion", False)),
            blocked_by_unreachable=bool(payload.get("blocked_by_unreachable", False)),
            candidate_source=payload.get("candidate_source"),
            blocking_reason_codes=list(payload.get("blocking_reason_codes", [])),
            surviving_unblocked_candidate_count=int(payload.get("surviving_unblocked_candidate_count", 0)),
            pre_filter_survived=bool(payload.get("pre_filter_survived", False)),
            hard_filter_applied=bool(payload.get("hard_filter_applied", False)),
            post_filter_survived=bool(payload.get("post_filter_survived", False)),
            rank_removed_reason=payload.get("rank_removed_reason"),
            pre_filter_rank_position=payload.get("pre_filter_rank_position"),
            post_filter_rank_position=payload.get("post_filter_rank_position"),
            selected_after_rerank=bool(payload.get("selected_after_rerank", False)),
            candidate_final_status=payload.get("candidate_final_status"),
            blocking_reason_details=dict(payload.get("blocking_reason_details", {})),
            final_score_breakdown=dict(payload.get("final_score_breakdown", {})),
        )


@dataclass(frozen=True)
class PlanResultV1:
    schema_version: str
    plan_id: str
    root_plan_node_id: str
    selected_plan_node_id: str
    selected_skill_id: Optional[str]
    selected_subgoal_id: Optional[str]
    planner_reason: str
    alternative_plan_node_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanResultV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.2")),
            plan_id=str(payload.get("plan_id", "")),
            root_plan_node_id=str(payload.get("root_plan_node_id", "")),
            selected_plan_node_id=str(payload.get("selected_plan_node_id", "")),
            selected_skill_id=payload.get("selected_skill_id"),
            selected_subgoal_id=payload.get("selected_subgoal_id"),
            planner_reason=str(payload.get("planner_reason", "")),
            alternative_plan_node_ids=list(payload.get("alternative_plan_node_ids", [])),
        )
