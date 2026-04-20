from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FailureReason = Literal[
    "no_moving_candidate",
    "ambiguous_avatar",
    "insufficient_support",
    "all_actions_blocked",
    "invalid_probe_capture",
]


@dataclass(frozen=True)
class ProbePlan:
    game_id: str
    level_id: str
    action_sequence: tuple[str, ...]


@dataclass(frozen=True)
class ProbeTransitionRecord:
    step_index: int
    action: str
    pre_frame: tuple[tuple[int, ...], ...] | None
    post_frame: tuple[tuple[int, ...], ...] | None
    invalid_action: bool
    blocked_action: bool
    terminal: bool
    levels_completed_before: int
    levels_completed_after: int
    reward_before: float | None = None
    reward_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateComponent:
    step_index: int
    action: str
    blocked_action: bool
    frame_width: int
    frame_height: int
    bbox: tuple[int, int, int, int]
    area: int
    pre_center: tuple[float, float]
    post_center: tuple[float, float]
    observed_dx: float
    observed_dy: float
    pre_non_background_cells: tuple[tuple[int, int], ...]
    post_non_background_cells: tuple[tuple[int, int], ...]
    value_histogram_pre: dict[int, int]
    value_histogram_post: dict[int, int]


@dataclass(frozen=True)
class ScoredStepCandidate:
    component: CandidateComponent
    score: float
    direction_agreement_score: float
    movement_consistency_score: float
    shape_consistency_score: float
    compactness_score: float


@dataclass(frozen=True)
class TrackCandidate:
    support_step_indices: tuple[int, ...]
    support_actions: tuple[str, ...]
    observed_motion_vectors: tuple[tuple[float, float], ...]
    entry_bbox: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    value_histogram_pre: dict[int, int]
    value_histogram_post: dict[int, int]
    direction_agreement_score: float
    shape_consistency_score: float
    track_consistency_score: float
    score: float
    support_count: int


@dataclass(frozen=True)
class AvatarCandidate:
    candidate_id: str
    entry_bbox: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    score: float
    support_step_indices: tuple[int, ...]
    support_actions: tuple[str, ...]
    observed_motion_vectors: tuple[tuple[float, float], ...]
    direction_agreement_score: float
    shape_consistency_score: float
    track_consistency_score: float
    value_histogram_pre: dict[int, int]
    value_histogram_post: dict[int, int]
    failure_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AvatarSelectedResult:
    selected_candidate_id: str | None
    selected_bbox: tuple[int, int, int, int] | None
    selected_center: tuple[float, float] | None
    confidence: float
    failure_reason: FailureReason | None
    ranking_margin_to_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AvatarDiagnostics:
    per_step_candidate_counts: dict[int, int] = field(default_factory=dict)
    per_step_top_scores: dict[int, tuple[float, ...]] = field(default_factory=dict)
    total_candidate_count: int = 0
    total_track_count: int = 0
    dropped_candidate_reasons: dict[str, int] = field(default_factory=dict)
    ambiguous_ranking: bool = False
    no_motion: bool = False
    all_blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AvatarIdentificationReport:
    candidates: tuple[AvatarCandidate, ...]
    selected: AvatarSelectedResult
    diagnostics: AvatarDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected": self.selected.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class ProbeEpisode:
    episode_index: int
    seed: int
    plan: ProbePlan
    transitions: tuple[ProbeTransitionRecord, ...]
    report: AvatarIdentificationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "plan": asdict(self.plan),
            "transitions": [record.to_dict() for record in self.transitions],
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class CrossResetAvatarEvidence:
    canonical_candidate_id: str
    episode_indices: tuple[int, ...]
    per_episode_candidate_ids: dict[int, str]
    bbox_sequence: tuple[tuple[int, int, int, int], ...]
    center_sequence: tuple[tuple[float, float], ...]
    value_histogram_pre_aggregate: dict[int, int]
    value_histogram_post_aggregate: dict[int, int]
    mean_score: float
    score_stddev: float
    shape_consistency_across_resets: float
    position_consistency_across_resets: float
    support_episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiResetDiagnostics:
    episode_count: int
    successful_episode_count: int
    failed_episode_count: int
    failure_reason_counts: dict[str, int]
    cross_reset_ambiguous: bool
    stable_avatar_found: bool
    confidence_accumulated: float
    dropped_episode_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiResetAvatarReport:
    episodes: tuple[ProbeEpisode, ...]
    cross_reset_evidence: tuple[CrossResetAvatarEvidence, ...]
    selected: AvatarSelectedResult
    diagnostics: MultiResetDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "cross_reset_evidence": [evidence.to_dict() for evidence in self.cross_reset_evidence],
            "selected": self.selected.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class POICandidate:
    poi_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int
    value_histogram: dict[int, int]
    seen_step_indices: tuple[int, ...]
    support_episode_indices: tuple[int, ...]
    source_kind: str
    near_avatar_steps: tuple[int, ...]
    min_avatar_distance: float
    confidence: float
    ambiguity_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class POISelectedResult:
    selected_poi_ids: tuple[str, ...]
    ambiguous: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class POIDiagnostics:
    per_step_component_counts: dict[int, int]
    static_inventory_count: int
    changed_component_count: int
    merged_candidate_count: int
    cross_reset_cluster_count: int
    dropped_candidate_reasons: dict[str, int]
    avatar_overlap_rejections: int
    ambiguous_candidates: int
    contact_log_count: int
    border_locked_rejections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class POIDiscoveryReport:
    candidates: tuple[POICandidate, ...]
    selected: POISelectedResult
    diagnostics: POIDiagnostics
    contact_logs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "selected": self.selected.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "contact_logs": [dict(item) for item in self.contact_logs],
        }


@dataclass(frozen=True)
class POIEpisode:
    episode_index: int
    avatar_report: AvatarIdentificationReport
    poi_report: POIDiscoveryReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "avatar_report": self.avatar_report.to_dict(),
            "poi_report": self.poi_report.to_dict(),
        }


@dataclass(frozen=True)
class CrossResetPOIEvidence:
    canonical_poi_id: str
    episode_indices: tuple[int, ...]
    per_episode_poi_ids: dict[int, str]
    bbox_sequence: tuple[tuple[int, int, int, int], ...]
    center_sequence: tuple[tuple[float, float], ...]
    value_histogram_aggregate: dict[int, int]
    mean_confidence: float
    position_consistency_across_resets: float
    support_episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactPolicy:
    policy_id: str
    poi_id: str
    episode_index: int
    planned_actions: tuple[str, ...]
    max_steps: int
    stop_on_contact: bool
    stop_on_screen_change: bool
    stop_on_terminal: bool
    interaction_mode: str = "overlap"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryCandidateRecord:
    trajectory_id: str
    level_id: str | None
    episode_index: int | None
    target_poi_id: str | None
    source: str
    actions: tuple[str, ...]
    planned_length: int
    net_dx: int
    net_dy: int
    first_action: str | None
    turn_count: int
    axis_order: str
    waypoints: tuple[tuple[int, int], ...]
    score_components: dict[str, float]
    rank_index: int | None
    selected_for_execution: bool
    validation_passed: bool = True
    rejection_reasons: tuple[str, ...] = ()
    plausibility_flags: tuple[str, ...] = ()
    hint_source: str | None = None
    start_avatar_center: tuple[float, float] | None = None
    target_center: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryAttemptRecord:
    trajectory_id: str
    level_id: str | None
    episode_index: int | None
    target_poi_id: str | None
    source: str
    actions: tuple[str, ...]
    planned_length: int
    executed_step_count: int
    completed_planned_route: bool
    stop_reason: str | None
    outcome_type: str | None
    solved: bool
    terminal: bool
    level_transition: bool
    blocked_step_count: int
    invalid_step_count: int
    screen_changed_step_count: int
    start_avatar_bbox: tuple[int, int, int, int] | None
    end_avatar_bbox: tuple[int, int, int, int] | None
    start_target_bbox: tuple[int, int, int, int] | None
    end_target_bbox: tuple[int, int, int, int] | None
    avatar_reacquire_mode: str | None = None
    target_reacquire_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryStatsReport:
    level_id: str
    solved: bool
    failure_reason: str | None
    generated_trajectory_count: int
    attempted_trajectory_count: int
    completed_trajectory_count: int
    min_steps_per_attempted_trajectory: int
    max_steps_per_attempted_trajectory: int
    mean_steps_per_attempted_trajectory: float
    total_executed_steps_across_attempted_trajectories: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactStepRecord:
    step_index: int
    action: str
    pre_frame: tuple[tuple[int, ...], ...] | None
    post_frame: tuple[tuple[int, ...], ...] | None
    invalid_action: bool
    blocked_action: bool
    terminal: bool
    levels_completed_before: int
    levels_completed_after: int
    reward_before: float | None
    reward_after: float | None
    avatar_bbox_before: tuple[int, int, int, int] | None
    avatar_bbox_after: tuple[int, int, int, int] | None
    poi_bbox_before: tuple[int, int, int, int] | None
    poi_bbox_after: tuple[int, int, int, int] | None
    screen_changed: bool
    hud_changed_only: bool
    contact_detected: bool
    avatar_reacquire_mode: str | None = None
    poi_reacquire_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactOutcome:
    outcome_type: str
    confidence: float
    contact_step_index: int | None
    screen_change_step_indices: tuple[int, ...]
    reward_change_step_indices: tuple[int, ...]
    object_removed: bool
    new_object_appeared: bool
    level_transition: bool
    terminal: bool
    hud_change_only: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestedPOIResult:
    poi_id: str
    episode_index: int
    policy: ContactPolicy
    steps: tuple[ContactStepRecord, ...]
    outcome: ContactOutcome
    initial_poi_bbox: tuple[int, int, int, int] | None
    final_poi_bbox: tuple[int, int, int, int] | None
    initial_avatar_bbox: tuple[int, int, int, int] | None
    final_avatar_bbox: tuple[int, int, int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "poi_id": self.poi_id,
            "episode_index": self.episode_index,
            "policy": self.policy.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
            "outcome": self.outcome.to_dict(),
            "initial_poi_bbox": self.initial_poi_bbox,
            "final_poi_bbox": self.final_poi_bbox,
            "initial_avatar_bbox": self.initial_avatar_bbox,
            "final_avatar_bbox": self.final_avatar_bbox,
        }


@dataclass(frozen=True)
class ContactExperimentEpisode:
    episode_index: int
    tested_pois: tuple[TestedPOIResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "tested_pois": [item.to_dict() for item in self.tested_pois],
        }


@dataclass(frozen=True)
class ContactExperimentReport:
    episodes: tuple[ContactExperimentEpisode, ...]
    tested_pois: tuple[TestedPOIResult, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [item.to_dict() for item in self.episodes],
            "tested_pois": [item.to_dict() for item in self.tested_pois],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class HUDCellSample:
    row: int
    col: int
    value: int
    episode_index: int
    step_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDRegion:
    hud_region_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int
    edge_side: str
    value_histogram: dict[int, int]
    seen_episode_indices: tuple[int, ...]
    seen_step_indices: tuple[int, ...]
    change_step_indices: tuple[int, ...]
    stability_score: float
    change_repeat_score: float
    world_overlap_rejection_score: float
    confidence: float
    ambiguity_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDMask:
    height: int
    width: int
    true_cell_count: int
    rows_active: tuple[int, ...]
    cols_active: tuple[int, ...]
    regions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDDiagnostics:
    edge_band_component_count: int
    repeated_change_component_count: int
    avatar_overlap_rejections: int
    world_motion_rejections: int
    persistent_region_count: int
    cross_reset_cluster_count: int
    ambiguous_regions: int
    text_or_color_sample_count: int
    static_edge_survivor_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDDetectionReport:
    mask: HUDMask
    regions: tuple[HUDRegion, ...]
    diagnostics: HUDDiagnostics
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask.to_dict(),
            "regions": [item.to_dict() for item in self.regions],
            "diagnostics": self.diagnostics.to_dict(),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class HUDEpisodeReport:
    episode_index: int
    hud_report: HUDDetectionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "hud_report": self.hud_report.to_dict(),
        }


@dataclass(frozen=True)
class CrossResetHUDEvidence:
    canonical_region_id: str
    episode_indices: tuple[int, ...]
    per_episode_region_ids: dict[int, str]
    bbox_sequence: tuple[tuple[int, int, int, int], ...]
    center_sequence: tuple[tuple[float, float], ...]
    value_histogram_aggregate: dict[int, int]
    mean_confidence: float
    position_consistency_across_resets: float
    support_episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDHintSummary:
    hud_region_id: str
    bbox: tuple[int, int, int, int]
    edge_side: str
    value_histogram: dict[int, int]
    dominant_values: tuple[int, ...]
    stable_value_count: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDPOIMatch:
    hud_region_id: str
    poi_id: str
    hud_values: dict[int, int]
    poi_values: dict[int, int]
    value_overlap_score: float
    dominant_value_match_score: float
    structural_compatibility_score: float
    support_episode_count: int
    confidence: float
    ambiguity_flags: tuple[str, ...]
    value_precision_score: float = 0.0
    poi_purity_score: float = 0.0
    region_specificity_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDTargetSelection:
    selected_poi_id: str | None
    ranked_poi_ids: tuple[str, ...]
    top_match_hud_region_ids: tuple[str, ...]
    ambiguous: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDHintDiagnostics:
    hud_region_count: int
    poi_candidate_count: int
    match_count: int
    ambiguous_match_count: int
    rejected_match_reasons: dict[str, int]
    selected_match_margin: float
    cross_reset_support_count: int
    best_match_win_counts: dict[str, int] = field(default_factory=dict)
    aggregate_score_gap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HUDHintReport:
    hud_hints: tuple[HUDHintSummary, ...]
    matches: tuple[HUDPOIMatch, ...]
    selected: HUDTargetSelection
    diagnostics: HUDHintDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "hud_hints": [item.to_dict() for item in self.hud_hints],
            "matches": [item.to_dict() for item in self.matches],
            "selected": self.selected.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class SolveStepRecord:
    step_index: int
    action: str
    pre_frame: tuple[tuple[int, ...], ...] | None
    post_frame: tuple[tuple[int, ...], ...] | None
    invalid_action: bool
    blocked_action: bool
    terminal: bool
    levels_completed_before: int
    levels_completed_after: int
    reward_before: float | None
    reward_after: float | None
    avatar_bbox_before: tuple[int, int, int, int] | None
    avatar_bbox_after: tuple[int, int, int, int] | None
    target_poi_id: str | None
    target_bbox_before: tuple[int, int, int, int] | None
    target_bbox_after: tuple[int, int, int, int] | None
    contact_detected: bool
    screen_changed: bool
    hud_changed_only: bool
    outcome_type: str
    source: str = "frontier_solve"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolveTargetState:
    target_poi_id: str | None
    source: str
    confidence: float
    attempt_count: int
    last_outcome_type: str | None
    active: bool
    route_feasibility: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolveEpisodeResult:
    episode_index: int
    initial_target: SolveTargetState
    steps: tuple[SolveStepRecord, ...]
    final_target: SolveTargetState
    solved: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "initial_target": self.initial_target.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
            "final_target": self.final_target.to_dict(),
            "solved": bool(self.solved),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SolveDiagnostics:
    episode_count: int
    solved_episode_count: int
    failed_episode_count: int
    failure_reason_counts: dict[str, int]
    retarget_count: int
    level_transition_count: int
    terminal_success_count: int
    terminal_failure_count: int
    step_budget_exhausted_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolveReport:
    episodes: tuple[SolveEpisodeResult, ...]
    diagnostics: SolveDiagnostics
    selected_target_id: str | None
    solved: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [item.to_dict() for item in self.episodes],
            "diagnostics": self.diagnostics.to_dict(),
            "selected_target_id": self.selected_target_id,
            "solved": bool(self.solved),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class TraversabilityCell:
    row: int
    col: int
    state: str
    confidence: float
    evidence_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraversabilityMap:
    height: int
    width: int
    cells: tuple[TraversabilityCell, ...]
    free_count: int
    blocked_count: int
    unknown_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "cells": [item.to_dict() for item in self.cells],
            "free_count": int(self.free_count),
            "blocked_count": int(self.blocked_count),
            "unknown_count": int(self.unknown_count),
        }


@dataclass(frozen=True)
class RoutePlan:
    target_poi_id: str
    waypoints: tuple[tuple[int, int], ...]
    planned_actions: tuple[str, ...]
    expected_length: int
    confidence: float
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteAttempt:
    target_poi_id: str
    planned_actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    blocked_step_indices: tuple[int, ...]
    replan_count: int
    reached_target: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingDiagnostics:
    route_attempt_count: int
    successful_route_count: int
    blocked_route_count: int
    replan_count: int
    avg_route_length: float
    failure_reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraversabilityReport:
    map: TraversabilityMap
    routes: tuple[RouteAttempt, ...]
    diagnostics: RoutingDiagnostics
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map.to_dict(),
            "routes": [item.to_dict() for item in self.routes],
            "diagnostics": self.diagnostics.to_dict(),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class POIMechanicEvidence:
    poi_id: str
    episode_index: int
    contact_count: int
    useful_change_count: int
    no_effect_count: int
    reward_change_count: int
    object_removed_count: int
    door_opens_count: int
    level_transition_count: int
    terminal_count: int
    hud_match_confidence: float
    support_step_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class POIMechanicState:
    poi_id: str
    mechanic_label: str
    confidence: float
    priority_score: float
    attempt_count: int
    success_count: int
    failure_count: int
    last_outcome_type: str | None
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicMemory:
    poi_states: tuple[POIMechanicState, ...]
    selected_poi_id: str | None
    retired_poi_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "poi_states": [item.to_dict() for item in self.poi_states],
            "selected_poi_id": self.selected_poi_id,
            "retired_poi_ids": list(self.retired_poi_ids),
        }


@dataclass(frozen=True)
class MechanicDecision:
    selected_poi_id: str | None
    reason_codes: tuple[str, ...]
    confidence: float
    retarget_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicDiagnostics:
    poi_count: int
    target_count: int
    decoy_count: int
    hazard_count: int
    exit_count: int
    door_or_switch_count: int
    retarget_count: int
    ambiguous_poi_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicReport:
    memory: MechanicMemory
    diagnostics: MechanicDiagnostics
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class AdaptiveTargetState:
    target_poi_id: str | None
    source: str
    confidence: float
    attempt_count: int
    last_outcome_type: str | None
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveStepRecord:
    step_index: int
    action: str
    pre_frame: tuple[tuple[int, ...], ...] | None
    post_frame: tuple[tuple[int, ...], ...] | None
    invalid_action: bool
    blocked_action: bool
    terminal: bool
    levels_completed_before: int
    levels_completed_after: int
    reward_before: float | None
    reward_after: float | None
    avatar_bbox_before: tuple[int, int, int, int] | None
    avatar_bbox_after: tuple[int, int, int, int] | None
    target_poi_id: str | None
    target_bbox_before: tuple[int, int, int, int] | None
    target_bbox_after: tuple[int, int, int, int] | None
    contact_detected: bool
    screen_changed: bool
    hud_changed_only: bool
    outcome_type: str
    retargeted: bool
    source: str = "frontier_solve"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveEpisodeResult:
    episode_index: int
    target_sequence: tuple[AdaptiveTargetState, ...]
    steps: tuple[AdaptiveStepRecord, ...]
    solved: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "target_sequence": [item.to_dict() for item in self.target_sequence],
            "steps": [item.to_dict() for item in self.steps],
            "solved": bool(self.solved),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class AdaptiveDiagnostics:
    episode_count: int
    solved_episode_count: int
    failed_episode_count: int
    retarget_count: int
    target_switch_count: int
    useful_change_count: int
    no_progress_count: int
    level_transition_count: int
    terminal_count: int
    step_budget_exhausted_count: int
    failure_reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveSolveReport:
    episodes: tuple[AdaptiveEpisodeResult, ...]
    diagnostics: AdaptiveDiagnostics
    selected_target_id: str | None
    solved: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [item.to_dict() for item in self.episodes],
            "diagnostics": self.diagnostics.to_dict(),
            "selected_target_id": self.selected_target_id,
            "solved": bool(self.solved),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class LevelRunRequest:
    game_id: str
    level_id: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelSolveAction:
    step_index: int
    action: str
    target_poi_id: str | None
    reason: str | None
    pre_level_index: int
    post_level_index: int
    source: str = "frontier_solve"
    pre_frame: tuple[tuple[int, ...], ...] | None = None
    post_frame: tuple[tuple[int, ...], ...] | None = None
    invalid_action: bool = False
    blocked_action: bool = False
    terminal: bool = False
    reward_before: float | None = None
    reward_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelSolution:
    game_id: str
    level_id: str
    solved: bool
    action_trace: tuple[LevelSolveAction, ...]
    step_count: int
    terminal: bool
    level_transition: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "solved": bool(self.solved),
            "action_trace": [item.to_dict() for item in self.action_trace],
            "step_count": int(self.step_count),
            "terminal": bool(self.terminal),
            "level_transition": bool(self.level_transition),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class PerLevelResult:
    level_id: str
    phase_status: dict[str, str]
    solved: bool
    failure_reason: str | None
    solution: LevelSolution
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "phase_status": dict(self.phase_status),
            "solved": bool(self.solved),
            "failure_reason": self.failure_reason,
            "solution": self.solution.to_dict(),
            "artifact_paths": dict(self.artifact_paths),
        }


@dataclass(frozen=True)
class GameLevelBatchDiagnostics:
    requested_level_count: int
    completed_level_count: int
    solved_level_count: int
    failed_level_count: int
    failure_reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GameLevelBatchReport:
    game_id: str
    levels: tuple[PerLevelResult, ...]
    diagnostics: GameLevelBatchDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "levels": [item.to_dict() for item in self.levels],
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class CampaignLevelState:
    game_id: str
    level_id: str
    status: str
    solved: bool
    solution_trace_path: str | None
    best_step_count: int | None
    attempt_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignRunStep:
    global_step_index: int
    level_id: str
    action: str
    source: str
    reason: str | None
    pre_levels_completed: int
    post_levels_completed: int
    pre_frame: tuple[tuple[int, ...], ...] | None = None
    post_frame: tuple[tuple[int, ...], ...] | None = None
    invalid_action: bool = False
    blocked_action: bool = False
    terminal: bool = False
    reward_before: float | None = None
    reward_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignLevelResult:
    level_id: str
    solved: bool
    step_count: int
    used_replay_prefix: bool
    replay_prefix_length: int
    solution: LevelSolution | None
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "solved": bool(self.solved),
            "step_count": int(self.step_count),
            "used_replay_prefix": bool(self.used_replay_prefix),
            "replay_prefix_length": int(self.replay_prefix_length),
            "solution": self.solution.to_dict() if self.solution is not None else None,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class CampaignRunReport:
    game_id: str
    levels: tuple[CampaignLevelResult, ...]
    global_action_trace: tuple[CampaignRunStep, ...]
    solved: bool
    highest_reached_level_id: str | None
    failure_reason: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "levels": [item.to_dict() for item in self.levels],
            "global_action_trace": [item.to_dict() for item in self.global_action_trace],
            "solved": bool(self.solved),
            "highest_reached_level_id": self.highest_reached_level_id,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class SavedLevelTrace:
    game_id: str
    level_id: str
    solved: bool
    action_trace: tuple[str, ...]
    step_count: int
    source_run_id: str | None
    trace_version: int
    replay_verified: bool
    action_sources: tuple[str, ...] | None = None
    trace_id: str | None = None
    optimized: bool | None = None
    optimized_at: str | None = None
    parent_trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "solved": bool(self.solved),
            "action_trace": list(self.action_trace),
            "step_count": int(self.step_count),
            "source_run_id": self.source_run_id,
            "trace_version": int(self.trace_version),
            "replay_verified": bool(self.replay_verified),
            "action_sources": list(self.action_sources) if self.action_sources is not None else None,
            "trace_id": self.trace_id,
            "optimized": self.optimized,
            "optimized_at": self.optimized_at,
            "parent_trace_id": self.parent_trace_id,
        }


@dataclass(frozen=True)
class TraceOptimizationTask:
    game_id: str
    level_id: str
    trace_path: str
    baseline_step_count: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceOptimizationCandidate:
    game_id: str
    level_id: str
    action_trace: tuple[str, ...]
    step_count: int
    improvement_vs_baseline: int
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "action_trace": list(self.action_trace),
            "step_count": int(self.step_count),
            "improvement_vs_baseline": int(self.improvement_vs_baseline),
            "verified": bool(self.verified),
        }


@dataclass(frozen=True)
class TraceOptimizationReport:
    game_id: str
    level_id: str
    baseline_trace: SavedLevelTrace
    best_candidate: TraceOptimizationCandidate
    candidates: tuple[TraceOptimizationCandidate, ...]
    failure_reason: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "baseline_trace": self.baseline_trace.to_dict(),
            "best_candidate": self.best_candidate.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }
