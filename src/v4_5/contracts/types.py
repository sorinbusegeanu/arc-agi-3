from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapProbePlan, HudAnalysisBundle, HudRegionResult
from .boardPerceptionReport import BoardPerceptionReport
from .avatarTypes import AvatarDetectionResult
from .constants import SCHEMA_VERSION
from .gameControlProfile import GameControlProfile
from .poiTypes import PoiAnalysisBundle, PoiRecord, PoiSet
from v4_5.memory.levelMemoryTypes import LevelMemoryRecord
from v4_5.memory.levelMemoryTypes import MemoryRegion


@dataclass(frozen=True)
class SceneSummary:
    schema_version: str
    agent_name: str
    round_id: str
    level_id: str
    avatar_bbox: tuple[int, int, int, int] | None = None
    avatar_position: tuple[float, float] | None = None
    hud_regions: tuple[MemoryRegion, ...] = ()
    life_regions: tuple[MemoryRegion, ...] = ()
    progress_regions: tuple[MemoryRegion, ...] = ()
    salient_changed_regions: tuple[MemoryRegion, ...] = ()
    pois: tuple[PoiRecord, ...] = ()
    candidate_mode_hints: tuple[str, ...] = ()
    observed_affordances: tuple[str, ...] = ()
    levels_completed: int | None = None
    terminal_flag: bool = False
    # raw_observation_payload may carry deterministic advisory region outputs such as
    # traversable, blocking, and unknown space summaries for downstream read-only use.
    raw_observation_payload: dict[str, Any] = field(default_factory=dict)
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentInput:
    schema_version: str
    agent_name: str
    round_id: str
    env_id: str
    level_id: str
    observation: Any | None = None
    parsed_state: Any | None = None
    game_control_profile: GameControlProfile | None = None
    loaded_level_memory: LevelMemoryRecord | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    prior_reports: dict[str, Any] = field(default_factory=dict)
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryReport:
    schema_version: str
    agent_name: str
    round_id: str
    scene_summary: SceneSummary
    game_control_profile: GameControlProfile | None
    bootstrap_required: bool
    bootstrap_probe_summary: tuple[str, ...] = ()
    bootstrap_plan: BootstrapProbePlan | None = None
    avatar_detection_result: AvatarDetectionResult | None = None
    bootstrap_report: BootstrapDiscoveryReport | None = None
    bootstrap_capture_bundle: BootstrapCaptureBundle | None = None
    bootstrap_analysis_bundle: HudAnalysisBundle | None = None
    poi_analysis_bundle: PoiAnalysisBundle | None = None
    board_perception_report: BoardPerceptionReport | None = None
    loaded_level_memory: LevelMemoryRecord | None = None
    poi_registry: POIRegistry | None = None
    stop_reason: str | None = None
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HypothesisItem:
    name: str
    rank: int
    support_flags: tuple[str, ...] = ()
    contradiction_flags: tuple[str, ...] = ()
    mode_label: str = "test"


@dataclass(frozen=True)
class HypothesisReport:
    schema_version: str
    agent_name: str
    round_id: str
    items: tuple[HypothesisItem, ...]
    active_mode_labels: tuple[str, ...]
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannerContext:
    schema_version: str
    agent_name: str
    round_id: str
    env_id: str
    level_id: str
    parsed_state: Any | None = None
    game_control_profile: GameControlProfile | None = None
    loaded_level_memory: LevelMemoryRecord | None = None
    discovery_report: DiscoveryReport | None = None
    board_perception_report: BoardPerceptionReport | None = None
    hypothesis_report: HypothesisReport | None = None
    poi_registry: POIRegistry | None = None
    trajectory_queue: TrajectoryQueue | None = None
    subgoals: tuple[str, ...] = ()
    memory: dict[str, Any] = field(default_factory=dict)
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanCandidate:
    schema_version: str
    agent_name: str
    round_id: str
    plugin_name: str
    candidate_id: str
    action_prefix: tuple[str, ...] = ()
    score: float = 0.0
    verified: bool = False
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanCandidateSet:
    schema_version: str
    agent_name: str
    round_id: str
    plugin_name: str
    candidates: tuple[PlanCandidate, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanDecision:
    schema_version: str
    agent_name: str
    round_id: str
    selected_candidate: PlanCandidate | None
    selected_prefix: tuple[str, ...]
    work_item_id: str | None = None
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeReport:
    schema_version: str
    agent_name: str
    round_id: str
    classification: str
    expected_effects: tuple[str, ...] = ()
    observed_effects: tuple[str, ...] = ()
    memory_updates: dict[str, Any] = field(default_factory=dict)
    hypothesis_updates: dict[str, Any] = field(default_factory=dict)
    poi_updates: tuple[POIUpdate, ...] = ()
    trajectory_updates: tuple[TrajectoryOutcome, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LevelOptimizationReport:
    schema_version: str
    agent_name: str
    round_id: str
    level_id: str
    reusable_hints: tuple[str, ...] = ()
    wasted_prefixes: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GameOptimizationReport:
    schema_version: str
    agent_name: str
    round_id: str
    env_id: str
    reusable_priors: tuple[str, ...] = ()
    mechanic_notes: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdvisoryRequest:
    schema_version: str
    agent_name: str
    round_id: str
    request_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdvisoryResponse:
    schema_version: str
    agent_name: str
    round_id: str
    advisory_only: bool
    suggestions: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class POIRecord:
    schema_version: str
    agent_name: str
    round_id: str
    poi_id: str
    game_id: str
    level_index: str
    type_hint: str
    source: str
    confidence: float
    status: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    colors: tuple[int, ...] = ()
    description: str | None = None
    hint: str | None = None
    times_targeted: int = 0
    times_reached: int = 0
    last_effect_type: str = ""
    linked_hypotheses: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class POIUpdate:
    schema_version: str
    agent_name: str
    round_id: str
    poi_id: str
    status: str
    last_effect_type: str = ""
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class POIRegistry:
    schema_version: str
    agent_name: str
    round_id: str
    records: tuple[POIRecord, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapDiscoveryReport:
    schema_version: str
    agent_name: str
    round_id: str
    plan: BootstrapProbePlan | None = None
    capture_bundle: BootstrapCaptureBundle | None = None
    avatar_detection_result: AvatarDetectionResult | None = None
    hud_analysis_bundle: HudAnalysisBundle | None = None
    poi_analysis_bundle: PoiAnalysisBundle | None = None
    selected_hud_result: HudRegionResult | None = None
    selected_poi_result: PoiSet | None = None
    status: str = "pending"
    error_message: str | None = None
    warnings: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryWorkItem:
    schema_version: str
    agent_name: str
    round_id: str
    work_item_id: str
    game_id: str
    level_index: str
    poi_id: str | None
    subgoal_id: str | None
    plan_prefix: tuple[str, ...]
    expected_contact_or_effect: str
    priority: float
    created_round: str
    attempt_count: int
    status: str
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryQueue:
    schema_version: str
    agent_name: str
    round_id: str
    items: tuple[TrajectoryWorkItem, ...] = ()
    rationale_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryOutcome:
    schema_version: str
    agent_name: str
    round_id: str
    work_item_id: str
    outcome_type: str
    updated_status: str
    replacement_work_item_id: str | None = None
    rationale_codes: tuple[str, ...] = ()
