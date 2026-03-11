from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AnalystConfigV2:
    connectivity: int = 4
    min_area: int = 1
    bg_border_weight: float = 0.35
    bg_frequency_weight: float = 0.45
    bg_connected_weight: float = 0.20
    hud_height_ratio: float = 0.15
    hud_width_ratio: float = 0.7
    avatar_motion_threshold: float = 0.5


@dataclass(frozen=True)
class TrajectoryAnalysisConfigV2:
    min_episodes: int = 1
    min_poi_persistence: int = 2
    motion_hotspot_threshold: float = 0.2
    consequence_hotspot_threshold: float = 0.2
    traversable_min_visits: int = 2
    episode_analysis_workers: int | None = None
    episode_analysis_chunk_size: int | None = None
    parallel_area_assign: bool = True


@dataclass(frozen=True)
class MemoryConfigV2:
    storage_dir: str = "runs_v2"
    atomic_writes: bool = True
    persist_round_archives: bool = True
    persist_world_model: bool = True
    persist_interventions: bool = True
    persist_events: bool = True
    persist_navigation_graph: bool = True


@dataclass(frozen=True)
class ControllerConfigV2:
    unguided_probe_fraction: float = 0.2
    exploit_confidence_threshold: float = 0.7
    route_confidence_threshold: float = 0.6
    info_gain_weight: float = 1.0
    confidence_weight: float = 1.0
    reachability_weight: float = 0.5
    random_seed: int = 0


@dataclass(frozen=True)
class ExecutorConfigV2:
    max_steps: int = 40
    blocked_repeat_limit: int = 4
    target_reach_distance: float = 1.5
    contact_hold_steps: int = 2
    relocalize_every_step: bool = True
    post_contact_observation_steps: int = 6
    max_local_probe_steps: int = 5


@dataclass(frozen=True)
class ScoringConfigV2:
    poi_rank_weight_info_gain: float = 1.0
    poi_rank_weight_confidence: float = 1.0
    poi_rank_weight_reachability: float = 0.5
    hypothesis_info_gain_weight: float = 1.0
    controller_target_weight: float = 1.0
    executor_progress_weight: float = 1.0
    consequence_weight_local: float = 0.7
    consequence_weight_global: float = 1.0


@dataclass(frozen=True)
class LoggingConfigV2:
    log_dir: str = "runs_v2/logs"


@dataclass(frozen=True)
class DatasetOrRolloutConfigV2:
    mode: str = "import_only"
    trajectory_path: Optional[str] = None
    episodes_per_round: int = 8
    max_steps_per_episode: int = 40


@dataclass(frozen=True)
class RuntimeConfigV2:
    max_rounds: int = 5
    resume_if_exists: bool = True
    fail_on_missing_live_env: bool = True


@dataclass(frozen=True)
class CollectionConfigGroupV2:
    initial_probe_episodes: int = 8
    directed_probe_episodes: int = 4
    max_steps_per_episode: int = 40
    max_steps_per_instruction: int = 40
    seed: int = 0
    action_repeat_limit: int = 4


@dataclass(frozen=True)
class RoutingConfigV2:
    use_graph_distance: bool = True
    route_stall_limit: int = 6
    blocked_retry_limit: int = 4
    local_probe_radius: int = 2
    use_action_semantics: bool = True
    route_replan_every_step: bool = True
    target_access_bias: float = 1.0
    blocked_edge_penalty: float = 2.0
    unknown_edge_penalty: float = 0.5
    transition_edge_penalty: float = 1.0


@dataclass(frozen=True)
class SchedulerConfigV2:
    unguided_min_fraction: float = 0.2
    discriminating_probe_fraction: float = 0.2
    poi_approach_fraction: float = 0.3
    poi_interaction_fraction: float = 0.2
    exploit_fraction: float = 0.1


@dataclass(frozen=True)
class StorageConfigV2:
    root_dir: str = "runs_v2"
    atomic_writes: bool = True
    keep_raw_frames: bool = True
    keep_raw_env_payloads: bool = False
    backend: str = "files"


@dataclass(frozen=True)
class VisualizationConfigV2:
    enable_round_one_poi_heatmap: bool = True
    enable_final_avatar_visit_heatmap: bool = True
    heatmap_grid_width: int = 64
    heatmap_grid_height: int = 64
    generate_heatmaps_postrun_only: bool = True


@dataclass(frozen=True)
class ResumeConfigV2:
    reload_latest_blackboard: bool = True
    reload_unfinished_round: bool = True


@dataclass(frozen=True)
class TargetScoringConfigV2:
    proximity_weight: float = 1.0
    information_gain_weight: float = 1.0
    consequence_weight: float = 1.0
    confidence_weight: float = 1.0
    stale_penalty: float = 0.2
    likely_hud_penalty: float = 0.5
    accessibility_weight: float = 1.0
    stale_route_penalty: float = 0.4
    event_novelty_weight: float = 1.0
    cross_area_mechanic_weight: float = 0.8


@dataclass(frozen=True)
class EnvConfigV2:
    env_factory: Optional[str] = None
    env_id: Optional[str] = None
    env_root: Optional[str] = None


@dataclass(frozen=True)
class DebugConfigV2:
    strict_state_hash: bool = False
    export_avatar_candidates: bool = False
    export_reachability_reasons: bool = False
    export_target_linkage: bool = False
    export_poi_rejections: bool = False
    fail_on_missing_target_link: bool = False
    keep_invalid_steps_for_debug: bool = False
    export_action_semantics: bool = False
    export_avatar_tracks: bool = False
    export_event_table: bool = False
    export_causal_links: bool = False
    export_area_table: bool = False
    export_mechanic_hypotheses: bool = False
    export_trigger_zones: bool = False
    export_event_graph: bool = False
    export_event_sequence_patterns: bool = False
    export_hidden_trigger_hypotheses: bool = False
    export_causal_chain_hypotheses: bool = False
    export_counterfactual_traces: bool = False


@dataclass(frozen=True)
class ActionSemanticsConfigV2:
    min_samples_per_action: int = 6
    min_samples_per_context: int = 3
    blocked_motion_threshold: float = 0.5
    noop_motion_threshold: float = 0.1
    interaction_change_threshold: float = 0.15
    transition_change_threshold: float = 0.6
    smoothing: float = 1.0


@dataclass(frozen=True)
class AvatarTrackingConfigV2:
    max_hypotheses: int = 5
    prediction_weight: float = 1.5
    appearance_weight: float = 1.0
    motion_weight: float = 1.2
    route_consistency_weight: float = 1.0
    missing_tolerance: int = 3
    track_confirm_threshold: float = 0.7
    track_prune_threshold: float = 0.2


@dataclass(frozen=True)
class EventExtractionConfigV2:
    min_region_change_ratio: float = 0.08
    local_radius: int = 3
    remote_min_distance: int = 6
    transition_frame_change_ratio: float = 0.55
    event_merge_gap: int = 2
    delayed_window_steps: int = 6
    max_post_contact_window: int = 10


@dataclass(frozen=True)
class CausalityConfigV2:
    immediate_window_steps: int = 2
    delayed_window_steps: int = 6
    cross_area_window_steps: int = 12
    same_area_bonus: float = 0.2
    repeatability_bonus: float = 0.3
    contradiction_penalty: float = 0.5
    min_link_confidence: float = 0.45


@dataclass(frozen=True)
class AreaModelConfigV2:
    area_match_iou_threshold: float = 0.5
    area_palette_overlap_threshold: float = 0.7
    topology_similarity_threshold: float = 0.6
    new_area_change_threshold: float = 0.5


@dataclass(frozen=True)
class MechanicInductionConfigV2:
    min_support_events: int = 2
    min_cross_area_support_events: int = 2
    mechanic_match_threshold: float = 0.65
    falsification_penalty: float = 0.4
    promotion_threshold: float = 0.75


@dataclass(frozen=True)
class HiddenTriggerConfigV2:
    min_region_visits_for_candidate: int = 3
    min_boundary_crossings_for_candidate: int = 2
    dwell_step_thresholds: List[int] = field(default_factory=lambda: [2, 4, 6])
    min_activation_support: int = 2
    null_penalty: float = 0.25
    contradiction_penalty: float = 0.5
    promotion_threshold: float = 0.72
    merge_iou_threshold: float = 0.5
    merge_cell_overlap_threshold: float = 0.6


@dataclass(frozen=True)
class CausalChainConfigV2:
    immediate_delay_max: int = 2
    delayed_delay_max: int = 8
    post_transition_delay_max: int = 16
    max_chain_hops: int = 4
    min_sequence_support: int = 2
    sequence_match_threshold: float = 0.7
    partial_match_weight: float = 0.35
    contradiction_penalty: float = 0.5
    promotion_threshold: float = 0.75


@dataclass(frozen=True)
class ProbeModeConfigV2:
    max_step_on_region_steps: int = 6
    max_dwell_probe_steps: int = 8
    max_action_in_region_steps: int = 6
    max_boundary_cross_steps: int = 6
    max_side_contact_steps: int = 6
    counterfactual_probe_budget: int = 10


@dataclass(frozen=True)
class SequenceMiningConfigV2:
    min_pattern_length: int = 2
    max_pattern_length: int = 5
    delay_bucket_edges: List[int] = field(default_factory=lambda: [0, 2, 8, 16])
    same_area_bonus: float = 0.15
    cross_area_bonus: float = 0.2
    topology_bonus: float = 0.2


@dataclass(frozen=True)
class ControllerExplorationConfigV2:
    hidden_trigger_probe_weight: float = 1.0
    causal_chain_verification_weight: float = 0.9
    counterfactual_probe_weight: float = 0.7
    null_result_penalty: float = 0.4
    fragment_novelty_weight: float = 0.8


@dataclass(frozen=True)
class V2Config:
    analyst: AnalystConfigV2 = field(default_factory=AnalystConfigV2)
    trajectory_analysis: TrajectoryAnalysisConfigV2 = field(default_factory=TrajectoryAnalysisConfigV2)
    memory: MemoryConfigV2 = field(default_factory=MemoryConfigV2)
    controller: ControllerConfigV2 = field(default_factory=ControllerConfigV2)
    executor: ExecutorConfigV2 = field(default_factory=ExecutorConfigV2)
    scoring: ScoringConfigV2 = field(default_factory=ScoringConfigV2)
    logging: LoggingConfigV2 = field(default_factory=LoggingConfigV2)
    dataset_or_rollout_source: DatasetOrRolloutConfigV2 = field(default_factory=DatasetOrRolloutConfigV2)
    runtime: RuntimeConfigV2 = field(default_factory=RuntimeConfigV2)
    collection: CollectionConfigGroupV2 = field(default_factory=CollectionConfigGroupV2)
    routing: RoutingConfigV2 = field(default_factory=RoutingConfigV2)
    scheduler: SchedulerConfigV2 = field(default_factory=SchedulerConfigV2)
    storage: StorageConfigV2 = field(default_factory=StorageConfigV2)
    visualization: VisualizationConfigV2 = field(default_factory=VisualizationConfigV2)
    resume: ResumeConfigV2 = field(default_factory=ResumeConfigV2)
    target_scoring: TargetScoringConfigV2 = field(default_factory=TargetScoringConfigV2)
    env: EnvConfigV2 = field(default_factory=EnvConfigV2)
    debug: DebugConfigV2 = field(default_factory=DebugConfigV2)
    action_semantics: ActionSemanticsConfigV2 = field(default_factory=ActionSemanticsConfigV2)
    avatar_tracking: AvatarTrackingConfigV2 = field(default_factory=AvatarTrackingConfigV2)
    event_extraction: EventExtractionConfigV2 = field(default_factory=EventExtractionConfigV2)
    causality: CausalityConfigV2 = field(default_factory=CausalityConfigV2)
    area_model: AreaModelConfigV2 = field(default_factory=AreaModelConfigV2)
    mechanic_induction: MechanicInductionConfigV2 = field(default_factory=MechanicInductionConfigV2)
    hidden_trigger: HiddenTriggerConfigV2 = field(default_factory=HiddenTriggerConfigV2)
    causal_chain: CausalChainConfigV2 = field(default_factory=CausalChainConfigV2)
    probe_mode: ProbeModeConfigV2 = field(default_factory=ProbeModeConfigV2)
    sequence_mining: SequenceMiningConfigV2 = field(default_factory=SequenceMiningConfigV2)
    controller_exploration: ControllerExplorationConfigV2 = field(default_factory=ControllerExplorationConfigV2)
    rounds: int = 2
    game_id: str = "unknown_game"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyst": self.analyst.__dict__,
            "trajectory_analysis": self.trajectory_analysis.__dict__,
            "memory": self.memory.__dict__,
            "controller": self.controller.__dict__,
            "executor": self.executor.__dict__,
            "scoring": self.scoring.__dict__,
            "logging": self.logging.__dict__,
            "dataset_or_rollout_source": self.dataset_or_rollout_source.__dict__,
            "runtime": self.runtime.__dict__,
            "collection": self.collection.__dict__,
            "routing": self.routing.__dict__,
            "scheduler": self.scheduler.__dict__,
            "storage": self.storage.__dict__,
            "visualization": self.visualization.__dict__,
            "resume": self.resume.__dict__,
            "target_scoring": self.target_scoring.__dict__,
            "env": self.env.__dict__,
            "debug": self.debug.__dict__,
            "action_semantics": self.action_semantics.__dict__,
            "avatar_tracking": self.avatar_tracking.__dict__,
            "event_extraction": self.event_extraction.__dict__,
            "causality": self.causality.__dict__,
            "area_model": self.area_model.__dict__,
            "mechanic_induction": self.mechanic_induction.__dict__,
            "hidden_trigger": self.hidden_trigger.__dict__,
            "causal_chain": self.causal_chain.__dict__,
            "probe_mode": self.probe_mode.__dict__,
            "sequence_mining": self.sequence_mining.__dict__,
            "controller_exploration": self.controller_exploration.__dict__,
            "rounds": self.rounds,
            "game_id": self.game_id,
        }


def load_config(payload: Dict[str, Any]) -> V2Config:
    return V2Config(
        analyst=AnalystConfigV2(**payload.get("analyst", {})),
        trajectory_analysis=TrajectoryAnalysisConfigV2(**payload.get("trajectory_analysis", {})),
        memory=MemoryConfigV2(**payload.get("memory", {})),
        controller=ControllerConfigV2(**payload.get("controller", {})),
        executor=ExecutorConfigV2(**payload.get("executor", {})),
        scoring=ScoringConfigV2(**payload.get("scoring", {})),
        logging=LoggingConfigV2(**payload.get("logging", {})),
        dataset_or_rollout_source=DatasetOrRolloutConfigV2(**payload.get("dataset_or_rollout_source", {})),
        runtime=RuntimeConfigV2(**payload.get("runtime", {})),
        collection=CollectionConfigGroupV2(**payload.get("collection", {})),
        routing=RoutingConfigV2(**payload.get("routing", {})),
        scheduler=SchedulerConfigV2(**payload.get("scheduler", {})),
        storage=StorageConfigV2(**payload.get("storage", {})),
        visualization=VisualizationConfigV2(**payload.get("visualization", {})),
        resume=ResumeConfigV2(**payload.get("resume", {})),
        target_scoring=TargetScoringConfigV2(**payload.get("target_scoring", {})),
        env=EnvConfigV2(**payload.get("env", {})),
        debug=DebugConfigV2(**payload.get("debug", {})),
        action_semantics=ActionSemanticsConfigV2(**payload.get("action_semantics", {})),
        avatar_tracking=AvatarTrackingConfigV2(**payload.get("avatar_tracking", {})),
        event_extraction=EventExtractionConfigV2(**payload.get("event_extraction", {})),
        causality=CausalityConfigV2(**payload.get("causality", {})),
        area_model=AreaModelConfigV2(**payload.get("area_model", {})),
        mechanic_induction=MechanicInductionConfigV2(**payload.get("mechanic_induction", {})),
        hidden_trigger=HiddenTriggerConfigV2(**payload.get("hidden_trigger", {})),
        causal_chain=CausalChainConfigV2(**payload.get("causal_chain", {})),
        probe_mode=ProbeModeConfigV2(**payload.get("probe_mode", {})),
        sequence_mining=SequenceMiningConfigV2(**payload.get("sequence_mining", {})),
        controller_exploration=ControllerExplorationConfigV2(**payload.get("controller_exploration", {})),
        rounds=int(payload.get("rounds", 2)),
        game_id=str(payload.get("game_id", "unknown_game")),
    )
