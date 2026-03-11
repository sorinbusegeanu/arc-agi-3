from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeSection:
    max_rounds: int = 8
    max_passes_per_round: int = 2
    no_progress_budget: int = 2
    enable_ray: bool = True
    stop_on_win: bool = True


@dataclass(frozen=True)
class RaySection:
    namespace: str = "v3_1"
    temp_dir: str | None = None
    local_mode: bool = False
    coordinator_cpus: float = 1.0
    service_cpus: float = 1.0
    worker_cpus: float = 1.0
    env_workers: int = 1
    analysis_workers: int = 1
    planning_helper_workers: int = 0


@dataclass(frozen=True)
class EnvironmentSection:
    env_factory: str | None = "arc_agi_agent.envs.arc_env_factory:create_env"
    env_id: str | None = "ls20"
    env_root: str | None = "/home/zodrak/zod/environment_files"
    seed: int = 0
    probe_steps: int = 40
    directed_steps: int = 40


@dataclass(frozen=True)
class AnalysisSection:
    min_object_area: int = 1
    poi_min_persistence: int = 2
    avatar_motion_threshold: float = 0.5
    min_change_region_area: int = 1
    background_confidence_threshold: float = 0.35
    avatar_candidate_score_threshold: float = 0.25


@dataclass(frozen=True)
class PlanningSection:
    max_candidates: int = 16
    novelty_weight: float = 0.6
    reward_weight: float = 0.4
    risk_penalty: float = 0.5
    reachability_weight: float = 0.55
    progress_weight: float = 1.0
    utility_weight: float = 1.0
    retry_penalty_weight: float = 0.18
    route_cost_weight: float = 0.12
    route_risk_weight: float = 0.35
    consequence_bonus_weight: float = 0.05
    trigger_bonus_weight: float = 0.08


@dataclass(frozen=True)
class ExecutionSection:
    target_reach_distance: float = 1.5
    stall_limit: int = 6
    blocked_repeat_limit: int = 4


@dataclass(frozen=True)
class MemorySection:
    retry_limit: int = 3
    cooldown_rounds: int = 2
    exhaustion_threshold: int = 4


@dataclass(frozen=True)
class StorageSection:
    root_dir: str = "runs_v3_1"
    export_json: bool = True
    export_sqlite: bool = True


@dataclass(frozen=True)
class VisualizationSection:
    enable_heatmaps: bool = True
    grid_width: int = 64
    grid_height: int = 64


@dataclass(frozen=True)
class DebuggingSection:
    keep_observations: bool = True
    trace_snapshots: bool = False
    strict_versions: bool = True


@dataclass(frozen=True)
class FeatureFlagsSection:
    enable_ranker: bool = False
    enable_helper_workers: bool = False
    enable_hypothesis_proposals: bool = False
    enable_candidate_expansion_helper: bool = True
    enable_route_analysis_helper: bool = True
    enable_score_feature_helper: bool = True
    enable_pruning_helper: bool = True


@dataclass(frozen=True)
class V31Config:
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    ray: RaySection = field(default_factory=RaySection)
    environment: EnvironmentSection = field(default_factory=EnvironmentSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    planning: PlanningSection = field(default_factory=PlanningSection)
    execution: ExecutionSection = field(default_factory=ExecutionSection)
    memory: MemorySection = field(default_factory=MemorySection)
    storage: StorageSection = field(default_factory=StorageSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)
    debugging: DebuggingSection = field(default_factory=DebuggingSection)
    feature_flags: FeatureFlagsSection = field(default_factory=FeatureFlagsSection)
