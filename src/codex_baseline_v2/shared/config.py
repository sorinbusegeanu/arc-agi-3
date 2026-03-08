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


@dataclass(frozen=True)
class MemoryConfigV2:
    storage_dir: str = "runs_v2"
    atomic_writes: bool = True


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
    resume: ResumeConfigV2 = field(default_factory=ResumeConfigV2)
    target_scoring: TargetScoringConfigV2 = field(default_factory=TargetScoringConfigV2)
    env: EnvConfigV2 = field(default_factory=EnvConfigV2)
    debug: DebugConfigV2 = field(default_factory=DebugConfigV2)
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
            "resume": self.resume.__dict__,
            "target_scoring": self.target_scoring.__dict__,
            "env": self.env.__dict__,
            "debug": self.debug.__dict__,
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
        resume=ResumeConfigV2(**payload.get("resume", {})),
        target_scoring=TargetScoringConfigV2(**payload.get("target_scoring", {})),
        env=EnvConfigV2(**payload.get("env", {})),
        debug=DebugConfigV2(**payload.get("debug", {})),
        rounds=int(payload.get("rounds", 2)),
        game_id=str(payload.get("game_id", "unknown_game")),
    )
