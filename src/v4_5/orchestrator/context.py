from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v4_5.contracts import BoardPerceptionReport
from v4_5.contracts import BootstrapDiscoveryReport, POIRegistry, TrajectoryQueue
from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapProbePlan, HudAnalysisBundle
from v4_5.contracts.poiTypes import PoiAnalysisBundle
from v4_5.memory.levelMemoryTypes import LevelMemoryRecord
from v4_5.orchestrator.stages import Stage


@dataclass
class OrchestratorContext:
    env_id: str
    level_id: str
    round_id: str
    observation: Any | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    stage: Stage = Stage.BOOTSTRAP
    unseen_level: bool = True
    force_bootstrap: bool = False
    bootstrap_complete: dict[str, bool] = field(default_factory=dict)
    bootstrap_plan: BootstrapProbePlan | None = None
    bootstrap_report: BootstrapDiscoveryReport | None = None
    bootstrap_capture_bundle: BootstrapCaptureBundle | None = None
    avatar_detection_result: AvatarDetectionResult | None = None
    hud_analysis_bundle: HudAnalysisBundle | None = None
    poi_analysis_bundle: PoiAnalysisBundle | None = None
    board_perception_report: BoardPerceptionReport | None = None
    poi_registry: POIRegistry | None = None
    trajectory_queue: TrajectoryQueue | None = None
    selected_work_item_id: str | None = None
    loaded_level_memory: LevelMemoryRecord | None = None
    live_snapshot: Any | None = None
    last_executed_prefix_result: Any | None = None
    last_committed_prefix: tuple[str, ...] = ()
    execution_committed: bool = False
