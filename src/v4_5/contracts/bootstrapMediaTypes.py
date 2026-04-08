from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.poiTypes import PoiAnalysisBundle


@dataclass(frozen=True)
class BootstrapSequenceConfig:
    schema_version: str
    plan_id: str
    game_id: str
    level_id: str
    primary_sequence: tuple[str, ...]
    fallback_sequences: tuple[tuple[str, ...], ...]
    stop_after_unique_avatar_found: bool
    capture_raw_observations: bool
    export_pngs: bool
    export_video: bool
    png_scale_factor: int
    video_fps: int
    status: str


@dataclass(frozen=True)
class BootstrapStepRecord:
    schema_version: str
    action: str
    status: str
    invalid_action: bool
    blocked_action: bool
    step_index: int
    raw_observation_ref: Any
    sequence_name: str = "primary"
    pre_observation_ref: Any | None = None
    post_observation_ref: Any | None = None


@dataclass(frozen=True)
class BootstrapPngArtifact:
    schema_version: str
    sequence_name: str
    png_paths: tuple[str, ...]
    scale_factor: int


@dataclass(frozen=True)
class BootstrapVideoArtifact:
    schema_version: str
    sequence_name: str
    video_path: str
    fps: int


@dataclass(frozen=True)
class BootstrapCaptureBundle:
    schema_version: str
    plan_id: str
    game_id: str
    level_id: str
    step_records: tuple[BootstrapStepRecord, ...]
    raw_observation_refs: tuple[Any, ...]
    status: str = "pending"
    png_artifacts: tuple[BootstrapPngArtifact, ...] = ()
    video_artifact: BootstrapVideoArtifact | None = None


@dataclass(frozen=True)
class BootstrapProbePlan:
    schema_version: str
    plan_id: str
    game_id: str
    level_id: str
    primary_sequence: tuple[str, ...]
    fallback_sequences: tuple[tuple[str, ...], ...]
    stop_after_unique_avatar_found: bool
    capture_raw_observations: bool
    export_pngs: bool
    export_video: bool
    png_scale_factor: int
    video_fps: int
    status: str
    capture_bundle: BootstrapCaptureBundle | None = None
    avatar_detection_result: AvatarDetectionResult | None = None
    hud_analysis_bundle: HudAnalysisBundle | None = None
    poi_analysis_bundle: PoiAnalysisBundle | None = None


@dataclass(frozen=True)
class HudRegionResult:
    schema_version: str
    source: str
    status: str
    hud_regions: tuple[str, ...] = ()
    life_regions: tuple[str, ...] = ()
    progress_regions: tuple[str, ...] = ()
    diagnostics: dict[str, Any] | None = None
    raw_response_text: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class HudAnalysisSelection:
    schema_version: str
    selected_source: str
    selected_result: HudRegionResult


@dataclass(frozen=True)
class HudAnalysisBundle:
    schema_version: str
    deterministic_result: HudRegionResult
    llm_text_result: HudRegionResult
    vlm_video_result: HudRegionResult
    selected_result: HudRegionResult
    selected_source: str
