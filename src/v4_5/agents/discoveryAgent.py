from __future__ import annotations

import json
from pathlib import Path

from v4_5.agents.poiRegistry import POIRegistryStore
from v4_5.agents.avatarDetector import AvatarDetector
from v4_5.adapters.stateAdapter import StateAdapter
from v4_5.bootstrap.bootstrapCapture import BootstrapCapture
from v4_5.bootstrap.bootstrapSequenceBuilder import BootstrapSequenceBuilder
from v4_5.bootstrap.hudAnalysisCoordinator import HudAnalysisCoordinator
from v4_5.bootstrap.hudDeterministicAnalyzer import HudDeterministicAnalyzer
from v4_5.bootstrap.hudTextAnalyzer import HudTextAnalyzer
from v4_5.bootstrap.hudVideoAnalyzer import HudVideoAnalyzer
from v4_5.bootstrap.poiAnalysisCoordinator import PoiAnalysisCoordinator
from v4_5.bootstrap.poiDeterministicAnalyzer import PoiDeterministicAnalyzer
from v4_5.bootstrap.poiTextAnalyzer import PoiTextAnalyzer
from v4_5.bootstrap.poiVideoAnalyzer import PoiVideoAnalyzer
from v4_5.bootstrap.pngExporter import PngExporter
from v4_5.bootstrap.videoExporter import VideoExporter
from v4_5.config.bootstrapConfig import BootstrapConfig, load_bootstrap_config
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapProbePlan
from v4_5.contracts import (
    AgentInput,
    BootstrapDiscoveryReport,
    ContractValidationError,
    DiscoveryReport,
    POIRecord,
    POIRegistry,
    PoiRecord,
    SCHEMA_VERSION,
    SceneSummary,
)
from v4_5.contracts.constants import POI_STATUS_CANDIDATE
from v4_5.contracts.errors import AvatarNotUniquelyIdentifiedError, BootstrapInvalidActionError
from v4_5.logging import BoundAgentLogger
from v4_5.memory.levelMemoryService import LevelMemoryService
from v4_5.memory.levelMemoryTypes import LevelMemoryRecord, MemoryRegion
from v4_5.perception.service import BoardPerceptionService
from v4_5.toon.toonClient import ToonClient


class DiscoveryAgent:
    agent_name = "DiscoveryAgent"
    bootstrap_budget = 6

    def __init__(
        self,
        state_adapter: StateAdapter | None = None,
        poi_registry: POIRegistryStore | None = None,
        avatar_detector: AvatarDetector | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        sequence_builder: BootstrapSequenceBuilder | None = None,
        bootstrap_capture: BootstrapCapture | None = None,
        png_exporter: PngExporter | None = None,
        video_exporter: VideoExporter | None = None,
        hud_analysis_coordinator: HudAnalysisCoordinator | None = None,
        poi_analysis_coordinator: PoiAnalysisCoordinator | None = None,
        board_perception_service: BoardPerceptionService | None = None,
        level_memory_service: LevelMemoryService | None = None,
        logger: BoundAgentLogger | None = None,
    ) -> None:
        self.state_adapter = state_adapter or StateAdapter()
        self.poi_registry = poi_registry or POIRegistryStore()
        self.avatar_detector = avatar_detector or AvatarDetector()
        self.bootstrap_config = bootstrap_config or load_bootstrap_config()
        self.sequence_builder = sequence_builder or BootstrapSequenceBuilder(self.bootstrap_config)
        self.bootstrap_capture = bootstrap_capture or BootstrapCapture()
        self.png_exporter = png_exporter or PngExporter()
        self.video_exporter = video_exporter or VideoExporter()
        if hud_analysis_coordinator is None:
            toon_client = ToonClient(self.bootstrap_config)
            hud_analysis_coordinator = HudAnalysisCoordinator(
                config=self.bootstrap_config,
                deterministic_analyzer=HudDeterministicAnalyzer(),
                text_analyzer=HudTextAnalyzer(toon_client),
                video_analyzer=HudVideoAnalyzer(toon_client),
                logger=logger,
            )
        self.hud_analysis_coordinator = hud_analysis_coordinator
        if poi_analysis_coordinator is None:
            toon_client = ToonClient(self.bootstrap_config)
            poi_analysis_coordinator = PoiAnalysisCoordinator(
                config=self.bootstrap_config,
                deterministic_analyzer=PoiDeterministicAnalyzer(),
                text_analyzer=PoiTextAnalyzer(toon_client, self.bootstrap_config),
                video_analyzer=PoiVideoAnalyzer(toon_client, self.bootstrap_config),
                logger=logger,
            )
        self.poi_analysis_coordinator = poi_analysis_coordinator
        self.board_perception_service = board_perception_service or BoardPerceptionService()
        self.level_memory_service = level_memory_service or LevelMemoryService()
        self.logger = logger
        self.last_summary: dict | None = None

    def run(self, agent_input: AgentInput, *, force_bootstrap: bool = False) -> DiscoveryReport:
        summary = self.state_adapter.summarize_observation(agent_input.observation)
        self.last_summary = summary
        loaded_level_memory = self._load_level_memory(agent_input)
        seen_levels = set(agent_input.memory.get("seen_levels", ()))
        bootstrap_required = force_bootstrap or agent_input.level_id not in seen_levels
        avatar_bbox = summary.get("avatar_bbox")
        avatar_position = summary.get("avatar_position")
        bootstrap_report = None
        bootstrap = ()
        bootstrap_plan = None
        bootstrap_capture_bundle = None
        bootstrap_analysis_bundle = None
        poi_analysis_bundle = None
        board_perception_report = None
        avatar_detection_result = None
        scene = None
        try:
            if bootstrap_required:
                if self.logger is not None:
                    self.logger.info(agent_input.env_id, "starting startup probing", round_id=agent_input.round_id, level_index=agent_input.level_id)
                bootstrap_plan = self._build_probe_plan(agent_input=agent_input)
                bootstrap_capture_bundle = self._collect_bootstrap_capture(agent_input, bootstrap_plan)
                avatar_detection_result = self._detect_avatar_from_capture(agent_input=agent_input, capture_bundle=bootstrap_capture_bundle)
                if avatar_detection_result is not None:
                    avatar_bbox = avatar_detection_result.avatar_bbox
                    avatar_position = avatar_detection_result.avatar_position
                if self.logger is not None and bootstrap_capture_bundle.step_records and bootstrap_plan.export_pngs:
                    self.logger.info(agent_input.env_id, "exporting bootstrap pngs", round_id=agent_input.round_id, level_index=agent_input.level_id)
                png_artifacts = ()
                video_artifact = None
                if bootstrap_capture_bundle.step_records and bootstrap_plan.export_pngs:
                    png_artifacts = self.png_exporter.export(
                        bundle=bootstrap_capture_bundle,
                        output_dir=str(agent_input.memory.get("artifact_dir", "artifacts/v4_5/bootstrap")),
                        scale_factor=bootstrap_plan.png_scale_factor,
                    )
                    bootstrap_capture_bundle = BootstrapCaptureBundle(
                        schema_version=bootstrap_capture_bundle.schema_version,
                        plan_id=bootstrap_capture_bundle.plan_id,
                        game_id=bootstrap_capture_bundle.game_id,
                        level_id=bootstrap_capture_bundle.level_id,
                        step_records=bootstrap_capture_bundle.step_records,
                        raw_observation_refs=bootstrap_capture_bundle.raw_observation_refs,
                        status=bootstrap_capture_bundle.status,
                        png_artifacts=png_artifacts,
                        video_artifact=bootstrap_capture_bundle.video_artifact,
                    )
                    if self.logger is not None and bootstrap_plan.export_video:
                        self.logger.info(agent_input.env_id, "exporting bootstrap video", round_id=agent_input.round_id, level_index=agent_input.level_id)
                    if bootstrap_plan.export_video:
                        video_artifact = self.video_exporter.export(
                            png_artifact=png_artifacts[0],
                            output_dir=str(agent_input.memory.get("artifact_dir", "artifacts/v4_5/bootstrap")),
                            fps=bootstrap_plan.video_fps,
                        )
                bootstrap_capture_bundle = BootstrapCaptureBundle(
                    schema_version=bootstrap_capture_bundle.schema_version,
                    plan_id=bootstrap_capture_bundle.plan_id,
                    game_id=bootstrap_capture_bundle.game_id,
                    level_id=bootstrap_capture_bundle.level_id,
                    step_records=bootstrap_capture_bundle.step_records,
                    raw_observation_refs=bootstrap_capture_bundle.raw_observation_refs,
                    status=bootstrap_capture_bundle.status,
                    png_artifacts=bootstrap_capture_bundle.png_artifacts,
                    video_artifact=video_artifact,
                )
                bootstrap = tuple(record.action for record in bootstrap_capture_bundle.step_records)
                poi_analysis_bundle = self.poi_analysis_coordinator.analyze(
                    game_id=agent_input.env_id,
                    capture_bundle=bootstrap_capture_bundle,
                    video_artifact=video_artifact,
                    avatar_bbox=avatar_bbox,
                    avatar_detection_result=avatar_detection_result,
                    hud_regions=(),
                    life_regions=(),
                    progress_regions=(),
                )
                bootstrap_analysis_bundle = self.hud_analysis_coordinator.analyze(
                    game_id=agent_input.env_id,
                    capture_bundle=bootstrap_capture_bundle,
                    video_artifact=video_artifact,
                    avatar_detection_result=avatar_detection_result,
                    poi_extraction_result=None if poi_analysis_bundle is None else poi_analysis_bundle.selected_pois,
                )
                bootstrap_plan = BootstrapProbePlan(
                    schema_version=bootstrap_plan.schema_version,
                    plan_id=bootstrap_plan.plan_id,
                    game_id=bootstrap_plan.game_id,
                    level_id=bootstrap_plan.level_id,
                    primary_sequence=bootstrap_plan.primary_sequence,
                    fallback_sequences=bootstrap_plan.fallback_sequences,
                    stop_after_unique_avatar_found=bootstrap_plan.stop_after_unique_avatar_found,
                    capture_raw_observations=bootstrap_plan.capture_raw_observations,
                    export_pngs=bootstrap_plan.export_pngs,
                    export_video=bootstrap_plan.export_video,
                    png_scale_factor=bootstrap_plan.png_scale_factor,
                    video_fps=bootstrap_plan.video_fps,
                    status="completed",
                    capture_bundle=bootstrap_capture_bundle,
                    avatar_detection_result=avatar_detection_result,
                    hud_analysis_bundle=bootstrap_analysis_bundle,
                    poi_analysis_bundle=poi_analysis_bundle,
                )
                bootstrap_report = BootstrapDiscoveryReport(
                    schema_version=SCHEMA_VERSION,
                    agent_name=self.agent_name,
                    round_id=agent_input.round_id,
                    plan=bootstrap_plan,
                    capture_bundle=bootstrap_capture_bundle,
                    avatar_detection_result=avatar_detection_result,
                    hud_analysis_bundle=bootstrap_analysis_bundle,
                    poi_analysis_bundle=poi_analysis_bundle,
                    selected_hud_result=(bootstrap_analysis_bundle.selected_result if bootstrap_analysis_bundle is not None else None),
                    selected_poi_result=(poi_analysis_bundle.selected_pois if poi_analysis_bundle is not None else None),
                    status="completed",
                    rationale_codes=("BOOTSTRAP_DISCOVERY",),
                )
            if self.logger is not None:
                self.logger.info(agent_input.env_id, "interpreting current level state", round_id=agent_input.round_id, level_index=agent_input.level_id)
            selected_hud = self._select_hud_regions(summary=summary, bootstrap_analysis_bundle=bootstrap_analysis_bundle)
            selected_pois = self._select_pois(summary=summary, poi_analysis_bundle=poi_analysis_bundle, avatar_bbox=avatar_bbox)
            background_summary = self._build_background_summary(summary=summary)
            raw_observation_payload = dict(summary.get("raw_observation_payload", {}))
            raw_observation_payload.update(background_summary)
            scene = SceneSummary(
                schema_version=SCHEMA_VERSION,
                agent_name=self.agent_name,
                round_id=agent_input.round_id,
                level_id=agent_input.level_id,
                avatar_bbox=avatar_bbox,
                avatar_position=avatar_position,
                hud_regions=selected_hud["hud_regions"],
                life_regions=selected_hud["life_regions"],
                progress_regions=selected_hud["progress_regions"],
                salient_changed_regions=self._memory_regions_from_serialized(tuple(summary.get("salient_changed_regions", ())), summary=summary),
                pois=selected_pois,
                candidate_mode_hints=summary["candidate_mode_hints"],
                observed_affordances=summary["observed_affordances"],
                levels_completed=summary.get("levels_completed"),
                terminal_flag=bool(summary.get("terminal_flag", False)),
                raw_observation_payload=raw_observation_payload,
                rationale_codes=("STATE_SUMMARY",),
            )
            if agent_input.game_control_profile is not None and agent_input.game_control_profile.control_category in {"movement_only", "move_and_click"}:
                if scene.avatar_bbox is None or scene.avatar_position is None:
                    self._write_discovery_round_artifact(
                        agent_input=agent_input,
                        bootstrap_required=bootstrap_required,
                        summary=summary,
                        bootstrap_plan=bootstrap_plan,
                        bootstrap_capture_bundle=bootstrap_capture_bundle,
                        avatar_detection_result=avatar_detection_result,
                        bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                        poi_analysis_bundle=poi_analysis_bundle,
                        scene=scene,
                        discovery_status="error_avatar_not_unique",
                        stop_reason="avatar_not_uniquely_identified",
                    )
                    raise AvatarNotUniquelyIdentifiedError("avatar not uniquely identified")
            if agent_input.game_control_profile is not None and agent_input.game_control_profile.control_category == "movement_only" and not scene.pois:
                raise ContractValidationError("movement-only deterministic POIs empty")
            board_requested = bootstrap_required or bool(agent_input.memory.get("request_board_perception"))
            if board_requested:
                observation_window = self.board_perception_service.observation_window_for_discovery(
                    observation=agent_input.observation,
                    bootstrap_capture_bundle=bootstrap_capture_bundle,
                    parsed_state=agent_input.parsed_state,
                )
                try:
                    board_perception_report = self.board_perception_service.build_report(
                        observations=observation_window,
                        round_id=agent_input.round_id,
                    )
                except Exception:
                    board_perception_report = None
                else:
                    agent_input.memory["board_perception_report"] = board_perception_report
                    agent_input.memory["board_perception_last_round_id"] = agent_input.round_id
            if self._no_identified_artifacts(scene=scene, summary=summary):
                report = DiscoveryReport(
                    schema_version=SCHEMA_VERSION,
                    agent_name=self.agent_name,
                    round_id=agent_input.round_id,
                    scene_summary=scene,
                    game_control_profile=agent_input.game_control_profile,
                    bootstrap_required=bootstrap_required,
                    bootstrap_probe_summary=bootstrap,
                    bootstrap_plan=bootstrap_plan,
                    avatar_detection_result=avatar_detection_result,
                    bootstrap_report=bootstrap_report,
                    bootstrap_capture_bundle=bootstrap_capture_bundle,
                    bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                    poi_analysis_bundle=poi_analysis_bundle,
                    board_perception_report=board_perception_report,
                    loaded_level_memory=loaded_level_memory,
                    poi_registry=None,
                    stop_reason="no_artifacts_identified",
                    rationale_codes=("BOOTSTRAP_REQUIRED",) if bootstrap_required else ("BOOTSTRAP_NOT_REQUIRED",),
                )
                self._write_discovery_round_artifact(
                    agent_input=agent_input,
                    bootstrap_required=bootstrap_required,
                    summary=summary,
                    bootstrap_plan=bootstrap_plan,
                    bootstrap_capture_bundle=bootstrap_capture_bundle,
                    avatar_detection_result=avatar_detection_result,
                    bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                    poi_analysis_bundle=poi_analysis_bundle,
                    scene=scene,
                    discovery_status="stop_no_artifacts",
                    stop_reason="no_artifacts_identified",
                )
                return report
            if self.logger is not None:
                self.logger.info(agent_input.env_id, "identifying points of interest", round_id=agent_input.round_id, level_index=agent_input.level_id)
            poi_registry = self._populate_poi_registry(agent_input=agent_input, scene=scene)
            if self.logger is not None:
                self.logger.info(agent_input.env_id, "identifying candidate hazards", round_id=agent_input.round_id, level_index=agent_input.level_id)
            if bootstrap_report is not None:
                bootstrap_report = BootstrapDiscoveryReport(
                    schema_version=bootstrap_report.schema_version,
                    agent_name=bootstrap_report.agent_name,
                    round_id=bootstrap_report.round_id,
                    plan=bootstrap_report.plan,
                    capture_bundle=bootstrap_report.capture_bundle,
                    avatar_detection_result=bootstrap_report.avatar_detection_result,
                    hud_analysis_bundle=bootstrap_report.hud_analysis_bundle,
                    poi_analysis_bundle=bootstrap_report.poi_analysis_bundle,
                    selected_hud_result=bootstrap_report.selected_hud_result,
                    selected_poi_result=bootstrap_report.selected_poi_result,
                    status=bootstrap_report.status,
                    error_message=bootstrap_report.error_message,
                    warnings=bootstrap_report.warnings,
                    rationale_codes=bootstrap_report.rationale_codes,
                )
            if self.logger is not None:
                self.logger.info(agent_input.env_id, "completing discovery report", round_id=agent_input.round_id, level_index=agent_input.level_id)
            self._save_level_memory(agent_input=agent_input, scene=scene)
            updated_seen_levels = tuple(sorted(seen_levels | {agent_input.level_id}))
            agent_input.memory["seen_levels"] = updated_seen_levels
            report = DiscoveryReport(
                schema_version=SCHEMA_VERSION,
                agent_name=self.agent_name,
                round_id=agent_input.round_id,
                scene_summary=scene,
                game_control_profile=agent_input.game_control_profile,
                bootstrap_required=bootstrap_required,
                bootstrap_probe_summary=bootstrap,
                bootstrap_plan=bootstrap_plan,
                avatar_detection_result=avatar_detection_result,
                bootstrap_report=bootstrap_report,
                bootstrap_capture_bundle=bootstrap_capture_bundle,
                bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                poi_analysis_bundle=poi_analysis_bundle,
                board_perception_report=board_perception_report,
                loaded_level_memory=loaded_level_memory,
                poi_registry=poi_registry,
                stop_reason=None,
                rationale_codes=("BOOTSTRAP_REQUIRED",) if bootstrap_required else ("BOOTSTRAP_NOT_REQUIRED",),
            )
            self._write_discovery_round_artifact(
                agent_input=agent_input,
                bootstrap_required=bootstrap_required,
                summary=summary,
                bootstrap_plan=bootstrap_plan,
                bootstrap_capture_bundle=bootstrap_capture_bundle,
                avatar_detection_result=avatar_detection_result,
                bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                poi_analysis_bundle=poi_analysis_bundle,
                scene=scene,
                discovery_status="ok",
                stop_reason=None,
            )
            return report
        except AvatarNotUniquelyIdentifiedError as exc:
            if avatar_detection_result is None:
                avatar_detection_result = getattr(exc, "avatar_detection_result", None)
            if scene is None:
                self._write_discovery_round_artifact(
                    agent_input=agent_input,
                    bootstrap_required=bootstrap_required,
                    summary=summary,
                    bootstrap_plan=bootstrap_plan,
                    bootstrap_capture_bundle=bootstrap_capture_bundle,
                    avatar_detection_result=avatar_detection_result,
                    bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                    poi_analysis_bundle=poi_analysis_bundle,
                    scene=scene,
                    discovery_status="error_avatar_not_unique",
                    stop_reason="avatar_not_uniquely_identified",
                )
            raise

    def _build_probe_plan(self, *, agent_input: AgentInput) -> BootstrapProbePlan:
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "building bootstrap plan", round_id=agent_input.round_id, level_index=agent_input.level_id)
        return self.sequence_builder.build(game_id=agent_input.env_id, level_id=agent_input.level_id)

    def _collect_bootstrap_capture(self, agent_input: AgentInput, bootstrap_plan: BootstrapProbePlan):
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "starting bootstrap capture", round_id=agent_input.round_id, level_index=agent_input.level_id)
        profile = agent_input.game_control_profile
        if profile is None or profile.control_category not in {"movement_only", "move_and_click"}:
            return BootstrapCaptureBundle(
                schema_version="v4.5",
                plan_id=bootstrap_plan.plan_id,
                game_id=bootstrap_plan.game_id,
                level_id=bootstrap_plan.level_id,
                step_records=(),
                raw_observation_refs=(),
                status="skipped",
            )
        executor = agent_input.prior_reports.get("bootstrap_capture_executor")
        if not callable(executor):
            raise AvatarNotUniquelyIdentifiedError("bootstrap capture executor missing")
        return self.bootstrap_capture.capture(
            plan=bootstrap_plan,
            execute_sequence=executor,
            detect_avatar=lambda bundle: self._detect_avatar_from_capture(agent_input=agent_input, capture_bundle=bundle, strict=False),
            game_id=agent_input.env_id,
        )

    def _detect_avatar_from_capture(self, *, agent_input: AgentInput, capture_bundle, strict: bool = True):
        profile = agent_input.game_control_profile
        if profile is None or profile.control_category == "click_only":
            return None
        sequences = self._split_bootstrap_records_by_sequence(capture_bundle)
        primary_sequence = sequences.get("primary")
        fallback_sequence = sequences.get("fallback")
        if not primary_sequence:
            if strict:
                raise AvatarNotUniquelyIdentifiedError("avatar probe frames missing")
            return None
        try:
            return self.avatar_detector.detect(primary_sequence=primary_sequence, fallback_sequence=fallback_sequence)
        except AvatarNotUniquelyIdentifiedError as exc:
            setattr(exc, "avatar_detection_result", getattr(self.avatar_detector, "last_result", None))
            authoritative_detector = agent_input.prior_reports.get("bootstrap_avatar_detector")
            if callable(authoritative_detector):
                authoritative_result = authoritative_detector()
                if authoritative_result is not None:
                    return authoritative_result
            if strict:
                raise exc
            return getattr(exc, "avatar_detection_result", None)

    def _split_bootstrap_records_by_sequence(self, capture_bundle) -> dict[str, tuple]:
        if capture_bundle is None:
            return {}
        sequences = {}
        for sequence_name in tuple(dict.fromkeys(record.sequence_name for record in capture_bundle.step_records)):
            records = tuple(record for record in capture_bundle.step_records if record.sequence_name == sequence_name)
            sequences["primary" if sequence_name == "primary" else "fallback"] = records
        return sequences

    def _populate_poi_registry(self, *, agent_input: AgentInput, scene: SceneSummary) -> POIRegistry:
        registry = self.poi_registry.create(round_id=agent_input.round_id)
        for idx, poi in enumerate(scene.pois):
            if scene.avatar_bbox is not None and self._bboxes_overlap(scene.avatar_bbox, poi.bbox):
                continue
            registry = self.poi_registry.insert_candidate_poi(registry, self._poi_registry_record(agent_input=agent_input, poi=poi, idx=idx))
        return registry

    def _load_level_memory(self, agent_input: AgentInput):
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "loading level memory", round_id=agent_input.round_id, level_index=agent_input.level_id)
        return self.level_memory_service.load_level_memory(agent_input.env_id, agent_input.level_id)

    def _save_level_memory(self, *, agent_input: AgentInput, scene: SceneSummary):
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "saving level memory", round_id=agent_input.round_id, level_index=agent_input.level_id)
        record = LevelMemoryRecord(
            game_id=agent_input.env_id,
            level_id=agent_input.level_id,
            memory_state="hypothesis",
            avatar=self._avatar_region(scene),
            hud_regions=tuple(scene.hud_regions),
            life_regions=tuple(scene.life_regions),
            progress_regions=tuple(scene.progress_regions),
            pois=tuple(self._poi_to_memory_region(item) for item in scene.pois),
            exit_regions=(),
            created_at="",
            updated_at="",
            schema_version=SCHEMA_VERSION,
        )
        return self.level_memory_service.save_hypothesis_level_memory(record)

    def _write_discovery_round_artifact(
        self,
        *,
        agent_input: AgentInput,
        bootstrap_required: bool,
        summary: dict,
        bootstrap_plan: BootstrapProbePlan | None,
        bootstrap_capture_bundle: BootstrapCaptureBundle | None,
        avatar_detection_result,
        bootstrap_analysis_bundle,
        poi_analysis_bundle,
        scene: SceneSummary | None,
        discovery_status: str,
        stop_reason: str | None,
    ) -> str:
        artifact_root = Path(agent_input.memory.get("artifact_dir", "artifacts/v4_5"))
        artifact_path = artifact_root / "discovery" / agent_input.env_id / agent_input.level_id / f"{agent_input.round_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "game_id": agent_input.env_id,
            "level_id": agent_input.level_id,
            "round_id": agent_input.round_id,
            "bootstrap_required": bool(bootstrap_required),
            "discovery_status": discovery_status,
            "summary": self._artifact_summary(summary),
            "bootstrap": self._artifact_bootstrap(
                bootstrap_plan=bootstrap_plan,
                bootstrap_capture_bundle=bootstrap_capture_bundle,
                avatar_detection_result=avatar_detection_result,
                bootstrap_analysis_bundle=bootstrap_analysis_bundle,
                poi_analysis_bundle=poi_analysis_bundle,
            ),
            "scene": self._artifact_scene(scene=scene, summary=summary),
            "identified_artifacts": self._artifact_identified_artifacts(scene=scene, summary=summary),
            "stop_reason": stop_reason,
        }
        artifact_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        counts = payload["identified_artifacts"]
        if self.logger is not None:
            self.logger.info(
                agent_input.env_id,
                "saved discovery round artifact",
                round_id=agent_input.round_id,
                level_index=agent_input.level_id,
                structured_fields={
                    "artifact_path": str(artifact_path),
                    "discovery_status": discovery_status,
                    "stop_reason": stop_reason,
                    "avatar_present": counts["avatar_present"],
                    "poi_count": counts["poi_count"],
                    "hud_region_count": counts["hud_region_count"],
                    "hazard_count": counts["hazard_count"],
                },
            )
        return str(artifact_path)

    def _artifact_summary(self, summary: dict) -> dict:
        return {
            "avatar_bbox": summary.get("avatar_bbox"),
            "avatar_position": summary.get("avatar_position"),
            "candidate_clickable_regions": tuple(summary.get("candidate_clickable_regions", ())),
            "hud_regions": tuple(summary.get("hud_regions", ())),
            "life_regions": tuple(summary.get("life_regions", ())),
            "progress_regions": tuple(summary.get("progress_regions", ())),
            "salient_changed_regions": tuple(summary.get("salient_changed_regions", ())),
            "candidate_hazards": tuple(summary.get("candidate_hazards", ())),
            "candidate_mode_hints": tuple(summary.get("candidate_mode_hints", ())),
            "observed_affordances": tuple(summary.get("observed_affordances", ())),
            "levels_completed": summary.get("levels_completed"),
            "terminal_flag": bool(summary.get("terminal_flag", False)),
        }

    def _artifact_bootstrap(self, *, bootstrap_plan, bootstrap_capture_bundle, avatar_detection_result, bootstrap_analysis_bundle, poi_analysis_bundle) -> dict:
        diagnostics = {} if avatar_detection_result is None else dict(getattr(avatar_detection_result, "diagnostics", {}) or {})
        hud_diagnostics = {} if bootstrap_analysis_bundle is None else dict(getattr(bootstrap_analysis_bundle.selected_result, "diagnostics", {}) or {})
        poi_diagnostics = {} if poi_analysis_bundle is None else dict(getattr(poi_analysis_bundle.selected_pois, "diagnostics", {}) or {})
        return {
            "plan_id": None if bootstrap_plan is None else bootstrap_plan.plan_id,
            "primary_sequence": () if bootstrap_plan is None else tuple(bootstrap_plan.primary_sequence),
            "fallback_sequences": () if bootstrap_plan is None else tuple(tuple(item) for item in bootstrap_plan.fallback_sequences),
            "executed_probe_actions": () if bootstrap_capture_bundle is None else tuple(record.action for record in bootstrap_capture_bundle.step_records),
            "capture_status": None if bootstrap_capture_bundle is None else bootstrap_capture_bundle.status,
            "step_records": [] if bootstrap_capture_bundle is None else [
                {
                    "sequence_name": record.sequence_name,
                    "step_index": record.step_index,
                    "action": record.action,
                    "status": record.status,
                    "invalid_action": bool(record.invalid_action),
                    "blocked_action": bool(record.blocked_action),
                }
                for record in bootstrap_capture_bundle.step_records
            ],
            "avatar_detection_result": None if avatar_detection_result is None else {
                "avatar_bbox": avatar_detection_result.avatar_bbox,
                "avatar_center": getattr(avatar_detection_result, "avatar_center", getattr(avatar_detection_result, "avatar_position", None)),
                "avatar_position": getattr(avatar_detection_result, "avatar_position", None),
                "support_actions": tuple(getattr(avatar_detection_result, "support_actions", ()) or ()),
                "support_step_indices": tuple(getattr(avatar_detection_result, "support_step_indices", ()) or ()),
                "confidence": float(getattr(avatar_detection_result, "confidence", 0.0) or 0.0),
                "avatar_value_candidates": tuple(getattr(avatar_detection_result, "avatar_value_candidates", ()) or ()),
                "failure_reason": getattr(avatar_detection_result, "failure_reason", None),
                "used_fallback": bool(getattr(avatar_detection_result, "used_fallback", False)),
            },
            "avatar_extractor_diagnostics": {
                "per_step_candidate_count": dict(diagnostics.get("per_step_candidate_count", {})),
                "top_candidate_scores_per_step": dict(diagnostics.get("top_candidate_scores_per_step", {})),
                "total_track_count": int(diagnostics.get("total_track_count", 0) or 0),
                "best_track_support_steps": tuple(diagnostics.get("best_track_support_steps", ()) or ()),
                "best_track_confidence": float(diagnostics.get("best_track_confidence", 0.0) or 0.0),
                "failure_reason": diagnostics.get("failure_reason"),
            },
            "selected_hud_result": None if bootstrap_analysis_bundle is None else {
                "hud_regions": tuple(bootstrap_analysis_bundle.selected_result.hud_regions),
                "life_regions": tuple(bootstrap_analysis_bundle.selected_result.life_regions),
                "progress_regions": tuple(bootstrap_analysis_bundle.selected_result.progress_regions),
            },
            "hud_extractor_diagnostics": {
                "per_frame_non_background_component_count": dict(hud_diagnostics.get("per_frame_non_background_component_count", {})),
                "hud_track_count": int(hud_diagnostics.get("hud_track_count", 0) or 0),
                "selected_hud_region_count": int(hud_diagnostics.get("selected_hud_region_count", 0) or 0),
                "progress_candidate_count": int(hud_diagnostics.get("progress_candidate_count", 0) or 0),
                "selected_progress_region_count": int(hud_diagnostics.get("selected_progress_region_count", 0) or 0),
                "rejection_counts": dict(hud_diagnostics.get("rejection_counts", {})),
                "best_hud_scores": tuple(hud_diagnostics.get("best_hud_scores", ()) or ()),
                "best_progress_scores": tuple(hud_diagnostics.get("best_progress_scores", ()) or ()),
                "hud_failure_reason": hud_diagnostics.get("hud_failure_reason", hud_diagnostics.get("failure_reason")),
                "progress_failure_reason": hud_diagnostics.get("progress_failure_reason"),
            },
            "selected_poi_result": None if poi_analysis_bundle is None else {
                "pois": [self._serialize_poi(poi) for poi in poi_analysis_bundle.selected_pois.pois],
            },
            "poi_extractor_diagnostics": {
                "per_frame_component_count_after_avatar_exclusion": dict(poi_diagnostics.get("per_frame_component_count_after_avatar_exclusion", {})),
                "non_avatar_track_count": int(poi_diagnostics.get("non_avatar_track_count", 0) or 0),
                "rejected_for_avatar_overlap_count": int(poi_diagnostics.get("rejected_for_avatar_overlap_count", 0) or 0),
                "rejected_for_motion_correlation_count": int(poi_diagnostics.get("rejected_for_motion_correlation_count", 0) or 0),
                "best_candidate_scores": tuple(poi_diagnostics.get("best_candidate_scores", ()) or ()),
                "selected_poi_count": int(poi_diagnostics.get("selected_poi_count", 0) or 0),
                "failure_reason": poi_diagnostics.get("failure_reason"),
            },
        }

    def _artifact_scene(self, *, scene: SceneSummary | None, summary: dict) -> dict:
        return {
            "avatar_bbox": None if scene is None else scene.avatar_bbox,
            "avatar_position": None if scene is None else scene.avatar_position,
            "hud_regions": [] if scene is None else [self._serialize_region(region) for region in scene.hud_regions],
            "life_regions": [] if scene is None else [self._serialize_region(region) for region in scene.life_regions],
            "progress_regions": [] if scene is None else [self._serialize_region(region) for region in scene.progress_regions],
            "pois": [] if scene is None else [self._serialize_poi(poi) for poi in scene.pois],
            "candidate_hazards": tuple(summary.get("candidate_hazards", ())),
            "candidate_mode_hints": () if scene is None else tuple(scene.candidate_mode_hints),
            "observed_affordances": () if scene is None else tuple(scene.observed_affordances),
        }

    def _artifact_identified_artifacts(self, *, scene: SceneSummary | None, summary: dict) -> dict:
        return {
            "avatar_present": bool(scene is not None and scene.avatar_bbox is not None and scene.avatar_position is not None),
            "hud_region_count": 0 if scene is None else len(scene.hud_regions),
            "life_region_count": 0 if scene is None else len(scene.life_regions),
            "progress_region_count": 0 if scene is None else len(scene.progress_regions),
            "poi_count": 0 if scene is None else len(scene.pois),
            "hazard_count": len(tuple(summary.get("candidate_hazards", ()))),
            "clickable_region_count": len(tuple(summary.get("candidate_clickable_regions", ()))),
        }

    def _serialize_region(self, region: MemoryRegion) -> dict:
        return {
            "bbox": region.bbox,
            "center": region.center,
            "colors": tuple(region.colors),
            "description": region.description,
            "hint": region.hint,
        }

    def _serialize_poi(self, poi: PoiRecord) -> dict:
        return {
            "bbox": poi.bbox,
            "center": poi.center,
            "colors": tuple(poi.colors),
            "support_step_indices": tuple(getattr(poi, "support_step_indices", ()) or ()),
            "value_candidates": tuple(getattr(poi, "value_candidates", ()) or ()),
            "stability_score": float(getattr(poi, "stability_score", 0.0) or 0.0),
            "reachability_score": float(getattr(poi, "reachability_score", 0.0) or 0.0),
            "poi_score": float(getattr(poi, "poi_score", 0.0) or 0.0),
            "rejected_as_avatar_overlap": bool(getattr(poi, "rejected_as_avatar_overlap", False)),
            "failure_reason": getattr(poi, "failure_reason", None),
            "description": poi.description,
            "hint": poi.hint,
        }

    def _no_identified_artifacts(self, *, scene: SceneSummary, summary: dict) -> bool:
        return (
            scene.avatar_bbox is None
            and scene.avatar_position is None
            and not scene.pois
            and not scene.hud_regions
            and not scene.life_regions
            and not scene.progress_regions
            and not tuple(summary.get("candidate_hazards", ()))
            and not tuple(summary.get("candidate_clickable_regions", ()))
        )

    def _avatar_region(self, scene: SceneSummary) -> MemoryRegion | None:
        if scene.avatar_bbox is None or scene.avatar_position is None:
            return None
        return MemoryRegion(
            bbox=scene.avatar_bbox,
            center=scene.avatar_position,
            colors=(),
            description=None,
            hint=None,
        )

    def _select_hud_regions(self, *, summary: dict, bootstrap_analysis_bundle) -> dict[str, tuple[MemoryRegion, ...]]:
        if bootstrap_analysis_bundle is not None:
            deterministic = bootstrap_analysis_bundle.deterministic_result
            return {
                "hud_regions": self._memory_regions_from_serialized(tuple(deterministic.hud_regions), summary=summary),
                "life_regions": self._memory_regions_from_serialized(tuple(deterministic.life_regions), summary=summary),
                "progress_regions": self._memory_regions_from_serialized(tuple(deterministic.progress_regions), summary=summary),
            }
        return {
            "hud_regions": self._memory_regions_from_serialized(tuple(summary.get("hud_regions", ())), summary=summary),
            "life_regions": self._memory_regions_from_serialized(tuple(summary.get("life_regions", ())), summary=summary),
            "progress_regions": self._memory_regions_from_serialized(tuple(summary.get("progress_regions", ())), summary=summary),
        }

    def _select_pois(self, *, summary: dict, poi_analysis_bundle, avatar_bbox) -> tuple[PoiRecord, ...]:
        if poi_analysis_bundle is not None:
            return tuple(
                poi
                for poi in poi_analysis_bundle.selected_pois.pois
                if not getattr(poi, "rejected_as_avatar_overlap", False)
                and getattr(poi, "failure_reason", None) is None
                and (avatar_bbox is None or not self._bboxes_overlap(avatar_bbox, poi.bbox))
            )
        poi_cells = tuple(summary.get("raw_observation_payload", {}).get("poi_cells", ()))
        return tuple(
            PoiRecord(
                bbox=(x, y, x, y),
                center=(float(x), float(y)),
                colors=self._colors_for_bbox((x, y, x, y), summary),
            )
            for x, y in poi_cells
            if avatar_bbox is None or not self._bboxes_overlap(avatar_bbox, (x, y, x, y))
        )

    def _build_background_summary(self, *, summary: dict) -> dict:
        payload = dict(summary.get("raw_observation_payload", {}))
        return {
            "traversable_regions": tuple(payload.get("traversable_regions", ())),
            "blocking_regions": tuple(payload.get("blocking_regions", ())),
            "unknown_regions": tuple(payload.get("unknown_regions", ())),
            "playfield_bbox": payload.get("playfield_bbox"),
        }

    def _memory_regions_from_serialized(self, regions: tuple[str, ...], *, summary: dict) -> tuple[MemoryRegion, ...]:
        parsed_regions = []
        for component in self._group_serialized_cells(regions):
            bbox = self._bbox_from_cells(component)
            if bbox is None:
                continue
            parsed_regions.append(
                MemoryRegion(
                    bbox=bbox,
                    center=self._center_from_bbox(bbox),
                    colors=self._colors_for_bbox(bbox, summary),
                    description=None,
                    hint=None,
                )
            )
        return tuple(parsed_regions)

    def _group_serialized_cells(self, regions: tuple[str, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
        cells = set()
        for region in regions:
            cells.update(self._cells_from_serialized(region))
        remaining = set(cells)
        groups = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            group = {start}
            while stack:
                x, y = stack.pop()
                for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        group.add(neighbor)
                        stack.append(neighbor)
            groups.append(tuple(sorted(group)))
        return tuple(groups)

    def _cells_from_serialized(self, region: str) -> set[tuple[int, int]]:
        if not isinstance(region, str):
            return set()
        if region.startswith("cell:"):
            parts = region.split("|")
            cells = set()
            for part in parts:
                if not part.startswith("cell:") or "," not in part:
                    continue
                x_str, y_str = part.removeprefix("cell:").split(",", 1)
                cells.add((int(x_str), int(y_str)))
            return cells
        return set()

    def _memory_region_from_string(self, region: str) -> MemoryRegion | None:
        if not isinstance(region, str) or not region.startswith("cell:"):
            return None
        x_str, y_str = region.removeprefix("cell:").split(",", 1)
        x = int(x_str)
        y = int(y_str)
        return MemoryRegion(
            bbox=(x, y, x, y),
            center=(float(x), float(y)),
            colors=(),
            description=None,
            hint=None,
        )

    def _poi_to_memory_region(self, poi: PoiRecord) -> MemoryRegion:
        return MemoryRegion(
            bbox=poi.bbox,
            center=poi.center,
            colors=poi.colors,
            description=poi.description,
            hint=poi.hint,
        )

    def _poi_registry_record(self, *, agent_input: AgentInput, poi: PoiRecord, idx: int) -> POIRecord:
        return POIRecord(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=agent_input.round_id,
            poi_id=f"{agent_input.env_id}:{agent_input.level_id}:poi:{idx:02d}",
            game_id=agent_input.env_id,
            level_index=agent_input.level_id,
            type_hint="poi",
            source="discovery",
            confidence=float(getattr(poi, "poi_score", 1.0) or 1.0),
            status=POI_STATUS_CANDIDATE,
            bbox=poi.bbox,
            center=poi.center,
            colors=poi.colors,
            description=poi.description,
            hint=poi.hint,
            linked_hypotheses=(),
            rationale_codes=("DISCOVERY_POI",),
        )

    def _bboxes_overlap(self, left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
        return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])

    def _colors_for_bbox(self, bbox: tuple[int, int, int, int], summary: dict) -> tuple[int, ...]:
        frame = summary.get("raw_observation_payload", {}).get("pre_frame") or ()
        if not frame:
            return ()
        colors = set()
        for y in range(bbox[1], bbox[3] + 1):
            for x in range(bbox[0], bbox[2] + 1):
                if 0 <= y < len(frame) and 0 <= x < len(frame[0]):
                    colors.add(int(frame[y][x]))
        return tuple(sorted(colors))

    def _center_from_bbox(self, bbox: tuple[int, int, int, int] | None) -> tuple[float, float] | None:
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _bbox_from_cells(self, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | set[tuple[int, int]]):
        if not cells:
            return None
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        return (min(xs), min(ys), max(xs), max(ys))
