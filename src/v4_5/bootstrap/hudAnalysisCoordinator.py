from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from v4_5.bootstrap.hudDeterministicAnalyzer import HudDeterministicAnalyzer
from v4_5.bootstrap.hudTextAnalyzer import HudTextAnalyzer
from v4_5.bootstrap.hudVideoAnalyzer import HudVideoAnalyzer
from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapVideoArtifact, HudAnalysisBundle, HudRegionResult
from v4_5.logging import BoundAgentLogger


class HudAnalysisCoordinator:
    def __init__(
        self,
        *,
        config: BootstrapConfig,
        deterministic_analyzer: HudDeterministicAnalyzer,
        text_analyzer: HudTextAnalyzer,
        video_analyzer: HudVideoAnalyzer,
        logger: BoundAgentLogger | None = None,
    ) -> None:
        self.config = config
        self.deterministic_analyzer = deterministic_analyzer
        self.text_analyzer = text_analyzer
        self.video_analyzer = video_analyzer
        self.logger = logger

    def analyze(
        self,
        *,
        game_id: str,
        capture_bundle: BootstrapCaptureBundle,
        video_artifact: BootstrapVideoArtifact | None,
        avatar_detection_result=None,
        poi_extraction_result=None,
    ) -> HudAnalysisBundle:
        if self.logger is not None:
            self.logger.info(game_id, "gathering HUD analysis results")
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.logger is not None:
                self.logger.info(game_id, "running deterministic HUD analysis")
            deterministic_future = executor.submit(
                self.deterministic_analyzer.analyze,
                capture_bundle=capture_bundle,
                avatar_detection_result=avatar_detection_result,
                poi_extraction_result=poi_extraction_result,
            )
            if self.logger is not None:
                self.logger.info(game_id, "running text HUD analysis")
            text_future = executor.submit(self._run_text, capture_bundle)
            if self.logger is not None:
                self.logger.info(game_id, "running video HUD analysis")
            video_future = executor.submit(self._run_video, video_artifact)
            deterministic_result = deterministic_future.result()
            text_result = text_future.result()
            video_result = video_future.result()
        if self.logger is not None:
            self.logger.info(game_id, "selecting deterministic HUD result")
        return HudAnalysisBundle(
            schema_version="v4.5",
            deterministic_result=deterministic_result,
            llm_text_result=text_result,
            vlm_video_result=video_result,
            selected_result=deterministic_result,
            selected_source="deterministic",
        )

    def _run_text(self, capture_bundle: BootstrapCaptureBundle) -> HudRegionResult:
        if not self.config.enable_llm_hud_analysis:
            return HudRegionResult(schema_version="v4.5", source="llm_text", status="disabled")
        try:
            return self.text_analyzer.analyze(capture_bundle=capture_bundle, prompt_path=self.config.hud_text_prompt_path)
        except Exception as exc:
            return HudRegionResult(schema_version="v4.5", source="llm_text", status="failed", error_message=str(exc))

    def _run_video(self, video_artifact: BootstrapVideoArtifact | None) -> HudRegionResult:
        if not self.config.enable_vlm_hud_analysis:
            return HudRegionResult(schema_version="v4.5", source="vlm_video", status="disabled")
        if video_artifact is None:
            return HudRegionResult(schema_version="v4.5", source="vlm_video", status="failed", error_message="video artifact missing")
        try:
            return self.video_analyzer.analyze(video_artifact=video_artifact, prompt_path=self.config.hud_video_prompt_path)
        except Exception as exc:
            return HudRegionResult(schema_version="v4.5", source="vlm_video", status="failed", error_message=str(exc))
