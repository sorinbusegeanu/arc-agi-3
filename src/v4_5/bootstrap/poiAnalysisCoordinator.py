from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from v4_5.bootstrap.poiDeterministicAnalyzer import PoiDeterministicAnalyzer
from v4_5.bootstrap.poiTextAnalyzer import PoiTextAnalyzer
from v4_5.bootstrap.poiVideoAnalyzer import PoiVideoAnalyzer
from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapVideoArtifact
from v4_5.contracts.poiTypes import PoiAnalysisBundle, PoiSet
from v4_5.logging import BoundAgentLogger


class PoiAnalysisCoordinator:
    def __init__(
        self,
        *,
        config: BootstrapConfig,
        deterministic_analyzer: PoiDeterministicAnalyzer,
        text_analyzer: PoiTextAnalyzer,
        video_analyzer: PoiVideoAnalyzer,
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
        avatar_bbox: tuple[int, int, int, int] | None,
        avatar_detection_result=None,
        hud_regions: tuple[str, ...],
        life_regions: tuple[str, ...],
        progress_regions: tuple[str, ...],
    ) -> PoiAnalysisBundle:
        if self.logger is not None:
            self.logger.info(game_id, "gathering point-of-interest results")
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.logger is not None:
                self.logger.info(game_id, "detecting deterministic points of interest")
            deterministic_future = executor.submit(
                self.deterministic_analyzer.analyze,
                capture_bundle=capture_bundle,
                avatar_bbox=avatar_bbox,
                avatar_detection_result=avatar_detection_result,
                hud_regions=hud_regions,
                life_regions=life_regions,
                progress_regions=progress_regions,
            )
            if self.logger is not None:
                self.logger.info(game_id, "running text point-of-interest analysis")
            text_future = executor.submit(self._run_text, capture_bundle)
            if self.logger is not None:
                self.logger.info(game_id, "running video point-of-interest analysis")
            video_future = executor.submit(self._run_video, video_artifact)
            deterministic_result = deterministic_future.result()
            text_result = text_future.result()
            video_result = video_future.result()
        if self.logger is not None:
            self.logger.info(game_id, "selecting deterministic point-of-interest result")
        return PoiAnalysisBundle(
            schema_version="v4.5",
            deterministic_pois=deterministic_result,
            llm_text_pois=text_result,
            vlm_video_pois=video_result,
            selected_pois=deterministic_result,
            selected_source="deterministic",
        )

    def _run_text(self, capture_bundle: BootstrapCaptureBundle) -> PoiSet:
        if not self.config.enable_llm_poi_analysis:
            return PoiSet(schema_version="v4.5", source="llm_text", status="disabled")
        try:
            return self.text_analyzer.analyze(capture_bundle=capture_bundle, prompt_path=self.config.poi_text_prompt_path)
        except Exception as exc:
            return PoiSet(schema_version="v4.5", source="llm_text", status="failed", error_message=str(exc))

    def _run_video(self, video_artifact: BootstrapVideoArtifact | None) -> PoiSet:
        if not self.config.enable_vlm_poi_analysis:
            return PoiSet(schema_version="v4.5", source="vlm_video", status="disabled")
        if video_artifact is None:
            return PoiSet(schema_version="v4.5", source="vlm_video", status="failed", error_message="video artifact missing")
        try:
            return self.video_analyzer.analyze(video_artifact=video_artifact, prompt_path=self.config.poi_video_prompt_path)
        except Exception as exc:
            return PoiSet(schema_version="v4.5", source="vlm_video", status="failed", error_message=str(exc))
