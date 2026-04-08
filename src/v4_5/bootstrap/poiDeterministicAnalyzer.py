from __future__ import annotations

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle
from v4_5.contracts.poiTypes import PoiSet
from v4_5.perception.board_builder.poiExtractor import extract_poi_candidates


class PoiDeterministicAnalyzer:
    def __init__(self) -> None:
        self.last_extraction = None

    def analyze(
        self,
        *,
        capture_bundle: BootstrapCaptureBundle,
        avatar_bbox: tuple[int, int, int, int] | None,
        avatar_detection_result: AvatarDetectionResult | None = None,
        hud_regions: tuple[str, ...],
        life_regions: tuple[str, ...],
        progress_regions: tuple[str, ...],
    ) -> PoiSet:
        avatar_result = avatar_detection_result
        if avatar_result is None and avatar_bbox is not None:
            avatar_result = AvatarDetectionResult(avatar_bbox=avatar_bbox, avatar_center=((avatar_bbox[0] + avatar_bbox[2]) / 2.0, (avatar_bbox[1] + avatar_bbox[3]) / 2.0))
        self.last_extraction = extract_poi_candidates(
            bootstrap_transition_records=tuple(capture_bundle.step_records),
            avatar_detection_result=avatar_result,
            hud_exclusion_regions=tuple(hud_regions),
            life_exclusion_regions=tuple(life_regions),
            progress_exclusion_regions=tuple(progress_regions),
        )
        return PoiSet(
            schema_version="v4.5",
            source="deterministic",
            status="ok",
            pois=tuple(self.last_extraction.selected_pois),
            ranked_candidates=tuple(self.last_extraction.ranked_candidates),
            diagnostics=dict(self.last_extraction.diagnostics),
        )
