from __future__ import annotations

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, HudRegionResult
from v4_5.contracts.errors import DeterministicHudAnalysisError
from v4_5.perception.board_builder.hudExtractor import extract_hud_candidates, extract_progress_candidates


class HudDeterministicAnalyzer:
    def __init__(self) -> None:
        self.last_extraction = None

    def analyze(
        self,
        *,
        capture_bundle: BootstrapCaptureBundle,
        avatar_detection_result: AvatarDetectionResult | None = None,
        poi_extraction_result=None,
    ) -> HudRegionResult:
        try:
            frames = tuple(record.pre_observation_ref for record in capture_bundle.step_records[:1] if record.pre_observation_ref is not None)
            frames = frames + tuple(record.post_observation_ref or record.raw_observation_ref for record in capture_bundle.step_records if (record.post_observation_ref or record.raw_observation_ref) is not None)
            if not self._frames_are_consistent(frames):
                return HudRegionResult(schema_version="v4.5", source="deterministic", status="ok", diagnostics={"hud_failure_reason": "inconsistent_frames", "progress_failure_reason": "inconsistent_frames"})
            hud_result = extract_hud_candidates(
                bootstrap_frames=frames,
                bootstrap_transition_records=tuple(capture_bundle.step_records),
                avatar_detection_result=avatar_detection_result,
                poi_extraction_result=poi_extraction_result,
            )
            progress_result = extract_progress_candidates(
                frames=frames,
                hud_tracks=hud_result.hud_tracks,
                avatar_detection_result=avatar_detection_result,
                poi_extraction_result=poi_extraction_result,
            )
            self.last_extraction = {
                "hud": hud_result,
                "progress": progress_result,
            }
            diagnostics = dict(hud_result.diagnostics)
            diagnostics.update(progress_result.diagnostics)
            diagnostics["hud_failure_reason"] = hud_result.failure_reason
            diagnostics["progress_failure_reason"] = progress_result.failure_reason
            return HudRegionResult(
                schema_version="v4.5",
                source="deterministic",
                status="ok",
                hud_regions=tuple(_serialize_bbox(track["hud_bbox"]) for track in hud_result.hud_tracks),
                life_regions=tuple(_serialize_bbox(track["hud_bbox"]) for track in hud_result.hud_tracks if track.get("life_like")),
                progress_regions=tuple(_serialize_bbox(candidate["container_bbox"]) for candidate in progress_result.selected_candidates),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            raise DeterministicHudAnalysisError(str(exc)) from exc

    def _frames_are_consistent(self, frames) -> bool:
        if not frames:
            return True
        first = frames[0]
        if not first:
            return False
        height = len(first)
        width = len(first[0]) if height else 0
        if height == 0 or width == 0:
            return False
        for frame in frames:
            if not frame or len(frame) != height:
                return False
            if any(len(row) != width for row in frame):
                return False
        return True


def _serialize_cells(cells) -> str:
    return "|".join(f"cell:{x},{y}" for x, y in cells)


def _serialize_bbox(bbox: tuple[int, int, int, int]) -> str:
    cells = []
    for y in range(bbox[1], bbox[3] + 1):
        for x in range(bbox[0], bbox[2] + 1):
            cells.append((x, y))
    return _serialize_cells(cells)
