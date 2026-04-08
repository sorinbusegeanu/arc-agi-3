from __future__ import annotations

from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.bootstrapMediaTypes import BootstrapProbePlan


class BootstrapSequenceBuilder:
    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config

    def build(self, *, game_id: str, level_id: str) -> BootstrapProbePlan:
        return BootstrapProbePlan(
            schema_version="v4.5",
            plan_id=f"{game_id}:{level_id}:bootstrap",
            game_id=game_id,
            level_id=level_id,
            primary_sequence=self.config.primary_sequence,
            fallback_sequences=self.config.fallback_sequences,
            stop_after_unique_avatar_found=self.config.stop_after_unique_avatar_found,
            capture_raw_observations=self.config.capture_raw_observations,
            export_pngs=self.config.export_pngs,
            export_video=self.config.export_video,
            png_scale_factor=self.config.png_scale_factor,
            video_fps=self.config.video_fps,
            status="planned",
        )
