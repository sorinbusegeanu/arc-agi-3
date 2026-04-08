from __future__ import annotations

from pathlib import Path

from v4_5.contracts.bootstrapMediaTypes import BootstrapPngArtifact, BootstrapVideoArtifact
from v4_5.contracts.errors import BootstrapVideoExportError
from vlm_v2.video_builder import build_episode_video


class VideoExporter:
    def export(self, *, png_artifact: BootstrapPngArtifact, output_dir: str, fps: int = 2) -> BootstrapVideoArtifact:
        try:
            video_path = build_episode_video(
                str((Path(output_dir) / png_artifact.sequence_name / "png").resolve()),
                fps=int(fps),
                output_name=f"{png_artifact.sequence_name}.mp4",
                frame_paths=png_artifact.png_paths,
            )
        except Exception as exc:
            raise BootstrapVideoExportError(str(exc)) from exc
        return BootstrapVideoArtifact(schema_version="v4.5", sequence_name=png_artifact.sequence_name, video_path=str(video_path), fps=int(fps))
