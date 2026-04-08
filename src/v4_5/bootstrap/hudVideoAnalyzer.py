from __future__ import annotations

from pathlib import Path

from v4_5.bootstrap.hudResultParser import HudResultParser
from v4_5.contracts.bootstrapMediaTypes import BootstrapVideoArtifact, HudRegionResult
from v4_5.toon.toonClient import ToonClient


class HudVideoAnalyzer:
    def __init__(self, client: ToonClient, parser: HudResultParser | None = None) -> None:
        self.client = client
        self.parser = parser or HudResultParser()

    def analyze(self, *, video_artifact: BootstrapVideoArtifact, prompt_path: str) -> HudRegionResult:
        prompt = Path(prompt_path).read_text(encoding="utf-8")
        response = self.client.call_video(prompt=prompt, video_path=video_artifact.video_path)
        return self.parser.parse(source="vlm_video", raw_text=response)
