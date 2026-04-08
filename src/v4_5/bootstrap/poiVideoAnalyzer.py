from __future__ import annotations

from pathlib import Path

from v4_5.bootstrap.poiResultParser import PoiResultParser
from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.bootstrapMediaTypes import BootstrapVideoArtifact
from v4_5.contracts.poiTypes import PoiSet
from v4_5.toon.toonClient import ToonClient


class PoiVideoAnalyzer:
    def __init__(self, client: ToonClient, config: BootstrapConfig, parser: PoiResultParser | None = None) -> None:
        self.client = client
        self.config = config
        self.parser = parser or PoiResultParser()

    def analyze(self, *, video_artifact: BootstrapVideoArtifact, prompt_path: str) -> PoiSet:
        prompt = Path(prompt_path).read_text(encoding="utf-8")
        response = self.client.call_video(
            prompt=prompt,
            video_path=video_artifact.video_path,
            endpoint_name=self.config.toon_poi_video_endpoint_name,
        )
        return self.parser.parse(source="vlm_video", raw_text=response)
