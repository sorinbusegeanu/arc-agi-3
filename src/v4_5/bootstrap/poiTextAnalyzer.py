from __future__ import annotations

from pathlib import Path

from v4_5.bootstrap.poiResultParser import PoiResultParser
from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle
from v4_5.contracts.poiTypes import PoiSet
from v4_5.toon.toonClient import ToonClient


class PoiTextAnalyzer:
    def __init__(self, client: ToonClient, config: BootstrapConfig, parser: PoiResultParser | None = None) -> None:
        self.client = client
        self.config = config
        self.parser = parser or PoiResultParser()

    def analyze(self, *, capture_bundle: BootstrapCaptureBundle, prompt_path: str) -> PoiSet:
        prompt = Path(prompt_path).read_text(encoding="utf-8")
        context_lines = []
        for record in capture_bundle.step_records:
            summary = getattr(record, "textual_summary", None)
            screen_text = getattr(record, "screen_text", None)
            context_lines.append(f"{record.sequence_name}:{record.step_index}:{record.action}:{screen_text or ''}:{summary or ''}")
        response = self.client.call_text(
            prompt=prompt,
            bootstrap_context="\n".join(context_lines),
            endpoint_name=self.config.toon_poi_text_endpoint_name,
        )
        return self.parser.parse(source="llm_text", raw_text=response)
