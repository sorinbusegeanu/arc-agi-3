from __future__ import annotations

from pathlib import Path

from v4_5.bootstrap.hudResultParser import HudResultParser
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, HudRegionResult
from v4_5.toon.toonClient import ToonClient


class HudTextAnalyzer:
    def __init__(self, client: ToonClient, parser: HudResultParser | None = None) -> None:
        self.client = client
        self.parser = parser or HudResultParser()

    def analyze(self, *, capture_bundle: BootstrapCaptureBundle, prompt_path: str) -> HudRegionResult:
        prompt = Path(prompt_path).read_text(encoding="utf-8")
        context = "\n".join(f"{record.sequence_name}:{record.step_index}:{record.action}" for record in capture_bundle.step_records)
        response = self.client.call_text(prompt=prompt, bootstrap_context=context)
        return self.parser.parse(source="llm_text", raw_text=response)
