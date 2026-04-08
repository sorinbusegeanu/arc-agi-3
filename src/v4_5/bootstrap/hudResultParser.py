from __future__ import annotations

from v4_5.contracts.bootstrapMediaTypes import HudRegionResult


class HudResultParser:
    def parse(self, *, source: str, raw_text: str) -> HudRegionResult:
        values = {}
        for line in str(raw_text).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = tuple(item for item in value.strip().split("|") if item)
        required = {"hud_regions", "life_regions", "progress_regions"}
        if not required.issubset(values):
            raise ValueError("malformed HUD analysis output")
        return HudRegionResult(
            schema_version="v4.5",
            source=source,
            status="ok",
            hud_regions=tuple(values["hud_regions"]),
            life_regions=tuple(values["life_regions"]),
            progress_regions=tuple(values["progress_regions"]),
            raw_response_text=str(raw_text),
        )
