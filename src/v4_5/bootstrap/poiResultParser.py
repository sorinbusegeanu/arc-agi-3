from __future__ import annotations

import ast

from v4_5.contracts.poiTypes import PoiRecord, PoiSet


class PoiResultParser:
    def parse(self, *, source: str, raw_text: str) -> PoiSet:
        try:
            payload = ast.literal_eval(str(raw_text).strip())
        except Exception as exc:
            raise ValueError("malformed POI analysis output") from exc
        if not isinstance(payload, dict) or "pois" not in payload or not isinstance(payload["pois"], list):
            raise ValueError("malformed POI analysis output")
        pois = []
        for item in payload["pois"]:
            if not isinstance(item, dict):
                raise ValueError("malformed POI record")
            bbox = tuple(item.get("bbox", ()))
            center = tuple(item.get("center", ()))
            colors = tuple(item.get("colors", ()))
            if len(bbox) != 4 or not all(isinstance(value, int) for value in bbox):
                raise ValueError("invalid POI bbox")
            if len(center) != 2 or not all(isinstance(value, (int, float)) for value in center):
                raise ValueError("invalid POI center")
            if not all(isinstance(value, int) for value in colors):
                raise ValueError("invalid POI colors")
            pois.append(PoiRecord(bbox=bbox, center=(float(center[0]), float(center[1])), colors=colors))
        return PoiSet(
            schema_version="v4.5",
            source=source,
            status="ok",
            pois=tuple(pois),
            raw_response_text=str(raw_text),
        )
