from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoiRecord:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    colors: tuple[int, ...] = ()
    support_step_indices: tuple[int, ...] = ()
    value_candidates: tuple[int, ...] = ()
    stability_score: float = 0.0
    reachability_score: float = 0.0
    poi_score: float = 0.0
    rejected_as_avatar_overlap: bool = False
    failure_reason: str | None = None
    description: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class PoiSet:
    schema_version: str
    source: str
    status: str
    pois: tuple[PoiRecord, ...] = ()
    ranked_candidates: tuple[PoiRecord, ...] = ()
    diagnostics: dict[str, object] | None = None
    raw_response_text: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PoiAnalysisBundle:
    schema_version: str
    deterministic_pois: PoiSet
    llm_text_pois: PoiSet
    vlm_video_pois: PoiSet
    selected_pois: PoiSet
    selected_source: str
