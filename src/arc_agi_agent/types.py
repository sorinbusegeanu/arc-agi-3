from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class Component:
    id: str
    color: int
    area: int
    bbox: BBox
    centroid: Tuple[float, float]
    perimeter: Optional[int] = None
    holes: Optional[int] = None
    grid_name: Optional[str] = None


@dataclass
class ObjectDelta:
    object_id: str
    color: int
    prev_bbox: Optional[BBox]
    curr_bbox: Optional[BBox]
    dy: float
    dx: float
    event: str


@dataclass
class EventSignature:
    kind: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSummary:
    name: str
    height: int
    width: int
    palette_sorted: List[int]
    bg_candidates: List[Tuple[int, float]]
    color_histogram: Dict[int, int]
    connected_components: List[Component]
    symmetry_candidates: Dict[str, float]
    static_regions: Optional[Dict[str, Any]] = None
    active_regions: Optional[Dict[str, Any]] = None
    periodicity: Optional[List[Dict[str, Any]]] = None


@dataclass
class StateSummary:
    step_idx: int
    grid_summaries: List[GridSummary]
    object_catalog: List[Component]
    invariants: List[Dict[str, Any]]


@dataclass
class DiffSummary:
    changed_cells_count: int
    changed_bbox: Optional[BBox]
    changed_colors: Dict[str, int]
    per_object_deltas: List[ObjectDelta]
    event_signatures: List[EventSignature]


@dataclass
class VizArtifacts:
    ascii_grid: Dict[str, str]
    overlay_grids: Dict[str, Dict[str, str]]
    save_paths: List[str] = field(default_factory=list)


@dataclass
class DebugInfo:
    schema_warnings: List[str]
    timings_ms: Dict[str, float]
    grid_hash: str


@dataclass
class FPReport:
    state_summary: StateSummary
    diff_summary: Optional[DiffSummary]
    viz_artifacts: VizArtifacts
    debug: DebugInfo
