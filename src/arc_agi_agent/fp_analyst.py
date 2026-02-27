from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .components import extract_components, track_components
from .config import FPAnalystConfig
from .grid_utils import (
    bg_candidates,
    border_color_invariant,
    changed_bbox,
    changed_colors,
    color_histogram,
    count_active_regions,
    diff_mask,
    grid_hash,
    palette,
    periodicity,
    symmetry_scores,
)
from .logger import get_logger
from .normalize import NormalizedObservation, normalize_observation
from .types import (
    Component,
    DiffSummary,
    EventSignature,
    FPReport,
    GridSummary,
    ObjectDelta,
    StateSummary,
    VizArtifacts,
    DebugInfo,
)
from .viz import (
    ascii_grid as render_ascii_grid,
    ascii_overlay,
    bbox_overlay,
    component_id_overlay,
    diff_mask_overlay,
    motion_overlay,
    save_grid_image,
)

logger = get_logger(__name__)


class FPAnalyst:
    def __init__(self, config: Optional[FPAnalystConfig] = None) -> None:
        self.config = config or FPAnalystConfig()
        self._cache: Dict[Tuple[str, str], Tuple[StateSummary, Dict[str, str], Dict[str, Dict[str, str]]]] = {}

    def analyze(
        self,
        observation: Any,
        prev_observation: Any = None,
        action_taken: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> FPReport:
        timings: Dict[str, float] = {}
        schema_warnings: List[str] = []
        t0 = time.time()
        norm = normalize_observation(observation, schema_warnings=schema_warnings)
        prev_norm = (
            normalize_observation(prev_observation, schema_warnings=schema_warnings)
            if prev_observation is not None
            else None
        )
        timings["normalize_ms"] = (time.time() - t0) * 1000

        grids = norm.grids
        if not grids:
            report = FPReport(
                state_summary=StateSummary(
                    step_idx=norm.step_idx,
                    grid_summaries=[],
                    object_catalog=[],
                    invariants=[],
                ),
                diff_summary=None,
                viz_artifacts=VizArtifacts(ascii_grid={}, overlay_grids={}),
                debug=DebugInfo(
                    schema_warnings=schema_warnings,
                    timings_ms=timings,
                    grid_hash="",
                ),
            )
            return report

        t1 = time.time()
        grid_hash_value = grid_hash(grids)
        game_id = str(norm.meta.get("game_id", "")) if norm.meta else ""
        cache_key = (game_id, grid_hash_value)
        cached = self._cache.get(cache_key)
        if cached is not None:
            state_summary, ascii_cache, overlay_cache = cached
        else:
            state_summary, ascii_cache, overlay_cache = self._build_state_summary(norm, prev_norm)
            self._cache[cache_key] = (state_summary, ascii_cache, overlay_cache)
        timings["state_summary_ms"] = (time.time() - t1) * 1000

        t2 = time.time()
        diff_summary = self._build_diff_summary(norm, prev_norm)
        timings["diff_summary_ms"] = (time.time() - t2) * 1000

        t3 = time.time()
        viz = self._build_viz(norm, prev_norm, state_summary, ascii_cache, overlay_cache)
        timings["viz_ms"] = (time.time() - t3) * 1000

        report = FPReport(
            state_summary=state_summary,
            diff_summary=diff_summary,
            viz_artifacts=viz,
            debug=DebugInfo(
                schema_warnings=schema_warnings,
                timings_ms=timings,
                grid_hash=grid_hash_value,
            ),
        )
        return report

    def render(self, report: FPReport, mode: str = "ascii") -> str:
        if mode == "ascii":
            parts = []
            for name, grid_str in report.viz_artifacts.ascii_grid.items():
                parts.append(f"[{name}]\n{grid_str}")
            return "\n\n".join(parts)
        if mode == "json":
            return str(asdict(report))
        raise ValueError(f"Unknown render mode: {mode}")

    def _build_state_summary(
        self,
        norm: NormalizedObservation,
        prev_norm: Optional[NormalizedObservation],
    ) -> Tuple[StateSummary, Dict[str, str], Dict[str, Dict[str, str]]]:
        grid_summaries: List[GridSummary] = []
        object_catalog: List[Component] = []
        invariants: List[Dict[str, Any]] = []
        ascii_cache: Dict[str, str] = {}
        overlay_cache: Dict[str, Dict[str, str]] = {}

        for idx, grid in enumerate(norm.grids):
            name = norm.grid_names[idx] if idx < len(norm.grid_names) else f"frame_{idx}"
            pal = palette(grid)
            bg_list = bg_candidates(grid, self.config.bg_detection_weights)
            bg_color = bg_list[0][0] if bg_list else None
            hist = color_histogram(grid)

            colors = [c for c in pal if c != bg_color]
            if bg_color is not None:
                border_color = border_color_invariant(grid)
                if border_color == bg_color:
                    colors.append(bg_color)
                    invariants.append(
                        {
                            "kind": "border_color",
                            "color": bg_color,
                            "confidence": 0.9,
                            "grid": name,
                        }
                    )

            comps = extract_components(
                grid=grid,
                colors=colors,
                connectivity=self.config.connectivity,
                min_area=self.config.min_area,
                max_objects=self.config.max_objects,
                grid_name=name,
            )
            object_catalog.extend(comps)

            symmetry = symmetry_scores(grid) if self.config.enable_symmetry else {}
            period = (
                periodicity(grid, self.config.max_period)
                if self.config.enable_periodicity
                else None
            )

            static_regions = None
            active_regions = None
            if prev_norm is not None and idx < len(prev_norm.grids):
                diff = diff_mask(grid, prev_norm.grids[idx])
                active_regions = count_active_regions(diff)
                static_regions = count_active_regions(~diff)

            grid_summaries.append(
                GridSummary(
                    name=name,
                    height=grid.shape[0],
                    width=grid.shape[1],
                    palette_sorted=pal,
                    bg_candidates=bg_list,
                    color_histogram=hist,
                    connected_components=comps,
                    symmetry_candidates=symmetry,
                    static_regions=static_regions,
                    active_regions=active_regions,
                    periodicity=period,
                )
            )

            ascii_cache[name] = render_ascii_grid(grid)
            overlay_cache[name] = {}

        state_summary = StateSummary(
            step_idx=norm.step_idx,
            grid_summaries=grid_summaries,
            object_catalog=object_catalog,
            invariants=invariants,
        )
        return state_summary, ascii_cache, overlay_cache

    def _build_diff_summary(
        self,
        norm: NormalizedObservation,
        prev_norm: Optional[NormalizedObservation],
    ) -> Optional[DiffSummary]:
        if prev_norm is None:
            return None
        total_changed = 0
        bbox_union: Optional[Tuple[int, int, int, int]] = None
        color_counts: Dict[str, int] = {}
        per_object_deltas: List[ObjectDelta] = []
        event_signatures: List[EventSignature] = []

        for idx, grid in enumerate(norm.grids):
            if idx >= len(prev_norm.grids):
                continue
            prev_grid = prev_norm.grids[idx]
            diff = diff_mask(grid, prev_grid)
            changed = int(diff.sum())
            total_changed += changed
            bbox = changed_bbox(diff)
            if bbox is not None:
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    y0, x0, y1, x1 = bbox_union
                    by0, bx0, by1, bx1 = bbox
                    bbox_union = (
                        min(y0, by0),
                        min(x0, bx0),
                        max(y1, by1),
                        max(x1, bx1),
                    )
            for k, v in changed_colors(grid, prev_grid).items():
                color_counts[k] = color_counts.get(k, 0) + v

            if self.config.enable_tracking:
                curr_comps = extract_components(
                    grid=grid,
                    colors=palette(grid),
                    connectivity=self.config.connectivity,
                    min_area=self.config.min_area,
                    max_objects=self.config.max_objects,
                )
                prev_comps = extract_components(
                    grid=prev_grid,
                    colors=palette(prev_grid),
                    connectivity=self.config.connectivity,
                    min_area=self.config.min_area,
                    max_objects=self.config.max_objects,
                )
                per_object_deltas.extend(
                    track_components(
                        prev=prev_comps,
                        curr=curr_comps,
                        iou_threshold=self.config.iou_threshold,
                        iou_soft_threshold=self.config.iou_soft_threshold,
                        centroid_distance_threshold=self.config.centroid_distance_threshold,
                    )
                )

        event_signatures.extend(_infer_events(color_counts, per_object_deltas, total_changed))

        return DiffSummary(
            changed_cells_count=total_changed,
            changed_bbox=bbox_union,
            changed_colors=color_counts,
            per_object_deltas=per_object_deltas,
            event_signatures=event_signatures,
        )

    def _build_viz(
        self,
        norm: NormalizedObservation,
        prev_norm: Optional[NormalizedObservation],
        state_summary: StateSummary,
        ascii_cache: Dict[str, str],
        overlay_cache: Dict[str, Dict[str, str]],
    ) -> VizArtifacts:
        ascii_out: Dict[str, str] = dict(ascii_cache)
        overlays_out: Dict[str, Dict[str, str]] = {k: dict(v) for k, v in overlay_cache.items()}
        save_paths: List[str] = []

        for grid_summary, grid in zip(state_summary.grid_summaries, norm.grids):
            name = grid_summary.name
            comps = grid_summary.connected_components
            overlays_out.setdefault(name, {})

            if "bbox_overlay" in self.config.overlays:
                overlay = bbox_overlay(grid, comps)
                overlays_out[name]["bbox_overlay"] = ascii_overlay(grid, overlay)
            if "component_id_overlay" in self.config.overlays:
                overlay = component_id_overlay(grid, comps)
                overlays_out[name]["component_id_overlay"] = ascii_overlay(grid, overlay)
            if prev_norm is not None and name in norm.grid_names:
                prev_idx = norm.grid_names.index(name)
                prev_grid = prev_norm.grids[prev_idx] if prev_idx < len(prev_norm.grids) else None
            else:
                prev_grid = None
            if prev_grid is not None and "diff_mask" in self.config.overlays:
                overlay = diff_mask_overlay(grid, diff_mask(grid, prev_grid))
                overlays_out[name]["diff_mask"] = ascii_overlay(grid, overlay)

            if prev_norm is not None and "object_motion_overlay" in self.config.overlays:
                motions = []
                if prev_grid is not None:
                    prev_comps = extract_components(
                        grid=prev_grid,
                        colors=palette(prev_grid),
                        connectivity=self.config.connectivity,
                        min_area=self.config.min_area,
                        max_objects=self.config.max_objects,
                    )
                    deltas = track_components(
                        prev=prev_comps,
                        curr=comps,
                        iou_threshold=self.config.iou_threshold,
                        iou_soft_threshold=self.config.iou_soft_threshold,
                        centroid_distance_threshold=self.config.centroid_distance_threshold,
                    )
                    for delta in deltas:
                        if delta.prev_bbox and delta.curr_bbox:
                            py = (delta.prev_bbox[0] + delta.prev_bbox[2]) / 2.0
                            px = (delta.prev_bbox[1] + delta.prev_bbox[3]) / 2.0
                            cy = (delta.curr_bbox[0] + delta.curr_bbox[2]) / 2.0
                            cx = (delta.curr_bbox[1] + delta.curr_bbox[3]) / 2.0
                            motions.append((py, px, cy, cx))
                overlay = motion_overlay(grid, motions)
                overlays_out[name]["object_motion_overlay"] = ascii_overlay(grid, overlay)

            if self.config.save_images:
                output_dir = f"{self.config.output_dir}/viz"
                try:
                    import os

                    os.makedirs(output_dir, exist_ok=True)
                    path = f"{output_dir}/{name}_step{state_summary.step_idx}.png"
                    saved = save_grid_image(path, grid)
                    if saved:
                        save_paths.append(saved)
                except Exception as e:
                    logger.warning("Failed to save image: %s", e)

        return VizArtifacts(ascii_grid=ascii_out, overlay_grids=overlays_out, save_paths=save_paths)


def _infer_events(
    color_counts: Dict[str, int],
    deltas: List[ObjectDelta],
    total_changed: int,
) -> List[EventSignature]:
    events: List[EventSignature] = []
    if total_changed == 0:
        return events

    moved = [d for d in deltas if d.event == "moved"]
    appeared = [d for d in deltas if d.event == "appeared"]
    disappeared = [d for d in deltas if d.event == "disappeared"]

    if moved:
        events.append(EventSignature(kind="translation", confidence=0.5, details={"moved": len(moved)}))

    if appeared:
        events.append(EventSignature(kind="spawn", confidence=0.4, details={"appeared": len(appeared)}))
    if disappeared:
        events.append(EventSignature(kind="despawn", confidence=0.4, details={"disappeared": len(disappeared)}))

    if color_counts:
        # Detect paint-like changes: dominant single transition
        dominant = max(color_counts.values())
        if dominant / max(total_changed, 1) > 0.7:
            events.append(EventSignature(kind="paint", confidence=0.4, details={"dominant": dominant}))

        # Detect toggle-like: two-way swaps between two colors
        if len(color_counts) == 2:
            keys = list(color_counts.keys())
            a_to_b = keys[0]
            b_to_a = keys[1]
            if a_to_b.split("->")[0] == b_to_a.split("->")[1] and a_to_b.split("->")[1] == b_to_a.split("->")[0]:
                events.append(EventSignature(kind="toggle", confidence=0.4, details={"pairs": keys}))

    if moved:
        vertical_moves = [d for d in moved if abs(d.dy) > abs(d.dx)]
        if vertical_moves and len(vertical_moves) / len(moved) > 0.6:
            events.append(EventSignature(kind="gravity", confidence=0.3, details={"vertical_moves": len(vertical_moves)}))

    return events
