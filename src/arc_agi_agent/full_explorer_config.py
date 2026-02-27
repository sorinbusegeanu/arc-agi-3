from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FullExplorerConfig:
    max_steps: int = 120
    max_unique_states: int = 300
    max_coords_per_state: int = 64
    attempts_per_coord_candidate: int = 1
    short_cycle_max_len: int = 4
    revisit_window_N: int = 30
    revisit_threshold_R: int = 8
    ban_noop_after: int = 2
    global_ban_noop_after: int = 4
    bfs_route_to_frontier: bool = True
    bfs_max_depth: int = 25
    save_trace: bool = True
    save_viz: bool = False
    selector_priority_order: List[str] = None  # type: ignore[assignment]
    topK_colors: int = 3
    samples_per_color: int = 4
    max_perimeter_points_per_region: int = 12
    topK_hotspots: int = 8
    topK_negative_zones: int = 8
    far_distance_threshold: int = 6

    def __post_init__(self) -> None:
        if self.selector_priority_order is None:
            object.__setattr__(
                self,
                "selector_priority_order",
                [
                    "changed_bbox_focus",
                    "adjacent_boundary_cells",
                    "region_frontier_cells",
                    "object_centroids",
                    "object_bbox_corners",
                    "color_hotspots",
                    "grid_edges_midpoints",
                    "grid_corners",
                ],
            )
