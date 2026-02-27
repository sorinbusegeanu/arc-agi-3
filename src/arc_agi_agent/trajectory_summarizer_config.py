from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectorySummarizerConfig:
    topK_coords: int = 8
    max_static_cells: int = 512
    keyframes_max: int = 6
    short_cycle_max_len: int = 4
    revisit_window_N: int = 25
    revisit_threshold_R: int = 6
    export_markdown: bool = False
