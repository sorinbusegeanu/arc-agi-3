from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimpleExplorerConfig:
    max_steps: int = 80
    attempts_per_action_per_state: int = 2
    max_unique_states: int = 200
    short_cycle_max_len: int = 4
    revisit_window_N: int = 20
    revisit_threshold_R: int = 6
    noop_edge_deprioritize: bool = True
    frontier_pick: Tuple[str, str] = ("most_untried", "least_recent")
    bfs_route_to_frontier: bool = True
    bfs_max_depth: int = 20
    save_trace: bool = True
    save_viz: bool = False
    save_representatives: bool = True
