from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwarmOrchestratorConfig:
    max_steps_total: int = 40
    probe_steps: int = 10
    exploit_steps: int = 30

    recompute_interval_steps: int = 3
    goal_window_steps: int = 10

    conflict_margin: float = 0.10
    resolution_margin: float = 0.20

    snapshot_every_steps: int = 5
    history_window_N: int = 50
    debug: bool = False
