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
    probe_steps_max: int = 12
    probe_every_k: int = 0
    conflict_open_delta: float = 0.05
    conflict_open_M: int = 4

    snapshot_every_steps: int = 0
    history_window_N: int = 50
    debug: bool = False

    memory_enabled: bool = True
    memory_persist_across_runs: bool = True
    memory_snapshot_every_steps: int = 0
    fp_save_mode: str = "buffer"
