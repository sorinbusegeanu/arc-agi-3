from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    persist_across_runs: bool = True
    snapshot_every_steps: int = 0
    state_action_table_max: int = 50000
    coord_table_max_per_action: int = 5000
    K_short: int = 5
    K_long: int = 25
    noop_rate_block_threshold: float = 0.9
    memory_dir: str = "memory_store"
