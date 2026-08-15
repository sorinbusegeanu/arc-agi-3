from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    actor_throttle_seconds: float
    peer_interval_seconds: float
    candidate_budget: int
    reason: str


class ResourceController:
    """Adapt producer pressure and peer breadth from queue and RAM capacity signals."""

    def decide(
        self,
        *,
        stage_depths: tuple[int, ...],
        shard_depths: tuple[int, ...],
        stage_capacity: int,
        shard_capacity: int,
        memory_count: int,
        memory_capacity: int,
    ) -> ResourceDecision:
        stage_ratio = max(stage_depths, default=0) / max(1, int(stage_capacity))
        shard_ratio = max(shard_depths, default=0) / max(1, int(shard_capacity))
        memory_ratio = int(memory_count) / max(1, int(memory_capacity))
        pressure = max(stage_ratio, shard_ratio, memory_ratio)
        if pressure >= 0.90:
            return ResourceDecision(0.010, 1.0, 32, "critical memory/backlog pressure")
        if pressure >= 0.75:
            return ResourceDecision(0.003, 0.75, 64, "high memory/backlog pressure")
        if pressure >= 0.50:
            return ResourceDecision(0.001, 0.50, 128, "moderate pressure")
        return ResourceDecision(0.0, 0.25, 256, "normal")
