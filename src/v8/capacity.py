from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from v8.snapshot import latest_complete_snapshot

DEFAULT_NODE_CAPACITY = 250_000
DEFAULT_EDGE_CAPACITY = 500_000
DEFAULT_ACTION_CAPACITY = 65_536

# Observed v8 memory growth is dominated by retained M0 episodes. M7 strategy
# identity is now reusable rather than trajectory-instance scoped. These factors
# intentionally leave substantial headroom for M1-M6 novelty and shard skew.
NODE_GROWTH_PER_EVENT = 4.0
EDGE_GROWTH_PER_EVENT = 4.0
ACTION_GROWTH_PER_EVENT = 0.75
FIXED_HEADROOM = 8_192


@dataclass(frozen=True, slots=True)
class SnapshotUsage:
    node_count: int = 0
    edge_count: int = 0
    node_capacity: int = 0
    edge_capacity: int = 0
    action_capacity: int = 0


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    node_capacity_per_shard: int
    edge_capacity_per_shard: int
    action_capacity_per_shard: int


def _arena_count(path: Path) -> int:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"invalid arena snapshot header: {path}")
    return int.from_bytes(raw, "little", signed=False)


def snapshot_usage(root: str | Path) -> SnapshotUsage:
    snapshot = latest_complete_snapshot(root)
    if snapshot is None:
        return SnapshotUsage()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    node_count = edge_count = 0
    node_capacity = edge_capacity = action_capacity = 0
    for shard in manifest.get("shards", []):
        node_spec = shard["nodes"]
        edge_spec = shard["edges"]
        action_spec = shard["actions"]
        node_count = max(node_count, _arena_count(snapshot / node_spec["file"]))
        edge_count = max(edge_count, _arena_count(snapshot / edge_spec["file"]))
        node_capacity = max(node_capacity, int(node_spec["capacity"]))
        edge_capacity = max(edge_capacity, int(edge_spec["capacity"]))
        action_capacity = max(action_capacity, int(action_spec["capacity"]))
    return SnapshotUsage(
        node_count=node_count,
        edge_count=edge_count,
        node_capacity=node_capacity,
        edge_capacity=edge_capacity,
        action_capacity=action_capacity,
    )


def plan_capacities(
    *,
    total_steps: int,
    shards: int,
    root: str | Path | None = None,
    restore: bool = True,
    node_override: int | None = None,
    edge_override: int | None = None,
    action_override: int | None = None,
) -> CapacityPlan:
    if total_steps < 0:
        raise ValueError("total_steps cannot be negative")
    if shards <= 0:
        raise ValueError("shards must be positive")

    prior = snapshot_usage(root) if restore and root is not None else SnapshotUsage()
    per_shard_steps = math.ceil(int(total_steps) / int(shards))

    if node_override is None:
        node_growth = math.ceil(per_shard_steps * NODE_GROWTH_PER_EVENT)
        node_capacity = max(
            DEFAULT_NODE_CAPACITY,
            prior.node_capacity,
            prior.node_count + node_growth + FIXED_HEADROOM,
        )
    else:
        node_capacity = max(int(node_override), prior.node_capacity)

    if edge_override is None:
        edge_growth = math.ceil(per_shard_steps * EDGE_GROWTH_PER_EVENT)
        edge_capacity = max(
            DEFAULT_EDGE_CAPACITY,
            prior.edge_capacity,
            prior.edge_count + edge_growth + FIXED_HEADROOM,
        )
    else:
        edge_capacity = max(int(edge_override), prior.edge_capacity)

    if action_override is None:
        action_growth = math.ceil(per_shard_steps * ACTION_GROWTH_PER_EVENT)
        # Action arenas are open-addressed tables, so preserve prior table size and
        # add headroom on continuation runs rather than risking a full probe table.
        continuation_floor = prior.action_capacity + action_growth if prior.action_capacity else 0
        action_capacity = max(
            DEFAULT_ACTION_CAPACITY,
            continuation_floor,
            action_growth + FIXED_HEADROOM,
        )
    else:
        action_capacity = max(int(action_override), prior.action_capacity)

    for name, value in (
        ("node_capacity_per_shard", node_capacity),
        ("edge_capacity_per_shard", edge_capacity),
        ("action_capacity_per_shard", action_capacity),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    return CapacityPlan(node_capacity, edge_capacity, action_capacity)
