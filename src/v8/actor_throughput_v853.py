from __future__ import annotations

"""v8.53 actor-throughput recovery after memory-efficiency capacity changes.

Arena occupancy is a storage/compaction signal, not evidence that actors must sleep.
Actor throttling remains driven by actual producer backlog and real RAM pressure.
Restored action tables are also allowed to shrink to occupied entries plus declared
run growth because v8.51 restore already rehashes action records across capacities.
"""

import math


_INSTALLED = False
_BASE_RESOURCE_DECIDE = None
_BASE_PLAN_CAPACITIES = None
_BASE_RUNTIME_METRICS = None
_BASE_MEMORY_SNAPSHOT = None


def _decision_payload(decision) -> dict[str, object]:
    return {
        "actor_throttle_seconds": float(decision.actor_throttle_seconds),
        "peer_interval_seconds": float(decision.peer_interval_seconds),
        "candidate_budget": int(decision.candidate_budget),
        "reason": str(decision.reason),
        "arena_occupancy_throttles_actors": False,
    }


def _resource_decide_v853(self, **kwargs):
    """Throttle actors for queue/real-RAM pressure, never compact arena occupancy."""
    inputs = dict(kwargs)
    # v8.51 intentionally sizes node/edge arenas close to retained occupancy. The
    # original scheduler treated that efficient packing as runtime pressure and
    # inserted a sleep before every environment action. Compaction already handles
    # arena fullness separately in ContinuousMemoryRuntime._resource_cadence.
    inputs["memory_count"] = 0
    inputs["memory_capacity"] = max(1, int(inputs.get("memory_capacity", 1)))
    decision = _BASE_RESOURCE_DECIDE(self, **inputs)
    self._v853_last_decision = decision
    return decision


def _plan_capacities_v853(
    *,
    total_steps: int,
    shards: int,
    root=None,
    restore: bool = True,
    node_override: int | None = None,
    edge_override: int | None = None,
    action_override: int | None = None,
):
    """Size restored ActionArena from occupancy, not historical table capacity."""
    from v8 import capacity as capacity_module
    from v8 import memory_efficiency_v851_integrity as integrity
    from v8 import memory_efficiency_v852_review_fix as v852

    base = _BASE_PLAN_CAPACITIES(
        total_steps=total_steps,
        shards=shards,
        root=root,
        restore=restore,
        node_override=node_override,
        edge_override=edge_override,
        action_override=action_override,
    )
    if not restore or root is None:
        return base

    per_shard_steps = math.ceil(max(0, int(total_steps)) / max(1, int(shards)))
    action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)
    occupied = int(v852._occupied_action_count(root))
    required = math.ceil(
        (occupied + int(action_growth) + int(capacity_module.FIXED_HEADROOM)) / 0.70
    )
    action_capacity = max(
        int(integrity._MIN_ACTION_CAPACITY),
        int(required),
        0 if action_override is None else int(action_override),
    )
    return capacity_module.CapacityPlan(
        int(base.node_capacity_per_shard),
        int(base.edge_capacity_per_shard),
        int(action_capacity),
    )


def _runtime_metrics_v853(self) -> dict[str, object]:
    payload = dict(_BASE_RUNTIME_METRICS(self))
    decision = getattr(getattr(self, "resource_controller", None), "_v853_last_decision", None)
    if decision is not None:
        payload["resource_decision"] = _decision_payload(decision)
    return payload


def _memory_snapshot_v853(runtime) -> dict[str, object]:
    payload = dict(_BASE_MEMORY_SNAPSHOT(runtime))
    decision = getattr(
        getattr(runtime, "resource_controller", None),
        "_v853_last_decision",
        None,
    )
    if decision is not None:
        payload["resource_decision"] = _decision_payload(decision)
    return payload


def install_actor_throughput_v853() -> None:
    global _INSTALLED, _BASE_RESOURCE_DECIDE, _BASE_PLAN_CAPACITIES
    global _BASE_RUNTIME_METRICS, _BASE_MEMORY_SNAPSHOT
    if _INSTALLED:
        return

    from v8 import capacity as capacity_module
    from v8 import memory_efficiency_v851 as memory
    from v8 import memory_storage_v851 as storage
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.scheduler import ResourceController

    # Keep v8.51 real-RAM pressure logic underneath this wrapper while removing
    # only the false occupancy-derived actor sleep.
    _BASE_RESOURCE_DECIDE = ResourceController.decide
    ResourceController.decide = _resource_decide_v853

    # v8.51 restore can safely rehash ActionArena into a different capacity.
    _BASE_PLAN_CAPACITIES = capacity_module.plan_capacities
    capacity_module.plan_capacities = _plan_capacities_v853
    storage._plan_capacities_v851 = _plan_capacities_v853

    # Surface the active decision in both runtime metrics and memory_efficiency.log.
    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v853
    _BASE_MEMORY_SNAPSHOT = memory.memory_efficiency_snapshot_v851
    memory.memory_efficiency_snapshot_v851 = _memory_snapshot_v853

    _INSTALLED = True
