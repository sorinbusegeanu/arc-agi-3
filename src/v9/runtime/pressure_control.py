from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_governor import MemoryGovernorState


@dataclass(frozen=True, slots=True)
class MemoryPressureBudget:
    state: MemoryGovernorState
    view_scale: float
    hgt_scale: float
    active_episode_scale: float
    producer_paused: bool
    compaction_passes: int


_POLICIES = {
    MemoryGovernorState.NORMAL: MemoryPressureBudget(MemoryGovernorState.NORMAL, 1.0, 1.0, 1.0, False, 1),
    MemoryGovernorState.COMPACTING: MemoryPressureBudget(MemoryGovernorState.COMPACTING, 0.75, 0.75, 0.75, False, 2),
    MemoryGovernorState.HARD_PRESSURE_DRAIN: MemoryPressureBudget(MemoryGovernorState.HARD_PRESSURE_DRAIN, 0.40, 0.40, 0.50, True, 8),
    MemoryGovernorState.RECOVERING: MemoryPressureBudget(MemoryGovernorState.RECOVERING, 0.60, 0.60, 0.75, False, 2),
}


def pressure_budget(state: MemoryGovernorState) -> MemoryPressureBudget:
    return _POLICIES[MemoryGovernorState(state)]


def install_pressure_control(runtime_cls: type, graph_cls: type, resident_manager_cls: type, training_module: Any, pipeline_cls: type) -> None:
    if getattr(runtime_cls, "_pressure_control_installed", False):
        return

    original_training_view = graph_cls.training_view
    original_bounded_view = graph_cls.bounded_view
    original_manager_publish = resident_manager_cls._publish_telemetry
    original_manager_compact = resident_manager_cls.maybe_compact
    original_pipeline_pump_ingest = pipeline_cls.pump_ingest_tasks
    original_pipeline_pump_derivation = pipeline_cls.pump_derivation_tasks
    original_train_hgt = training_module.train_hgt_epoch
    original_transition_rows = training_module.transition_training_rows

    def graph_budget(graph: Any) -> MemoryPressureBudget:
        state = getattr(graph, "_memory_pressure_state", MemoryGovernorState.NORMAL)
        try:
            return pressure_budget(MemoryGovernorState(state))
        except ValueError:
            return _POLICIES[MemoryGovernorState.NORMAL]

    def training_view(self: Any, *, max_nodes: int = 800, max_edges: int = 4000):
        budget = graph_budget(self)
        nodes = max(8, int(max_nodes * budget.view_scale))
        edges = max(32, int(max_edges * budget.view_scale))
        self._memory_pressure_effective_view_nodes = nodes
        self._memory_pressure_effective_view_edges = edges
        return original_training_view(self, max_nodes=nodes, max_edges=edges)

    def bounded_view(self: Any, *, max_nodes: int = 800, max_edges: int = 4000, seed_uids=(), per_level_quotas=None, selection_reason: str = "bounded_runtime_view"):
        budget = graph_budget(self)
        nodes = max(8, int(max_nodes * budget.view_scale))
        edges = max(32, int(max_edges * budget.view_scale))
        quotas = None
        if per_level_quotas is not None:
            quotas = {level: max(1, int(value * budget.view_scale)) for level, value in per_level_quotas.items() if int(value) > 0}
        self._memory_pressure_effective_view_nodes = nodes
        self._memory_pressure_effective_view_edges = edges
        return original_bounded_view(
            self,
            max_nodes=nodes,
            max_edges=edges,
            seed_uids=seed_uids,
            per_level_quotas=quotas,
            selection_reason=selection_reason,
        )

    def publish_telemetry(self: Any) -> None:
        original_manager_publish(self)
        snapshot = self._last_snapshot or self.governor.sample(backlog=self.backlog())
        budget = pressure_budget(snapshot.state)
        graph = self.runtime.graph
        graph._memory_pressure_state = snapshot.state
        gauges = getattr(self.runtime.unified_telemetry, "gauges", None)
        if isinstance(gauges, dict):
            gauges.update({
                "memory_pressure_view_scale": float(budget.view_scale),
                "memory_pressure_hgt_scale": float(budget.hgt_scale),
                "memory_pressure_active_episode_scale": float(budget.active_episode_scale),
                "memory_pressure_producer_paused": int(budget.producer_paused),
                "memory_pressure_effective_view_nodes": int(getattr(graph, "_memory_pressure_effective_view_nodes", 0)),
                "memory_pressure_effective_view_edges": int(getattr(graph, "_memory_pressure_effective_view_edges", 0)),
            })

    def maybe_compact(self: Any, *, force: bool = False) -> int:
        snapshot = self.governor.sample(backlog=self.backlog())
        self._last_snapshot = snapshot
        self.runtime.graph._memory_pressure_state = snapshot.state
        return original_manager_compact(self, force=force or snapshot.state is MemoryGovernorState.HARD_PRESSURE_DRAIN)

    def runtime_pressure_policy(self: Any) -> MemoryPressureBudget:
        manager = getattr(self, "_resident_memory", None)
        if manager is None:
            return _POLICIES[MemoryGovernorState.NORMAL]
        snapshot = manager.governor.sample(backlog=manager.backlog())
        manager._last_snapshot = snapshot
        self.graph._memory_pressure_state = snapshot.state
        return pressure_budget(snapshot.state)

    def pump_ingest_tasks(self: Any) -> bool:
        policy = getattr(self.runtime, "memory_pressure_policy", lambda: _POLICIES[MemoryGovernorState.NORMAL])()
        if policy.producer_paused:
            manager = getattr(self.runtime, "_resident_memory", None)
            if manager is not None:
                manager.maybe_compact(force=True)
            return False
        return original_pipeline_pump_ingest(self)

    def pump_derivation_tasks(self: Any) -> bool:
        policy = getattr(self.runtime, "memory_pressure_policy", lambda: _POLICIES[MemoryGovernorState.NORMAL])()
        if policy.producer_paused:
            return False
        return original_pipeline_pump_derivation(self)

    def transition_rows(path: Any, *, max_rows: int | None = None, active_episode_limit: int | None = None):
        scale = float(getattr(training_module, "_memory_pressure_hgt_scale", 1.0))
        base_rows = 8192 if max_rows is None else max(1, int(max_rows))
        base_episodes = 128 if active_episode_limit is None else max(1, int(active_episode_limit))
        return original_transition_rows(
            path,
            max_rows=max(64, int(base_rows * scale)),
            active_episode_limit=max(8, int(base_episodes * scale)),
        )

    def train_hgt_epoch(runtime: Any, **kwargs: Any):
        policy = runtime_pressure_policy(runtime)
        existing = float(kwargs.get("_budget_scale", 1.0))
        kwargs["_budget_scale"] = min(existing, float(policy.hgt_scale))
        previous = float(getattr(training_module, "_memory_pressure_hgt_scale", 1.0))
        training_module._memory_pressure_hgt_scale = float(policy.active_episode_scale)
        runtime.set_telemetry_gauge("memory_pressure_hgt_scale", float(policy.hgt_scale))
        runtime.set_telemetry_gauge("memory_pressure_producer_paused", int(policy.producer_paused))
        try:
            return original_train_hgt(runtime, **kwargs)
        finally:
            training_module._memory_pressure_hgt_scale = previous

    graph_cls.training_view = training_view
    graph_cls.bounded_view = bounded_view
    resident_manager_cls._publish_telemetry = publish_telemetry
    resident_manager_cls.maybe_compact = maybe_compact
    runtime_cls.memory_pressure_policy = runtime_pressure_policy
    pipeline_cls.pump_ingest_tasks = pump_ingest_tasks
    pipeline_cls.pump_derivation_tasks = pump_derivation_tasks
    training_module.transition_training_rows = transition_rows
    training_module.train_hgt_epoch = train_hgt_epoch
    runtime_cls._pressure_control_installed = True
