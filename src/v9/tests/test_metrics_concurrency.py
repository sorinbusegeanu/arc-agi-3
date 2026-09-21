from __future__ import annotations

from v9 import ContinuousMemoryRuntime, RuntimeConfig
from v9.memory.identity import MemoryUid


class _LockGuardedDict(dict):
    def __init__(self, rows, lock) -> None:
        super().__init__(rows)
        self._lock = lock

    def _assert_locked(self) -> None:
        assert self._lock._is_owned(), "metrics traversed mutable runtime state without the runtime lock"

    def items(self):
        self._assert_locked()
        return super().items()

    def values(self):
        self._assert_locked()
        return super().values()

    def __iter__(self):
        self._assert_locked()
        return super().__iter__()

    def copy(self):
        self._assert_locked()
        return dict(self)


def test_composed_metrics_hold_runtime_lock_across_all_layers(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False)
    )
    runtime.graph.nodes = _LockGuardedDict(runtime.graph.nodes, runtime._lock)
    runtime.grounding.states = _LockGuardedDict(runtime.grounding.states, runtime._lock)

    assert isinstance(runtime.metrics(), dict)
    assert isinstance(runtime.dashboard_metrics(), dict)


def test_dashboard_metrics_tolerate_retirement_between_payload_and_node_removal(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False)
    )
    orphan = MemoryUid(6512798212089775223, 13334826863637042881)
    runtime.graph.payloads[orphan] = {
        "reliability_trials": 1,
        "reliability_successes": 1,
    }

    try:
        result = runtime.dashboard_metrics()
    finally:
        runtime.graph.payloads.pop(orphan, None)
        runtime.close(normal=False)

    assert result["m7_strategy_success_rate"] == 0.0
