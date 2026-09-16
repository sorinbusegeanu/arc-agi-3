from __future__ import annotations

from v9 import ContinuousMemoryRuntime, RuntimeConfig


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
