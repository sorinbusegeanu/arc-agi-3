from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from v9.runtime import ScientificConfig
from v9.runtime.memory_governor import MemoryGovernorState, RuntimeMemoryGovernor
from v9.runtime.residency import ResidentMemoryManager


def test_accounting_categories_are_disjoint_and_reconciled() -> None:
    scientific = ScientificConfig(memory_rss_high_watermark_bytes=10000, memory_rss_hard_watermark_bytes=20000, memory_swap_high_watermark_bytes=10000)
    governor = RuntimeMemoryGovernor(scientific)
    with patch("v9.runtime.memory_governor._process_tree_memory", return_value=(900, 500, 0)), patch("v9.runtime.memory_governor._mem_available_bytes", return_value=10_000):
        snapshot = governor.sample(tracked_shm_bytes=200, snapshot_external_buffer_bytes=100, other_external_buffer_bytes=50)
    assert snapshot.accounted_working_set_bytes == 850
    assert snapshot.state is MemoryGovernorState.NORMAL


def test_memavailable_emergency_floor_forces_hard_drain() -> None:
    governor = RuntimeMemoryGovernor(ScientificConfig())
    with patch("v9.runtime.memory_governor._process_tree_memory", return_value=(10, 10, 0)), patch("v9.runtime.memory_governor._mem_available_bytes", return_value=100):
        assert governor.sample(mem_available_emergency_floor_bytes=101).state is MemoryGovernorState.HARD_PRESSURE_DRAIN


def test_residency_governor_accounts_live_transport_shm() -> None:
    captured = {}
    manager = object.__new__(ResidentMemoryManager)
    manager.runtime = SimpleNamespace(_tracked_shm_bytes=64 * 1024 * 1024)
    manager.backlog = lambda: 17
    manager.governor = SimpleNamespace(sample=lambda **kwargs: captured.update(kwargs) or "snapshot")

    assert manager.sample_memory() == "snapshot"
    assert captured == {"backlog": 17, "tracked_shm_bytes": 64 * 1024 * 1024}
