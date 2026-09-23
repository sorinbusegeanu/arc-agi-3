from __future__ import annotations

from v9.runtime.shared_batch_transport import SlabOwnership, TransportSlabPool


def test_ownership_accounting_is_disjoint() -> None:
    with TransportSlabPool(slab_count=2, slab_bytes=64, global_byte_ceiling=128) as pool:
        descriptor = pool.write(b"row", producer_id=1, start_sequence=1, end_sequence=1, rows=1)
        accounting = pool.ownership_bytes()
        assert sum(accounting.values()) == pool.tracked_bytes
        assert accounting[SlabOwnership.WORKER_OWNED.value] == 64
        pool.transfer_to_coordinator(descriptor)
        accounting = pool.ownership_bytes()
        assert sum(accounting.values()) == pool.tracked_bytes
        assert accounting[SlabOwnership.COORDINATOR_OWNED.value] == 64
