from __future__ import annotations

import pytest

from v9.runtime.shared_batch_transport import SlabOwnership, TransportSlabPool


def test_worker_cannot_reclaim_coordinator_owned_slab() -> None:
    with TransportSlabPool(slab_count=1, slab_bytes=128, global_byte_ceiling=128) as pool:
        descriptor = pool.write(b"row", producer_id=1, start_sequence=1, end_sequence=1, rows=1)
        pool.transfer_to_coordinator(descriptor)
        with pytest.raises(RuntimeError, match="ownership mismatch"):
            pool.release(descriptor, owner=SlabOwnership.WORKER_OWNED)
        pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
