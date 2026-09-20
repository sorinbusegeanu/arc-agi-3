from __future__ import annotations

import pytest

from v9.runtime.shared_batch_transport import SlabOwnership, TransportSlabDescriptor, TransportSlabPool


def test_fixed_descriptor_and_slab_round_trip() -> None:
    with TransportSlabPool(slab_count=2, slab_bytes=1024, global_byte_ceiling=2048) as pool:
        descriptor = pool.write(b"payload", producer_id=7, start_sequence=1, end_sequence=1, rows=1)
        packed = descriptor.pack()
        assert len(packed) == TransportSlabDescriptor.binary_size()
        assert TransportSlabDescriptor.unpack(packed) == descriptor
        pool.transfer_to_coordinator(descriptor)
        assert pool.read(descriptor) == b"payload"
        pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
        assert pool.ownership_bytes()[SlabOwnership.GRANT_FREE.value] == 2048


def test_slab_pool_enforces_byte_credit() -> None:
    with TransportSlabPool(slab_count=1, slab_bytes=8, global_byte_ceiling=8) as pool:
        descriptor = pool.write(b"12345678", producer_id=1, start_sequence=1, end_sequence=1, rows=1)
        with pytest.raises(BufferError, match="free credit"):
            pool.write(b"x", producer_id=1, start_sequence=2, end_sequence=2, rows=1)
        pool.release(descriptor, owner=SlabOwnership.WORKER_OWNED)
        with pytest.raises(OverflowError, match="exceeds"):
            pool.write(b"123456789", producer_id=1, start_sequence=2, end_sequence=2, rows=1)
