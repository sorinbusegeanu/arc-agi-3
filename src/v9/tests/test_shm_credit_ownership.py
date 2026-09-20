from __future__ import annotations

import multiprocessing as mp

import pytest

from v9.runtime.memory_worker_topology import MemoryWorkerTopology
from v9.runtime.multiprocess import ProcessTopology
from v9.runtime.shared_batch_transport import SlabOwnership, TransportSlabPool


def test_worker_cannot_reclaim_coordinator_owned_slab() -> None:
    with TransportSlabPool(slab_count=1, slab_bytes=128, global_byte_ceiling=128) as pool:
        descriptor = pool.write(b"row", producer_id=1, start_sequence=1, end_sequence=1, rows=1)
        pool.transfer_to_coordinator(descriptor)
        with pytest.raises(RuntimeError, match="ownership mismatch"):
            pool.release(descriptor, owner=SlabOwnership.WORKER_OWNED)
        pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)


def test_live_transport_and_compiled_result_subpools_share_one_global_ceiling() -> None:
    ctx = mp.get_context("spawn")
    transport = ProcessTopology(
        actors=30,
        stage_workers=2,
        shards=4,
        queue_capacity=256,
        start_method="spawn",
    )
    compiled = MemoryWorkerTopology(
        ctx,
        ingest_workers=4,
        derivation_workers=4,
        ingest_queue_capacity=256,
        derivation_queue_capacity=256,
    )
    try:
        assert transport.tracked_shm_bytes == 60 * 1024 * 1024
        assert compiled.ingest_result_pool.tracked_bytes == 48 * 1024 * 1024
        assert compiled.derivation_result_pool.tracked_bytes == 16 * 1024 * 1024
        assert transport.tracked_shm_bytes + compiled.tracked_shm_bytes <= 128 * 1024 * 1024
    finally:
        compiled.close(drain=False)
        transport.close(drain=False)


def test_compiled_result_recovery_preserves_coordinator_owned_credit() -> None:
    ctx = mp.get_context("spawn")
    compiled = MemoryWorkerTopology(
        ctx,
        ingest_workers=1,
        derivation_workers=1,
        ingest_queue_capacity=4,
        derivation_queue_capacity=4,
    )
    try:
        pool = compiled.ingest_result_pool
        descriptor = pool.write(b"compiled", producer_id=0, start_sequence=1, end_sequence=1, rows=1)
        pool.transfer_to_coordinator(descriptor)
        assert pool.recover_worker_death() == 0
        assert pool.ownership_bytes()[SlabOwnership.COORDINATOR_OWNED.value] == pool.slab_bytes
        pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
    finally:
        compiled.close(drain=False)
