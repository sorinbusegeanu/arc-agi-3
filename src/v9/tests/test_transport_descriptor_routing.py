from __future__ import annotations

from v9.runtime.shared_batch_transport import ProducerAffinityRouter


def test_producer_affinity_ignores_sequence_and_schedule() -> None:
    router = ProducerAffinityRouter(8)
    assert router.shard_for(1234) == router.shard_for(1234)
    assert 0 <= router.shard_for(9999) < 8
