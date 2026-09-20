from __future__ import annotations

from v9.runtime.derivation_merge import DerivationLeaseManager, DerivationTaskIdentity


def test_expired_lease_retries_with_new_epoch_and_attempt() -> None:
    now = [0.0]
    manager = DerivationLeaseManager(pending_limit=2, inflight_limit=1, clock=lambda: now[0])
    identity = DerivationTaskIdentity(7, 10, 1)
    assert manager.enqueue(identity)
    first = manager.lease(duration_seconds=1.0)[0]
    now[0] = 2.0
    assert manager.expire() == (identity,)
    second = manager.lease(duration_seconds=1.0)[0]
    assert second.lease_epoch > first.lease_epoch
    assert second.attempt == 2
    assert not manager.complete(first, "stale")
    assert manager.complete(second, "accepted")
